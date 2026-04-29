import asyncio
import json
import math
import os
import time
from collections.abc import Awaitable, Callable, Sequence

import numpy as np
import tinker
import torch
import torch.nn.functional as F
from tinker import types as tinker_types

import wandb
from tourno.logger import get_logger, log_metrics, trace
from tourno.training.grpo import (
    get_learning_rate,
    save_checkpoint_and_get_sampling_client,
)
from tourno.training.models import get_sampling_client, get_training_client
from tourno.training.types import DPOConfig, DPOPair

# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------


def _sequence_datum(
    obs: tinker_types.ModelInput,
    sequence: tinker_types.SampledSequence,
) -> tinker_types.Datum:
    obs_tokens = obs.to_ints()
    all_tokens = [*obs_tokens, *sequence.tokens]
    weights = [0.0] * len(obs_tokens) + [1.0] * len(sequence.tokens)

    return tinker_types.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(all_tokens[1:])),
            "weights": tinker.TensorData.from_torch(torch.tensor(weights[1:])),
        },
    )


def _flatten_pairs(pairs: Sequence[DPOPair]) -> list[tinker_types.Datum]:
    datums: list[tinker_types.Datum] = []
    for pair in pairs:
        datums.extend(
            [
                _sequence_datum(pair.obs, pair.chosen),
                _sequence_datum(pair.obs, pair.rejected),
            ]
        )

    return datums


def _full_sequence(datum: tinker_types.Datum) -> tinker.ModelInput:
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
    if target_tokens.numel() == 0:
        return datum.model_input

    return datum.model_input.append_int(int(target_tokens[-1].item()))


def _batch_pairs(
    pairs: Sequence[DPOPair],
    *,
    batch_size: int,
    epoch: int,
    seed: int,
) -> list[list[DPOPair]]:
    indices = np.arange(len(pairs))
    rng = np.random.default_rng(seed + epoch)
    rng.shuffle(indices)
    return [
        [pairs[int(i)] for i in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


# ---------------------------------------------------------------------------
# DPO loss & metrics
# ---------------------------------------------------------------------------


def compute_dpo_loss(
    *,
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_log_ratio = torch.stack(
        [lp - ref for lp, ref in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - ref for lp, ref in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    chosen_rewards = beta * chosen_log_ratio
    rejected_rewards = beta * rejected_log_ratio
    margins = chosen_rewards - rejected_rewards
    losses = -F.logsigmoid(margins)
    loss = losses.mean()

    return loss, {
        "dpo/loss": float(loss.item()),
        "dpo/accuracy": float((chosen_log_ratio > rejected_log_ratio).float().mean().item()),
        "dpo/margin": float(margins.mean().item()),
        "dpo/chosen_reward": float(chosen_rewards.mean().item()),
        "dpo/rejected_reward": float(rejected_rewards.mean().item()),
    }


async def compute_reference_logprobs(
    reference_client: tinker.SamplingClient,
    datums: Sequence[tinker_types.Datum],
) -> list[torch.Tensor]:
    full_sequences = [_full_sequence(datum) for datum in datums]
    logprobs = await asyncio.gather(
        *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
    )

    return [torch.tensor(seq[1:], dtype=torch.float32) for seq in logprobs]


def compute_batch_metrics(pairs: Sequence[DPOPair]) -> dict[str, float]:
    chosen_tokens = [len(pair.chosen.tokens) for pair in pairs]
    rejected_tokens = [len(pair.rejected.tokens) for pair in pairs]
    return {
        "n_pairs": float(len(pairs)),
        "tokens/chosen": float(np.mean(chosen_tokens)) if chosen_tokens else 0.0,
        "tokens/rejected": float(np.mean(rejected_tokens)) if rejected_tokens else 0.0,
    }


def _weighted_logprob_sum(
    logprobs: torch.Tensor,
    datum: tinker_types.Datum,
) -> torch.Tensor:
    weights = datum.loss_fn_inputs["weights"].to_torch().float().to(logprobs.device)
    if logprobs.shape[0] != weights.shape[0]:
        raise ValueError(
            f"logprobs/weights length mismatch: {logprobs.shape[0]} != {weights.shape[0]}"
        )

    return torch.dot(logprobs.float(), weights)


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------


async def _get_reference_client(
    config: DPOConfig,
    training_client: tinker.TrainingClient,
) -> tinker.SamplingClient:
    if config.reference_model_path is not None:
        return await get_sampling_client(
            load_checkpoint_path=config.reference_model_path,
            base_url=config.base_url,
        )

    if config.reference_model is not None:
        return await get_sampling_client(
            base_model=config.reference_model,
            base_url=config.base_url,
        )

    return await get_sampling_client(training_client=training_client)


async def dpo_train_step(
    pairs: Sequence[DPOPair],
    training_client: tinker.TrainingClient,
    reference_client: tinker.SamplingClient,
    *,
    learning_rate: float,
    beta: float,
    num_substeps: int,
) -> dict[str, float]:
    if len(pairs) == 0:
        return {}

    k, m = divmod(len(pairs), num_substeps)
    batches = [pairs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_substeps)]
    batches = [batch for batch in batches if batch]

    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    metrics: dict[str, float] = {}

    async def _run_batch(batch: Sequence[DPOPair]):
        datums = _flatten_pairs(batch)
        ref_logprobs = await compute_reference_logprobs(reference_client, datums)
        chosen_data, rejected_data = datums[0::2], datums[1::2]
        chosen_ref, rejected_ref = ref_logprobs[0::2], ref_logprobs[1::2]

        def dpo_loss_fn(
            data: list[tinker_types.Datum],
            logprobs_list: list[torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, float]]:
            chosen_logprob_seqs, rejected_logprob_seqs = logprobs_list[0::2], logprobs_list[1::2]
            chosen_logprobs: list[torch.Tensor] = []
            rejected_logprobs: list[torch.Tensor] = []
            chosen_ref_logprobs: list[torch.Tensor] = []
            rejected_ref_logprobs: list[torch.Tensor] = []

            for idx in range(len(chosen_data)):
                chosen_logprobs.append(
                    _weighted_logprob_sum(chosen_logprob_seqs[idx], chosen_data[idx])
                )
                rejected_logprobs.append(
                    _weighted_logprob_sum(rejected_logprob_seqs[idx], rejected_data[idx])
                )
                chosen_ref_logprobs.append(_weighted_logprob_sum(chosen_ref[idx], chosen_data[idx]))
                rejected_ref_logprobs.append(
                    _weighted_logprob_sum(rejected_ref[idx], rejected_data[idx])
                )

            return compute_dpo_loss(
                chosen_logprobs=chosen_logprobs,
                rejected_logprobs=rejected_logprobs,
                chosen_ref_logprobs=chosen_ref_logprobs,
                rejected_ref_logprobs=rejected_ref_logprobs,
                beta=beta,
            )

        return await training_client.forward_backward_custom_async(datums, dpo_loss_fn)

    for batch in batches:
        fwd_bwd_future = await _run_batch(batch)
        fwd_bwd_result = await fwd_bwd_future.result_async()
        if fwd_bwd_result.metrics:
            metrics.update({k: float(v) for k, v in fwd_bwd_result.metrics.items()})

        optim_future = await training_client.optim_step_async(adam_params)
        optim_result = await optim_future.result_async()
        if optim_result.metrics:
            metrics.update({k: float(v) for k, v in optim_result.metrics.items()})

    return metrics


@trace
async def training_loop(
    config: DPOConfig,
    pairs: Sequence[DPOPair],
    on_checkpoint_save: Callable[[int, str, str], Awaitable[None] | None] | None = None,
) -> None:
    log = get_logger("dpo")
    if len(pairs) == 0:
        raise ValueError("DPO training requires at least one preference pair")

    wandb_run = None
    if config.wandb_project:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=config.model_dump(),
        )

    os.makedirs(config.log_path, exist_ok=True)
    with open(os.path.join(config.log_path, "base_model.json"), "w") as f:
        json.dump({"base_model": config.base_model}, f)

    training_client = await get_training_client(
        config.base_model,
        lora_rank=config.lora_rank,
        load_checkpoint_path=config.load_checkpoint_path,
        resume_optimizer=config.resume_optimizer,
        base_url=config.base_url,
    )
    reference_client = await _get_reference_client(config, training_client)

    step = 0
    done = False
    n_batches = math.ceil(len(pairs) / config.batch_size)
    total_steps = min(config.n_steps, config.n_epochs * n_batches)
    log.info(
        f"Starting DPO training: {len(pairs)} pairs, {n_batches} batches/epoch, {total_steps} steps"
    )

    for epoch in range(config.n_epochs):
        for batch in _batch_pairs(
            pairs,
            batch_size=config.batch_size,
            epoch=epoch,
            seed=config.seed,
        ):
            if step >= config.n_steps:
                done = True
                break

            t_start = time.time()
            lr = get_learning_rate(step, config)
            metrics: dict[str, float | int | str] = {
                "step": step,
                "epoch": epoch,
                "optim/lr": lr,
                "progress/done_frac": (step + 1) / max(total_steps, 1),
            }
            metrics.update(compute_batch_metrics(batch))
            metrics.update(
                await dpo_train_step(
                    batch,
                    training_client,
                    reference_client,
                    learning_rate=lr,
                    beta=config.beta,
                    num_substeps=config.num_substeps,
                )
            )

            completed_step = step + 1
            if (
                config.save_every > 0
                and completed_step > 0
                and completed_step % config.save_every == 0
            ):
                _, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
                    training_client,
                    completed_step,
                    config.log_path,
                    config.ttl_seconds,
                    on_checkpoint_save=on_checkpoint_save,
                )
                metrics.update(checkpoint_metrics)

            metrics["time/total"] = time.time() - t_start
            log_metrics(metrics, step=step)
            if wandb_run is not None:
                wandb.log(metrics, step=step)

            step += 1

        if done or step >= config.n_steps:
            break

    final_step = step
    already_saved = config.save_every > 0 and final_step > 0 and final_step % config.save_every == 0
    if final_step > 0 and not already_saved:
        log.info(f"Saving final DPO checkpoint at step {final_step}")
        _, final_save_metrics = await save_checkpoint_and_get_sampling_client(
            training_client,
            final_step,
            config.log_path,
            config.ttl_seconds,
            on_checkpoint_save=on_checkpoint_save,
        )
        log_metrics(final_save_metrics, step=final_step - 1)
        if wandb_run is not None:
            wandb.log(final_save_metrics, step=final_step - 1)

    if wandb_run is not None:
        wandb_run.finish()
