import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Iterable

import numpy as np
import tinker
import torch
import torch.nn.functional as F
from tinker import types as tinker_types
from tinker_cookbook.display import colorize_example
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

import wandb
from tourno.logger import get_logger, log_metrics, trace
from tourno.training.models import get_sampling_client, get_training_client
from tourno.training.types import DPOConfig, DPOPair
from tourno.training.utils import (
    get_learning_rate,
    save_checkpoint_and_get_sampling_client,
)


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


def compute_batch_metrics(pairs: Sequence[DPOPair]) -> dict[str, float]:
    chosen_tokens = [len(pair.chosen.tokens) for pair in pairs]
    rejected_tokens = [len(pair.rejected.tokens) for pair in pairs]
    return {
        "n_pairs": float(len(pairs)),
        "tokens/chosen": float(np.mean(chosen_tokens)) if chosen_tokens else 0.0,
        "tokens/rejected": float(np.mean(rejected_tokens)) if rejected_tokens else 0.0,
    }


def _full_sequence(datum: tinker_types.Datum) -> tinker.ModelInput:
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
    if target_tokens.numel() == 0:
        return datum.model_input

    return datum.model_input.append_int(int(target_tokens[-1].item()))


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


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------


async def dpo_train_step(
    pairs: Sequence[DPOPair],
    training_client: tinker.TrainingClient,
    reference_client: tinker.SamplingClient,
    *,
    tokenizer: Tokenizer,
    learning_rate: float,
    beta: float,
) -> dict[str, float]:
    if len(pairs) == 0:
        return {}

    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    ### Construct datums ###
    metrics: dict[str, float] = {}
    datums: list[tinker_types.Datum] = []
    for pair in pairs:
        obs = pair.obs.to_ints()
        full_chosen_tokens = [*obs, *pair.chosen.tokens]
        full_rejected_tokens = [*obs, *pair.rejected.tokens]
        weights_chosen = ([0.0] * len(obs) + [1.0] * len(pair.chosen.tokens))[1:]
        weights_rejected = ([0.0] * len(obs) + [1.0] * len(pair.rejected.tokens))[1:]

        datums.extend(
            [
                tinker_types.Datum(
                    model_input=tinker.ModelInput.from_ints(full_chosen_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(
                            torch.tensor(full_chosen_tokens[1:])
                        ),
                        "weights": tinker.TensorData.from_torch(torch.tensor(weights_chosen)),
                    },
                ),
                tinker_types.Datum(
                    model_input=tinker.ModelInput.from_ints(full_rejected_tokens[:-1]),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(
                            torch.tensor(full_rejected_tokens[1:])
                        ),
                        "weights": tinker.TensorData.from_torch(torch.tensor(weights_rejected)),
                    },
                ),
            ]
        )

    ### Log colorized training examples ###
    log = get_logger("dpo")
    log.debug("\nChosen example:\n" + colorize_example(datums[0], tokenizer))
    log.debug("\nRejected example:\n" + colorize_example(datums[1], tokenizer))

    ### Compute reference logprobs ###
    full_sequences = [_full_sequence(datum) for datum in datums]
    raw_ref_logprobs = await asyncio.gather(
        *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
    )
    ref_logprobs = [torch.tensor(seq[1:], dtype=torch.float32) for seq in raw_ref_logprobs]
    chosen_ref, rejected_ref = ref_logprobs[0::2], ref_logprobs[1::2]

    ### Mask data and compute loss ###
    def dpo_loss_fn(
        data: list[tinker_types.Datum],
        logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chosen_data, rejected_data = data[0::2], data[1::2]
        chosen_logprob_seqs, rejected_logprob_seqs = logprobs_list[0::2], logprobs_list[1::2]

        chosen_logprobs: list[torch.Tensor] = []
        rejected_logprobs: list[torch.Tensor] = []
        chosen_ref_logprobs: list[torch.Tensor] = []
        rejected_ref_logprobs: list[torch.Tensor] = []
        for idx in range(len(chosen_data)):
            chosen_datum, rejected_datum = chosen_data[idx], rejected_data[idx]
            chosen_weights, rejected_weights = (
                chosen_datum.loss_fn_inputs["weights"].to_torch().float(),
                rejected_datum.loss_fn_inputs["weights"].to_torch().float(),
            )

            chosen_logprobs.append(chosen_weights @ chosen_logprob_seqs[idx].float())
            rejected_logprobs.append(rejected_weights @ rejected_logprob_seqs[idx].float())
            chosen_ref_logprobs.append(chosen_ref[idx].float() @ chosen_weights)
            rejected_ref_logprobs.append(rejected_ref[idx].float() @ rejected_weights)

        return compute_dpo_loss(
            chosen_logprobs=chosen_logprobs,
            rejected_logprobs=rejected_logprobs,
            chosen_ref_logprobs=chosen_ref_logprobs,
            rejected_ref_logprobs=rejected_ref_logprobs,
            beta=beta,
        )

    ### Forward-backward and optimizer step ###
    fwd_bwd_future = await training_client.forward_backward_custom_async(datums, dpo_loss_fn)
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
    train_data: Iterable[tuple[int, Sequence[DPOPair]]],
    on_checkpoint_save: Callable[[int, str, str], Awaitable[None] | None] | None = None,
) -> None:
    log = get_logger("dpo")
    log.info("Starting DPO training...")

    wandb_run = None
    if config.wandb_project:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=config.model_dump(),
        )

    ### Write base model metadata ###
    os.makedirs(config.log_path, exist_ok=True)
    with open(os.path.join(config.log_path, "base_model.json"), "w") as f:
        json.dump({"base_model": config.base_model}, f)

    ### Initialize training client and reference client ###
    tokenizer = get_tokenizer(config.base_model)
    training_client = await get_training_client(
        config.base_model,
        lora_rank=config.lora_rank,
        load_checkpoint_path=config.load_checkpoint_path,
        resume_optimizer=config.resume_optimizer,
        base_url=config.base_url,
    )
    reference_client = await _get_reference_client(config, training_client)

    ### Main training loop ###
    completed_step = 0
    for step, (epoch, batch) in enumerate(train_data):
        if step >= config.n_steps:
            break
        if len(batch) != config.batch_size:
            raise ValueError(
                f"Data Loader batch size mismatch, expected {config.batch_size} but got"
                f" {len(batch)}"
            )

        t_start = time.time()

        ### Compute learning rate and initial metrics ###
        lr = get_learning_rate(step, config)
        metrics: dict[str, float | int | str] = {
            "step": step,
            "epoch": epoch,
            "optim/lr": lr,
            "progress/done_frac": (step + 1) / max(config.n_steps, 1),
        }
        metrics.update(compute_batch_metrics(batch))

        ### Run train step ###
        metrics.update(
            await dpo_train_step(
                batch,
                training_client,
                reference_client,
                tokenizer=tokenizer,
                beta=config.beta,
                learning_rate=lr,
            )
        )

        ### Save checkpoint and log to wandb ###
        completed_step = step + 1
        if config.save_every > 0 and completed_step > 0 and completed_step % config.save_every == 0:
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

    ### Save final checkpoint and log to wandb ###
    final_step = completed_step
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
