import inspect
import json
import os
import time
from typing import Any, Awaitable, Callable

import numpy as np
import tinker
import torch
from tinker import types as tinker_types
from tinker_cookbook.display import colorize_example
from tinker_cookbook.tokenizer_utils import get_tokenizer

import wandb
from tourno.logger import get_logger, log_metrics, trace
from tourno.training.models import get_sampling_client, get_training_client
from tourno.training.types import (
    GRPOConfig,
    TrainingQueue,
    TrajectoryGroup,
)
from tourno.training.utils import (
    compute_kl_metrics,
    compute_post_kl,
    get_learning_rate,
    incorporate_kl_penalty,
    save_checkpoint_and_get_sampling_client,
)

# ---------------------------------------------------------------------------
# Datum construction (multi-turn)
# ---------------------------------------------------------------------------


def process_trajectory_group(traj_group: TrajectoryGroup) -> list[tinker_types.Datum]:
    rewards = torch.tensor(traj_group.rewards, dtype=torch.float32)
    advantages = rewards - rewards.mean()

    if torch.all(advantages == 0):
        return []

    datums: list[tinker_types.Datum] = []
    for trajectory, advantage in zip(traj_group.trajectories, advantages):
        all_tokens: list[int] = []
        logprobs: list[float] = []
        advs: list[float] = []
        mask: list[float] = []

        for turn in trajectory:
            obs_ints = turn.obs.to_ints()
            all_tokens.extend(obs_ints)
            all_tokens.extend(turn.ac.tokens)

            n_obs, n_ac = len(obs_ints), len(turn.ac.tokens)
            logprobs.extend([0.0] * n_obs + list(turn.ac.logprobs))
            advs.extend([0.0] * n_obs + [advantage] * n_ac)
            mask.extend([0.0] * n_obs + [1.0] * n_ac)

        model_input = tinker.ModelInput.from_ints(all_tokens[:-1])
        target_tokens = all_tokens[1:]

        n = model_input.length
        assert n == len(target_tokens) == len(logprobs) - 1 == len(advs) - 1 == len(mask) - 1

        datums.append(
            tinker_types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs[1:])),
                    "advantages": tinker.TensorData.from_torch(torch.tensor(advs[1:])),
                    "mask": tinker.TensorData.from_torch(torch.tensor(mask[1:])),
                },
            )
        )

    return datums


# ---------------------------------------------------------------------------
# Trajectory & reward metrics
# ---------------------------------------------------------------------------


def compute_batch_metrics(
    trajectory_groups: list[TrajectoryGroup],
    n_datums: int,
) -> dict[str, float]:
    all_rewards = [r for tg in trajectory_groups for r in tg.rewards]
    rewards_arr = np.array(all_rewards)

    flat_turns = [turn for tg in trajectory_groups for traj in tg.trajectories for turn in traj]
    ac_tokens = [len(turn.ac.tokens) for turn in flat_turns]
    ob_tokens = [turn.obs.length for turn in flat_turns]
    turns_per_traj = [len(traj) for tg in trajectory_groups for traj in tg.trajectories]

    n_groups = len(trajectory_groups)
    n_mixed = n_good = n_bad = 0
    for tg in trajectory_groups:
        if all(r == tg.rewards[0] for r in tg.rewards):
            if tg.rewards[0] >= 0.5:
                n_good += 1
            else:
                n_bad += 1
        else:
            n_mixed += 1

    return {
        "reward/mean": float(rewards_arr.mean()),
        "reward/std": float(rewards_arr.std()),
        "reward/min": float(rewards_arr.min()),
        "reward/max": float(rewards_arr.max()),
        "n_datums": n_datums,
        "ac_tokens_per_turn": sum(ac_tokens) / max(len(ac_tokens), 1),
        "ob_tokens_per_turn": sum(ob_tokens) / max(len(ob_tokens), 1),
        "turns_per_episode": sum(turns_per_traj) / max(len(turns_per_traj), 1),
        "total_ac_tokens": sum(ac_tokens),
        "total_ob_tokens": sum(ob_tokens),
        "by_group/frac_mixed": n_mixed / max(n_groups, 1),
        "by_group/frac_all_good": n_good / max(n_groups, 1),
        "by_group/frac_all_bad": n_bad / max(n_groups, 1),
    }


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------


def _remove_mask(datum: tinker_types.Datum) -> tinker_types.Datum:
    return tinker_types.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


async def pull_minibatch(
    training_queue: TrainingQueue,
    step: int,
    minibatch_size: int,
    max_steps_off_policy: int,
) -> list[TrajectoryGroup] | None:
    log = get_logger()
    trajectory_groups: list[TrajectoryGroup] = []

    while len(trajectory_groups) < minibatch_size:
        item = await training_queue.get()
        if item is None:
            log.info("Received shutdown signal")
            return None

        sampled_at_step, traj_group = item
        if step - sampled_at_step > max_steps_off_policy:
            log.warning(f"Step {step}: dropping stale group (sampled at step {sampled_at_step})")
            continue

        trajectory_groups.append(traj_group)

    return trajectory_groups


async def train_step(
    datums: list[tinker_types.Datum],
    training_client: tinker.TrainingClient,
    *,
    learning_rate: float,
    num_substeps: int,
    loss_fn: str,
    loss_fn_config: dict[str, Any] | None,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    k, m = divmod(len(datums), num_substeps)
    batches = [datums[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_substeps)]
    if not batches:
        return [], {}

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logprobs: list[torch.Tensor] = []
    optim_metrics: dict[str, float] = {}

    fwd_bwd_future = await training_client.forward_backward_async(
        [_remove_mask(d) for d in batches[0]], loss_fn=loss_fn, loss_fn_config=loss_fn_config
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        if i + 1 < len(batches):
            next_fwd_bwd = await training_client.forward_backward_async(
                [_remove_mask(d) for d in batches[i + 1]],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            next_optim = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd = None
            next_optim = None

        fwd_bwd_result = await fwd_bwd_future.result_async()
        training_logprobs.extend(
            out["logprobs"].to_torch() for out in fwd_bwd_result.loss_fn_outputs
        )
        optim_result = await optim_future.result_async()
        if optim_result.metrics:
            optim_metrics.update(optim_result.metrics)

        if next_fwd_bwd is not None and next_optim is not None:
            fwd_bwd_future = next_fwd_bwd
            optim_future = next_optim

    return training_logprobs, optim_metrics


@trace
async def training_loop(
    config: GRPOConfig,
    training_queue: TrainingQueue,
    update_sampling_client: Callable[[tinker.SamplingClient, int], None],
    on_checkpoint_save: Callable[[int, str, str], Awaitable[None] | None] | None = None,
    extra_metrics_fn: Callable[
        [int, list[TrajectoryGroup]], dict[str, Any] | Awaitable[dict[str, Any]]
    ]
    | None = None,
    validation_fn: Callable[[int], Awaitable[dict[str, Any]]] | None = None,
) -> None:
    log = get_logger("grpo")
    log.info("Starting training loop...")

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

    ### Initialize training client and set sampling client ###
    tokenizer = get_tokenizer(config.base_model)
    training_client = await get_training_client(
        config.base_model,
        lora_rank=config.lora_rank,
        load_checkpoint_path=config.load_checkpoint_path,
        resume_optimizer=config.resume_optimizer,
        base_url=config.base_url,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    update_sampling_client(sampling_client, -1)

    ### Initialize reference client if kl_coef > 0 ###
    reference_client: tinker.SamplingClient | None = None
    if config.kl_coef > 0:
        if config.kl_reference_model:
            reference_client = await get_sampling_client(
                base_model=config.kl_reference_model,
                base_url=config.base_url,
            )
        else:
            reference_client = await get_sampling_client(
                base_model=config.base_model,
                base_url=config.base_url,
            )

    ### Main training loop ###
    step = 0
    while step < config.n_steps:
        t_start = time.time()

        ### Pull batch of trajectory groups ###
        trajectory_groups = await pull_minibatch(
            training_queue,
            step,
            minibatch_size=config.batch_size,
            max_steps_off_policy=config.max_steps_off_policy,
        )

        if trajectory_groups is None:
            return

        ### Process trajectory groups into datums ###
        datums: list[tinker_types.Datum] = []
        for tg in trajectory_groups:
            datums.extend(process_trajectory_group(tg))

        if not datums:
            log.warning(f"Step {step}: all advantages zero, skipping")
            # TODO: in the future we should requeue the bad trajectory groups into
            # a sampling queue
            continue

        if len(datums) != config.batch_size * config.group_size:
            log.warning(
                f"Step {step}: expected {config.batch_size * config.group_size} datums, got"
                f" {len(datums)}"
            )

        step_judge_calls = sum(tg.judge_calls for tg in trajectory_groups)

        lr = get_learning_rate(step, config)
        metrics: dict[str, Any] = {
            "step": step,
            "optim/lr": lr,
            "progress/done_frac": (step + 1) / config.n_steps,
            "judge_calls": step_judge_calls,
        }
        metrics.update(compute_batch_metrics(trajectory_groups, len(datums)))

        ### Mutate datums with KL penalty if enabled ###
        if config.kl_coef > 0 and reference_client is not None:
            kl_metrics = await incorporate_kl_penalty(
                datums, reference_client, config.kl_coef, config.kl_discount_factor
            )
            metrics.update(kl_metrics)

        ### Log colorized training examples ###
        log.debug("\n" + colorize_example(datums[0], tokenizer, key="advantages"))

        ### Train step ###
        training_logprobs, optim_metrics = await train_step(
            datums,
            training_client,
            learning_rate=lr,
            num_substeps=config.num_substeps,
            loss_fn=config.loss_fn,
            loss_fn_config=config.loss_fn_config,
        )

        metrics.update(optim_metrics)
        metrics.update(compute_kl_metrics(datums, training_logprobs))

        ### Checkpoint + update sampling client ###
        completed_step = step + 1
        should_save_checkpoint = (
            config.save_every > 0 and completed_step > 0 and completed_step % config.save_every == 0
        )
        if should_save_checkpoint:
            sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
                training_client,
                completed_step,
                config.log_path,
                config.ttl_seconds,
                on_checkpoint_save=on_checkpoint_save,
            )
        else:
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            checkpoint_metrics = {}
        update_sampling_client(sampling_client, step)
        metrics.update(checkpoint_metrics)

        ### Post-update KL ###
        if config.compute_post_kl:
            post_kl_metrics = await compute_post_kl(datums, sampling_client)
            metrics.update(post_kl_metrics)

        ### Caller-supplied extra metrics + validation ###
        if extra_metrics_fn is not None:
            extra = extra_metrics_fn(step, trajectory_groups)
            if inspect.isawaitable(extra):
                extra = await extra
            if extra:
                metrics.update(extra)
        if validation_fn is not None and should_save_checkpoint:
            val_metrics = await validation_fn(completed_step)
            if val_metrics:
                metrics.update(val_metrics)

        metrics["time/total"] = time.time() - t_start
        log_metrics(metrics, step=step)
        if wandb_run is not None:
            wandb.log(metrics, step=step)

        step += 1

    final_step = config.n_steps
    already_saved = config.save_every > 0 and final_step > 0 and final_step % config.save_every == 0
    if final_step > 0 and not already_saved:
        log.info(f"Saving final checkpoint at step {final_step}")
        sampling_client, final_save_metrics = await save_checkpoint_and_get_sampling_client(
            training_client,
            final_step,
            config.log_path,
            config.ttl_seconds,
            on_checkpoint_save=on_checkpoint_save,
        )
        update_sampling_client(sampling_client, final_step - 1)
        log_metrics(final_save_metrics, step=final_step - 1)
        if wandb_run is not None:
            wandb.log(final_save_metrics, step=final_step - 1)

    if wandb_run is not None:
        wandb_run.finish()
