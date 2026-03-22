import asyncio
import json
import math
import os
import time
from typing import Any, Callable, cast

import numpy as np
import scipy.signal
import tinker
import torch
import wandb
from tinker import types as tinker_types
from tinker_cookbook.display import colorize_example
from tinker_cookbook.tokenizer_utils import get_tokenizer

from pioneer.logger import get_logger, log_metrics, trace
from pioneer.models import get_sampling_client, get_training_client
from pioneer.types import (
    TrainConfig,
    TrainingQueue,
    TrajectoryGroup,
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
# KL penalty & metrics
# ---------------------------------------------------------------------------


def _discounted_future_sum(x: np.ndarray, gamma: float) -> np.ndarray:
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1].astype(x.dtype)


def compute_kl_metrics(
    datums: list[tinker_types.Datum],
    training_logprobs: list[torch.Tensor],
) -> dict[str, float]:
    all_diffs: list[torch.Tensor] = []
    all_sampling_logprobs: list[torch.Tensor] = []

    for datum, tlp in zip(datums, training_logprobs):
        slp = datum.loss_fn_inputs["logprobs"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        slp_masked = slp[mask]
        tlp_masked = tlp[mask]
        if len(slp_masked) > 0:
            all_diffs.append(slp_masked - tlp_masked)
            all_sampling_logprobs.append(slp_masked)

    if not all_diffs:
        return {}

    flat_diffs = torch.cat(all_diffs)
    flat_slp = torch.cat(all_sampling_logprobs)

    return {
        "kl/sample_train_v1": flat_diffs.mean().item(),
        "kl/sample_train_v2": 0.5 * (flat_diffs**2).mean().item(),
        "entropy": -flat_slp.mean().item(),
    }


async def compute_post_kl(
    datums: list[tinker_types.Datum],
    post_sampling_client: tinker.SamplingClient,
) -> dict[str, float]:
    full_seqs = [
        d.model_input.append_int(cast(int, d.loss_fn_inputs["target_tokens"].data[-1]))
        for d in datums
    ]
    new_logprobs_list = await asyncio.gather(
        *[post_sampling_client.compute_logprobs_async(seq) for seq in full_seqs]
    )

    flat_diffs: list[torch.Tensor] = []
    for datum, new_lp in zip(datums, new_logprobs_list):
        prev_lp = datum.loss_fn_inputs["logprobs"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        diff = (prev_lp - torch.tensor(new_lp[1:]))[mask]
        if len(diff) > 0:
            flat_diffs.append(diff)

    all_diffs = torch.cat(flat_diffs)

    return {
        "kl/pre_post_v1": all_diffs.mean().item(),
        "kl/pre_post_v2": 0.5 * (all_diffs**2).mean().item(),
    }


async def incorporate_kl_penalty(
    datums: list[tinker_types.Datum],
    reference_client: tinker.SamplingClient,
    kl_coef: float,
    kl_discount_factor: float = 0.0,
) -> dict[str, float]:
    full_seqs = [
        d.model_input.append_int(cast(int, d.loss_fn_inputs["target_tokens"].data[-1]))
        for d in datums
    ]
    ref_logprobs_list = await asyncio.gather(
        *[reference_client.compute_logprobs_async(seq) for seq in full_seqs]
    )

    sampling_logprobs = [d.loss_fn_inputs["logprobs"].to_torch() for d in datums]
    masks = [d.loss_fn_inputs["mask"].to_torch().float() for d in datums]

    kl_diffs = [
        (slp - torch.tensor(rlp[1:])) * m
        for slp, rlp, m in zip(sampling_logprobs, ref_logprobs_list, masks)
    ]
    avg_kl = sum(d.sum() for d in kl_diffs) / sum(m.sum() for m in masks)

    for datum, kl_diff, mask in zip(datums, kl_diffs, masks):
        kl_advantage = kl_coef * mask * (avg_kl - kl_diff)
        if kl_discount_factor > 0:
            kl_advantage = torch.tensor(
                _discounted_future_sum(kl_advantage.numpy(), kl_discount_factor)
            )

        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantage
        )

    return {"kl/policy_vs_reference": float(avg_kl)}


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------


def _remove_mask(datum: tinker_types.Datum) -> tinker_types.Datum:
    return tinker_types.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def get_learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.lr_warmup_steps:
        return config.learning_rate * (step + 1) / config.lr_warmup_steps

    if config.lr_schedule == "none":
        return config.learning_rate

    decay_steps = config.n_steps - config.lr_warmup_steps
    t = (step - config.lr_warmup_steps) / max(decay_steps, 1)

    if config.lr_schedule == "linear":
        return config.lr_min + (config.learning_rate - config.lr_min) * (1 - t)

    return config.lr_min + (config.learning_rate - config.lr_min) * 0.5 * (
        1 + math.cos(math.pi * t)
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


async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    step: int,
    log_path: str,
    save_every: int,
    ttl_seconds: int | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    log = get_logger()
    metrics: dict[str, Any] = {}
    if save_every > 0 and step > 0 and step % save_every == 0:
        name = f"{step:06d}"
        state_future = await training_client.save_state_async(name, ttl_seconds=ttl_seconds)
        sampler_future = await training_client.save_weights_for_sampler_async(
            name, ttl_seconds=ttl_seconds
        )
        state_result = await state_future.result_async()
        sampler_result = await sampler_future.result_async()

        paths = {"state_path": state_result.path, "sampler_path": sampler_result.path}
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
            f.write(json.dumps({"name": name, "step": step, **paths}) + "\n")

        log.info(f"Saved checkpoint at step {step}: {paths}")
        metrics["checkpoint"] = name
        return training_client.create_sampling_client(sampler_result.path), metrics

    return await training_client.save_weights_and_get_sampling_client_async(), metrics


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
    config: TrainConfig,
    training_queue: TrainingQueue,
    update_sampling_client: Callable[[tinker.SamplingClient, int], None],
) -> None:
    log = get_logger("train")
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
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        step=0,
        log_path=config.log_path,
        save_every=config.save_every,
        ttl_seconds=config.ttl_seconds,
    )
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
        sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
            training_client,
            step + 1,
            config.log_path,
            config.save_every,
            config.ttl_seconds,
        )
        update_sampling_client(sampling_client, step)
        metrics.update(checkpoint_metrics)

        ### Post-update KL ###
        if config.compute_post_kl:
            post_kl_metrics = await compute_post_kl(datums, sampling_client)
            metrics.update(post_kl_metrics)

        metrics["time/total"] = time.time() - t_start
        log_metrics(metrics, step=step)
        if wandb_run is not None:
            wandb.log(metrics, step=step)

        step += 1

    if wandb_run is not None:
        wandb_run.finish()
