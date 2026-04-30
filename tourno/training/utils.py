import asyncio
import inspect
import json
import math
import os
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

import numpy as np
import scipy.signal
import tinker
import torch
from tinker import types as tinker_types

from tourno.logger import get_logger


class LearningRateConfig(Protocol):
    learning_rate: float
    lr_schedule: str
    lr_warmup_steps: int
    lr_min: float
    n_steps: int


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


def get_learning_rate(step: int, config: LearningRateConfig) -> float:
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


async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    step: int,
    log_path: str,
    ttl_seconds: int | None = None,
    on_checkpoint_save: Callable[[int, str, str], Awaitable[None] | None] | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    log = get_logger()
    name = f"{step:06d}"
    state_future = await training_client.save_state_async(name, ttl_seconds=ttl_seconds)
    sampler_future = await training_client.save_weights_for_sampler_async(
        name, ttl_seconds=ttl_seconds
    )
    state_result = await state_future.result_async()
    sampler_result = await sampler_future.result_async()

    paths = {"state_path": state_result.path, "sampler_path": sampler_result.path}
    checkpoint = {"name": name, "step": step, **paths}
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(checkpoint) + "\n")

    log.info(f"Saved checkpoint at step {step}: {paths}")
    if on_checkpoint_save is not None:
        result = on_checkpoint_save(step, name, sampler_result.path)
        if inspect.isawaitable(result):
            await result

    metrics: dict[str, Any] = {"checkpoint": name}
    return training_client.create_sampling_client(sampler_result.path), metrics
