from typing import Awaitable, Callable

import numpy as np
import tinker
import tracelog
from tulu_types import TuluSample
from tinker_cookbook.renderers import Renderer

from pioneer.logger import get_logger, log_agent_run, trace
from pioneer.types import (
    TrainingQueue,
    TrajectoryGroup,
    TrajectoryTurn,
)


def _get_stop_token_ids(renderer: Renderer) -> set[int]:
    stop_ids: set[int] = set()
    for s in renderer.get_stop_sequences():
        if isinstance(s, int):
            stop_ids.add(s)
        elif isinstance(s, str):
            ids = renderer.tokenizer.encode(s)
            if ids:
                stop_ids.add(ids[-1])
    return stop_ids


def decode_trajectories(
    trajectories: list[list[TrajectoryTurn]],
    renderer: Renderer,
) -> list[str]:
    stop_ids = _get_stop_token_ids(renderer)
    texts: list[str] = []
    for traj in trajectories:
        tokens = list(traj[0].ac.tokens)
        if tokens and tokens[-1] in stop_ids:
            tokens = tokens[:-1]
        texts.append(renderer.tokenizer.decode(tokens))
    return texts


def _wandb_log_safe(payload: dict) -> None:
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log(payload)
    except Exception:
        pass


@trace
async def group_worker(
    id: str,
    sample: TuluSample,
    renderer: Renderer,
    sampling_client_with_step: tuple[tinker.SamplingClient, int],
    training_queue: TrainingQueue,
    *,
    get_rewards: Callable[
        [TuluSample, list[str], list[str]], Awaitable[tuple[list[float], int]]
    ],
    group_size: int = 8,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> TrajectoryGroup:
    log = get_logger(f"worker{id}")
    sampling_client, _ = sampling_client_with_step

    obs = renderer.build_generation_prompt(sample.prompt)
    completions = await sampling_client.sample_async(
        prompt=obs,
        num_samples=group_size,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    trajectories = [[TrajectoryTurn(obs=obs, ac=seq)] for seq in completions.sequences]
    completion_texts = decode_trajectories(trajectories, renderer)
    rollout_ids = [f"{sample.row_id}_{i}" for i in range(len(completion_texts))]
    rewards, judge_calls = await get_rewards(sample, completion_texts, rollout_ids)

    log.debug(f"{sample.prompt=}\n{completion_texts[0]=}\n{rewards[0]=}")
    for completion, reward, rollout_id in zip(completion_texts, rewards, rollout_ids):
        log_agent_run(
            sample.prompt + [{"content": completion, "role": "assistant"}],
            {
                "reward": reward,
                "type": "trajectory",
                "rollout_id": rollout_id,
                "prompt_id": sample.prompt_id,
                "primary_tag": sample.primary_tag,
                "row_id": sample.row_id,
            },
        )

    rewards_np = np.array(rewards)
    token_lens = [len(t.tokens) for t in completions.sequences]
    # Group-relative advantages — same formula GRPO uses (r_i - r̄).
    advantages = rewards_np - rewards_np.mean()
    abs_adv = np.abs(advantages)
    prompt_text = sample.user_query

    log.info(
        f"Rewards for {sample.prompt_id} ({sample.primary_tag or 'wildchat'}): "
        f"mean={rewards_np.mean():.3f} std={rewards_np.std():.3f}"
    )

    # Capture every model output as a row in the wandb traces table (full prompt + completion).
    for completion, reward, rollout_id, tlen in zip(
        completion_texts, rewards, rollout_ids, token_lens
    ):
        tracelog.add_rollout(
            row_id=sample.row_id,
            rollout_id=rollout_id,
            prompt_id=sample.prompt_id,
            scenario_id=sample.primary_tag or "wildchat",
            category=sample.primary_tag or "",
            reward=float(reward),
            token_len=int(tlen),
            prompt=prompt_text,
            completion=completion,
        )

    # Per-group rollout stats. wandb auto-increments its internal step counter when no step= is
    # passed — these appear as a separate high-frequency series alongside the step-level metrics.
    _wandb_log_safe(
        {
            f"category/{sample.primary_tag or 'wildchat'}/reward_mean": float(rewards_np.mean()),
            f"category/{sample.primary_tag or 'wildchat'}/reward_std": float(rewards_np.std()),
            "rollout/reward_mean": float(rewards_np.mean()),
            "rollout/reward_std": float(rewards_np.std()),
            "rollout/reward_min": float(rewards_np.min()),
            "rollout/reward_max": float(rewards_np.max()),
            "rollout/reward_p50": float(np.percentile(rewards_np, 50)),
            "rollout/reward_p95": float(np.percentile(rewards_np, 95)),
            "rollout/token_len_mean": float(np.mean(token_lens)),
            "rollout/token_len_max": float(np.max(token_lens)),
            "rollout/token_len_min": float(np.min(token_lens)),
            "rollout/token_len_p50": float(np.percentile(token_lens, 50)),
            "rollout/token_len_p95": float(np.percentile(token_lens, 95)),
            "rollout/judge_calls": int(judge_calls),
            # GRPO advantages (group-relative). At step level, frac_mixed already shows
            # "useful" groups; here we expose magnitudes for diagnostics.
            "rollout/adv_mean": float(advantages.mean()),
            "rollout/adv_std": float(advantages.std()),
            "rollout/adv_abs_mean": float(abs_adv.mean()),
            "rollout/adv_abs_max": float(abs_adv.max()),
            "rollout/frac_zero_adv": float(np.mean(abs_adv < 1e-8)),
        }
    )

    traj_group = TrajectoryGroup(
        group_size=group_size,
        trajectories=trajectories,
        rewards=rewards,
        judge_calls=judge_calls,
    )
    await training_queue.put((sampling_client_with_step[1], traj_group))
    return traj_group
