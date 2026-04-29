from typing import Awaitable, Callable

import numpy as np
import tinker
from tinker_cookbook.renderers import Renderer
from truthfulqa_types import TruthfulQASample

from pioneer.logger import current_step, get_logger, log_trace, trace
from pioneer.types import (
    TrainingQueue,
    TrajectoryGroup,
    TrajectoryTurn,
)


def _serialize_prompt(prompt: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in prompt)


def decode_trajectories(
    trajectories: list[list[TrajectoryTurn]],
    renderer: Renderer,
) -> list[str]:
    stops = renderer.get_stop_sequences()
    stop_token_ids = {s for s in stops if isinstance(s, int)}
    stop_strings = [s for s in stops if isinstance(s, str)]
    texts: list[str] = []
    for traj in trajectories:
        tokens = list(traj[0].ac.tokens)
        if tokens and tokens[-1] in stop_token_ids:
            tokens = tokens[:-1]
        text = renderer.tokenizer.decode(tokens, skip_special_tokens=True)
        for s in stop_strings:
            if text.endswith(s):
                text = text[: -len(s)]
                break
        texts.append(text)

    return texts


@trace
async def group_worker(
    id: str,
    sample: TruthfulQASample,
    renderer: Renderer,
    sampling_client_with_step: tuple[tinker.SamplingClient, int],
    training_queue: TrainingQueue,
    *,
    get_rewards: Callable[
        [TruthfulQASample, list[str], list[str]], Awaitable[tuple[list[float], int]]
    ],
    group_size: int = 8,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> TrajectoryGroup:
    log = get_logger(f"worker{id}")
    sampling_client, sampling_client_step = sampling_client_with_step

    obs = renderer.build_generation_prompt(sample.prompt)
    completions = await sampling_client.sample_async(
        prompt=obs,
        num_samples=group_size,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )

    trajectories = [[TrajectoryTurn(obs=obs, ac=seq)] for seq in completions.sequences]

    completion_texts = decode_trajectories(trajectories, renderer)
    rollout_ids = [f"{sample.row_id}_{i}" for i in range(len(completion_texts))]

    # Stamp every judge trace row spawned during reward computation with the
    # current training step. Worker tasks each have their own context, so this
    # never leaks across sibling samples.
    current_step.set(sampling_client_step)
    rewards, judge_calls = await get_rewards(sample, completion_texts, rollout_ids)

    log.debug(f"{sample.prompt=}\n{completion_texts[0]=}\n{rewards[0]=}")
    prompt_str = _serialize_prompt(sample.prompt)
    for traj, completion, reward, rollout_id in zip(
        trajectories, completion_texts, rewards, rollout_ids
    ):
        log_trace(
            "traces/rollouts",
            step=sampling_client_step,
            row_id=sample.row_id,
            rollout_id=rollout_id,
            prompt_id=sample.prompt_id,
            category=sample.category,
            reward=reward,
            token_len=len(traj[0].ac.tokens),
            prompt=prompt_str,
            completion=completion,
        )

    rewards_np = np.array(rewards)
    log.debug(f"{rewards=}")
    log.info(
        f"Rewards for {sample.prompt_id}: mean={rewards_np.mean():.3f} std={rewards_np.std():.3f}"
    )

    traj_group = TrajectoryGroup(
        group_size=group_size,
        trajectories=trajectories,
        rewards=rewards,
        judge_calls=judge_calls,
    )

    await training_queue.put((sampling_client_step, traj_group))
    return traj_group
