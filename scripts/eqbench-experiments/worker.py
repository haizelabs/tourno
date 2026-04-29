from typing import Awaitable, Callable

import numpy as np
import tinker
from eqbench_types import EQBenchSample
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


@trace
async def group_worker(
    id: str,
    sample: EQBenchSample,
    renderer: Renderer,
    sampling_client_with_step: tuple[tinker.SamplingClient, int],
    training_queue: TrainingQueue,
    *,
    get_rewards: Callable[
        [EQBenchSample, list[str], list[str]], Awaitable[tuple[list[float], int]]
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
                "scenario_id": sample.scenario_id,
                "task_type": sample.task_type,
                "row_id": sample.row_id,
            },
        )

    rewards_np = np.array(rewards)
    log.info(
        f"Rewards for {sample.prompt_id} ({sample.task_type}): "
        f"mean={rewards_np.mean():.3f} std={rewards_np.std():.3f}"
    )

    traj_group = TrajectoryGroup(
        group_size=group_size,
        trajectories=trajectories,
        rewards=rewards,
        judge_calls=judge_calls,
    )
    await training_queue.put((sampling_client_with_step[1], traj_group))
    return traj_group
