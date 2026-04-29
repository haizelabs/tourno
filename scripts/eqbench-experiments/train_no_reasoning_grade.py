import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import tinker
from data import EQBenchDataLoader
from dotenv import load_dotenv
from eqbench_types import EQBenchSample, TaskType
from judges import EQBenchPairwiseJudge, EQBenchPointwiseJudge
from openai import AsyncOpenAI
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from worker import group_worker

from pioneer.logger import get_logger, init_docent, log_agent_run, setup, trace
from pioneer.loop import training_loop
from pioneer.types import TrainConfig, TrainingQueue
from tourno.tournament import (
    batched_elo_reward_fn,
    pointwise_reward_fn,
)

load_dotenv()


async def get_eqbench_pointwise_rewards(
    sample: EQBenchSample,
    completions: list[str],
    judge: EQBenchPointwiseJudge,
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    rewards = await pointwise_reward_fn(
        sample.scenario_text,
        completions,
        lambda _prompt, comps: judge(sample.scenario_text, comps, sample.task_type),
    )

    for i, completion in enumerate(completions):
        log_agent_run(
            sample.prompt + [{"content": completion, "role": "assistant"}],
            {
                "type": "pointwise_reward",
                "prompt_id": sample.prompt_id,
                "scenario_id": sample.scenario_id,
                "task_type": sample.task_type,
                "row_id": sample.row_id,
                "rollout_id": rollout_ids[i],
                "reward": rewards[i],
            },
        )

    return rewards, len(completions)


async def get_eqbench_pairwise_rewards(
    sample: EQBenchSample,
    completions: list[str],
    judge: EQBenchPairwiseJudge,
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    task_type: TaskType = sample.task_type
    rewards, n_calls = await batched_elo_reward_fn(
        sample.scenario_text,
        completions,
        judge_fn=lambda pairwise_samples: judge(pairwise_samples, task_type),
    )

    for i, completion in enumerate(completions):
        log_agent_run(
            sample.prompt + [{"content": completion, "role": "assistant"}],
            {
                "type": "pairwise_reward",
                "prompt_id": sample.prompt_id,
                "scenario_id": sample.scenario_id,
                "task_type": sample.task_type,
                "row_id": sample.row_id,
                "rollout_id": rollout_ids[i],
                "reward": rewards[i],
            },
        )

    return rewards, n_calls


async def get_eqbench_mixture_rewards(
    sample: EQBenchSample,
    completions: list[str],
    pointwise_judge: EQBenchPointwiseJudge,
    pairwise_judge: EQBenchPairwiseJudge,
    pairwise_alpha: float,
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    (pw, pw_calls), (pair, pair_calls) = await asyncio.gather(
        get_eqbench_pointwise_rewards(sample, completions, pointwise_judge, rollout_ids),
        get_eqbench_pairwise_rewards(sample, completions, pairwise_judge, rollout_ids),
    )

    r_point = np.array(pw)
    r_pair = np.array(pair)
    mixed = r_point + np.exp(-pairwise_alpha * r_point.mean()) * r_pair

    return mixed.tolist(), pw_calls + pair_calls


def make_training_loop(
    config: TrainConfig,
    training_queue: TrainingQueue,
    update_sampling_client: Callable[[tinker.SamplingClient, int], None],
) -> asyncio.Task:
    log = get_logger()

    def _on_done(task: asyncio.Task) -> None:
        if task.exception() is not None:
            log.error(f"Training loop failed: {task.exception()}")
        else:
            log.info("Training loop finished")

    task = asyncio.create_task(training_loop(config, training_queue, update_sampling_client))
    task.add_done_callback(_on_done)
    return task


@trace
async def main(
    config: TrainConfig,
    get_rewards: Callable[
        [EQBenchSample, list[str], list[str]], Awaitable[tuple[list[float], int]]
    ],
    data_loader: EQBenchDataLoader,
    num_workers: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    renderer_name: str | None,
) -> None:
    log = get_logger()

    tokenizer = get_tokenizer(config.base_model)
    if renderer_name is None:
        renderer_name = get_recommended_renderer_name(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer)
    log.info(f"Renderer: {renderer_name}, model: {config.base_model}")

    sampling_client_with_step: tuple[tinker.SamplingClient, int] | None = None
    sampling_client_ready = asyncio.Event()

    def update_sampling_client(client: tinker.SamplingClient, step: int) -> None:
        nonlocal sampling_client_with_step
        sampling_client_with_step = (client, step)
        if not sampling_client_ready.is_set():
            sampling_client_ready.set()

    training_queue: TrainingQueue = asyncio.Queue()
    training_loop_task = make_training_loop(config, training_queue, update_sampling_client)
    work_queue: asyncio.Queue[EQBenchSample | None] = asyncio.Queue(maxsize=num_workers)

    log.info("Waiting for sampling client...")
    await asyncio.wait(
        [asyncio.create_task(sampling_client_ready.wait()), training_loop_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if training_loop_task.done():
        log.error("Training loop exited before providing a sampling client")
        training_loop_task.result()
        return

    assert sampling_client_with_step is not None
    log.info("Sampling client ready, starting workers")

    async def producer() -> None:
        row_id = 0
        async for _, batch in data_loader:
            for sample in batch:
                sample.row_id = row_id
                await work_queue.put(sample)
                row_id += 1

    async def worker(wid: int) -> None:
        while True:
            sample = await work_queue.get()
            if sample is None:
                break
            try:
                await group_worker(
                    str(wid),
                    sample=sample,
                    renderer=renderer,
                    sampling_client_with_step=sampling_client_with_step,
                    training_queue=training_queue,
                    get_rewards=get_rewards,
                    group_size=group_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception:
                log.exception(f"Worker {wid} error", exc_info=True)
                raise

    producer_task = asyncio.create_task(producer())
    worker_tasks = [asyncio.create_task(worker(i)) for i in range(num_workers)]
    all_tasks = [training_loop_task, producer_task, *worker_tasks]

    try:
        done, _ = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc:
                raise exc
    finally:
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        log.info("Training complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EQ-Bench 3 RL training")

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument(
        "--judge-type",
        type=str,
        choices=["pairwise", "pointwise", "mixture"],
        default="pointwise",
    )
    p.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    p.add_argument("--pairwise-alpha", type=float, default=3.0)

    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-filter", type=str, default=None)
    p.add_argument("--docent-collection", type=str, default=None)

    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--renderer", type=str, default=None)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument(
        "--n-epochs",
        type=float,
        default=None,
        help="If set, overrides --n-steps based on dataset size",
    )

    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--log-path", type=str, default="./eqbench3-rl")
    p.add_argument("--kl-reference-model", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--loss-fn", type=str, default="importance_sampling")
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--kl-discount-factor", type=float, default=0.0)
    p.add_argument("--compute-post-kl", action="store_true")
    p.add_argument("--max-steps-off-policy", type=int, default=3)
    p.add_argument("--wandb-project", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(level=getattr(logging, args.log_level.upper()), filter_pattern=args.log_filter)

    data_loader = EQBenchDataLoader(max_length=args.max_samples)
    n_steps = args.n_steps
    if args.n_epochs is not None:
        n_steps = math.ceil(args.n_epochs * data_loader.num_rows / args.batch_size)

    config = TrainConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        judge_type=args.judge_type,
        judge_model=args.judge_model,
        pairwise_alpha=args.pairwise_alpha,
        kl_reference_model=args.kl_reference_model,
        base_url=args.base_url,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        group_size=args.group_size,
        loss_fn=args.loss_fn,
        kl_coef=args.kl_coef,
        kl_discount_factor=args.kl_discount_factor,
        compute_post_kl=args.compute_post_kl,
        max_steps_off_policy=args.max_steps_off_policy,
        n_steps=n_steps,
        save_every=args.save_every,
        log_path=args.log_path,
        wandb_project=args.wandb_project,
    )
    config.log_path = (Path(config.log_path) / config.run_name).as_posix()

    docent_collection = args.docent_collection
    if not docent_collection and os.environ.get("DOCENT_API_KEY"):
        docent_collection = config.run_name
    if docent_collection:
        init_docent(docent_collection)

    grader_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if args.judge_type == "mixture":
        point_judge = EQBenchPointwiseJudge(client=grader_client, model=args.judge_model)
        pair_judge = EQBenchPairwiseJudge(client=grader_client, model=args.judge_model)
        get_rewards = (  # noqa: E731
            lambda sample, completions, rollout_ids: get_eqbench_mixture_rewards(
                sample,
                completions,
                point_judge,
                pair_judge,
                config.pairwise_alpha,
                rollout_ids,
            )
        )
    elif args.judge_type == "pairwise":
        judge = EQBenchPairwiseJudge(client=grader_client, model=args.judge_model)
        get_rewards = (  # noqa: E731
            lambda sample, completions, rollout_ids: get_eqbench_pairwise_rewards(
                sample, completions, judge, rollout_ids
            )
        )
    else:
        judge = EQBenchPointwiseJudge(client=grader_client, model=args.judge_model)
        get_rewards = (  # noqa: E731
            lambda sample, completions, rollout_ids: get_eqbench_pointwise_rewards(
                sample, completions, judge, rollout_ids
            )
        )

    asyncio.run(
        main(
            config=config,
            get_rewards=get_rewards,
            data_loader=data_loader,
            num_workers=args.num_workers,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            renderer_name=args.renderer,
        )
    )
