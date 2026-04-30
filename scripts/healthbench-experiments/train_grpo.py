import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Awaitable, Callable

import tinker
from data import HealthBenchDataLoader, HealthBenchSample
from openai import AsyncOpenAI
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tourno.logger import get_logger, setup, trace
from tourno.training.grpo import training_loop
from tourno.training.types import (
    GRPOConfig,
    TrainingQueue,
    TrajectoryGroup,
    TrajectoryTurn,
)

DATASETS_DIR = Path("datasets")


def _decode_trajectories(
    trajectories: list[list[TrajectoryTurn]],
    renderer: Renderer,
) -> list[str]:
    stop_ids: set[int] = set()
    for stop in renderer.get_stop_sequences():
        if isinstance(stop, int):
            stop_ids.add(stop)
        elif isinstance(stop, str):
            ids = renderer.tokenizer.encode(stop)
            if ids:
                stop_ids.add(ids[-1])

    texts: list[str] = []
    for traj in trajectories:
        tokens = list(traj[0].ac.tokens)
        if tokens and tokens[-1] in stop_ids:
            tokens = tokens[:-1]

        texts.append(renderer.tokenizer.decode(tokens, skip_special_tokens=True))

    return texts


@trace
async def rollout(
    worker_id: int,
    sample: HealthBenchSample,
    renderer: Renderer,
    sampling_client_with_step: tuple[tinker.SamplingClient, int],
    training_queue: TrainingQueue,
    *,
    get_rewards: Callable[
        [HealthBenchSample, list[str], list[str]], Awaitable[tuple[list[float], int]]
    ],
    group_size: int,
    max_tokens: int,
    temperature: float,
) -> TrajectoryGroup:
    log = get_logger(f"worker{worker_id}")
    sampling_client, sampling_client_step = sampling_client_with_step

    log.debug(f"Sampling {group_size} completions from tinker client...")
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
    completion_texts = _decode_trajectories(trajectories, renderer)
    rollout_ids = [f"{sample.row_id}_{i}" for i in range(len(completion_texts))]
    rewards, judge_calls = await get_rewards(sample, completion_texts, rollout_ids)

    log.debug(f"{sample.prompt=}\n{completion_texts[0]=}\n{rewards[0]=}")

    traj_group = TrajectoryGroup(
        group_size=group_size,
        trajectories=trajectories,
        rewards=rewards,
        judge_calls=judge_calls,
    )
    await training_queue.put((sampling_client_step, traj_group))

    return traj_group


@trace
async def main(
    config: GRPOConfig,
    get_rewards: Callable[
        [HealthBenchSample, list[str], list[str]], Awaitable[tuple[list[float], int]]
    ],
    dataloader: HealthBenchDataLoader,
    num_workers: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    renderer_name: str | None,
    on_checkpoint_save: Callable[[int, str, str], Awaitable[None] | None] | None = None,
):
    ### Setup tokenizer and renderer ###
    log = get_logger()
    tokenizer = get_tokenizer(config.base_model)
    renderer_name = renderer_name or get_recommended_renderer_name(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer)
    log.info(f"Model: {config.base_model}, Renderer: {renderer_name}")

    ### Sampling client state ###
    sampling_client_with_step: tuple[tinker.SamplingClient, int] | None = None
    sampling_client_ready = asyncio.Event()

    def update_sampling_client(client: tinker.SamplingClient, step: int) -> None:
        nonlocal sampling_client_with_step
        sampling_client_with_step = (client, step)
        if not sampling_client_ready.is_set():
            sampling_client_ready.set()

    ### Start training loop ###
    training_queue: TrainingQueue = asyncio.Queue()
    training_loop_task = asyncio.create_task(
        training_loop(
            config,
            training_queue,
            update_sampling_client,
            on_checkpoint_save,
        )
    )

    ### Wait for sampling client ###
    log.info("Waiting for sampling client...")
    await asyncio.wait(
        [asyncio.create_task(sampling_client_ready.wait()), training_loop_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if training_loop_task.done():
        log.error("Training loop exited before providing a sampling client")
        training_loop_task.result()
        return

    ### Define worker ###
    async def worker(worker_id: int) -> None:
        async for _, batch in dataloader:
            if len(batch) > 1:
                log.warning(
                    f"Batch size {len(batch)} is greater than 1, only the first sample will be"
                    " processed"
                )

            sample = batch[0]
            try:
                assert sampling_client_with_step is not None
                await rollout(
                    worker_id,
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
                log.exception(f"Worker {worker_id} error")
                raise

    ### Spawn workers and wait for completion ###
    worker_tasks = [asyncio.create_task(worker(i)) for i in range(num_workers)]
    all_tasks = [training_loop_task, *worker_tasks]

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
    parser = argparse.ArgumentParser(description="HealthBench GRPO training")

    ### Worker / generation settings ###
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)

    ### Judge settings ###
    parser.add_argument(
        "--judge-type",
        type=str,
        choices=["pairwise", "pointwise", "mixture"],
        default="pointwise",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-2025-04-14")
    parser.add_argument("--pairwise-alpha", type=float, default=0.5)

    ### Logging settings ###
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-filter", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)

    ### Model / optimizer settings ###
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--renderer", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-5)

    ### Training schedule settings ###
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument(
        "--n-epochs",
        type=float,
        default=None,
        help="If set, overrides --n-steps based on dataset size",
    )

    ### Checkpoint / runtime settings ###
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-path", type=str, default="./healthbench-rl")
    parser.add_argument("--base-url", type=str, default=None)

    ### Loss / KL settings ###
    parser.add_argument("--loss-fn", type=str, default="importance_sampling")
    parser.add_argument("--kl-reference-model", type=str, default=None)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--kl-discount-factor", type=float, default=0.0)
    parser.add_argument("--compute-post-kl", action="store_true")
    parser.add_argument("--max-steps-off-policy", type=int, default=3)

    ### Evaluation settings ###
    parser.add_argument("--eval-judge-model", type=str, default=None)
    parser.add_argument(
        "--eval-dataset", type=str, default=(DATASETS_DIR / "healthbench_val.jsonl").as_posix()
    )
    parser.add_argument("--eval-max-samples", type=int, default=64)
    parser.add_argument("--eval-max-tokens", type=int, default=1024)
    parser.add_argument("--eval-gen-concurrency", type=int, default=32)
    parser.add_argument("--eval-num-completions", type=int, default=1)
    parser.add_argument("--no-eval", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    ### Initialize logging ###
    args = parse_args()
    setup(
        level=getattr(logging, args.log_level.upper()),
        filter_pattern=args.log_filter,
    )

    ### Initialize dataloader ###
    dataloader = HealthBenchDataLoader(batch_size=1, max_length=args.max_samples)

    ### Initialize training config ###
    n_steps = args.n_steps
    if args.n_epochs is not None:
        n_steps = math.ceil(args.n_epochs * dataloader.num_rows / args.batch_size)

    run_prefix = "healthbench-grpo"
    model_short = args.base_model.split("/")[-1]
    name = f"{model_short}_lr{args.learning_rate}_bs{args.batch_size}_lora{args.lora_rank}_{args.judge_type}_judge{args.judge_model}"
    if args.judge_type == "mixture" and args.pairwise_alpha > 0:
        name += f"_alpha{args.pairwise_alpha}"

    config = GRPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        judge_type=args.judge_type,
        judge_model=args.judge_model,
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
        log_path=(Path(args.log_path) / f"{run_prefix}_{name}").as_posix(),
        run_name=f"{run_prefix}_{name}",
        wandb_project=args.wandb_project,
    )

    ### Initialize judge client ###
    if os.getenv("OPENAI_API_KEY") is not None:
        judge_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif os.getenv("OPENROUTER_API_KEY") is not None:
        judge_client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError("No LLM Provider API key found")

    ### Initialize get_rewards function ###
    get_rewards = ...

    ### Initialize evaluation callback ###
    async def on_checkpoint_save(step: int, _name: str, sampler_path: str) -> None:
        print("hello world")
        ...

    ### Run training ###
    asyncio.run(
        main(
            config=config,
            get_rewards=get_rewards,
            dataloader=dataloader,
            num_workers=args.num_workers,
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            renderer_name=args.renderer,
            on_checkpoint_save=on_checkpoint_save,
        )
    )
