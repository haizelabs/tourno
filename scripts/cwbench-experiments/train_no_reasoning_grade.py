import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import tinker
from cwbench_types import CreativeBenchSample
import tracelog
from data import CreativeBenchDataLoader, TRAIN_DATASET_PATH
from dotenv import load_dotenv
from judges import CreativeBenchPairwiseJudge, CreativeBenchPointwiseJudge
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

# Keys the pointwise-reward side reads the writing_prompt string FROM (not the
# master-wrapped policy prompt) — same string the judge uses in its template.


async def get_cwbench_pointwise_rewards(
    sample: CreativeBenchSample,
    completions: list[str],
    judge: CreativeBenchPointwiseJudge,
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    rewards = await pointwise_reward_fn(
        sample.writing_prompt,
        completions,
        lambda _prompt, comps: judge(sample.writing_prompt, comps),
    )
    for i, completion in enumerate(completions):
        log_agent_run(
            sample.prompt + [{"content": completion, "role": "assistant"}],
            {
                "type": "pointwise_reward",
                "prompt_id": sample.prompt_id,
                "scenario_id": sample.scenario_id,
                "row_id": sample.row_id,
                "rollout_id": rollout_ids[i],
                "reward": rewards[i],
            },
        )
    return rewards, len(completions)


async def get_cwbench_pairwise_rewards(
    sample: CreativeBenchSample,
    completions: list[str],
    judge: CreativeBenchPairwiseJudge,
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    rewards, n_calls = await batched_elo_reward_fn(
        sample.writing_prompt,
        completions,
        judge_fn=judge,
    )
    for i, completion in enumerate(completions):
        log_agent_run(
            sample.prompt + [{"content": completion, "role": "assistant"}],
            {
                "type": "pairwise_reward",
                "prompt_id": sample.prompt_id,
                "scenario_id": sample.scenario_id,
                "row_id": sample.row_id,
                "rollout_id": rollout_ids[i],
                "reward": rewards[i],
            },
        )
    return rewards, n_calls


async def get_cwbench_mixture_rewards(
    sample: CreativeBenchSample,
    completions: list[str],
    pointwise_judge: CreativeBenchPointwiseJudge,
    pairwise_judge: CreativeBenchPairwiseJudge,
    pairwise_alpha: float,
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    (pw, pw_calls), (pair, pair_calls) = await asyncio.gather(
        get_cwbench_pointwise_rewards(sample, completions, pointwise_judge, rollout_ids),
        get_cwbench_pairwise_rewards(sample, completions, pairwise_judge, rollout_ids),
    )
    r_point = np.array(pw)
    r_pair = np.array(pair)
    mixture_weight = float(np.exp(-pairwise_alpha * r_point.mean()))
    mixed = r_point + mixture_weight * r_pair

    # Track the TournO mixing weight live — important for paper figures.
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "mixture/pairwise_weight": mixture_weight,
                    "mixture/pointwise_mean": float(r_point.mean()),
                    "mixture/pairwise_mean": float(r_pair.mean()),
                    "mixture/mixed_mean": float(mixed.mean()),
                }
            )
    except Exception:
        pass
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


def _build_judge_client(base_url: str | None, api_key_env: str) -> AsyncOpenAI:
    """OpenAI-compatible client.

    Default: OpenRouter (supports Claude Sonnet 4.5 via `anthropic/claude-sonnet-4.5`).
    Pass --judge-base-url='' to use raw OpenAI.
    """
    kwargs: dict = {"api_key": os.environ[api_key_env]}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def _log_run_config_to_wandb(config: TrainConfig, args: argparse.Namespace, data_loader) -> None:
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.config.update(
            {
                "dataset/train_rows": data_loader.num_rows,
                "dataset/train_path": TRAIN_DATASET_PATH,
                "dataset/test_path": "datasets/cwbench_test.jsonl (cwbench v3, 321)",
                "judge/base_url": args.judge_base_url or "openai-default",
                "judge/api_key_env": args.judge_api_key_env,
                "judge/model": args.judge_model,
                "judge/temperature": args.judge_temperature,
                "policy/temperature": args.temperature,
                "policy/max_tokens": args.max_tokens,
                "hparams/pairwise_alpha": config.pairwise_alpha,
                "hparams/learning_rate": config.learning_rate,
                "hparams/batch_size": config.batch_size,
                "hparams/group_size": config.group_size,
                "hparams/lora_rank": config.lora_rank,
                "hparams/judge_type": config.judge_type,
                "hparams/n_steps": config.n_steps,
                "hparams/loss_fn": config.loss_fn,
                "hparams/kl_coef": config.kl_coef,
            },
            allow_val_change=True,
        )
    except Exception as exc:
        get_logger().warning(f"Failed to log config to wandb: {exc}")


def _log_judge_stats_to_wandb(judges: list[tuple[str, object]]) -> None:
    try:
        import wandb
        if wandb.run is None:
            return
        payload: dict = {}
        for name, j in judges:
            s = getattr(j, "stats", None)
            if s is None:
                continue
            payload[f"judge_stats/{name}/success"] = s.success
            payload[f"judge_stats/{name}/failure"] = s.failure
            payload[f"judge_stats/{name}/total_latency_s"] = s.total_latency_s
            payload[f"judge_stats/{name}/total_tokens_in"] = s.total_tokens_in
            payload[f"judge_stats/{name}/total_tokens_out"] = s.total_tokens_out
            total = s.success + s.failure
            if total > 0:
                payload[f"judge_stats/{name}/failure_rate"] = s.failure / total
                payload[f"judge_stats/{name}/avg_latency_s"] = s.total_latency_s / total
        if payload:
            wandb.summary.update(payload)
    except Exception:
        pass


@trace
async def main(
    config: TrainConfig,
    get_rewards: Callable[
        [CreativeBenchSample, list[str], list[str]], Awaitable[tuple[list[float], int]]
    ],
    data_loader: CreativeBenchDataLoader,
    num_workers: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    renderer_name: str | None,
    args: argparse.Namespace,
    judges_for_stats: list[tuple[str, object]],
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
    work_queue: asyncio.Queue[CreativeBenchSample | None] = asyncio.Queue(maxsize=num_workers)

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
    _log_run_config_to_wandb(config, args, data_loader)

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
        _log_judge_stats_to_wandb(judges_for_stats)
        tracelog.flush_all()  # capture trailing rows below the periodic-flush threshold
        log.info("Training complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Creative Writing Bench v3 RL training (OWB → cwbench v3)")

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument(
        "--judge-type",
        type=str,
        choices=["pairwise", "pointwise", "mixture"],
        default="mixture",
    )
    p.add_argument("--judge-model", type=str, default="anthropic/claude-sonnet-4.5")
    p.add_argument(
        "--judge-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Judge API base URL. Default: OpenRouter. Set to '' for raw OpenAI.",
    )
    p.add_argument(
        "--judge-api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable to read judge API key from.",
    )
    # Upstream cwbench leaderboard scoring params (eqbench.com): temp=1.0, top_p=0.95,
    # max_tokens=2048. Matched here for leaderboard-comparable reward distributions.
    p.add_argument("--judge-temperature", type=float, default=1.0)
    p.add_argument("--judge-top-p", type=float, default=0.95)
    p.add_argument("--judge-max-tokens", type=int, default=2048)
    p.add_argument("--pairwise-alpha", type=float, default=3.0)

    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-filter", type=str, default=None)
    p.add_argument("--docent-collection", type=str, default=None)

    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    # `qwen3` (default) forces every assistant turn to start with `<think>\n`, which makes
    # Qwen3-8B emit long planning preambles that eat the token budget. Use the disable-thinking
    # variant which prefills `</think>\n\n` and lets the model write prose directly.
    p.add_argument("--renderer", type=str, default="qwen3_disable_thinking")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--n-epochs", type=float, default=None)

    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--log-path", type=str, default="./cwbench-rl")
    p.add_argument("--kl-reference-model", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None, help="Tinker service URL")
    p.add_argument("--loss-fn", type=str, default="importance_sampling")
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--kl-discount-factor", type=float, default=0.0)
    p.add_argument("--compute-post-kl", action="store_true")
    p.add_argument("--max-steps-off-policy", type=int, default=3)
    p.add_argument("--wandb-project", type=str, default="tourno-cwbench")
    p.add_argument("--wandb-run-name", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(level=getattr(logging, args.log_level.upper()), filter_pattern=args.log_filter)

    data_loader = CreativeBenchDataLoader(max_length=args.max_samples)
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
        wandb_run_name=args.wandb_run_name,
    )
    # Allow a custom W&B run name (paper-friendly: `pointwise_sonnet_lr4e-5_s42`).
    config.log_path = (
        Path(config.log_path) / (args.wandb_run_name or config.run_name)
    ).as_posix()

    docent_collection = args.docent_collection
    if not docent_collection and os.environ.get("DOCENT_API_KEY"):
        docent_collection = args.wandb_run_name or config.run_name
    if docent_collection:
        init_docent(docent_collection)

    judge_base_url = args.judge_base_url or None
    judge_client = _build_judge_client(judge_base_url, args.judge_api_key_env)
    judge_sampling_args: dict = {
        "temperature": args.judge_temperature,
        "top_p": args.judge_top_p,
        "max_tokens": args.judge_max_tokens,
    }

    judges_for_stats: list[tuple[str, object]] = []
    if args.judge_type == "mixture":
        point_judge = CreativeBenchPointwiseJudge(
            client=judge_client, model=args.judge_model, judge_sampling_args=judge_sampling_args
        )
        pair_judge = CreativeBenchPairwiseJudge(
            client=judge_client, model=args.judge_model, judge_sampling_args=judge_sampling_args
        )
        judges_for_stats = [("pointwise", point_judge), ("pairwise", pair_judge)]
        get_rewards = (  # noqa: E731
            lambda sample, completions, rollout_ids: get_cwbench_mixture_rewards(
                sample,
                completions,
                point_judge,
                pair_judge,
                config.pairwise_alpha,
                rollout_ids,
            )
        )
    elif args.judge_type == "pairwise":
        judge = CreativeBenchPairwiseJudge(
            client=judge_client, model=args.judge_model, judge_sampling_args=judge_sampling_args
        )
        judges_for_stats = [("pairwise", judge)]
        get_rewards = (  # noqa: E731
            lambda sample, completions, rollout_ids: get_cwbench_pairwise_rewards(
                sample, completions, judge, rollout_ids
            )
        )
    else:
        judge = CreativeBenchPointwiseJudge(
            client=judge_client, model=args.judge_model, judge_sampling_args=judge_sampling_args
        )
        judges_for_stats = [("pointwise", judge)]
        get_rewards = (  # noqa: E731
            lambda sample, completions, rollout_ids: get_cwbench_pointwise_rewards(
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
            args=args,
            judges_for_stats=judges_for_stats,
        )
    )
