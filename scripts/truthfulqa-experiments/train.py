import argparse
import asyncio
import functools
import logging
import math
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, cast

import numpy as np
import tinker
from data import TruthfulQADataLoader
from dotenv import load_dotenv
from eval_common import (
    DATASETS_DIR,
    EvalPipeline,
    load_samples,
    make_openai_client,
    prefixed_mean_metrics,
    serialize_conversation,
)
from gold_judges import get_gold_judge
from judges import TruthfulQAPairwiseJudge, TruthfulQAPointwiseJudge
from openai import AsyncOpenAI
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from truthfulqa_types import TruthfulQASample
from worker import group_worker

from pioneer.logger import get_logger, setup, trace
from pioneer.loop import PostSaveCallback, training_loop
from pioneer.types import TrainConfig, TrainingQueue
from tourno.tournament import batched_elo_reward_fn, pointwise_reward_fn

load_dotenv()


# Schemas for the wandb trace tables; one INCREMENTAL Table is created per key
# in pioneer.loop after wandb.init. log_trace(name, **fields) populates rows.
TRACE_SCHEMAS: dict[str, list[str]] = {
    "traces/rollouts": [
        "step",
        "row_id",
        "rollout_id",
        "prompt_id",
        "category",
        "reward",
        "token_len",
        "prompt",
        "completion",
    ],
    "traces/pointwise_judge": [
        "step",
        "judge_model",
        "prompt_id",
        "score",
        "judge_prompt",
        "judge_response",
    ],
    "traces/pairwise_judge": [
        "step",
        "judge_model",
        "prompt_id",
        "p_a_wins",
        "completion_a",
        "completion_b",
        "judge_prompt",
        "judge_response",
    ],
}


async def _pointwise_rewards(
    judge: TruthfulQAPointwiseJudge,
    sample: TruthfulQASample,
    completions: list[str],
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    conversation = serialize_conversation(sample.prompt)
    raw_scores = await pointwise_reward_fn(
        conversation, completions, lambda _p, comps: judge(conversation, comps, sample)
    )

    return [judge.normalize(s) for s in raw_scores], len(completions)


async def _pairwise_rewards(
    judge: TruthfulQAPairwiseJudge,
    sample: TruthfulQASample,
    completions: list[str],
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    conversation = serialize_conversation(sample.prompt)
    return await batched_elo_reward_fn(
        conversation, completions, judge_fn=lambda pairs: judge(pairs, sample)
    )


async def _mixture_rewards(
    point_judge: TruthfulQAPointwiseJudge,
    pair_judge: TruthfulQAPairwiseJudge,
    alpha: float,
    sample: TruthfulQASample,
    completions: list[str],
    rollout_ids: list[str],
) -> tuple[list[float], int]:
    (pw, pw_calls), (pair, pair_calls) = await asyncio.gather(
        _pointwise_rewards(point_judge, sample, completions, rollout_ids),
        _pairwise_rewards(pair_judge, sample, completions, rollout_ids),
    )
    r_point = np.array(pw)
    mixed = r_point + np.exp(-alpha * r_point.mean()) * np.array(pair)
    return mixed.tolist(), pw_calls + pair_calls


def build_get_rewards(
    judge_type: str,
    judge_model: str,
    grader_client: AsyncOpenAI,
    pairwise_alpha: float,
) -> Callable[[TruthfulQASample, list[str], list[str]], Awaitable[tuple[list[float], int]]]:
    """Return a get_rewards callable matching the worker's expected signature."""
    if judge_type == "pointwise":
        judge = TruthfulQAPointwiseJudge(client=grader_client, model=judge_model)
        return functools.partial(_pointwise_rewards, judge)
    if judge_type == "pairwise":
        judge = TruthfulQAPairwiseJudge(client=grader_client, model=judge_model)
        return functools.partial(_pairwise_rewards, judge)
    if judge_type == "mixture":
        return functools.partial(
            _mixture_rewards,
            TruthfulQAPointwiseJudge(client=grader_client, model=judge_model),
            TruthfulQAPairwiseJudge(client=grader_client, model=judge_model),
            pairwise_alpha,
        )
    raise ValueError(f"Unknown judge_type: {judge_type!r}")


def make_in_training_eval_callback(
    config: TrainConfig,
    judge_kinds: list[str],
    judge_model: str,
    judge_provider: str,
    n_samples_per_prompt: int,
    max_samples: int | None,
    max_tokens: int,
    temperature: float,
    gen_concurrency: int,
    judge_concurrency: int,
) -> PostSaveCallback:
    log = get_logger()

    val_path = DATASETS_DIR / "truthfulqa_val.jsonl"
    val_samples = load_samples(val_path, max_samples=max_samples)
    log.info(
        f"In-training eval: loaded {len(val_samples)} val samples from {val_path}, "
        f"judges={judge_kinds}, n_samples_per_prompt={n_samples_per_prompt}"
    )

    judge_client = make_openai_client(judge_provider)
    judges = [get_gold_judge(kind, client=judge_client, model=judge_model) for kind in judge_kinds]
    service = tinker.ServiceClient(base_url=config.base_url)
    run_name = config.run_name

    eval_dir = Path(config.log_path) / "evals"

    async def callback(step: int, sampling_client: tinker.SamplingClient) -> dict[str, Any]:
        results_path = eval_dir / f"eval_step_{step:06d}.jsonl"
        log.info(f"[in-training eval] step {step}: writing to {results_path}")

        pipeline = EvalPipeline(
            samples=val_samples,
            judges=judges,
            service=service,
            results_path=results_path,
            n_samples_per_prompt=n_samples_per_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            gen_concurrency=gen_concurrency,
            judge_concurrency=judge_concurrency,
            client_overrides={(run_name, step): sampling_client},
        )

        eval_target = (run_name, step)
        records = await pipeline.run([eval_target])
        metrics: dict[str, Any] = prefixed_mean_metrics(records, judge_kinds, eval_target, "val")

        log.info(f"[in-training eval] step {step} metrics: {metrics}")
        return metrics

    return callback


def make_training_loop(
    config: TrainConfig,
    training_queue: TrainingQueue,
    update_sampling_client: Callable[[tinker.SamplingClient, int], None],
    post_save_callback: PostSaveCallback | None = None,
    trace_schemas: dict[str, list[str]] | None = None,
) -> asyncio.Task:
    log = get_logger()

    def _on_done(task: asyncio.Task) -> None:
        if task.exception() is not None:
            log.error(f"Training loop failed: {task.exception()}")
        else:
            log.info("Training loop finished")

    task = asyncio.create_task(
        training_loop(
            config,
            training_queue,
            update_sampling_client,
            post_save_callback=post_save_callback,
            trace_schemas=trace_schemas,
        )
    )
    task.add_done_callback(_on_done)
    return task


@trace
async def main(
    config: TrainConfig,
    get_rewards: Callable[
        [TruthfulQASample, list[str], list[str]], Awaitable[tuple[list[float], int]]
    ],
    data_loader: TruthfulQADataLoader,
    num_workers: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    renderer_name: str | None,
    post_save_callback: PostSaveCallback | None = None,
    trace_schemas: dict[str, list[str]] | None = None,
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
    training_loop_task = make_training_loop(
        config,
        training_queue,
        update_sampling_client,
        post_save_callback=post_save_callback,
        trace_schemas=trace_schemas,
    )
    work_queue: asyncio.Queue[TruthfulQASample | None] = asyncio.Queue(maxsize=num_workers)

    log.info("Waiting for sampling client...")
    await asyncio.wait(
        [asyncio.create_task(sampling_client_ready.wait()), training_loop_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if training_loop_task.done():
        log.error("Training loop exited before providing a sampling client")
        training_loop_task.result()
        return

    assert cast(sampling_client_with_step, (tinker.SamplingClient, int)) is not None
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
    p = argparse.ArgumentParser(description="TruthfulQA RL training")

    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument(
        "--judge-type",
        type=str,
        choices=["pairwise", "pointwise", "mixture"],
        default="pointwise",
    )
    p.add_argument("--judge-model", type=str, default="gpt-4.1-2025-04-14")
    p.add_argument(
        "--judge-provider",
        type=str,
        choices=["openai", "openrouter"],
        default="openai",
        help=(
            "Where to route judge calls. 'openai' uses OPENAI_API_KEY against the default "
            "OpenAI endpoint; 'openrouter' uses OPENROUTER_API_KEY against "
            "https://openrouter.ai/api/v1."
        ),
    )
    p.add_argument("--pairwise-alpha", type=float, default=0.5)

    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-filter", type=str, default=None)

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
    p.add_argument("--log-path", type=str, default="./truthfulqa-rl")

    ### In-training gold-judge eval ###
    p.add_argument(
        "--eval-every-checkpoint",
        action="store_true",
        help="After every checkpoint save, run gold-judge eval on the val set.",
    )
    p.add_argument(
        "--eval-judges",
        nargs="+",
        choices=["strict", "lenient"],
        default=["lenient"],
        help="Which gold judges to run per checkpoint (default: lenient only).",
    )
    p.add_argument("--eval-judge-model", type=str, default="openai/gpt-5.4")
    p.add_argument(
        "--eval-judge-provider",
        type=str,
        choices=["openai", "openrouter"],
        default="openrouter",
    )
    p.add_argument(
        "--eval-n-samples",
        type=int,
        default=1,
        help="Completions per prompt for in-training eval. Higher = lower variance, more cost.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Cap val set size for in-training eval (default: use full val set).",
    )
    p.add_argument("--eval-gen-concurrency", type=int, default=32)
    p.add_argument("--eval-judge-concurrency", type=int, default=64)

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

    setup(
        level=getattr(logging, args.log_level.upper()),
        filter_pattern=args.log_filter,
    )

    data_loader = TruthfulQADataLoader(max_length=args.max_samples)
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

    if args.judge_provider == "openrouter":
        grader_client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        grader_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    get_rewards = build_get_rewards(
        args.judge_type, args.judge_model, grader_client, config.pairwise_alpha
    )

    post_save_callback: PostSaveCallback | None = None
    if args.eval_every_checkpoint:
        post_save_callback = make_in_training_eval_callback(
            config=config,
            judge_kinds=args.eval_judges,
            judge_model=args.eval_judge_model,
            judge_provider=args.eval_judge_provider,
            n_samples_per_prompt=args.eval_n_samples,
            max_samples=args.eval_max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gen_concurrency=args.eval_gen_concurrency,
            judge_concurrency=args.eval_judge_concurrency,
        )

    exit_code = 0
    try:
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
                post_save_callback=post_save_callback,
                trace_schemas=TRACE_SCHEMAS,
            )
        )
    except BaseException:
        get_logger().exception("Training run failed")
        exit_code = 1

    os._exit(exit_code)
