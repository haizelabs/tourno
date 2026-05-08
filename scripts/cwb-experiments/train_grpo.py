import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import tinker
import weave
from data import CreativeBenchDataLoader, CreativeBenchSample
from dotenv import load_dotenv
from eval import main as run_eval
from judges import CreativeBenchPairwiseJudge, CreativeBenchPointwiseJudge
from openai import AsyncOpenAI
from paths import PAIRWISE_PROMPT_PATH, POINTWISE_PROMPT_PATH
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

import wandb
from tourno.logger import get_logger, init_weave, setup, trace
from tourno.tournament import adaptive_pointwise_rewards, batched_elo_rewards
from tourno.training.grpo import training_loop
from tourno.training.types import (
    GRPOConfig,
    TrainingQueue,
    TrajectoryGroup,
    TrajectoryTurn,
)

load_dotenv()


RewardFn = Callable[
    [CreativeBenchSample, list[str], list[str]], Awaitable[tuple[list[float], int]]
]


def _stop_token_ids(renderer: Renderer) -> list[int]:
    ids: set[int] = set()
    for stop in renderer.get_stop_sequences():
        if isinstance(stop, int):
            ids.add(stop)
        elif isinstance(stop, str):
            encoded = renderer.tokenizer.encode(stop)
            if encoded:
                ids.add(encoded[-1])

    return sorted(ids)


async def _get_pointwise_rewards(
    sample: CreativeBenchSample,
    completions: list[str],
    judge: CreativeBenchPointwiseJudge,
    total_samples: int,
) -> list[float]:
    return await adaptive_pointwise_rewards(
        sample.writing_prompt,
        completions,
        judge,
        total_samples=total_samples,
    )


def make_get_rewards(
    *,
    judge_type: str,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_sampling_kwargs: dict,
    pairwise_alpha: float,
    pointwise_total_samples: int | None = None,
) -> RewardFn:
    pointwise_template = POINTWISE_PROMPT_PATH.read_text()
    pairwise_template = PAIRWISE_PROMPT_PATH.read_text()

    if judge_type == "pointwise":
        judge = CreativeBenchPointwiseJudge(
            judge_client, judge_model, pointwise_template, judge_sampling_kwargs
        )

        @weave.op
        async def pointwise(
            sample: CreativeBenchSample, completions: list[str], _rollout_ids: list[str]
        ) -> tuple[list[float], int]:
            total_samples = pointwise_total_samples or len(completions)
            rewards = await _get_pointwise_rewards(
                sample=sample,
                completions=completions,
                judge=judge,
                total_samples=total_samples,
            )
            return rewards, -1

        return pointwise

    elif judge_type == "pairwise":
        judge = CreativeBenchPairwiseJudge(
            judge_client, judge_model, pairwise_template, judge_sampling_kwargs
        )

        @weave.op
        async def pairwise(
            sample: CreativeBenchSample, completions: list[str], _rollout_ids: list[str]
        ) -> tuple[list[float], int]:
            rewards = await batched_elo_rewards(sample.writing_prompt, completions, judge)
            return rewards, -1

        return pairwise

    elif judge_type == "tourno":
        point_judge = CreativeBenchPointwiseJudge(
            judge_client, judge_model, pointwise_template, judge_sampling_kwargs
        )
        pair_judge = CreativeBenchPairwiseJudge(
            judge_client, judge_model, pairwise_template, judge_sampling_kwargs
        )

        @weave.op
        async def tourno(
            sample: CreativeBenchSample, completions: list[str], _rollout_ids: list[str]
        ) -> tuple[list[float], int]:
            pointwise_rewards, pairwise_rewards = await asyncio.gather(
                _get_pointwise_rewards(
                    sample, completions, point_judge, total_samples=len(completions)
                ),
                batched_elo_rewards(sample.writing_prompt, completions, pair_judge),
            )

            pointwise_arr = np.array(pointwise_rewards)
            pairwise_arr = np.array(pairwise_rewards)
            decay_coeff = np.exp(-pairwise_alpha * pointwise_arr.mean())
            mixed_rewards = pointwise_arr + decay_coeff * pairwise_arr

            return mixed_rewards.tolist(), -1

        return tourno

    raise ValueError(f"Invalid judge type: {judge_type}")


@weave.op
@trace
async def rollout(
    worker_id: int,
    sample: CreativeBenchSample,
    renderer: Renderer,
    sampling_client_with_step: tuple[tinker.SamplingClient, int],
    training_queue: TrainingQueue,
    *,
    get_rewards: RewardFn,
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
            stop=_stop_token_ids(renderer),
        ),
    )

    trajectories = [[TrajectoryTurn(obs=obs, ac=seq)] for seq in completions.sequences]
    completion_texts = [
        renderer.tokenizer.decode(seq.tokens, skip_special_tokens=True)
        for seq in completions.sequences
    ]
    rollout_ids = [f"{sample.prompt_id}_{i}" for i in range(len(completion_texts))]
    rewards, judge_calls = await get_rewards(sample, completion_texts, rollout_ids)

    log.debug(f"{sample.writing_prompt=}\n{completion_texts[0]=}\n{rewards[0]=}")

    traj_group = TrajectoryGroup(
        group_size=group_size,
        trajectories=trajectories,
        rewards=rewards,
        judge_calls=judge_calls,
        prompt=sample.writing_prompt,
        completions=completion_texts,
        sample_id=sample.prompt_id,
        worker_id=worker_id,
    )
    await training_queue.put((sampling_client_step, traj_group))

    return traj_group


@trace
async def main(
    config: GRPOConfig,
    get_rewards: RewardFn,
    dataloader: CreativeBenchDataLoader,
    num_workers: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    renderer_name: str | None,
    on_checkpoint_save: Callable[[int, str, str], Awaitable[None] | None] | None = None,
    extra_metrics_fn: (
        Callable[[int, list[TrajectoryGroup]], Awaitable[dict[str, Any]]] | None
    ) = None,
    pending_eval_tasks: list[asyncio.Task] | None = None,
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
            extra_metrics_fn=extra_metrics_fn,
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

        if pending_eval_tasks:
            outstanding = [t for t in pending_eval_tasks if not t.done()]
            if outstanding:
                log.info(f"Waiting for {len(outstanding)} in-flight eval task(s) to finish...")
                await asyncio.gather(*outstanding, return_exceptions=True)

        log.info("Training complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creative Writing Bench v3 GRPO training")

    ### Worker / generation settings ###
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)

    ### Dataset settings ###
    parser.add_argument(
        "--dataset",
        type=str,
        default=Path("datasets/cwbench_train.jsonl").as_posix(),
    )

    ### Judge settings ###
    parser.add_argument(
        "--judge-type",
        type=str,
        choices=["pointwise", "pairwise", "tourno"],
        default="tourno",
    )
    parser.add_argument("--judge-model", type=str, default="anthropic/claude-sonnet-4.5")
    parser.add_argument("--judge-base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--judge-top-p", type=float, default=0.95)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--pairwise-alpha", type=float, default=3.0)
    parser.add_argument(
        "--pointwise-total-samples",
        type=int,
        default=None,
        help=(
            "Total pointwise judge calls per group (distributed adaptively across "
            "completions). Defaults to --group-size."
        ),
    )

    ### Logging settings ###
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-filter", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="tourno-cwbench")
    parser.add_argument("--enable-weave", action="store_true")
    parser.add_argument("--no-trace-table", action="store_true")

    ### Model / optimizer settings ###
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--renderer", type=str, default="qwen3_disable_thinking")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-5)

    ### Training schedule settings ###
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument(
        "--n-epochs",
        type=float,
        default=None,
        help="If set, overrides --n-steps based on dataset size",
    )

    ### Checkpoint / runtime settings ###
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-path", type=str, default="./cwbench-rl")
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
        "--eval-dataset",
        type=str,
        default=Path("datasets/cwbench_val.jsonl").as_posix(),
    )
    parser.add_argument("--eval-max-samples", type=int, default=64)
    parser.add_argument("--eval-num-completions", type=int, default=1)
    parser.add_argument("--eval-max-tokens", type=int, default=4096)
    parser.add_argument("--eval-temperature", type=float, default=0.7)
    parser.add_argument("--eval-judge-temperature", type=float, default=0.0)
    parser.add_argument("--eval-judge-max-tokens", type=int, default=2048)
    parser.add_argument("--eval-gen-concurrency", type=int, default=32)
    parser.add_argument("--eval-judge-concurrency", type=int, default=128)
    parser.add_argument("--no-eval", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    ### Initialize logging ###
    args = parse_args()

    if not args.enable_weave:
        os.environ.setdefault("WANDB_DISABLE_WEAVE", "true")
        os.environ.setdefault("WEAVE_DISABLED", "true")

    setup(
        level=getattr(logging, args.log_level.upper()),
        filter_pattern=args.log_filter,
    )

    ### Initialize weave tracing (before any OpenAI client is created) ###
    if args.enable_weave and args.wandb_project:
        init_weave(args.wandb_project)

    ### Initialize dataloader ###
    dataloader = CreativeBenchDataLoader(
        args.dataset, batch_size=1, max_length=args.max_samples
    )

    ### Initialize training config ###
    n_steps = args.n_steps
    if args.n_epochs is not None:
        n_steps = math.ceil(args.n_epochs * dataloader.num_rows / args.batch_size)

    run_prefix = "cwbench-grpo"
    model_short = args.base_model.split("/")[-1]
    name = (
        f"{model_short}_lr{args.learning_rate}_bs{args.batch_size}_lora{args.lora_rank}_"
        f"{args.judge_type}_judge{args.judge_model}"
    )
    if args.judge_type == "tourno" and args.pairwise_alpha > 0:
        name += f"_alpha{args.pairwise_alpha}"
    if (
        args.judge_type == "pointwise"
        and args.pointwise_total_samples is not None
        and args.pointwise_total_samples > args.group_size
    ):
        name += f"_judge-n{args.pointwise_total_samples}"

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
    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise ValueError(f"Missing judge API key in env var {args.judge_api_key_env}")
    judge_client = AsyncOpenAI(api_key=api_key, base_url=args.judge_base_url or None)
    judge_sampling_kwargs: dict = {
        "temperature": args.judge_temperature,
        "top_p": args.judge_top_p,
        "max_tokens": args.judge_max_tokens,
    }

    ### Initialize get_rewards function ###
    get_rewards = make_get_rewards(
        judge_type=args.judge_type,
        judge_client=judge_client,
        judge_model=args.judge_model,
        judge_sampling_kwargs=judge_sampling_kwargs,
        pairwise_alpha=args.pairwise_alpha,
        pointwise_total_samples=args.pointwise_total_samples,
    )

    ### Initialize evaluation callback (parallel / fire-and-forget) ###
    pending_eval_tasks: list[asyncio.Task] = []
    eval_log = get_logger("eval")

    async def _run_validation(step: int, sampler_path: str) -> None:
        try:
            summary = await run_eval(
                label=f"step_{step:06d}",
                dataset_path=Path(args.eval_dataset),
                output_dir=Path(config.log_path) / "evals",
                base_model=args.base_model,
                sampler_path=sampler_path,
                base_url=args.base_url,
                judge_client=judge_client,
                judge_model=args.eval_judge_model or args.judge_model,
                renderer_name=args.renderer,
                max_samples=args.eval_max_samples,
                num_completions=args.eval_num_completions,
                max_tokens=args.eval_max_tokens,
                temperature=args.eval_temperature,
                gen_concurrency=args.eval_gen_concurrency,
                judge_temperature=args.eval_judge_temperature,
                judge_max_tokens=args.eval_judge_max_tokens,
                judge_concurrency=args.eval_judge_concurrency,
            )
        except Exception:
            eval_log.exception(f"Validation at step {step} failed")
            return

        if wandb.run is None or not summary:
            return

        eval_metrics: dict[str, Any] = {
            "eval_step": step,
            "eval/eqbench_creative_score_0_100": summary.get("eqbench_creative_score_0_100"),
            "eval/overall_score_0_20": summary.get("overall_score_0_20"),
            "eval/n_pieces_scored": summary.get("n_pieces_scored"),
            "eval/n_judge_errors": summary.get("n_judge_errors"),
        }
        wandb.log(eval_metrics)
        eval_log.info(f"Logged validation metrics for step {step} to wandb")

    async def on_checkpoint_save(step: int, _name: str, sampler_path: str) -> None:
        if args.no_eval:
            return
        task = asyncio.create_task(_run_validation(step, sampler_path))
        pending_eval_tasks.append(task)
        pending_eval_tasks[:] = [t for t in pending_eval_tasks if not t.done()]

    ### Initialize wandb trace Table (incremental — uploads only deltas) ###
    trace_state: dict[str, Any] = {"table": None, "metrics_defined": False}

    async def extra_metrics_fn(
        step: int, trajectory_groups: list[TrajectoryGroup]
    ) -> dict[str, Any]:
        if wandb.run is None:
            return {}

        if not trace_state["metrics_defined"]:
            wandb.define_metric("eval_step")
            wandb.define_metric("eval/*", step_metric="eval_step")
            trace_state["metrics_defined"] = True

        if args.no_trace_table:
            return {}

        if trace_state["table"] is None:
            trace_state["table"] = wandb.Table(
                columns=[
                    "step",
                    "worker_id",
                    "sample_id",
                    "prompt",
                    "completion",
                    "reward",
                    "judge_calls",
                ],
                log_mode="INCREMENTAL",
            )

        table = trace_state["table"]
        for tg in trajectory_groups:
            if tg.completions is None or tg.prompt is None:
                continue
            for completion, reward in zip(tg.completions, tg.rewards):
                table.add_data(
                    step,
                    tg.worker_id if tg.worker_id is not None else -1,
                    tg.sample_id or "",
                    tg.prompt,
                    completion,
                    float(reward),
                    int(tg.judge_calls),
                )

        wandb.log({"rollouts": table}, step=step)
        return {}

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
            extra_metrics_fn=extra_metrics_fn,
            pending_eval_tasks=pending_eval_tasks,
        )
    )
