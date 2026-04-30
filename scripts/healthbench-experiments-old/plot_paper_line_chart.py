"""Paper figure: line chart showing performance across training checkpoints.

One line per training method (Pointwise, Pairwise, TournO), plotting mean
HealthBench score at each checkpoint step for a single judge. Step 0 (base
model) is shared across all methods.

Usage:
    uv run scripts/healthbench-experiments/plot_paper_line_chart.py \
        --judge gpt-5.2 \
        --steps 0 100 200 300 400 500 600 700 \
        --base-model Qwen/Qwen3-4B-Instruct-2507 \
        --output figures/out/Qwen3-4B/line_chart.pdf \
        --cache-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as healthbench_eval
from eval import (
    DATASETS_DIR,
    POINTWISE_PROMPT,
    ResultCache,
    discover_run,
    evaluate_checkpoint,
    get_available_steps,
    get_base_model,
    get_sampler_path,
    load_samples,
)

from tourno.eval.judges import PointwiseJudge

load_dotenv()
log = logging.getLogger(__name__)

METHODS = ["pointwise", "pairwise", "mixture"]
METHOD_LABELS = {"pointwise": "Pointwise", "pairwise": "Pairwise", "mixture": "TournO"}
METHOD_COLORS = {
    "base": "#e5e3dc",
    "pointwise": "#faedcd",
    "pairwise": "#e9edc9",
    "mixture": "#d5bdaf",
}
METHOD_LINE_COLORS = {
    "pointwise": "#c4a35a",
    "pairwise": "#7a9a3a",
    "mixture": "#a0725a",
}


def collect_line_data(
    cache: ResultCache,
    judge: str,
    steps: list[int],
    base_model_short: str,
    mixture_alpha: float,
    prompt_ids: set[str],
) -> dict[str, dict[int, float]]:
    """Returns {method: {step: mean}} for steps with cached data.

    Step 0 (base model) is shared: we find the first run with cached step-0
    scores and reuse that single value for every method.
    """
    base_mean: float | None = None
    for method in METHODS:
        if base_mean is not None:
            break
        run = discover_run(base_model_short, method, judge, mixture_alpha)
        if not run:
            continue
        scores_dict = cache.get_scores(run, 0, prompt_ids)
        if scores_dict:
            base_mean = float(np.mean(list(scores_dict.values())))

    result: dict[str, dict[int, float]] = {}
    for method in METHODS:
        run = discover_run(base_model_short, method, judge, mixture_alpha)
        if not run:
            log.warning(f"No run found for {method}/{judge}, skipping")
            continue
        available = get_available_steps(run)
        step_data: dict[int, float] = {}
        if base_mean is not None and 0 in steps:
            step_data[0] = base_mean
        for step in steps:
            if step == 0:
                continue
            if step not in available:
                continue
            scores_dict = cache.get_scores(run, step, prompt_ids)
            if not scores_dict:
                continue
            step_data[step] = float(np.mean(list(scores_dict.values())))
        if step_data:
            result[method] = step_data
            log.info(f"{method}/{judge}: {len(step_data)} steps with data")
    return result


def _plot_on_ax(ax, data: dict[str, dict[int, float]], title: str | None = None) -> None:
    for method in METHODS:
        if method not in data:
            continue
        step_data = data[method]
        xs = sorted(step_data.keys())
        means = [step_data[s] for s in xs]
        line_color = METHOD_LINE_COLORS[method]

        ax.plot(
            xs,
            means,
            color=line_color,
            linewidth=1.8,
            marker="o",
            markersize=4,
            label=METHOD_LABELS[method],
            zorder=3,
        )

    all_means = [m for d in data.values() for m in d.values()]
    if all_means:
        y_min = min(all_means) - 0.015
        y_max = max(all_means) + 0.015
        ax.set_ylim(max(0, y_min), min(1.0, y_max))

    ax.set_xlabel("Training Step", fontsize=11)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    if title:
        ax.set_title(title, fontsize=11)


def plot_line_chart(
    data: dict[str, dict[int, float]],
    judge: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _plot_on_ax(ax, data)
    ax.set_ylabel("HealthBench Normalized Score", fontsize=11)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    log.info(f"Figure saved to {output}")

    stem = output.stem
    for ext in [".pdf", ".png"]:
        p = output.with_name(stem + ext)
        if p != output:
            plt.savefig(p, dpi=200, bbox_inches="tight")
            log.info(f"Also saved {p}")

    plt.close()


JUDGE_DISPLAY = {
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1": "GPT-4.1",
}


def plot_line_chart_side_by_side(
    data_by_judge: dict[str, dict[str, dict[int, float]]],
    output: Path,
) -> None:
    judges = list(data_by_judge.keys())
    fig, axes = plt.subplots(1, len(judges), figsize=(6 * len(judges), 4.5), sharey=True)
    if len(judges) == 1:
        axes = [axes]

    for ax, judge in zip(axes, judges):
        _plot_on_ax(
            ax, data_by_judge[judge], title=f"Training Judge: {JUDGE_DISPLAY.get(judge, judge)}"
        )

    axes[0].set_ylabel("HealthBench Normalized Score", fontsize=11)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight")
    log.info(f"Figure saved to {output}")

    stem = output.stem
    for ext in [".pdf", ".png"]:
        p = output.with_name(stem + ext)
        if p != output:
            plt.savefig(p, dpi=200, bbox_inches="tight")
            log.info(f"Also saved {p}")

    plt.close()


async def run(args: argparse.Namespace) -> None:
    if args.healthbench_dir:
        healthbench_eval.HEALTHBENCH_DIR = Path(args.healthbench_dir).resolve()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model_short = args.base_model.split("/")[-1]

    dataset_label = args.dataset
    samples = load_samples(DATASETS_DIR / f"healthbench_{dataset_label}.jsonl")
    prompt_ids = {s.prompt_id for s in samples}
    log.info(f"Loaded {len(samples)} {dataset_label} samples")

    legacy_dirs = list(output_dir.glob("*/")) if output_dir.exists() else []
    cache = ResultCache(output_dir / "cache.jsonl")
    cache.load(legacy_dirs=legacy_dirs)
    log.info(f"Cache has {len(cache)} entries")

    judges = args.judges if args.judges else [args.judge]

    all_runs: list[str] = []
    for judge in judges:
        for method in METHODS:
            run_name = discover_run(base_model_short, method, judge, args.mixture_alpha)
            if run_name:
                all_runs.append(run_name)

    if not args.cache_only:
        import tinker

        judge_client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        judge_inst = PointwiseJudge(
            client=judge_client,
            model=args.eval_judge_model,
            judge_template=POINTWISE_PROMPT,
        )
        service = tinker.ServiceClient(base_url=args.base_url)

        run_steps: dict[str, list[int]] = {}
        for run_name in all_runs:
            available = get_available_steps(run_name)
            valid = [s for s in args.steps if s == 0 or s in available]
            run_steps[run_name] = valid

        needed: list[tuple[str, int]] = []
        for run_name in all_runs:
            for step in run_steps[run_name]:
                cached = cache.get_scores(run_name, step, prompt_ids)
                if len(cached) < len(prompt_ids):
                    needed.append((run_name, step))

        if needed:
            log.info(
                f"{len(needed)} (model, step) pairs need evaluation ({len(samples)} samples each)"
            )
            for run_name, step in needed:
                cached = cache.get_scores(run_name, step, prompt_ids)
                missing = [sample for sample in samples if sample.prompt_id not in cached]
                if not missing:
                    continue
                log.info(
                    f"{run_name} step {step}: {len(cached)} cached, {len(missing)} to evaluate"
                )
                results = await evaluate_checkpoint(
                    missing,
                    sampler_path=None if step == 0 else get_sampler_path(run_name, step),
                    base_model=get_base_model(run_name),
                    judge=judge_inst,
                    service=service,
                    num_completions=args.num_completions,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    gen_concurrency=args.gen_concurrency,
                )
                for result in results:
                    if math.isnan(result.mean_normalized):
                        continue
                    cache.put(
                        result.sample.prompt_id,
                        run_name,
                        step,
                        {
                            "completions": result.completions,
                            "raw_scores": result.raw_scores,
                            "normalized_scores": result.normalized_scores,
                            "normalized_score": result.mean_normalized,
                        },
                    )

    output = Path(args.output)

    if len(judges) > 1:
        data_by_judge: dict[str, dict[str, dict[int, float]]] = {}
        for judge in judges:
            data_by_judge[judge] = collect_line_data(
                cache,
                judge,
                args.steps,
                base_model_short,
                args.mixture_alpha,
                prompt_ids,
            )
        plot_line_chart_side_by_side(data_by_judge, output)
    else:
        line_data = collect_line_data(
            cache,
            judges[0],
            args.steps,
            base_model_short,
            args.mixture_alpha,
            prompt_ids,
        )
        plot_line_chart(line_data, judges[0], output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper line chart: performance across checkpoints")
    p.add_argument("--judge", type=str, default="gpt-4.1")
    p.add_argument("--judges", nargs="+", type=str, default=None)
    p.add_argument("--steps", nargs="+", type=int, default=[0, 100, 200, 300, 400, 500, 600, 700])
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--mixture-alpha", type=float, default=3.0)
    p.add_argument("--dataset", type=str, default="val", choices=["val", "test"])
    p.add_argument("--output", type=str, default="figures/out/line_chart.pdf")
    p.add_argument("--output-dir", type=str, default="healthbench-results/Qwen3-8B")
    p.add_argument("--eval-judge-model", type=str, default="anthropic/claude-opus-4.5")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-completions", type=int, default=1)
    p.add_argument("--gen-concurrency", type=int, default=32)
    p.add_argument("--judge-concurrency", type=int, default=128)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--healthbench-dir", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))
