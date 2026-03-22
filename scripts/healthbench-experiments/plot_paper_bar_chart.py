"""Paper figure: grouped bar chart comparing training methods across judges.

Layout: standalone Base Model bar on the left, then one group per training judge
(e.g. gpt-4.1-mini, gpt-4.1, gpt-5.2).  Each judge group has 3 bars: Pointwise,
Pairwise, TournO.  All bars use validation-based checkpoint selection and
bootstrapped 1-SE error bars evaluated on the test set.

Usage:
    uv run scripts/healthbench-experiments/plot_paper_bar_chart.py \
        --judges gpt-4.1 gpt-5.2 \
        --candidate-steps 0 100 200 300 400 500 600 700 \
        --output figures/out/paper_bar_chart.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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
from eval_common import (
    DATASETS_DIR,
    HEALTHBENCH_DIR,
    EvalPipeline,
    ResultCache,
    bootstrap_se,
    discover_run,
    get_available_steps,
    load_samples,
)
from judges import HealthBenchPointwiseJudge

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


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------


def select_best_step(
    cache: ResultCache,
    model: str,
    steps: list[int],
    val_prompt_ids: set[str],
) -> tuple[int, float] | None:
    best_step, best_avg = -1, -1.0
    for step in steps:
        scores = cache.get_scores(model, step, val_prompt_ids)
        if not scores:
            continue
        avg = float(np.mean(list(scores.values())))
        if avg > best_avg:
            best_avg = avg
            best_step = step
    if best_step < 0:
        return None
    return best_step, best_avg


# ---------------------------------------------------------------------------
# Collect results for the bar chart
# ---------------------------------------------------------------------------


def collect_bar_data(
    cache: ResultCache,
    judges: list[str],
    candidate_steps: list[int],
    base_model_short: str,
    mixture_alpha: float,
    val_prompt_ids: set[str],
    test_prompt_ids: set[str],
    n_bootstrap: int,
) -> tuple[dict, dict]:
    """Returns (bar_data, meta) where bar_data maps labels -> (mean, se, scores)."""
    bar_data: dict[str, tuple[float, float, np.ndarray]] = {}
    meta: dict[str, dict] = {}

    any_run = None
    for judge in judges:
        for method in METHODS:
            run = discover_run(base_model_short, method, judge, mixture_alpha)
            if run:
                any_run = run
                break
        if any_run:
            break

    if any_run:
        base_scores_dict = cache.get_scores(any_run, 0, test_prompt_ids)
        if base_scores_dict:
            scores = np.array(list(base_scores_dict.values()))
            bar_data["Base Model"] = (
                float(scores.mean()),
                bootstrap_se(scores, n_bootstrap),
                scores,
            )
            meta["Base Model"] = {"model": any_run, "step": 0}
        else:
            log.warning("No cached base model results found for step 0")

    for judge in judges:
        for method in METHODS:
            run = discover_run(base_model_short, method, judge, mixture_alpha)
            if not run:
                log.warning(f"No run found for {method}/{judge}, skipping")
                continue

            result = select_best_step(cache, run, candidate_steps, val_prompt_ids)
            if result is None:
                log.warning(f"No val results for {method}/{judge}, skipping")
                continue
            best_step, val_avg = result
            log.info(f"{method}/{judge}: best step={best_step} (val avg={val_avg:.4f})")

            test_scores_dict = cache.get_scores(run, best_step, test_prompt_ids)
            if not test_scores_dict:
                log.warning(f"No test results for {run} step {best_step}")
                continue

            scores = np.array(list(test_scores_dict.values()))
            label = f"{METHOD_LABELS[method]}\n({judge})"
            bar_data[label] = (float(scores.mean()), bootstrap_se(scores, n_bootstrap), scores)
            meta[label] = {"model": run, "step": best_step, "val_avg": val_avg}

    return bar_data, meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_bar_chart(
    bar_data: dict[str, tuple[float, float, np.ndarray]],
    judges: list[str],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(bar_data)), 4.5))

    labels = list(bar_data.keys())
    means = [bar_data[k][0] for k in labels]
    ses = [bar_data[k][1] for k in labels]

    positions: list[float] = []
    colors: list[str] = []
    x = 0.0
    group_centers: list[tuple[float, str]] = []

    for i, label in enumerate(labels):
        if label == "Base Model":
            positions.append(x)
            colors.append(METHOD_COLORS["base"])
            x += 2.0
        else:
            method_key = None
            for mk, ml in METHOD_LABELS.items():
                if label.startswith(ml):
                    method_key = mk
                    break
            colors.append(METHOD_COLORS.get(method_key or "base", "#999999"))

            judge_name = label.split("(")[-1].rstrip(")")
            is_first_in_group = i == 0 or not labels[i - 1].endswith(f"({judge_name})")
            is_last_in_group = i == len(labels) - 1 or not labels[i + 1].endswith(f"({judge_name})")

            if is_first_in_group:
                group_start = x

            positions.append(x)
            x += 0.95

            if is_last_in_group:
                group_end = x - 0.95
                group_centers.append(((group_start + group_end) / 2, judge_name))
                x += 1.0

    bar_width = 0.85
    bars = ax.bar(
        positions,
        means,
        yerr=ses,
        width=bar_width,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        error_kw={"linewidth": 1.0, "capthick": 1.0, "color": "#555555"},
    )

    for bar, avg, se in zip(bars, means, ses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + se + 0.003,
            f"{avg:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="medium",
        )

    method_labels_short = []
    for label in labels:
        if label == "Base Model":
            method_labels_short.append("Base\nModel")
        else:
            method_labels_short.append(label.split("\n")[0])

    ax.set_xticks(positions)
    ax.set_xticklabels(method_labels_short, fontsize=9)

    if positions:
        ax.set_xlim(min(positions) - bar_width, max(positions) + bar_width)

    y_vals = [m + s for m, s in zip(means, ses)]
    if y_vals:
        y_min = min(means) - 0.05
        y_max = max(y_vals) + 0.03
        ax.set_ylim(max(0, y_min), min(1.0, y_max))

    trans = ax.get_xaxis_transform()
    for center, judge_name in group_centers:
        ax.text(
            center,
            -0.10,
            judge_name,
            ha="center",
            va="top",
            fontsize=9,
            fontstyle="italic",
            transform=trans,
        )

    ax.set_ylabel("HealthBench Normalized Score", fontsize=11)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(bottom=0.15)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    import eval_common

    if args.healthbench_dir:
        eval_common.HEALTHBENCH_DIR = Path(args.healthbench_dir).resolve()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_short = args.base_model.split("/")[-1]

    val_samples = load_samples(DATASETS_DIR / "healthbench_val.jsonl")
    test_samples = load_samples(DATASETS_DIR / "healthbench_test.jsonl")
    val_prompt_ids = {s.prompt_id for s in val_samples}
    test_prompt_ids = {s.prompt_id for s in test_samples}
    log.info(f"Loaded {len(val_samples)} val, {len(test_samples)} test samples")

    legacy_dirs = list((output_dir).glob("*/")) if output_dir.exists() else []
    cache = ResultCache(output_dir / "cache.jsonl")
    cache.load(legacy_dirs=legacy_dirs)
    log.info(f"Cache has {len(cache)} entries")

    all_runs: list[str] = []
    for judge in args.judges:
        for method in METHODS:
            run_name = discover_run(base_model_short, method, judge, args.mixture_alpha)
            if run_name:
                all_runs.append(run_name)

    log.info(f"Evaluating the following runs:\n{'\n'.join([f'- {r}' for r in all_runs])}")

    all_steps = args.candidate_steps

    if not args.cache_only:
        import tinker

        judge_client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        judge_inst = HealthBenchPointwiseJudge(client=judge_client, model=args.eval_judge_model)
        service = tinker.ServiceClient(base_url=args.base_url)

        # --- Phase 1: evaluate all candidate steps on VAL only ---
        # Filter candidate steps to only those that exist as checkpoints per run
        run_steps: dict[str, list[int]] = {}
        for run_name in all_runs:
            available = get_available_steps(run_name)
            valid = [s for s in all_steps if s == 0 or s in available]
            run_steps[run_name] = valid
            if len(valid) < len(all_steps):
                log.info(f"{run_name}: {len(valid)}/{len(all_steps)} candidate steps available")

        val_needed: list[tuple[str, int]] = []
        for run_name in all_runs:
            for step in run_steps[run_name]:
                cached = cache.get_scores(run_name, step, val_prompt_ids)
                if len(cached) < len(val_prompt_ids):
                    val_needed.append((run_name, step))

        if val_needed:
            log.info(
                f"Phase 1 (val): {len(val_needed)} (model, step) pairs need evaluation "
                f"({len(val_samples)} samples each)"
            )
            val_pipeline = EvalPipeline(
                samples=val_samples,
                judge=judge_inst,
                cache=cache,
                service=service,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                gen_concurrency=args.gen_concurrency,
                judge_concurrency=args.judge_concurrency,
            )
            val_models = list({r for r, _ in val_needed})
            val_steps = sorted({s for _, s in val_needed})
            await val_pipeline.run(val_models, val_steps)

        # --- Phase 2: pick best step per run, evaluate only that on TEST ---
        test_needed_runs: list[tuple[str, int]] = []
        for run_name in all_runs:
            result = select_best_step(cache, run_name, run_steps[run_name], val_prompt_ids)
            if result is None:
                continue
            best_step, val_avg = result
            log.info(f"Val selection: {run_name} -> step {best_step} (avg={val_avg:.4f})")
            cached = cache.get_scores(run_name, best_step, test_prompt_ids)
            if len(cached) < len(test_prompt_ids):
                test_needed_runs.append((run_name, best_step))

        # also ensure base model (step 0) is evaluated on test
        if all_runs:
            base_cached = cache.get_scores(all_runs[0], 0, test_prompt_ids)
            if len(base_cached) < len(test_prompt_ids):
                test_needed_runs.append((all_runs[0], 0))

        if test_needed_runs:
            log.info(
                f"Phase 2 (test): {len(test_needed_runs)} (model, step) pairs need evaluation "
                f"({len(test_samples)} samples each)"
            )
            test_pipeline = EvalPipeline(
                samples=test_samples,
                judge=judge_inst,
                cache=cache,
                service=service,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                gen_concurrency=args.gen_concurrency,
                judge_concurrency=args.judge_concurrency,
            )
            test_models = list({r for r, _ in test_needed_runs})
            test_steps = sorted({s for _, s in test_needed_runs})
            await test_pipeline.run(test_models, test_steps)

    bar_data, meta = collect_bar_data(
        cache=cache,
        judges=args.judges,
        candidate_steps=all_steps,
        base_model_short=base_model_short,
        mixture_alpha=args.mixture_alpha,
        val_prompt_ids=val_prompt_ids,
        test_prompt_ids=test_prompt_ids,
        n_bootstrap=args.n_bootstrap,
    )

    for label, info in meta.items():
        log.info(f"  {label}: {info}")

    output = Path(args.output)
    plot_bar_chart(bar_data, args.judges, output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper bar chart: methods x judges")
    p.add_argument("--judges", nargs="+", default=["gpt-4.1-mini", "gpt-4.1", "gpt-5.2"])
    p.add_argument(
        "--candidate-steps", nargs="+", type=int, default=[0, 100, 200, 300, 400, 500, 600, 700]
    )
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--mixture-alpha", type=float, default=3.0)
    p.add_argument("--output", type=str, default="figures/out/paper_bar_chart.pdf")
    p.add_argument("--output-dir", type=str, default="healthbench-results/Qwen3-8B")
    p.add_argument("--eval-judge-model", type=str, default="anthropic/claude-opus-4.5")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-concurrency", type=int, default=32)
    p.add_argument("--judge-concurrency", type=int, default=128)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached results; skip Tinker evaluation for missing data",
    )
    p.add_argument("--healthbench-dir", type=str, default=HEALTHBENCH_DIR)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))
