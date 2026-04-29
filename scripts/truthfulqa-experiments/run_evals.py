from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import tinker
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (
    DATASETS_DIR,
    EvalPipeline,
    discover_runs,
    load_samples,
    make_openai_client,
    mean_score_summary,
    resolve_eval_targets,
)
from gold_judges import get_gold_judge

load_dotenv()
log = logging.getLogger(__name__)

METHODS = ["pointwise", "pairwise", "mixture"]
DEFAULT_STEPS = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 143]


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        DATASETS_DIR / f"truthfulqa_{args.dataset}.jsonl", max_samples=args.max_samples
    )
    log.info(f"Loaded {len(samples)} {args.dataset} samples")

    runs_by_method = discover_runs(
        base_model_short=args.base_model.split("/")[-1],
        methods=METHODS,
        training_judge=args.training_judge,
        mixture_alpha=args.mixture_alpha,
    )
    if not runs_by_method:
        log.error("No runs discovered; nothing to evaluate")
        return

    eval_targets = resolve_eval_targets(
        runs_by_method=runs_by_method,
        steps=args.steps,
        explicit_targets=args.eval_targets,
    )
    if not eval_targets:
        log.error("No valid eval targets; nothing to evaluate")
        return

    judge_client = make_openai_client(args.judge_provider)
    service = tinker.ServiceClient(base_url=args.base_url)
    judges = [
        get_gold_judge(kind, client=judge_client, model=args.judge_model) for kind in args.judges
    ]
    results_path = output_dir / "results.jsonl"
    log.info(
        f"=== Eval with judges {[j.name for j in judges]} (model={args.judge_model}, "
        f"n_samples_per_prompt={args.n_samples_per_prompt}); writing {results_path} ==="
    )

    pipeline = EvalPipeline(
        samples=samples,
        judges=judges,
        service=service,
        results_path=results_path,
        n_samples_per_prompt=args.n_samples_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        gen_concurrency=args.gen_concurrency,
        judge_concurrency=args.judge_concurrency,
    )
    records = await pipeline.run(eval_targets)

    log.info("=== Mean normalized score per (run, step, judge) ===")
    for (run_name, step, kind), (score, n_prompts) in mean_score_summary(
        records, args.judges, eval_targets
    ).items():
        value = f"{score:.3f}" if score is not None else "-"
        log.info(f"run={run_name} step={step} judge={kind} mean={value} n={n_prompts}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gold-judge evaluation for TruthfulQA")

    p.add_argument(
        "--judges",
        nargs="+",
        choices=["strict", "lenient"],
        default=["strict", "lenient"],
    )
    p.add_argument("--judge-model", type=str, default="openai/gpt-5.4")
    p.add_argument(
        "--judge-provider",
        type=str,
        choices=["openai", "openrouter"],
        default="openrouter",
    )

    p.add_argument(
        "--training-judge",
        type=str,
        default="meta-llama/llama-3.2-3b-instruct",
        help="Substring of the training-time judge model used to disambiguate runs.",
    )
    p.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    p.add_argument(
        "--eval-targets",
        "--pairs",
        dest="eval_targets",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Explicit '<method>:<step>' eval targets. Overrides --steps. "
            "Example: --eval-targets pointwise:15 pairwise:60 mixture:60"
        ),
    )
    p.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--mixture-alpha", type=float, default=3.0)

    p.add_argument("--dataset", type=str, default="val", choices=["val", "test"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="truthfulqa-results/Llama-3.2-1B")

    p.add_argument(
        "--n-samples-per-prompt",
        type=int,
        default=1,
        help="Completions per prompt; their scores are averaged.",
    )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-concurrency", type=int, default=32)
    p.add_argument(
        "--judge-concurrency",
        type=int,
        default=64,
        help="Total in-flight judge calls across all judges (single shared cap).",
    )
    p.add_argument("--base-url", type=str, default=None)

    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    exit_code = 0
    try:
        asyncio.run(run(args))
    except BaseException:
        log.exception("Eval run failed")
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

    os._exit(exit_code)
