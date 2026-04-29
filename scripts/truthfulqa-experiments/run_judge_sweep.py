from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "truthfulqa-experiments" / "train.py"
LOG_DIR = REPO_ROOT / "logs" / "sweep"
CHAIN_LOG = LOG_DIR / "chain.log"
TRUTHFULQA_DIR = REPO_ROOT / "truthfulqa-rl"

DEFAULT_JUDGES = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "openai/gpt-4.1",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2.6",
]
DEFAULT_METHODS = ["pointwise", "mixture"]

# Held-constant training config (matches prior 3B-judge sweep).
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B"
LEARNING_RATE = 4e-5
BATCH_SIZE = 4
LORA_RANK = 8
GROUP_SIZE = 8
N_EPOCHS = 1.0
SAVE_EVERY = 15
MAX_TOKENS = 1024
PAIRWISE_ALPHA = 3.0
NUM_WORKERS = 8

# In-training eval config (n=4 lenient+strict on full val with gpt-5.4).
EVAL_JUDGE_MODEL = "openai/gpt-5.4"
EVAL_JUDGE_PROVIDER = "openrouter"
EVAL_JUDGES = ["lenient", "strict"]
EVAL_N_SAMPLES = 4
EVAL_GEN_CONCURRENCY = 64
EVAL_JUDGE_CONCURRENCY = 64


def _short_name(model: str) -> str:
    """Filesystem-safe abbreviation: 'meta-llama/llama-3.1-70b-instruct' -> 'llama-3.1-70b-instruct'."""
    return model.split("/")[-1]


def _run_dir(judge_model: str, method: str, base_model: str) -> Path:
    """Mirror pioneer.types.TrainConfig.run_name so we can detect prior runs."""
    name = (
        f"{_short_name(base_model)}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_lora{LORA_RANK}"
        f"_{method}_judge{judge_model}"
    )
    if PAIRWISE_ALPHA > 0:
        name += f"_alpha{PAIRWISE_ALPHA}"
    name += "_importance_sampling"
    return TRUTHFULQA_DIR / name


def _build_command(
    judge_model: str, method: str, wandb_project: str | None, base_model: str
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        str(TRAIN_SCRIPT),
        "--base-model",
        base_model,
        "--judge-type",
        method,
        "--judge-model",
        judge_model,
        "--judge-provider",
        "openrouter",
        "--pairwise-alpha",
        str(PAIRWISE_ALPHA),
        "--learning-rate",
        str(LEARNING_RATE),
        "--batch-size",
        str(BATCH_SIZE),
        "--group-size",
        str(GROUP_SIZE),
        "--lora-rank",
        str(LORA_RANK),
        "--n-epochs",
        str(N_EPOCHS),
        "--save-every",
        str(SAVE_EVERY),
        "--max-tokens",
        str(MAX_TOKENS),
        "--num-workers",
        str(NUM_WORKERS),
        "--eval-every-checkpoint",
        "--eval-judges",
        *EVAL_JUDGES,
        "--eval-judge-model",
        EVAL_JUDGE_MODEL,
        "--eval-judge-provider",
        EVAL_JUDGE_PROVIDER,
        "--eval-n-samples",
        str(EVAL_N_SAMPLES),
        "--eval-gen-concurrency",
        str(EVAL_GEN_CONCURRENCY),
        "--eval-judge-concurrency",
        str(EVAL_JUDGE_CONCURRENCY),
    ]
    if wandb_project:
        cmd += ["--wandb-project", wandb_project]
    return cmd


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _append_chain(line: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAIN_LOG, "a") as f:
        f.write(line + "\n")


async def _run_one(
    judge_model: str,
    method: str,
    sem: asyncio.Semaphore,
    dry_run: bool,
    wandb_project: str | None,
    base_model: str,
    start_delay: float = 0.0,
) -> tuple[str, str, int]:
    """Run a single training job, gated by the global semaphore.

    ``start_delay`` is applied before acquiring the semaphore so that the
    initial wave of parallel tasks is spread out in time (avoids hammering
    Tinker setup or judge endpoints all at once). After the first wave, tasks
    naturally stagger as semaphore slots free up, so the delay has no further
    effect.
    """
    if start_delay > 0 and not dry_run:
        await asyncio.sleep(start_delay)
    async with sem:
        log_path = LOG_DIR / f"{_short_name(base_model)}__{_short_name(judge_model)}_{method}.log"
        cmd = _build_command(judge_model, method, wandb_project, base_model)
        tag = f"{_short_name(base_model)}/{_short_name(judge_model)}/{method}"

        pretty = " ".join(shlex.quote(c) for c in cmd)
        if dry_run:
            print(f"[dry-run] {tag} -> {log_path}")
            print(f"          {pretty}")
            return judge_model, method, 0

        _append_chain(f"=== {_now()}: starting {tag} (log={log_path.name}) ===")
        print(f"[{_now()}] starting {tag}; logging to {log_path}")
        sys.stdout.flush()

        with open(log_path, "wb") as f_out:
            f_out.write(f"# command: {pretty}\n".encode("utf-8"))
            f_out.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=f_out,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=os.environ.copy(),
            )
            rc = await proc.wait()

        _append_chain(f"=== {_now()}: {tag} exited rc={rc} ===")
        print(f"[{_now()}] finished {tag} rc={rc}")
        sys.stdout.flush()
        return judge_model, method, rc


async def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base policy model (HuggingFace path, e.g. 'Qwen/Qwen3-8B-Base').",
    )
    p.add_argument(
        "--judges",
        nargs="+",
        default=DEFAULT_JUDGES,
        help="OpenRouter judge model slugs to sweep over.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["pointwise", "pairwise", "mixture"],
        default=DEFAULT_METHODS,
        help="Reward methods to run for each judge (default: pointwise + mixture).",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Number of training runs to execute concurrently.",
    )
    p.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.0,
        help=(
            "Delay between launching the i-th and (i+1)-th tasks in the queue."
            " Useful for spreading out the initial parallel burst when several"
            " runs would otherwise hit the same upstream service simultaneously."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a run's checkpoint directory already exists under truthfulqa-rl/, "
            "skip it. Use --no-skip-existing to force a rerun."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without launching anything.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT"),
        help="Wandb project (defaults to $WANDB_PROJECT). Pass empty string to disable.",
    )
    args = p.parse_args()
    wandb_project = args.wandb_project or None

    if args.max_parallel < 1:
        print("--max-parallel must be >= 1", file=sys.stderr)
        return 2

    matrix: list[tuple[str, str]] = [(j, m) for j in args.judges for m in args.methods]
    print(
        f"Sweep matrix size: {len(matrix)} ({len(args.judges)} judges x"
        f" {len(args.methods)} methods)"
    )
    print(f"base model: {args.base_model}")
    print(f"wandb project: {wandb_project!r}")

    todo: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for judge_model, method in matrix:
        run_dir = _run_dir(judge_model, method, args.base_model)
        ckpt = run_dir / "checkpoints.jsonl"
        if args.skip_existing and ckpt.exists():
            skipped.append((judge_model, method))
        else:
            todo.append((judge_model, method))

    if skipped:
        print(f"Skipping {len(skipped)} existing run(s):")
        for j, m in skipped:
            print(f"  - {_short_name(j)} / {m}")
    print(f"Will launch {len(todo)} run(s) with max_parallel={args.max_parallel}")

    if not todo:
        print("Nothing to do.")
        return 0

    if not args.dry_run:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _append_chain(
            f"=== {_now()}: sweep launch ({len(todo)} runs, max_parallel={args.max_parallel}) ==="
        )

    sem = asyncio.Semaphore(args.max_parallel)
    started = time.time()
    results = await asyncio.gather(
        *[
            _run_one(
                j,
                method=m,
                sem=sem,
                dry_run=args.dry_run,
                wandb_project=wandb_project,
                base_model=args.base_model,
                start_delay=args.stagger_seconds * i,
            )
            for i, (j, m) in enumerate(todo)
        ],
        return_exceptions=True,
    )

    if args.dry_run:
        return 0

    elapsed = time.time() - started
    failures = []
    for r in results:
        if isinstance(r, BaseException):
            failures.append(("?", "?", repr(r)))
            continue
        j, m, rc = r
        if rc != 0:
            failures.append((j, m, f"rc={rc}"))

    summary = (
        f"=== {_now()}: sweep done in {elapsed/60:.1f} min,"
        f" {len(todo) - len(failures)}/{len(todo)} succeeded ==="
    )
    print(summary)
    _append_chain(summary)
    if failures:
        for j, m, why in failures:
            line = f"FAILED {_short_name(j) if '/' in j else j} / {m}: {why}"
            print(line)
            _append_chain(line)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
