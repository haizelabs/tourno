"""Fan out all 5 training methods as parallel subprocesses.

Each method runs in its own subprocess (wandb is process-global, so 5 parallel
runs need 5 processes), each with its own wandb run + log file. All logs land
in the same project + entity so they overlay automatically in W&B.

Usage:
  # 5-step smoke test on all 5 methods
  uv run --env-file .env scripts/tulu-experiment/train_all.py --smoke

  # Full 400-step run on all 5 methods
  uv run --env-file .env scripts/tulu-experiment/train_all.py

  # Subset
  uv run --env-file .env scripts/tulu-experiment/train_all.py \\
      --methods tourno,grpo-point --n-steps 200

The launcher passes through OPENROUTER_API_KEY / TINKER_API_KEY to each
subprocess via inherited env. Run under `uv run --env-file .env` so those are
present in the parent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ALL_METHODS = ("tourno", "grpo-point", "grpo-pair", "dpo", "dpo-online")
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


async def run_one(method: str, args: list[str], log_path: Path) -> tuple[str, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--method", method, *args]
    with log_path.open("w") as f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=f,
            stderr=asyncio.subprocess.STDOUT,
        )
        await proc.wait()
        return method, proc.returncode or 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="5-step smoke: small batch/group, skip checkpoints + val.")
    p.add_argument("--methods", type=str, default="all",
                   help="Comma-separated methods, or 'all'.")
    p.add_argument("--log-dir", type=Path, default=Path("/tmp/tulu-logs"))

    # Pass-through args (mirror train.py defaults; --smoke overrides below)
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B-Base")
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--wandb-project", type=str, default="tourno-tulu")
    p.add_argument("--wandb-entity", type=str, default="Haize-Research")
    p.add_argument("--log-root", type=Path, default=Path("./tulu-runs"))
    p.add_argument("--train-path", type=Path, default=Path("datasets/tulu_if_train.jsonl"))
    p.add_argument("--val-path", type=Path, default=Path("datasets/tulu_if_val.jsonl"))
    p.add_argument("--dpo-train-path", type=Path, default=Path("datasets/dpo/train.jsonl"))
    p.add_argument("--judge-concurrency", type=int, default=128)
    p.add_argument("--val-n-prompts", type=int, default=64)
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    if args.methods == "all":
        methods = list(ALL_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        for m in methods:
            if m not in ALL_METHODS:
                raise SystemExit(f"Unknown method: {m}. Choose from {ALL_METHODS}.")

    if args.smoke:
        args.n_steps = 5
        args.batch_size = 8
        args.group_size = 8
        args.num_workers = 8
        # Save (and validate) once at the end of the smoke run so val/* shows in W&B.
        args.save_every = 5
        args.max_tokens = 256
        args.judge_concurrency = 64
        args.val_n_prompts = 16
        # tulu_if_train.jsonl already carries chosen/rejected from the source
        # dataset, so the offline-DPO method works without waiting for
        # build_dpo.py — perfect for a smoke test.
        args.dpo_train_path = args.train_path
        run_name_suffix = "_smoke"
    else:
        run_name_suffix = ""

    log_dir: Path = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--base-model", args.base_model,
        "--n-steps", str(args.n_steps),
        "--batch-size", str(args.batch_size),
        "--group-size", str(args.group_size),
        "--num-workers", str(args.num_workers),
        "--save-every", str(args.save_every),
        "--max-tokens", str(args.max_tokens),
        "--wandb-project", args.wandb_project,
        "--wandb-entity", args.wandb_entity,
        "--log-root", str(args.log_root),
        "--train-path", str(args.train_path),
        "--val-path", str(args.val_path),
        "--dpo-train-path", str(args.dpo_train_path),
        "--judge-concurrency", str(args.judge_concurrency),
        "--val-n-prompts", str(args.val_n_prompts),
    ]

    model_short = args.base_model.split("/")[-1]
    print(f"Launching {len(methods)} method(s) in parallel:")
    print(f"  base_model:       {args.base_model}")
    print(f"  n_steps:          {args.n_steps}")
    print(f"  batch×group:      {args.batch_size}×{args.group_size}")
    print(f"  wandb project:    {args.wandb_entity}/{args.wandb_project}")
    print(f"  log dir:          {log_dir}")
    print()
    tasks = []
    for m in methods:
        run_name = f"tulu_{m}_{model_short}{run_name_suffix}"
        method_args = [*base_args, "--run-name", run_name]
        log_path = log_dir / f"{m}.log"
        print(f"  {m:12} → {log_path}  (run: {run_name})")
        tasks.append(run_one(m, method_args, log_path))

    print()
    results = await asyncio.gather(*tasks)
    print()
    print("=== Results ===")
    for method, code in results:
        status = "PASS" if code == 0 else f"FAIL (exit {code})"
        print(f"  {method:12} {status}  → {log_dir / f'{method}.log'}")
    if any(code != 0 for _, code in results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
