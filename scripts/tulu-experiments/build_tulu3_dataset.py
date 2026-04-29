"""Build Tulu-3 personas-IF train/val/test splits.

Source: allenai/tulu-3-pref-personas-instruction-following (~19,890 rows).
Each row has:
  - prompt:     the user's persona/scenario instruction (visible to the policy model)
  - constraints: explicit list of verifiable constraints (passed to the JUDGE only,
                 NOT the policy — the policy sees only `prompt`)

We use a fixed seed so train/val/test are deterministic and identical across runs.

Outputs:
    datasets/tulu3if_train.jsonl
    datasets/tulu3if_val.jsonl
    datasets/tulu3if_test.jsonl
    datasets/tulu3if_all.jsonl  (train + val combined for the default training pool)

Usage:
    uv run scripts/wildchat-experiments/build_tulu3_dataset.py \
        [--n-train 800 --n-val 50 --n-test 256 --seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "datasets"


def _to_sample(row: dict, idx: int) -> dict:
    constraints = list(row.get("constraints") or [])
    prompt_text = (row.get("prompt") or "").strip()
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "prompt_id": f"tulu3if_{row.get('id', f'idx{idx}')}",
        "user_query": prompt_text,
        "history": "",
        "constraints": constraints,
        "checklist": [],
        "primary_tag": "",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val",   type=int, default=50)
    ap.add_argument("--n-test",  type=int, default=256)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--min-constraints", type=int, default=1,
                    help="Drop rows with fewer than this many constraints")
    args = ap.parse_args()

    print("Loading allenai/tulu-3-pref-personas-instruction-following …")
    ds = load_dataset(
        "allenai/tulu-3-pref-personas-instruction-following",
        split="train",
    )
    n_total = len(ds)
    print(f"  {n_total} total rows")

    # Materialize a deterministic shuffle by seed (same indices every run)
    indices = list(range(n_total))
    rng = random.Random(args.seed)
    rng.shuffle(indices)

    samples: list[dict] = []
    for i in indices:
        row = ds[i]
        cons = row.get("constraints") or []
        if len(cons) < args.min_constraints:
            continue
        prompt_text = (row.get("prompt") or "").strip()
        if not prompt_text or len(prompt_text) < 30:
            continue
        samples.append(_to_sample(row, i))
        if len(samples) >= args.n_train + args.n_val + args.n_test:
            break

    print(f"  {len(samples)} samples after filter (need {args.n_train + args.n_val + args.n_test})")
    train = samples[: args.n_train]
    val   = samples[args.n_train : args.n_train + args.n_val]
    test  = samples[args.n_train + args.n_val : args.n_train + args.n_val + args.n_test]

    write_jsonl(OUT_DIR / "tulu3if_train.jsonl", train)
    write_jsonl(OUT_DIR / "tulu3if_val.jsonl",   val)
    write_jsonl(OUT_DIR / "tulu3if_test.jsonl",  test)
    write_jsonl(OUT_DIR / "tulu3if_all.jsonl",   train + val)

    # Quick stats
    n_constraints_train = [len(s["constraints"]) for s in train]
    if n_constraints_train:
        avg_c = sum(n_constraints_train) / len(n_constraints_train)
        print(f"\n  avg constraints / prompt (train): {avg_c:.2f} "
              f"(min {min(n_constraints_train)}, max {max(n_constraints_train)})")

    print("\n=== Summary ===")
    print(f"  train: {len(train)} prompts")
    print(f"  val:   {len(val)} prompts")
    print(f"  test:  {len(test)} prompts")
    print(f"  (deterministic seed={args.seed} — same prompts for both pointwise + tourno runs)")


if __name__ == "__main__":
    main()
