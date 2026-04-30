"""Pull creative-writing data:

  - Train/val: random sample from `euclaise/writingprompts` (Reddit r/WritingPrompts).
  - Test: full CWB v3 leaderboard set (32 prompts, all metadata preserved so
    you can reuse the EQ-Bench runner for ELO-comparable evaluation).

Outputs to datasets/:
  cw_train.jsonl                 (1000 rows)
  cw_val.jsonl                   (250 rows)
  cw_test.jsonl                  (32 rows, JSONL form of CWB v3 prompts)
  cw_test_eqbench_format.json    (32 prompts in EQ-Bench's exact dict format,
                                  drop-in for `creative_writing_bench.py`)

Schema for train/val rows:
  id, row_id, prompt (chat-format list), raw_prompt

Schema for test rows (matches CWB v3 keys):
  id, row_id, prompt (chat-format list), raw_prompt, title, category,
  seed_modifiers (list[str], typically 10), iterations (int = 3, the EQ-Bench default)

Usage:
    uv run scripts/prepare_creative_writing.py
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path("datasets")
SEED = 42
N_TRAIN = 1000
N_VAL = 250
WRITINGPROMPTS_HF = "euclaise/writingprompts"
CWB_RAW_URL = (
    "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/"
    "data/creative_writing_prompts_v3.json"
)
CWB_DEFAULT_ITERATIONS = 3


def _strip_writingprompt(text: str) -> str:
    """Reddit r/WritingPrompts prompts are usually prefixed with [WP], [EU], etc."""
    text = text.strip()
    # Drop tags like "[WP]", "[ EU ]", "[CW]", "[OT]"
    if text.startswith("[") and "]" in text[:10]:
        text = text[text.index("]") + 1 :].lstrip()
    return text


def fetch_writingprompts(n_train: int, n_val: int, seed: int) -> tuple[list[str], list[str]]:
    print(f"Loading {WRITINGPROMPTS_HF} (this can take ~1 min the first time)...")
    ds = load_dataset(WRITINGPROMPTS_HF, split="train")
    print(f"  loaded {len(ds)} rows")

    rng = random.Random(seed)
    # Pick a column for the prompt (varies by HF version)
    candidate_cols = [c for c in ("prompt", "writing_prompt", "title") if c in ds.column_names]
    if not candidate_cols:
        raise SystemExit(f"No prompt column found in {ds.column_names}")
    col = candidate_cols[0]
    print(f"  using column {col!r}")

    n_total = n_train + n_val
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    # Walk shuffled indices and skip empties
    train: list[str] = []
    val: list[str] = []
    for idx in indices:
        text = (ds[idx][col] or "").strip()
        text = _strip_writingprompt(text)
        if len(text) < 20:
            continue
        if len(train) < n_train:
            train.append(text)
        elif len(val) < n_val:
            val.append(text)
        if len(train) >= n_train and len(val) >= n_val:
            break

    if len(train) < n_train or len(val) < n_val:
        raise SystemExit(
            f"Only got {len(train)} train + {len(val)} val (needed {n_train}+{n_val})"
        )
    return train, val


def fetch_cwb() -> dict:
    print(f"Fetching CWB v3 prompts from {CWB_RAW_URL} ...")
    with urllib.request.urlopen(CWB_RAW_URL) as resp:
        return json.loads(resp.read())


def write_train_val(prompts: list[str], path: Path, prefix: str) -> None:
    with path.open("w") as f:
        for i, raw in enumerate(prompts):
            row = {
                "id": f"{prefix}_{i:05d}",
                "row_id": i,
                "prompt": [{"role": "user", "content": raw}],
                "raw_prompt": raw,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(prompts)} rows -> {path}")


def write_cwb_test(cwb: dict, jsonl_path: Path, eqbench_path: Path) -> None:
    # JSONL form (matches our train/val schema + extra CWB metadata)
    with jsonl_path.open("w") as f:
        for k, v in cwb.items():
            raw = v["writing_prompt"]
            row = {
                "id": f"cwb_{k}",
                "row_id": int(k) - 1,
                "prompt": [{"role": "user", "content": raw}],
                "raw_prompt": raw,
                "title": v.get("title", ""),
                "category": v.get("category", ""),
                "seed_modifiers": v.get("seed_modifiers", []),
                "iterations": CWB_DEFAULT_ITERATIONS,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(cwb)} rows -> {jsonl_path}")

    # EQ-Bench's exact dict format (for their `creative_writing_bench.py` runner)
    with eqbench_path.open("w") as f:
        json.dump(cwb, f, ensure_ascii=False, indent=2)
    print(f"  wrote {len(cwb)} prompts -> {eqbench_path} (EQ-Bench native format)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train, val = fetch_writingprompts(N_TRAIN, N_VAL, SEED)
    write_train_val(train, OUT_DIR / "cw_train.jsonl", prefix="wp")
    write_train_val(val, OUT_DIR / "cw_val.jsonl", prefix="wp")

    cwb = fetch_cwb()
    write_cwb_test(
        cwb,
        OUT_DIR / "cw_test.jsonl",
        OUT_DIR / "cw_test_eqbench_format.json",
    )


if __name__ == "__main__":
    main()
