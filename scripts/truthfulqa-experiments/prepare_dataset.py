"""Download TruthfulQA from HuggingFace and split into train/val/test JSONL files.

Usage:
    uv run scripts/truthfulqa-experiments/prepare_dataset.py

Writes to datasets/truthfulqa_{train,val,test}.jsonl
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42


def make_prompt_id(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("domenicrosati/TruthfulQA", split="train")

    ds = ds.shuffle(seed=SEED)
    n = len(ds)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": ds.select(range(n_train)),
        "val": ds.select(range(n_train, n_train + n_val)),
        "test": ds.select(range(n_train + n_val, n)),
    }

    for split_name, split_ds in splits.items():
        out_path = DATASETS_DIR / f"truthfulqa_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for row in split_ds:
                sample = {
                    "prompt": [{"role": "user", "content": row["Question"]}],
                    "prompt_id": make_prompt_id(row["Question"]),
                    "best_answer": row["Best Answer"],
                    "correct_answers": [ans.strip() for ans in row["Correct Answers"].split(";")],
                    "incorrect_answers": [
                        ans.strip() for ans in row["Incorrect Answers"].split(";")
                    ],
                    "category": row["Category"],
                }
                f.write(json.dumps(sample) + "\n")

        print(f"Wrote {len(split_ds)} samples to {out_path}")

    print(f"\nTotal: {n} samples -> train={n_train}, val={n_val}, test={n - n_train - n_val}")


if __name__ == "__main__":
    main()
