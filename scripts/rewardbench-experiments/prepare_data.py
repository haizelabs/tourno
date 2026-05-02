import json
import random
from pathlib import Path
from typing import Any

from paths import DATASETS_DIR, TEST_DATASET_PATH, TRAIN_DATASET_PATH, VAL_DATASET_PATH

from datasets import load_dataset

DATASET_NAME = "allenai/reward-bench-2"
TRAIN_SIZE = 1000
VAL_SIZE = 100
SPLIT_SEED = 42


def download_rewardbench2() -> list[dict[str, Any]]:
    dataset = load_dataset(DATASET_NAME, split="test")
    return [dict(row) for row in dataset]


def build_splits(
    rows: list[dict[str, Any]],
    *,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < train_size + val_size:
        raise ValueError(f"Dataset has {len(rows)} rows, need at least {train_size + val_size}")

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    shuffled = [rows[i] for i in indices]
    train = shuffled[:train_size]
    val = shuffled[train_size : train_size + val_size]
    test = shuffled[train_size + val_size :]
    return train, val, test


def _source_id(row: dict[str, Any]) -> str:
    return str(row["id"])


def expand_train_row(row: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    chosen = list(row["chosen"])
    rejected = list(row["rejected"])
    if not chosen:
        raise ValueError(f"Row {_source_id(row)} has no chosen responses")
    if not rejected:
        raise ValueError(f"Row {_source_id(row)} has no rejected responses")

    base = {
        "prompt": row["prompt"],
        "subset": row["subset"],
        "source_id": _source_id(row),
    }
    return [
        {
            **base,
            "response": chosen[0],
            "is_chosen": True,
        },
        {
            **base,
            "response": rng.choice(rejected),
            "is_chosen": False,
        },
    ]


def normalize_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "chosen": list(row["chosen"]),
        "rejected": list(row["rejected"]),
        "subset": row["subset"],
        "source_id": _source_id(row),
    }


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_splits(out_dir: Path = DATASETS_DIR) -> None:
    rows = download_rewardbench2()
    train_rows, val_rows, test_rows = build_splits(rows)

    train_rng = random.Random(SPLIT_SEED)
    train = [sample for row in train_rows for sample in expand_train_row(row, train_rng)]
    val = [normalize_eval_row(row) for row in val_rows]
    test = [normalize_eval_row(row) for row in test_rows]

    _write_jsonl(out_dir / Path(TRAIN_DATASET_PATH).name, train)
    _write_jsonl(out_dir / Path(VAL_DATASET_PATH).name, val)
    _write_jsonl(out_dir / Path(TEST_DATASET_PATH).name, test)

    print(f"train: {len(train):>5} -> {TRAIN_DATASET_PATH}")
    print(f"val:   {len(val):>5} -> {VAL_DATASET_PATH}")
    print(f"test:  {len(test):>5} -> {TEST_DATASET_PATH}")


if __name__ == "__main__":
    write_splits()
