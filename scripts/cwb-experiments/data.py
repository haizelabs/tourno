import json
import random
from pathlib import Path
from typing import Any

import pydantic

from tourno.data import DataLoader
from tourno.training.types import PreferenceSample

DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
SOURCE_POOL_PATH = DATASETS_DIR / "cwbench_all.jsonl"
SOURCE_TEST_PATH = DATASETS_DIR / "cwbench_test.jsonl"
TRAIN_DATASET_PATH = (DATASETS_DIR / "cwbench_train.jsonl").as_posix()
VAL_DATASET_PATH = (DATASETS_DIR / "cwbench_val.jsonl").as_posix()
TEST_DATASET_PATH = (DATASETS_DIR / "cwbench_test.jsonl").as_posix()

TRAIN_SIZE = 1000
VAL_SIZE = 100
SPLIT_SEED = 42


class CreativeBenchSample(pydantic.BaseModel):
    prompt: list[dict[str, str]]
    prompt_id: str
    scenario_id: str
    seed_index: int
    category: str
    title: str
    writing_prompt: str
    row_id: int | None = None


class CreativeBenchDataLoader(DataLoader[CreativeBenchSample]):
    def __init__(
        self,
        dataset_path: str = TRAIN_DATASET_PATH,
        *,
        batch_size: int = 1,
        max_length: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        epoch: int = 0,
        epoch_step: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_path,
            CreativeBenchSample,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            epoch_step=epoch_step,
            **kwargs,
        )


class PreferenceDataLoader(DataLoader[PreferenceSample]):
    def __init__(
        self,
        dataset_path: str,
        *,
        batch_size: int = 1,
        max_length: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        epoch: int = 0,
        epoch_step: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            dataset_path,
            PreferenceSample,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            epoch_step=epoch_step,
            **kwargs,
        )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_splits() -> None:
    pool = _read_jsonl(SOURCE_POOL_PATH)
    test = _read_jsonl(SOURCE_TEST_PATH)

    if len(pool) < TRAIN_SIZE + VAL_SIZE:
        raise ValueError(
            f"Pool has {len(pool)} rows, need at least {TRAIN_SIZE + VAL_SIZE}"
        )

    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(pool)))
    rng.shuffle(indices)

    train = [pool[i] for i in indices[:TRAIN_SIZE]]
    val = [pool[i] for i in indices[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]]

    _write_jsonl(Path(TRAIN_DATASET_PATH), train)
    _write_jsonl(Path(VAL_DATASET_PATH), val)
    _write_jsonl(Path(TEST_DATASET_PATH), test)

    print(f"train: {len(train):>5} -> {TRAIN_DATASET_PATH}")
    print(f"val:   {len(val):>5} -> {VAL_DATASET_PATH}")
    print(f"test:  {len(test):>5} -> {TEST_DATASET_PATH}")


if __name__ == "__main__":
    build_splits()
