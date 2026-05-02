import json
from pathlib import Path
from typing import Any

import pydantic
from paths import TRAIN_DATASET_PATH

from tourno.data import DataLoader


class RewardBenchSample(pydantic.BaseModel):
    prompt: str
    response: str
    is_chosen: bool
    subset: str
    source_id: str
    row_id: int | None = None


class RewardBenchEvalSample(pydantic.BaseModel):
    prompt: str
    chosen: list[str]
    rejected: list[str]
    subset: str
    source_id: str
    row_id: int | None = None


class RewardBenchDataLoader(DataLoader[RewardBenchSample]):
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
            RewardBenchSample,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            epoch_step=epoch_step,
            **kwargs,
        )


def load_eval_samples(path: Path, max_samples: int | None = None) -> list[RewardBenchEvalSample]:
    samples: list[RewardBenchEvalSample] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                samples.append(RewardBenchEvalSample.model_validate(json.loads(line)))

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples
