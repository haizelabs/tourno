from typing import Any

import pydantic

from tourno.data import DataLoader
from tourno.training.types import PreferenceSample

TRAIN_DATASET_PATH = "./datasets/healthbench_train.jsonl"


class Rubric(pydantic.BaseModel):
    criterion: str
    points: int
    tags: list[str]


class HealthBenchSample(pydantic.BaseModel):
    prompt: list[dict[str, str]]
    prompt_id: str
    rubrics: list[Rubric]
    canary: str
    row_id: int | None = None


class HealthBenchDataLoader(DataLoader[HealthBenchSample]):
    def __init__(
        self,
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
            TRAIN_DATASET_PATH,
            HealthBenchSample,
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
