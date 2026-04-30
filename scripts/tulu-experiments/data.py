from typing import Any

import pydantic

from tourno.data import DataLoader
from tourno.training.types import PreferenceSample


class TuluSample(pydantic.BaseModel):
    id: str
    prompt: str
    constraints: list[str]
    row_id: int | None = None


class TuluDataLoader(DataLoader[TuluSample]):
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
            TuluSample,
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
