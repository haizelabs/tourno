import asyncio
from pathlib import Path
from typing import Any, Generic, TypeVar

import pydantic

from datasets import Dataset, DatasetDict, load_dataset

SampleT = TypeVar("SampleT", bound=pydantic.BaseModel)


class DataLoader(Generic[SampleT]):
    def __init__(
        self,
        dataset_path: str,
        sample_model: type[SampleT] | None = None,
        *,
        batch_size: int = 1,
        max_length: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        epoch: int = 0,
        epoch_step: int = 0,
        split: str | None = None,
        **kwargs: Any,
    ):
        self.sample_model = sample_model
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        self.curr_epoch = epoch
        self.curr_epoch_step = epoch_step

        self.batch_size = batch_size
        self._iter_lock = asyncio.Lock()

        self._curr_dataset = self._load_dataset(dataset_path, split=split, **kwargs)
        if max_length is not None:
            self._curr_dataset = self._curr_dataset.shuffle(seed=self.seed).select(
                range(max_length)
            )

        self.num_rows = len(self._curr_dataset)
        self._set_dataset_to_epoch(self.curr_epoch)

    def __aiter__(self):
        return self

    async def __anext__(self) -> tuple[int, list[SampleT | dict[str, Any]]]:
        async with self._iter_lock:
            remaining = self.num_rows - self.curr_epoch_step
            n = min(self.batch_size, remaining)
            if n == 0:
                self.curr_epoch += 1
                self.curr_epoch_step = 0
                self._set_dataset_to_epoch(self.curr_epoch)
                n = min(self.batch_size, self.num_rows)

            rows = self._curr_dataset[self.curr_epoch_step : self.curr_epoch_step + n]
            self.curr_epoch_step += n

            batch = [{k: rows[k][i] for k in rows} for i in range(n)]
            if self.sample_model is not None:
                return self.curr_epoch, [self.sample_model.model_validate(row) for row in batch]

            return self.curr_epoch, batch

    def _load_dataset(self, dataset_path: str, *, split: str | None, **kwargs: Any) -> Dataset:
        path = Path(dataset_path)
        if path.exists():
            if path.suffix != ".jsonl":
                raise ValueError("Only JSONL files are supported for local dataset loading")
            return Dataset.from_json(path.as_posix())

        dataset = load_dataset(dataset_path, split=split, **kwargs)
        if isinstance(dataset, Dataset):
            return dataset
        if isinstance(dataset, DatasetDict):
            return dataset[split or "train"]

        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    def _set_dataset_to_epoch(self, epoch: int) -> None:
        if self.shuffle:
            self._curr_dataset = self._curr_dataset.shuffle(seed=self.seed + epoch)
