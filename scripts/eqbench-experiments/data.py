import asyncio
from pathlib import Path
from typing import Any

from eqbench_types import EQBenchSample

from datasets import Dataset, load_dataset

TRAIN_DATASET_PATH = "./datasets/eqbench3_train.jsonl"


class EQBenchDataLoader:
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
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        self.curr_epoch = epoch
        self.curr_epoch_step = epoch_step

        self.batch_size = batch_size
        self._iter_lock = asyncio.Lock()

        self._curr_dataset: Dataset
        if Path(TRAIN_DATASET_PATH).exists():
            if not TRAIN_DATASET_PATH.endswith(".jsonl"):
                raise ValueError("Only JSONL files are supported for local dataset loading")
            self._curr_dataset = Dataset.from_json(TRAIN_DATASET_PATH)
        else:
            self._curr_dataset = load_dataset(TRAIN_DATASET_PATH, **kwargs)

        if max_length is not None:
            self._curr_dataset = self._curr_dataset.shuffle(seed=self.seed).select(
                range(self.max_length)
            )

        self.num_rows = len(self._curr_dataset)
        self._set_dataset_to_epoch(self.curr_epoch)

    def __aiter__(self):
        return self

    async def __anext__(self) -> tuple[int, list[EQBenchSample]]:
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

            return self.curr_epoch, [
                EQBenchSample.model_validate({k: rows[k][i] for k in rows}) for i in range(n)
            ]

    def _set_dataset_to_epoch(self, epoch: int) -> None:
        if self.shuffle:
            self._curr_dataset = self._curr_dataset.shuffle(seed=self.seed + epoch)
