from typing import Protocol, TypedDict


class PairwiseComparison(TypedDict):
    prompt: str
    completion1: str
    completion2: str


class PairwiseJudge(Protocol):
    async def __call__(self, samples: list[PairwiseComparison]) -> list[float]:
        """
        Return a list of probabilities that completion1 > completion2.

        For binary classifiers, note that 0 = completion2 and 1 = completion1 under this scheme.
        """
        ...


class PointwiseJudge(Protocol):
    async def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:
        """
        Return a list of scores for each (prompt, completion) pair.
        """
        ...
