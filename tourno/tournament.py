import hashlib
from collections.abc import Awaitable, Iterator
from typing import Callable

import numpy as np

from tourno.types import (
    PairwiseComparison,
    PairwiseJudge,
)

MIN_SIMILARITY_WEIGHT = 0.25
DISTANCE_WEIGHT_POWER = 1.0


### Pointwise Reward ###
async def pointwise_reward_fn(
    prompt: str,
    completions: list[str],
    reward_fn: Callable[[str, list[str]], Awaitable[list[float]]],
) -> list[float]:
    return await reward_fn(prompt, completions)


def _should_swap_stable(prompt: str, completion_a: str, completion_b: str) -> bool:
    h_a = hashlib.sha256(f"{prompt}|{completion_a}".encode("utf-8")).hexdigest()
    h_b = hashlib.sha256(f"{prompt}|{completion_b}".encode("utf-8")).hexdigest()
    if h_a <= h_b:
        return False
    else:
        return True


### Pairwise Round Robin ###
async def compute_round_robin_wins(
    prompt: str, completions: list[str], judge_fn: PairwiseJudge
) -> np.ndarray:
    n = len(completions)
    judge_inputs: list[PairwiseComparison] = []
    index_pairs: list[tuple[int, int, bool]] = []

    for i in range(n):
        for j in range(i + 1, n):
            swap = _should_swap_stable(prompt, completions[i], completions[j])
            if swap:
                judge_inputs.append(
                    {
                        "prompt": prompt,
                        "completion1": completions[j],
                        "completion2": completions[i],
                    }
                )
            else:
                judge_inputs.append(
                    {
                        "prompt": prompt,
                        "completion1": completions[i],
                        "completion2": completions[j],
                    }
                )

            index_pairs.append((i, j, swap))

    judge_outputs = await judge_fn(judge_inputs)
    wins = np.zeros((n, n))
    for idx, (i, j, swapped) in enumerate(index_pairs):
        completion1_won = int(judge_outputs[idx] > 0.5)
        if swapped:
            wins[j, i] += completion1_won
            wins[i, j] += 1 - completion1_won
        else:
            wins[j, i] += 1 - completion1_won
            wins[i, j] += completion1_won

    return wins


async def round_robin_reward_fn(
    prompt: str, completions: list[str], judge_fn: PairwiseJudge
) -> list[float]:
    n = len(completions)
    wins = await compute_round_robin_wins(prompt, completions, judge_fn)
    win_rates: np.ndarray = wins.sum(axis=1) / (n - 1)

    return win_rates.tolist()


async def weighted_round_robin_reward_fn(
    prompt: str,
    completions: list[str],
    judge_fn: PairwiseJudge,
    *,
    min_similarity_weight: float = MIN_SIMILARITY_WEIGHT,
    distance_weight_power: float = DISTANCE_WEIGHT_POWER,
) -> list[float]:
    """
    Two-pass scoring:
    1) Estimate baseline performance from unweighted win rates.
    2) Reweight each pair result by performance distance so that:
       - similar-performance opponents are downweighted
       - different-performance opponents are upweighted
    """
    n = len(completions)
    wins = await compute_round_robin_wins(prompt, completions, judge_fn)
    baseline_win_rates: np.ndarray = wins.sum(axis=1) / (n - 1)

    performance_distance = np.abs(
        baseline_win_rates[:, np.newaxis] - baseline_win_rates[np.newaxis, :]
    )
    max_distance = performance_distance.max()
    if max_distance > 0:
        normalized_distance = performance_distance / max_distance
    else:
        normalized_distance = performance_distance
    weights = min_similarity_weight + (
        (1.0 - min_similarity_weight) * np.power(normalized_distance, distance_weight_power)
    )
    np.fill_diagonal(weights, 0.0)

    weighted_wins = (wins * weights).sum(axis=1)
    total_weights = weights.sum(axis=1)

    return (weighted_wins / total_weights).tolist()


### Batched ELO ###
def _sample_matches(
    elo: np.ndarray,
    match_counts: np.ndarray,
    batch_size: int,
    alpha: float,
    rng: np.random.Generator,
) -> Iterator[tuple[int, int]]:
    n = len(elo)
    pair_i, pair_j = np.triu_indices(n, k=1)
    n_pairs = len(pair_i)

    elo_diff = elo[pair_i] - elo[pair_j]
    win_ij = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    win_ij_var = win_ij * (1.0 - win_ij)
    exploration = 1.0 / (1.0 + match_counts[pair_i, pair_j])

    priority = win_ij_var + alpha * exploration
    priority /= priority.sum()

    selected = rng.choice(n_pairs, size=min(batch_size, n_pairs), p=priority, replace=False)
    swaps = rng.random(len(selected)) < 0.5
    for idx, swap in zip(selected, swaps):
        if swap:
            yield (int(pair_j[idx]), int(pair_i[idx]))
        else:
            yield (int(pair_i[idx]), int(pair_j[idx]))


async def batched_elo_reward_fn(
    prompt: str,
    completions: list[str],
    judge_fn: PairwiseJudge,
    *,
    k: float = 32.0,
    convergence_threshold: float = 8.0,
    convergence_patience: int = 3,
    seed: int = 42,
) -> tuple[list[float], int]:
    assert len(completions) > 1

    rng = np.random.default_rng(seed)
    n = len(completions)
    batch_size = min(n // 2, 4)
    total_comparisons = int(np.ceil(n * np.log2(n)))
    max_iters = max(convergence_patience, total_comparisons // batch_size)

    elo = np.zeros(n)
    match_counts = np.zeros((n, n), dtype=int)
    converged_rounds = 0
    total_judge_calls = 0

    for _ in range(max_iters):
        elo_prev = elo.copy()

        matches = list(
            _sample_matches(
                elo,
                match_counts,
                batch_size=batch_size,
                alpha=1.0,
                rng=rng,
            )
        )

        judge_inputs: list[PairwiseComparison] = [
            {"prompt": prompt, "completion1": completions[a], "completion2": completions[b]}
            for a, b in matches
        ]
        judge_outputs = await judge_fn(judge_inputs)
        total_judge_calls += len(judge_inputs)

        for idx, (a, b) in enumerate(matches):
            a_won = int(judge_outputs[idx] > 0.5)
            expected_a = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
            elo[a] += k * (a_won - expected_a)
            elo[b] -= k * (a_won - expected_a)
            match_counts[a, b] += 1
            match_counts[b, a] += 1

        if np.max(np.abs(elo - elo_prev)) > convergence_threshold:
            converged_rounds = 0
        else:
            converged_rounds += 1
            if converged_rounds >= convergence_patience:
                break

    elo_range = max(elo.max() - elo.min(), 1e-4)
    return ((elo - elo.min()) / elo_range).tolist(), total_judge_calls
