import asyncio
import hashlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from tourno.eval.judges import PairwiseJudge, PointwiseJudge

MIN_SIMILARITY_WEIGHT = 0.25
DISTANCE_WEIGHT_POWER = 1.0


async def adaptive_pointwise_rewards(
    prompt: str,
    completions: list[str],
    judge: PointwiseJudge,
    *,
    total_samples: int,
    seed: int = 42,
    **template_kwargs: Any,
) -> list[float]:
    n = len(completions)
    assert n >= 1 and total_samples >= n

    phase1 = await asyncio.gather(
        *(judge(prompt=prompt, completion=c, **template_kwargs) for c in completions)
    )
    sums = np.array(phase1, dtype=float)
    total_counts = np.ones(n, dtype=int)
    rng = np.random.default_rng(seed)

    remaining = total_samples - n
    while remaining > 0:
        ranks = np.argsort(np.argsort(sums)) + 1
        weights = (ranks * (n + 1 - ranks)).astype(float) / (total_counts * (total_counts + 1))

        batch_size = min(n // 2, remaining)  # TODO: This is a pretty arbitray batch size
        selected = rng.choice(n, size=batch_size, p=weights / weights.sum(), replace=True).tolist()
        results = await asyncio.gather(
            *(judge(prompt=prompt, completion=completions[i], **template_kwargs) for i in selected)
        )

        for i, score in zip(selected, results):
            sums[i] += score
            total_counts[i] += 1

        remaining -= batch_size

    return (sums / total_counts).tolist()


def _should_swap_stable(prompt: str, completion_a: str, completion_b: str) -> bool:
    h_a = hashlib.sha256(f"{prompt}|{completion_a}".encode("utf-8")).hexdigest()
    h_b = hashlib.sha256(f"{prompt}|{completion_b}".encode("utf-8")).hexdigest()
    if h_a <= h_b:
        return False
    else:
        return True


async def _compute_round_robin_wins(
    prompt: str,
    completions: list[str],
    judge: PairwiseJudge,
    **template_kwargs: Any,
) -> np.ndarray:
    n = len(completions)
    judge_inputs: list[tuple[str, str]] = []
    index_pairs: list[tuple[int, int, bool]] = []

    for i in range(n):
        for j in range(i + 1, n):
            swap = _should_swap_stable(prompt, completions[i], completions[j])
            if swap:
                judge_inputs.append((completions[j], completions[i]))
            else:
                judge_inputs.append((completions[i], completions[j]))

            index_pairs.append((i, j, swap))

    judge_outputs = await asyncio.gather(
        *(
            judge(
                prompt=prompt,
                completion_a=completion_a,
                completion_b=completion_b,
                **template_kwargs,
            )
            for completion_a, completion_b in judge_inputs
        )
    )
    wins = np.zeros((n, n))
    for idx, (i, j, swapped) in enumerate(index_pairs):
        a_won = 1.0 - judge_outputs[idx]
        if swapped:
            wins[j, i] += a_won
            wins[i, j] += 1.0 - a_won
        else:
            wins[i, j] += a_won
            wins[j, i] += 1.0 - a_won

    return wins


async def round_robin_rewards(
    prompt: str,
    completions: list[str],
    judge: PairwiseJudge,
    **template_kwargs: Any,
) -> list[float]:
    n = len(completions)
    wins = await _compute_round_robin_wins(prompt, completions, judge, **template_kwargs)
    win_rates: np.ndarray = wins.sum(axis=1) / (n - 1)

    return win_rates.tolist()


async def weighted_round_robin_rewards(
    prompt: str,
    completions: list[str],
    judge: PairwiseJudge,
    *,
    min_similarity_weight: float = MIN_SIMILARITY_WEIGHT,
    distance_weight_power: float = DISTANCE_WEIGHT_POWER,
    **template_kwargs: Any,
) -> list[float]:
    """
    Two-pass scoring:
    1) Estimate baseline performance from unweighted win rates.
    2) Reweight each pair result by performance distance so that:
       - similar-performance opponents are downweighted
       - different-performance opponents are upweighted
    """
    n = len(completions)
    wins = await _compute_round_robin_wins(prompt, completions, judge, **template_kwargs)
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


async def batched_elo_rewards(
    prompt: str,
    completions: list[str],
    judge: PairwiseJudge,
    *,
    k: float = 32.0,
    convergence_threshold: float = 8.0,
    convergence_patience: int = 3,
    seed: int = 42,
    **template_kwargs: Any,
) -> list[float]:
    assert len(completions) > 1

    rng = np.random.default_rng(seed)
    n = len(completions)
    batch_size = min(n // 2, 4)
    total_comparisons = int(np.ceil(n * np.log2(n)))
    max_iters = max(convergence_patience, total_comparisons // batch_size)

    elo = np.zeros(n)
    match_counts = np.zeros((n, n), dtype=int)
    converged_rounds = 0

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

        judge_outputs = await asyncio.gather(
            *(
                judge(
                    prompt=prompt,
                    completion_a=completions[a],
                    completion_b=completions[b],
                    **template_kwargs,
                )
                for a, b in matches
            )
        )

        for idx, (a, b) in enumerate(matches):
            a_won = 1.0 - judge_outputs[idx]
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
    return ((elo - elo.min()) / elo_range).tolist()
