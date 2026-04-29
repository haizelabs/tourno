"""Estimate cwbench v3 leaderboard ELO for our three models using rubric → ELO
interpolation, exactly per `core/elo.py:interpolate_elo_from_rubric_scores_cw`.

This gives the same number cwbench v3 uses as the *initial* Elo before running
pairwise battles. For full leaderboard parity you'd still need to run pairwise
battles vs the model pool (~thousands of judge calls). The interpolated number
is what gets reported when `--no-elo` is set.

Usage:
    uv run scripts/cwbench-experiments/interpolate_elo.py
"""

from __future__ import annotations

import json
from pathlib import Path

# Path to the unzipped cwbench v3 leaderboard data.
LEADERBOARD_ELO = Path("/tmp/creative-writing-bench/elo_results.json")

# Our eval results — we use the eqbench_creative_score (0-100) directly.
OUR_MODELS = {
    # display label                  : (eval JSON path, alias used in cwbench)
    "Qwen3-8B base":            (Path("cwbench-rl/eval/base.json"),       "qwen-3-8b-base-tourno"),
    "Pointwise GRPO (ours)":    (Path("cwbench-rl/eval/pointwise.json"),  "qwen-3-8b-pointwise-tourno"),
    "TournO mixture (ours)":    (Path("cwbench-rl/eval/tourno-step120.json"), "qwen-3-8b-tourno-mixture"),
}


def load_eval_score(p: Path) -> float:
    """Mean piece_score_0_20 (matches `compute_creative_scores` in cwbench v3)."""
    d = json.loads(p.read_text())
    pieces = [
        blk["piece_score_0_20"]
        for r in d["rows"]
        for blk in r["iterations"]
        if blk.get("piece_score_0_20") is not None
    ]
    return sum(pieces) / len(pieces)


def interpolate(rubric: float, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation matching cwbench v3's logic.
    `points` = list of (rubric_0_20, elo) sorted by rubric ascending.
    """
    if rubric <= points[0][0]:
        return points[0][1]
    if rubric >= points[-1][0]:
        return points[-1][1]
    for (r1, e1), (r2, e2) in zip(points, points[1:]):
        if r1 <= rubric <= r2:
            if r2 == r1:
                return (e1 + e2) / 2
            return e1 + (rubric - r1) / (r2 - r1) * (e2 - e1)
    return points[-1][1]  # fallback


def main() -> None:
    print(f"Loading leaderboard ELO from {LEADERBOARD_ELO} ...")
    leaderboard = json.loads(LEADERBOARD_ELO.read_text())

    # Build (rubric_score_0_20, elo, elo_norm) for every existing model
    points_raw: list[tuple[float, float, str]] = []
    points_norm: list[tuple[float, float, str]] = []
    for model, info in leaderboard.items():
        if model == "__metadata__":
            continue
        rubric = info.get("creative_writing_rubric_score_agg")
        elo = info.get("elo")
        elo_norm = info.get("elo_norm")
        if rubric is None or elo is None:
            continue
        rubric = float(rubric)
        points_raw.append((rubric, float(elo), model))
        if elo_norm is not None:
            points_norm.append((rubric, float(elo_norm), model))

    points_raw.sort(key=lambda x: x[0])
    points_norm.sort(key=lambda x: x[0])

    print(f"Pool: {len(points_raw)} models with rubric+elo, "
          f"{len(points_norm)} with elo_norm")
    print(f"Rubric range: {points_raw[0][0]:.2f} ({points_raw[0][2]}) "
          f"-> {points_raw[-1][0]:.2f} ({points_raw[-1][2]})")
    print(f"ELO range:    {min(p[1] for p in points_raw):.0f} -> "
          f"{max(p[1] for p in points_raw):.0f}")
    print(f"Norm-ELO range (anchored deepseek-r1=1500, ministral-3b=200): "
          f"{min(p[1] for p in points_norm):.0f} -> "
          f"{max(p[1] for p in points_norm):.0f}")
    print()

    # Strip leaderboard model name from interpolation table for cleaner output
    raw_table = [(r, e) for r, e, _ in points_raw]
    norm_table = [(r, e) for r, e, _ in points_norm]

    print("=" * 86)
    print(f"{'Model':<26} {'rubric_0-20':>12} {'eqbench_0-100':>14} "
          f"{'~ELO (raw)':>12} {'~ELO (norm)':>12}")
    print("-" * 86)

    # Interpolate each of our models
    for label, (path, _) in OUR_MODELS.items():
        rubric = load_eval_score(path)
        eqbench = rubric * 5
        est_elo = interpolate(rubric, raw_table)
        est_elo_norm = interpolate(rubric, norm_table)
        print(f"{label:<26} {rubric:>12.3f} {eqbench:>14.2f} "
              f"{est_elo:>12.1f} {est_elo_norm:>12.1f}")
    print()

    # Show context — the closest neighbours in the leaderboard for each rubric tier
    print("Nearest leaderboard neighbours by rubric:")
    print("-" * 86)
    for label, (path, _) in OUR_MODELS.items():
        rubric = load_eval_score(path)
        # Find nearest 2 leaderboard models by absolute rubric distance
        nearest = sorted(points_raw, key=lambda x: abs(x[0] - rubric))[:3]
        nearest_str = ", ".join(
            f"{name} (rubric={r:.2f}, elo_norm="
            f"{next((en for rr, en, n in points_norm if n == name), float('nan')):.0f})"
            for r, _, name in nearest
        )
        print(f"  {label}: {nearest_str}")


if __name__ == "__main__":
    main()
