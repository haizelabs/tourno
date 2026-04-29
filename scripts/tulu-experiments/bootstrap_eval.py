"""Bootstrap 95% CI on eqbench_creative_score from saved eval JSONs.

Mirrors core/scoring.py:bootstrap_benchmark_stability_creative from upstream cwbench v3:
sample tasks with replacement, recompute score, repeat N times.

Usage:
    uv run scripts/cwbench-experiments/bootstrap_eval.py \
        cwbench-rl/eval/base.json \
        cwbench-rl/eval/pointwise.json \
        cwbench-rl/eval/tourno.json \
        cwbench-rl/eval/tourno-step120.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np


def piece_scores_from_eval(path: str) -> tuple[str, list[float], list[int]]:
    d = json.load(open(path))
    label = d["summary"]["label"]
    pieces: list[float] = []
    token_lens: list[int] = []
    for r in d.get("rows", []):
        for blk in r.get("iterations", []):
            ps = blk.get("piece_score_0_20")
            if ps is not None:
                pieces.append(ps)
                token_lens.append(blk.get("token_len", 0))
    return label, pieces, token_lens


def bootstrap_ci(scores: list[float], n_boot: int = 1000, ci: float = 0.95) -> dict:
    rng = random.Random(42)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [scores[rng.randrange(len(scores))] for _ in range(len(scores))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int((1 - ci) / 2 * n_boot)]
    hi = boots[int((1 + ci) / 2 * n_boot) - 1]
    point = sum(scores) / len(scores)
    return {
        "n": len(scores),
        "point_0_20": round(point, 4),
        "ci_lo_0_20": round(lo, 4),
        "ci_hi_0_20": round(hi, 4),
        "point_0_100": round(point * 5, 2),
        "ci_lo_0_100": round(lo * 5, 2),
        "ci_hi_0_100": round(hi * 5, 2),
        "ci_halfwidth_0_100": round((hi - lo) * 5 / 2, 2),
        "se_0_100": round(np.std(boots) * 5, 3),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = sys.argv[1:]
    print()
    print("Bootstrap (n=1000, 95% CI on eqbench_creative_score)")
    print("=" * 86)
    print(
        f"{'label':<22} {'n':>5}  {'eqbench (95% CI)':<22}  {'half-width':>10}  {'avg tok_len':>11}"
    )
    print("-" * 86)

    rows = []
    for p in paths:
        label, scores, tok_lens = piece_scores_from_eval(p)
        if not scores:
            print(f"{label:<22} (no piece scores)")
            continue
        ci = bootstrap_ci(scores)
        avg_len = (sum(tok_lens) / len(tok_lens)) if tok_lens else 0
        rows.append((label, ci, avg_len))
        rng_str = f"{ci['point_0_100']:.2f}  [{ci['ci_lo_0_100']:.2f}, {ci['ci_hi_0_100']:.2f}]"
        print(f"{label:<22} {ci['n']:>5}  {rng_str:<22}  ±{ci['ci_halfwidth_0_100']:>5.2f}     {avg_len:>7.0f}")

    # Pairwise differences (point estimate of difference + bootstrap-ish CI via SE addition)
    print()
    print("Pairwise differences (Δ eqbench, with naive ±2σ CI from bootstrap SEs)")
    print("-" * 86)
    for i, (la, cia, _) in enumerate(rows):
        for j, (lb, cib, _) in enumerate(rows):
            if j <= i:
                continue
            delta = cib["point_0_100"] - cia["point_0_100"]
            se = (cia["se_0_100"] ** 2 + cib["se_0_100"] ** 2) ** 0.5
            sig = "**" if abs(delta) > 2 * se else ""
            print(f"  {lb} − {la}:  {delta:+.2f}  (±{2*se:.2f}, ~95% CI) {sig}")
    print()
    print('** = difference is outside the ±2σ band (≈ statistically distinguishable at 95%)')


if __name__ == "__main__":
    main()
