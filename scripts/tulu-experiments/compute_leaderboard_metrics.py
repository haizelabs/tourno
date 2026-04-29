"""Compute the cwbench v3 leaderboard's auxiliary metrics on our trained models.

Mirrors the metric pipeline from `generate_results_html.ipynb` in upstream
creative-writing-bench v3 (see /tmp/creative-writing-bench):
  - Style       = vocab_complexity = `core/metrics.py:calculate_complexity_index`
                  (Flesch-Kincaid grade + polysyllabic-word ratio, 0-100)
  - Slop        = `core/metrics.py:calculate_slop_index_new`
                  (slop word + 2*bigram + 8*trigram hits per 1000 tokens)
  - Repetition  = n-gram repetition score from `get_multi_prompt_ngrams`
                  ((top-40 over-used bigram count + top-40 trigram count) /
                   total_text_words * 1000), considering only n-grams that
                  appear in >= 3 distinct prompts (matches notebook).
  - Length      = mean character count per completion (avg_length in notebook)
                  + mean token_len from our eval JSONs for reference.

Usage:
    uv run scripts/cwbench-experiments/compute_leaderboard_metrics.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# --- Wire up upstream cwbench v3 metrics ---------------------------------
UPSTREAM = Path("/tmp/creative-writing-bench")
sys.path.insert(0, str(UPSTREAM))
# The slop_index loaders use relative paths like 'data/slop_list.json',
# so we must chdir into the upstream repo before invoking those funcs.
ORIG_CWD = Path.cwd()

from core.metrics import (  # noqa: E402
    calculate_complexity_index,
    calculate_slop_index_new,
    get_multi_prompt_ngrams,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "cwbench-rl" / "eval"
OUT_PATH = EVAL_DIR / "leaderboard_metrics.json"

EVAL_FILES = [
    ("base", EVAL_DIR / "base.json"),
    ("pointwise", EVAL_DIR / "pointwise.json"),
    ("pairwise", EVAL_DIR / "pairwise.json"),
    ("tourno", EVAL_DIR / "tourno-step120.json"),
]


def load_completions(path: Path):
    """Return (label, prompts_data dict[prompt_id -> [text...]], all_completions list, token_lens)."""
    d = json.loads(path.read_text())
    label = d.get("summary", {}).get("label", path.stem)
    prompts_data: dict[str, list[str]] = defaultdict(list)
    flat: list[str] = []
    token_lens: list[int] = []
    for r in d.get("rows", []):
        pid = r.get("prompt_id") or r.get("scenario_id") or "unknown"
        for it in r.get("iterations", []):
            text = it.get("completion") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            prompts_data[pid].append(text)
            flat.append(text)
            token_lens.append(it.get("token_len", 0))
    return label, dict(prompts_data), flat, token_lens


def compute_repetition_ngram_score(prompts_data: dict[str, list[str]]) -> dict:
    """Replicates the notebook's n-gram repetition score.

    repetition_score = (sum top-40 overused bigrams freq +
                        sum top-40 overused trigrams freq) / total_words * 1000
    Both sourced via `get_multi_prompt_ngrams` with min_prompt_ids=3, top_k=200.
    """
    # Total words across all texts (notebook uses .split() on the flat corpus).
    total_words = sum(len(t.split()) for texts in prompts_data.values() for t in texts)

    top_bigrams = get_multi_prompt_ngrams(
        prompts_data,
        n=2,
        top_k=200,
        min_prompt_ids=3,
        human_profile_path=UPSTREAM / "data" / "human_writing_profile.json",
    )
    top_trigrams = get_multi_prompt_ngrams(
        prompts_data,
        n=3,
        top_k=200,
        min_prompt_ids=3,
        human_profile_path=UPSTREAM / "data" / "human_writing_profile.json",
    )

    top_bigram_count = sum(freq for _, freq in top_bigrams[:40])
    top_trigram_count = sum(freq for _, freq in top_trigrams[:40])

    if total_words == 0:
        return {
            "repetition_score": 0.0,
            "top_bigram_count": 0,
            "top_trigram_count": 0,
            "total_words": 0,
            "top_bigrams_sample": [],
            "top_trigrams_sample": [],
        }
    score = (top_bigram_count + top_trigram_count) / total_words * 1000
    return {
        "repetition_score": round(score, 4),
        "top_bigram_count": int(top_bigram_count),
        "top_trigram_count": int(top_trigram_count),
        "total_words": int(total_words),
        "top_bigrams_sample": [(" ".join(ng), c) for ng, c in top_bigrams[:10]],
        "top_trigrams_sample": [(" ".join(ng), c) for ng, c in top_trigrams[:10]],
    }


def piece_score_stats(path: Path) -> dict:
    d = json.loads(path.read_text())
    pieces: list[float] = []
    for r in d.get("rows", []):
        for it in r.get("iterations", []):
            ps = it.get("piece_score_0_20")
            if ps is not None:
                pieces.append(float(ps))
    if not pieces:
        return {"n": 0, "mean_0_100": 0.0}
    mean = sum(pieces) / len(pieces)
    return {
        "n": len(pieces),
        "mean_0_20": round(mean, 4),
        "mean_0_100": round(mean * 5, 2),
        "piece_scores": pieces,
    }


def main() -> None:
    results: dict[str, dict] = {}

    # Need to run slop_index_new from inside upstream dir (relative paths).
    os.chdir(UPSTREAM)
    try:
        for label, path in EVAL_FILES:
            print(f"\n=== {label} ({path.name}) ===", flush=True)
            if not path.exists():
                print(f"  MISSING: {path}")
                continue
            _orig_label, prompts_data, flat, token_lens = load_completions(path)
            n_prompts = len(prompts_data)
            n_completions = len(flat)
            print(f"  prompts={n_prompts}  completions={n_completions}", flush=True)

            # Length: mean characters per completion (avg_length in notebook),
            # plus mean tokens for our reference (from eval JSON token_len).
            mean_chars = sum(len(t) for t in flat) / max(1, n_completions)
            mean_tokens = sum(token_lens) / max(1, len(token_lens)) if token_lens else 0.0

            # Style: combine all texts, like notebook does ("all_text_combined")
            all_text = "\n\n".join(flat)
            print("  -> calculating vocab_complexity ...", flush=True)
            vocab = calculate_complexity_index(all_text)

            print("  -> calculating slop_index ...", flush=True)
            slop = calculate_slop_index_new(all_text, debug=False)

            print("  -> calculating n-gram repetition ...", flush=True)
            rep = compute_repetition_ngram_score(prompts_data)

            print("  -> bootstrap piece score (mean only) ...", flush=True)
            ps = piece_score_stats(path)

            results[label] = {
                "label": label,
                "source": str(path.relative_to(REPO_ROOT)),
                "n_prompts": n_prompts,
                "n_completions": n_completions,
                "avg_length_chars": round(mean_chars, 2),
                "avg_token_len": round(mean_tokens, 2),
                "vocab_complexity": round(float(vocab), 4),
                "slop_score": round(float(slop), 4),
                "repetition_score": rep["repetition_score"],
                "repetition_detail": {
                    k: v for k, v in rep.items() if k != "repetition_score"
                },
                "piece_score_mean_0_100": ps.get("mean_0_100"),
                "piece_score_n": ps.get("n"),
            }
            print(
                f"  vocab={vocab:.2f}  slop={slop:.2f}  "
                f"rep={rep['repetition_score']:.3f}  chars={mean_chars:.0f}  "
                f"tok={mean_tokens:.0f}",
                flush=True,
            )
    finally:
        os.chdir(ORIG_CWD)

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}")

    # ----- Markdown comparison table -----
    print("\n## cwbench v3 leaderboard auxiliary metrics\n")
    cols = [
        ("Model", "label"),
        ("Rubric (0-100)", "piece_score_mean_0_100"),
        ("Style (vocab)", "vocab_complexity"),
        ("Slop", "slop_score"),
        ("Repetition", "repetition_score"),
        ("Length (chars)", "avg_length_chars"),
        ("Length (tok)", "avg_token_len"),
    ]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    print(header)
    print(sep)
    for label, _ in EVAL_FILES:
        if label not in results:
            continue
        r = results[label]
        cells = []
        for h, k in cols:
            v = r.get(k)
            if isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
