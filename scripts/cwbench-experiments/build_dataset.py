"""Build TournO train/val/test splits for Creative Writing Bench v3.

Pipeline:
  train/val  ← Open-Writing-Bench (11 files, deduped) ∖ Creative-Writing-Bench v3
  test       ← full Creative-Writing-Bench v3 (the canonical 321-prompt leaderboard eval)

This gives a clean, leaderboard-compatible setup: the eval is the public cwbench v3
benchmark at eqbench.com/creative_writing.html, and training uses held-out sibling
prompts from OWB (same authors, same 22-criterion 0-20 rubric, zero leakage).

Splitting the training pool is scenario-stratified by (source_file, scenario_id):
all seeds of a given scenario stay in the same split.

Usage:
    uv run scripts/cwbench-experiments/build_dataset.py \
        [--seed 42] [--train 0.95] [--val 0.05]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OWB_DIR = ROOT / "datasets" / "owb_raw"
CWBENCH_V3 = ROOT / "datasets" / "cwbench_prompts.json"
OUT_DIR = ROOT / "datasets"


def _short_source(fname: str) -> str:
    """Shorten a source filename to a human-readable abbreviation."""
    stem = fname.replace("creative_writing_prompts_v3_", "").replace(".json", "")
    return stem or "v3"


def _prompt_key(writing_prompt: str, seed_modifier: str) -> str:
    """Stable hash for deduplication and leakage checking."""
    h = hashlib.sha256()
    h.update(writing_prompt.encode("utf-8"))
    h.update(b"\x00")
    h.update(seed_modifier.encode("utf-8"))
    return h.hexdigest()[:16]


def _expand_file(path: Path, source_tag: str) -> list[dict]:
    """Load one JSON of {scenario_id: {writing_prompt, seed_modifiers, ...}}
    and yield one record per (scenario, seed_modifier) pair.

    A missing/empty seed_modifiers list produces a single record with seed='' and
    the raw writing_prompt as the final prompt.
    """
    raw = json.loads(path.read_text())
    out: list[dict] = []
    for scenario_id, obj in raw.items():
        base = obj.get("writing_prompt", "")
        mods = obj.get("seed_modifiers") or [""]
        for i, modifier in enumerate(mods):
            final = base.replace("<SEED>", modifier) if "<SEED>" in base else base
            out.append(
                {
                    "key": _prompt_key(base, modifier),
                    "source": source_tag,
                    "orig_scenario_id": str(scenario_id),
                    "seed_index": i,
                    "category": obj.get("category", ""),
                    "title": obj.get("title", ""),
                    "writing_prompt": final,
                    "base_template": base,
                    "seed_modifier": modifier,
                }
            )
    return out


def load_cwbench_v3() -> list[dict]:
    return _expand_file(CWBENCH_V3, source_tag="cwbench_v3")


def load_owb_pool() -> list[dict]:
    files = sorted(OWB_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No OWB files in {OWB_DIR}. Vendor the 11 creative_writing_prompts_v3_*.json "
            "files from https://github.com/EQ-bench/open-writing-bench/tree/main/data"
        )
    records: list[dict] = []
    for f in files:
        records.extend(_expand_file(f, source_tag=_short_source(f.name)))
    return records


def dedupe(records: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in records:
        if r["key"] not in seen:
            seen[r["key"]] = r
    return list(seen.values())


def to_tourno_sample(r: dict) -> dict:
    """Final shape matching CreativeBenchSample (scripts/cwbench-experiments/cwbench_types.py)."""
    source = r["source"]
    return {
        "prompt": [{"role": "user", "content": r["writing_prompt"]}],
        "prompt_id": f"{source}_{r['orig_scenario_id']}_s{r['seed_index']}",
        "scenario_id": f"{source}:{r['orig_scenario_id']}",
        "seed_index": r["seed_index"],
        "category": r["category"],
        "title": r["title"],
        "writing_prompt": r["writing_prompt"],
    }


def scenario_stratified_split(
    records: list[dict], train_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split by (source, orig_scenario_id) so all seeds of a scenario stay together."""
    by_scenario: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["source"], r["orig_scenario_id"])
        by_scenario.setdefault(key, []).append(r)

    scenario_keys = sorted(by_scenario.keys())
    rng = random.Random(seed)
    rng.shuffle(scenario_keys)

    n = len(scenario_keys)
    n_train = max(1, int(round(train_frac * n)))
    train_keys = scenario_keys[:n_train]
    val_keys = scenario_keys[n_train:]

    train = [r for k in train_keys for r in by_scenario[k]]
    val = [r for k in val_keys for r in by_scenario[k]]
    return train, val


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.95, help="Train fraction of OWB∖v3 pool")
    ap.add_argument("--val", type=float, default=0.05, help="Val fraction (remainder if omitted)")
    args = ap.parse_args()

    # --- Load test set (full cwbench v3) ---
    test_records = load_cwbench_v3()
    test_keys = {r["key"] for r in test_records}
    print(f"test (cwbench v3): {len(test_records)} samples, {len(test_keys)} unique keys")

    # --- Load OWB pool and dedupe ---
    owb_raw = load_owb_pool()
    owb_deduped = dedupe(owb_raw)
    print(f"OWB (deduped): {len(owb_deduped)} samples from {len(owb_raw)} raw entries")

    # --- Subtract test to prevent leakage ---
    before = len(owb_deduped)
    owb_held_out = [r for r in owb_deduped if r["key"] not in test_keys]
    removed = before - len(owb_held_out)
    print(f"  removed {removed} entries that overlap with cwbench v3 (leakage prevention)")

    # --- Stratified train/val split ---
    train_records, val_records = scenario_stratified_split(
        owb_held_out, train_frac=args.train, seed=args.seed
    )

    # --- Leakage check: no train/val key in test ---
    train_keys = {r["key"] for r in train_records}
    val_keys = {r["key"] for r in val_records}
    assert not (train_keys & test_keys), f"LEAKAGE: {len(train_keys & test_keys)} train ∩ test"
    assert not (val_keys & test_keys), f"LEAKAGE: {len(val_keys & test_keys)} val ∩ test"
    assert not (train_keys & val_keys), f"{len(train_keys & val_keys)} train ∩ val"

    # --- Convert to CreativeBenchSample shape and write ---
    train_samples = [to_tourno_sample(r) for r in train_records]
    val_samples = [to_tourno_sample(r) for r in val_records]
    test_samples = [to_tourno_sample(r) for r in test_records]

    write_jsonl(OUT_DIR / "cwbench_train.jsonl", train_samples)
    write_jsonl(OUT_DIR / "cwbench_val.jsonl", val_samples)
    write_jsonl(OUT_DIR / "cwbench_test.jsonl", test_samples)

    def summarize(name: str, rows: list[dict]) -> str:
        n = len(rows)
        sources = sorted({r["scenario_id"].split(":", 1)[0] for r in rows})
        scenarios = len({r["scenario_id"] for r in rows})
        cats = len({r["category"] for r in rows if r["category"]})
        return (
            f"  {name:5s} n={n:4d}  scenarios={scenarios:3d}  categories={cats:2d}  "
            f"sources={sources}"
        )

    print()
    print(f"leakage check: zero overlap between train∪val and test ✓")
    print(summarize("train", train_samples))
    print(summarize("val", val_samples))
    print(summarize("test", test_samples))


if __name__ == "__main__":
    main()
