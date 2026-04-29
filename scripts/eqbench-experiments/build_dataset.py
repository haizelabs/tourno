"""Parse EQ-Bench 3 scenarios → Tourno JSONL train/val/test splits.

Each (scenario_id, sub_prompt_index) becomes one sample. Multi-turn role-play
sub-prompts are flattened to independent single-turn samples — a deliberate
simplification for RL training where each rollout is scored in isolation.

Task type is derived from scenario_id:
- 401-420  → "analysis" (uses the analysis master/rubric)
- others   → "standard"

Usage:
    uv run scripts/eqbench-experiments/build_dataset.py
        [--seed 42] [--train 0.85] [--val 0.075]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIOS_FILE = ROOT / "datasets" / "eqbench3_scenarios.txt"
MASTER_STANDARD = ROOT / "prompts" / "eqbench3_master_standard.txt"
MASTER_ANALYSIS = ROOT / "prompts" / "eqbench3_master_analysis.txt"

ANALYSIS_IDS = {str(i) for i in range(401, 421)}

SCENARIO_HEADER = re.compile(r"^######## +(\d+) +\|")
SUBPROMPT_HEADER = re.compile(r"^####### +Prompt(\d+)\s*$")


def parse_scenarios(path: Path) -> list[dict]:
    """Return [{scenario_id, sub_prompt_index, text}, ...]."""
    samples: list[dict] = []
    current_id: str | None = None
    current_sub: int | None = None
    current_lines: list[str] = []

    def flush():
        if current_id is not None and current_sub is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                samples.append(
                    {
                        "scenario_id": current_id,
                        "sub_prompt_index": current_sub,
                        "text": text,
                    }
                )

    lines = path.read_text().splitlines()
    for line in lines:
        m_scenario = SCENARIO_HEADER.match(line)
        m_sub = SUBPROMPT_HEADER.match(line)
        if m_scenario:
            flush()
            current_id = m_scenario.group(1)
            current_sub = None
            current_lines = []
            # Analysis scenarios have no explicit Prompt1 header in the file,
            # so pre-seed sub_prompt_index = 1 for analysis IDs.
            if current_id in ANALYSIS_IDS:
                current_sub = 1
        elif m_sub:
            flush()
            current_sub = int(m_sub.group(1))
            current_lines = []
        else:
            if current_id is not None and current_sub is not None:
                current_lines.append(line)
    flush()
    return samples


def build_samples(raw: list[dict]) -> list[dict]:
    master_standard = MASTER_STANDARD.read_text()
    master_analysis = MASTER_ANALYSIS.read_text()

    out: list[dict] = []
    for r in raw:
        task_type = "analysis" if r["scenario_id"] in ANALYSIS_IDS else "standard"
        master = master_analysis if task_type == "analysis" else master_standard
        user_content = master.format(scenario_prompt=r["text"])
        prompt_id = f"eq3_{r['scenario_id']}_p{r['sub_prompt_index']}"

        out.append(
            {
                "prompt": [{"role": "user", "content": user_content}],
                "prompt_id": prompt_id,
                "scenario_id": r["scenario_id"],
                "sub_prompt_index": r["sub_prompt_index"],
                "task_type": task_type,
                "scenario_text": r["text"],
            }
        )
    return out


def stratified_split(
    samples: list[dict], train_frac: float, val_frac: float, seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split keeping all sub-prompts of a given scenario in the same split.

    Prevents train/test leakage where the model sees Prompt1 of a scenario at
    training time and Prompt2 of the same scenario at test time.
    """
    by_scenario: dict[str, list[dict]] = {}
    for s in samples:
        by_scenario.setdefault(s["scenario_id"], []).append(s)

    standard_ids = sorted(sid for sid in by_scenario if sid not in ANALYSIS_IDS)
    analysis_ids = sorted(sid for sid in by_scenario if sid in ANALYSIS_IDS)

    rng = random.Random(seed)
    rng.shuffle(standard_ids)
    rng.shuffle(analysis_ids)

    def split_ids(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(ids)
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        n_val = min(n_val, n - n_train - 1) if n - n_train - 1 > 0 else 0
        return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]

    std_tr, std_vl, std_te = split_ids(standard_ids)
    an_tr, an_vl, an_te = split_ids(analysis_ids)

    def gather(ids: list[str]) -> list[dict]:
        out: list[dict] = []
        for sid in ids:
            out.extend(by_scenario[sid])
        return out

    return gather(std_tr + an_tr), gather(std_vl + an_vl), gather(std_te + an_te)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.85)
    ap.add_argument("--val", type=float, default=0.075)
    args = ap.parse_args()

    raw = parse_scenarios(SCENARIOS_FILE)
    samples = build_samples(raw)

    train, val, test = stratified_split(samples, args.train, args.val, args.seed)

    out_dir = ROOT / "datasets"
    write_jsonl(out_dir / "eqbench3_train.jsonl", train)
    write_jsonl(out_dir / "eqbench3_val.jsonl", val)
    write_jsonl(out_dir / "eqbench3_test.jsonl", test)

    def summarize(name: str, rows: list[dict]) -> str:
        std = sum(1 for r in rows if r["task_type"] == "standard")
        an = sum(1 for r in rows if r["task_type"] == "analysis")
        scenarios = len({r["scenario_id"] for r in rows})
        return f"  {name:5s} n={len(rows):3d}  scenarios={scenarios:2d}  standard={std:3d}  analysis={an:3d}"

    print(f"Parsed {len(samples)} samples from {len({s['scenario_id'] for s in samples})} scenarios")
    print(summarize("train", train))
    print(summarize("val", val))
    print(summarize("test", test))


if __name__ == "__main__":
    main()
