"""Build WildChat train/val splits + load WildBench test set.

- Train/val: pulled from `allenai/WildChat-1M` (filtered to English, single-turn, length-bounded).
- Test: pulled from `allenai/WildBench` v2-hard (256 prompts) for fast eval, or v2 (1024) for full.

Output:
    datasets/tulu3if_train.jsonl   (~750 prompts by default)
    datasets/tulu3if_val.jsonl     (~50 prompts)
    datasets/tulu3if_all.jsonl     (train + val combined)
    datasets/tulu3if_test.jsonl   (256 or 1024 prompts with checklists)

Usage:
    uv run scripts/wildchat-experiments/build_dataset.py [--n-train 750 --n-val 50 --wildbench-version v2-hard]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "datasets"


def first_user_turn(conversation) -> str | None:
    for msg in conversation:
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            return content if content else None
    return None


def build_wildchat_splits(n_train: int, n_val: int, seed: int) -> tuple[list[dict], list[dict]]:
    print("Loading allenai/WildChat-1M (streaming)…")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    rng = random.Random(seed)
    candidates: list[dict] = []
    target = n_train + n_val
    needed = target * 8

    for row in ds:
        if row.get("toxic"):
            continue
        if row.get("language") != "English":
            continue
        if (row.get("turn") or 0) < 1:
            continue
        first = first_user_turn(row.get("conversation") or [])
        if not first:
            continue
        n_chars = len(first)
        if n_chars < 80 or n_chars > 4000:
            continue
        candidates.append(
            {
                "conversation_hash": row["conversation_hash"],
                "user_query": first,
                "language": row.get("language") or "English",
            }
        )
        if len(candidates) >= needed:
            break

    print(f"  collected {len(candidates)} candidates after filter")
    rng.shuffle(candidates)
    candidates = candidates[:target]

    samples: list[dict] = []
    for c in candidates:
        samples.append(
            {
                "prompt": [{"role": "user", "content": c["user_query"]}],
                "prompt_id": f"wildchat_{c['conversation_hash'][:16]}",
                "user_query": c["user_query"],
                "history": "",
                "checklist": [],
                "primary_tag": "",
            }
        )
    train = samples[:n_train]
    val = samples[n_train:]
    return train, val


def build_wildbench_test(version: str) -> list[dict]:
    if version == "v2-hard":
        name = "v2-hard"
    elif version == "v2":
        name = "v2"
    elif version == "v1":
        name = "v1"
    else:
        raise ValueError(f"Unknown wildbench version: {version}")
    print(f"Loading allenai/WildBench {name}…")
    ds = load_dataset("allenai/WildBench", name, split="test")
    out: list[dict] = []
    for row in ds:
        ci = row.get("conversation_input") or []
        last_user_idx = max(
            (i for i, m in enumerate(ci) if m.get("role") == "user"), default=-1
        )
        if last_user_idx < 0:
            continue
        user_query = (ci[last_user_idx].get("content") or "").strip()
        history_msgs = ci[:last_user_idx]
        history_str = ""
        if history_msgs:
            chunks = []
            for m in history_msgs:
                role = m.get("role", "user").upper()
                chunks.append(f"[{role}]\n{m.get('content', '')}")
            history_str = "\n\n".join(chunks)

        if history_str:
            prompt_messages = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in history_msgs
            ] + [{"role": "user", "content": user_query}]
        else:
            prompt_messages = [{"role": "user", "content": user_query}]

        out.append(
            {
                "prompt": prompt_messages,
                "prompt_id": f"wildbench_{row.get('session_id', row.get('id'))}",
                "user_query": user_query,
                "history": history_str,
                "checklist": list(row.get("checklist") or []),
                "primary_tag": row.get("primary_tag") or "",
            }
        )
    print(f"  loaded {len(out)} test prompts")
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=750)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--wildbench-version", type=str, default="v2-hard",
        choices=["v2", "v2-hard", "v1"],
    )
    args = ap.parse_args()

    train, val = build_wildchat_splits(args.n_train, args.n_val, args.seed)
    write_jsonl(OUT_DIR / "tulu3if_train.jsonl", train)
    write_jsonl(OUT_DIR / "tulu3if_val.jsonl", val)
    write_jsonl(OUT_DIR / "tulu3if_all.jsonl", train + val)

    test = build_wildbench_test(args.wildbench_version)
    write_jsonl(OUT_DIR / "tulu3if_test.jsonl", test)

    print("\n=== Summary ===")
    print(f"  train: {len(train)} prompts")
    print(f"  val:   {len(val)} prompts")
    print(f"  test:  {len(test)} prompts ({args.wildbench_version})")


if __name__ == "__main__":
    main()
