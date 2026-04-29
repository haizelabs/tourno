"""Evaluate a Qwen3-8B model (base or LoRA checkpoint) on the cwbench v3 test set
(321 prompts) using the OFFICIAL EQ-bench creative-writing-bench judging prompt
+ scoring formula. Output is leaderboard-comparable.

Usage:
    # Base Qwen3-8B (no LoRA), single iteration
    uv run scripts/cwbench-experiments/eval_on_test.py \
        --label base --base-model Qwen/Qwen3-8B \
        --out cwbench-rl/eval/base.json

    # Pointwise final LoRA
    uv run scripts/cwbench-experiments/eval_on_test.py \
        --label pointwise \
        --model-path tinker://3be33ba1-aa26-59bf-82b0-bf3ac51fca92:train:0/sampler_weights/000120 \
        --out cwbench-rl/eval/pointwise.json

The eval generates `n_iterations` completions per prompt, scores each with Sonnet 4.5
through the official 22-criterion 0-20 rubric, and reports:
  - overall_score          0-20  (per cwbench v3 scoring.py)
  - eqbench_creative_score 0-100 (overall_score × 5; the leaderboard number)
  - per_criterion          mean per criterion, direction-corrected
  - per_category           mean per cwbench category
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import tinker
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

load_dotenv()

PROMPTS_DIR = Path("./prompts")
OFFICIAL_TEMPLATE = (PROMPTS_DIR / "cwbench_judge_official.txt").read_text()
POSITIVE_CRITERIA = (PROMPTS_DIR / "cwbench_criteria.txt").read_text().strip().splitlines()
NEGATIVE_CRITERIA = (PROMPTS_DIR / "cwbench_negative_criteria.txt").read_text().strip().splitlines()
ALL_CRITERIA = POSITIVE_CRITERIA  # Note: official criteria.txt already mixes positive + negative
NEGATIVE_SET = set(c.strip() for c in NEGATIVE_CRITERIA)
SCORE_RANGE_MAX = 20.0

log = logging.getLogger("cwbench_eval")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)


# ---------- Official scoring (mirrors cwbench v3 core/scoring.py) -----------


_SCORE_PAT_1 = re.compile(r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)")
_SCORE_PAT_2 = re.compile(r"(.*?):\s*\[(-?\d+(?:\.\d+)?)\]")


def parse_judge_scores(judge_response: str) -> dict[str, float]:
    """Match cwbench v3 parse_judge_scores_creative byte-for-byte."""
    scores: dict[str, float] = {}
    for pat in (_SCORE_PAT_1, _SCORE_PAT_2):
        for m in pat.findall(judge_response):
            metric = m[0].strip()
            try:
                val = float(m[1])
            except ValueError:
                continue
            if val <= SCORE_RANGE_MAX:
                scores[metric] = val
    # Filter to only the 22 canonical criteria (drops noise from the [Analysis] section)
    canon = set(POSITIVE_CRITERIA + NEGATIVE_CRITERIA)
    return {k: v for k, v in scores.items() if k in canon}


def piece_score_0_20(scores: dict[str, float]) -> float:
    """Per-piece average score on 0-20 scale, with negative criteria inverted."""
    vals: list[float] = []
    for metric, val in scores.items():
        adj = (SCORE_RANGE_MAX - val) if metric in NEGATIVE_SET else val
        vals.append(adj)
    return sum(vals) / len(vals) if vals else 0.0


# ---------- Tinker generation ----------


def _stop_token_ids(renderer) -> set[int]:
    out: set[int] = set()
    for s in renderer.get_stop_sequences():
        if isinstance(s, int):
            out.add(s)
        elif isinstance(s, str):
            ids = renderer.tokenizer.encode(s)
            if ids:
                out.add(ids[-1])
    return out


async def generate_for_prompt(
    sampling_client: tinker.SamplingClient,
    renderer,
    prompt_text: str,
    *,
    n_iterations: int,
    max_tokens: int,
    temperature: float,
) -> list[tuple[str, int]]:
    """Returns a list of (completion_text, n_tokens) of length n_iterations."""
    obs = renderer.build_generation_prompt([{"role": "user", "content": prompt_text}])
    completions = await sampling_client.sample_async(
        prompt=obs,
        num_samples=n_iterations,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens, temperature=temperature
        ),
    )
    stop_ids = _stop_token_ids(renderer)
    out: list[tuple[str, int]] = []
    for seq in completions.sequences:
        tokens = list(seq.tokens)
        if tokens and tokens[-1] in stop_ids:
            tokens = tokens[:-1]
        out.append((renderer.tokenizer.decode(tokens), len(tokens)))
    return out


# ---------- Sonnet judge ----------


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(log, logging.WARNING),
    retry_error_callback=lambda _state: None,
)
async def judge_one(
    client: AsyncOpenAI,
    model: str,
    writing_prompt: str,
    completion: str,
    *,
    request_timeout_s: float = 120.0,
    sampling_args: dict | None = None,
) -> str | None:
    """Returns raw judge response text or None on retry exhaustion."""
    user_prompt = OFFICIAL_TEMPLATE.format(
        writing_prompt=writing_prompt,
        test_model_response=completion,
        lower_is_better_criteria="\n".join(NEGATIVE_CRITERIA),
        creative_writing_criteria="\n".join(ALL_CRITERIA),
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=request_timeout_s,
        stream=False,
        **(sampling_args or {}),
    )
    content = resp.choices[0].message.content
    if not content:
        finish = resp.choices[0].finish_reason
        raise RuntimeError(f"Empty judge content (finish_reason={finish!r})")
    return content


# ---------- Eval driver ----------


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="Tag for this eval run (e.g. base, pointwise, tourno)")
    p.add_argument("--model-path", default=None, help="tinker:// LoRA sampler weights path. Omit for base.")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--renderer", default="qwen3_disable_thinking")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Generation temperature; cwbench v3 leaderboard default = 0.7")
    p.add_argument("--n-iterations", type=int, default=1, help="Completions per prompt")
    p.add_argument("--n-prompts", type=int, default=None, help="Limit prompts (smoke test). Default = all 321.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--judge-model", default="anthropic/claude-sonnet-4.5")
    p.add_argument("--judge-base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--judge-api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--test-set", default="datasets/tulu3if_test.jsonl")
    p.add_argument("--out", required=True, help="Output JSON path (per-prompt + summary)")
    args = p.parse_args()

    # --- Load prompts ---
    prompts_data: list[dict] = []
    with open(args.test_set) as f:
        for line in f:
            prompts_data.append(json.loads(line))
    if args.n_prompts:
        prompts_data = prompts_data[: args.n_prompts]
    log.info(f"Loaded {len(prompts_data)} prompts from {args.test_set}")

    # --- Build sampling client ---
    service = tinker.ServiceClient()
    if args.model_path:
        log.info(f"Loading LoRA from {args.model_path}")
        sampling_client = await service.create_sampling_client_async(model_path=args.model_path)
    else:
        log.info(f"Loading base model {args.base_model}")
        sampling_client = await service.create_sampling_client_async(base_model=args.base_model)
    tokenizer = get_tokenizer(args.base_model)
    renderer = get_renderer(args.renderer, tokenizer)
    log.info(f"Renderer: {args.renderer}")

    # --- Build judge client ---
    judge_kwargs: dict = {"api_key": os.environ[args.judge_api_key_env]}
    if args.judge_base_url:
        judge_kwargs["base_url"] = args.judge_base_url
    judge_client = AsyncOpenAI(**judge_kwargs)
    judge_sampling_args = {"temperature": args.judge_temperature, "max_tokens": 2048}

    # --- Generate + judge in parallel per (prompt, iteration) ---
    sem = asyncio.Semaphore(args.num_workers)

    results: list[dict] = []  # one entry per prompt

    async def process_prompt(idx: int, row: dict) -> None:
        async with sem:
            t0 = time.monotonic()
            try:
                pairs = await generate_for_prompt(
                    sampling_client,
                    renderer,
                    row["writing_prompt"],
                    n_iterations=args.n_iterations,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:
                log.error(f"[{idx}] generation failed: {exc}")
                results.append({**row, "error": f"generation: {exc}", "iterations": []})
                return
            iter_blocks: list[dict] = []
            for it_idx, (completion_text, n_tok) in enumerate(pairs):
                judge_response = await judge_one(
                    judge_client,
                    args.judge_model,
                    row["writing_prompt"],
                    completion_text,
                    sampling_args=judge_sampling_args,
                )
                if judge_response is None:
                    iter_blocks.append({
                        "iteration": it_idx,
                        "completion": completion_text,
                        "token_len": n_tok,
                        "judge_scores": {},
                        "judge_response": None,
                        "piece_score_0_20": None,
                    })
                    continue
                scores = parse_judge_scores(judge_response)
                ps = piece_score_0_20(scores)
                iter_blocks.append({
                    "iteration": it_idx,
                    "completion": completion_text,
                    "token_len": n_tok,
                    "judge_scores": scores,
                    "judge_response": judge_response,
                    "piece_score_0_20": ps,
                })
            elapsed = time.monotonic() - t0
            results.append({
                **{k: row[k] for k in ("prompt_id", "scenario_id", "category", "writing_prompt") if k in row},
                "iterations": iter_blocks,
                "elapsed_s": elapsed,
            })
            done = len(results)
            mean_so_far = np.mean([
                blk["piece_score_0_20"]
                for r in results for blk in r.get("iterations", [])
                if blk.get("piece_score_0_20") is not None
            ])
            log.info(
                f"[{done}/{len(prompts_data)}] {row.get('prompt_id','?')} "
                f"({elapsed:.1f}s) running mean 0-20={mean_so_far:.3f}"
            )

    log.info(f"Starting {len(prompts_data)} prompts × {args.n_iterations} iterations "
             f"with {args.num_workers} workers...")
    await asyncio.gather(*[process_prompt(i, r) for i, r in enumerate(prompts_data)])

    # --- Aggregate (matches official compute_creative_scores) ---
    piece_scores = [
        blk["piece_score_0_20"]
        for r in results for blk in r.get("iterations", [])
        if blk.get("piece_score_0_20") is not None
    ]
    overall_0_20 = float(np.mean(piece_scores)) if piece_scores else 0.0
    eqbench_score = round(overall_0_20 * 5.0, 2)

    # Per-criterion (direction-corrected so higher = better)
    per_crit: dict[str, list[float]] = {c: [] for c in POSITIVE_CRITERIA + NEGATIVE_CRITERIA}
    for r in results:
        for blk in r.get("iterations", []):
            for crit, val in (blk.get("judge_scores") or {}).items():
                if crit in NEGATIVE_SET:
                    val = SCORE_RANGE_MAX - val
                per_crit.setdefault(crit, []).append(val)
    per_crit_mean = {c: (float(np.mean(vs)) if vs else None) for c, vs in per_crit.items()}

    # Per-category
    per_cat: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("category", "") or "unknown"
        for blk in r.get("iterations", []):
            if blk.get("piece_score_0_20") is not None:
                per_cat.setdefault(cat, []).append(blk["piece_score_0_20"])
    per_cat_mean = {c: float(np.mean(vs)) for c, vs in per_cat.items()}

    summary = {
        "label": args.label,
        "model_path": args.model_path,
        "base_model": args.base_model,
        "renderer": args.renderer,
        "judge_model": args.judge_model,
        "n_prompts": len(prompts_data),
        "n_iterations": args.n_iterations,
        "n_pieces_scored": len(piece_scores),
        "overall_score_0_20": round(overall_0_20, 4),
        "eqbench_creative_score_0_100": eqbench_score,
        "per_criterion_0_20": {c: (round(v, 3) if v is not None else None) for c, v in per_crit_mean.items()},
        "per_category_0_20": {c: round(v, 3) for c, v in per_cat_mean.items()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "rows": results}, f, ensure_ascii=False)

    print()
    print(f"=== {args.label} ===")
    print(f"overall_score (0-20):        {summary['overall_score_0_20']}")
    print(f"eqbench_creative_score (0-100): {summary['eqbench_creative_score_0_100']}")
    print(f"pieces scored: {summary['n_pieces_scored']}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
