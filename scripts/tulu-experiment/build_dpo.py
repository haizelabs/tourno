"""Build DPO preference pairs from multi-model completions.

For each prompt, runs a round-robin pairwise tournament with Claude Sonnet 4.5
as judge over all available completions, then picks the highest-win-rate
completion as `chosen` and the lowest as `rejected`.

USING MODELS:
qwen/qwen3-14b
meta-llama/llama-3-8b 
google/gemma-4-31b-it
google/gemini-3-flash-preview
deepseek/deepseek-v3-base
openai/gpt-5.1

Output JSONL matches `tourno.training.types.PreferenceSample` for direct use
with `train_dpo.py`:
    {prompt: [{role, content}], chosen: str, rejected: str, row_id: int, ...}

Usage:
    uv run scripts/tulu-experiment/build_dpo.py \\
        --completions datasets/dpo/train_completions.jsonl \\
        --samples     datasets/tulu_if_train.jsonl \\
        --output      datasets/dpo/train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from jinja2 import Template
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm as atqdm

from tourno.tournament import compute_round_robin_wins
from tourno.types import PairwiseComparison

JUDGE_MODEL = "anthropic/claude-sonnet-4-5"
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
TEMPLATE_PATH = PROMPTS_DIR / "tulu_pairwise_judge.jinja"

log = logging.getLogger("build_dpo")


class SonnetTuluPairwiseJudge:
    """Pairwise judge backed by Sonnet 4.5 + the Tulu Jinja template.

    Closes over `constraints` so that callers can use the standard
    `PairwiseJudge` protocol (which only carries prompt/completion1/completion2)
    while still passing the constraint list into the rendered template.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        constraints: list[str],
        *,
        model: str,
        template: Template,
        sem: asyncio.Semaphore,
        max_tokens: int,
        timeout_s: float,
        retry_attempts: int,
    ):
        self.client = client
        self.constraints = constraints
        self.model = model
        self.template = template
        self.sem = sem
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retry_attempts = retry_attempts

    async def __call__(self, samples: list[PairwiseComparison]) -> list[float]:
        return await asyncio.gather(*(self._one(s) for s in samples))

    async def _one(self, s: PairwiseComparison) -> float:
        message = self.template.render(
            prompt=s["prompt"],
            completion1=s["completion1"],
            completion2=s["completion2"],
            constraints=self.constraints,
        )

        async def _call() -> str:
            async with self.sem:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": message}],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    timeout=self.timeout_s,
                )
            return (resp.choices[0].message.content or "").strip()

        text = ""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.retry_attempts),
                wait=wait_exponential(multiplier=2, min=2, max=30),
                retry=retry_if_exception_type(Exception),
                reraise=True,
                before_sleep=before_sleep_log(log, logging.WARNING),
            ):
                with attempt:
                    text = await _call()
        except Exception as exc:
            log.warning(f"judge call failed after retries: {exc!r}; defaulting to 0.5")
            return 0.5

        # Per template: "0" → completion1 better, "1" → completion2 better.
        # We return P(completion1 ≻ completion2): 1.0 if "0", 0.0 if "1".
        if not text:
            return 0.5
        first = text[0]
        if first == "0":
            return 1.0
        if first == "1":
            return 0.0
        return 0.5


def _load_completions(path: Path) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("completion"):
                by_id[row["id"]].append(row)
    return by_id


def _load_samples(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_id[row["id"]] = row
    return by_id


def _load_existing(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


async def process_one_prompt(
    *,
    sample: dict,
    completions_for_prompt: list[dict],
    client: AsyncOpenAI,
    template: Template,
    judge_sem: asyncio.Semaphore,
    judge_model: str,
    judge_max_tokens: int,
    judge_timeout_s: float,
    judge_retry_attempts: int,
) -> dict | None:
    # Add Tulu's gpt-4o pairs as additional tournament candidates. The original
    # `chosen` was constructed to satisfy all constraints; the original
    # `rejected` was constructed to violate exactly one. Both are useful.
    extra_candidates: list[dict] = []
    if sample.get("chosen"):
        extra_candidates.append(
            {"model": "openai/gpt-4o (tulu chosen)", "completion": sample["chosen"]}
        )
    if sample.get("rejected"):
        extra_candidates.append(
            {"model": "openai/gpt-4o (tulu rejected)", "completion": sample["rejected"]}
        )

    all_candidates = [c for c in (*completions_for_prompt, *extra_candidates) if c.get("completion")]
    if len(all_candidates) < 2:
        log.warning(f"id={sample['id']}: only {len(all_candidates)} candidates; skipping")
        return None

    completions = [c["completion"] for c in all_candidates]
    models = [c["model"] for c in all_candidates]

    judge = SonnetTuluPairwiseJudge(
        client,
        constraints=sample["constraints"],
        model=judge_model,
        template=template,
        sem=judge_sem,
        max_tokens=judge_max_tokens,
        timeout_s=judge_timeout_s,
        retry_attempts=judge_retry_attempts,
    )

    wins = await compute_round_robin_wins(
        prompt=sample["raw_prompt"],
        completions=completions,
        judge_fn=judge,
    )
    n = len(completions)
    win_rates = wins.sum(axis=1) / max(n - 1, 1)
    win_rates_list = win_rates.tolist()

    best_idx = int(max(range(n), key=lambda i: win_rates_list[i]))
    worst_idx = int(min(range(n), key=lambda i: win_rates_list[i]))

    if best_idx == worst_idx or win_rates_list[best_idx] == win_rates_list[worst_idx]:
        log.info(f"id={sample['id']}: no winner separation; skipping")
        return None

    return {
        "id": sample["id"],
        "row_id": sample["row_id"],
        "prompt": sample["prompt"],
        "raw_prompt": sample["raw_prompt"],
        "constraints": sample["constraints"],
        "chosen": completions[best_idx],
        "rejected": completions[worst_idx],
        "chosen_model": models[best_idx],
        "rejected_model": models[worst_idx],
        "chosen_score": win_rates_list[best_idx],
        "rejected_score": win_rates_list[worst_idx],
        "all_models": models,
        "all_scores": win_rates_list,
    }


async def main(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples_by_id = _load_samples(args.samples)
    completions_by_id = _load_completions(args.completions)
    existing = _load_existing(args.output)

    log.info(
        f"Loaded {len(samples_by_id)} samples, "
        f"{sum(len(v) for v in completions_by_id.values())} completions across "
        f"{len(completions_by_id)} prompts; "
        f"{len(existing)} pairs already built."
    )

    work_ids = [pid for pid in completions_by_id if pid not in existing and pid in samples_by_id]
    if not work_ids:
        log.info("Nothing to do.")
        return

    template = Template(TEMPLATE_PATH.read_text())
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=0,
    )
    judge_sem = asyncio.Semaphore(args.judge_concurrency)
    write_lock = asyncio.Lock()
    output_fh = args.output.open("a")

    try:

        async def run(prompt_id: str) -> None:
            sample = samples_by_id[prompt_id]
            completions = completions_by_id[prompt_id]
            try:
                result = await process_one_prompt(
                    sample=sample,
                    completions_for_prompt=completions,
                    client=client,
                    template=template,
                    judge_sem=judge_sem,
                    judge_model=args.judge_model,
                    judge_max_tokens=args.judge_max_tokens,
                    judge_timeout_s=args.judge_timeout_s,
                    judge_retry_attempts=args.judge_retry_attempts,
                )
            except Exception:
                log.exception(f"id={prompt_id}: tournament failed")
                return
            if result is None:
                return
            async with write_lock:
                output_fh.write(json.dumps(result) + "\n")
                output_fh.flush()

        await atqdm.gather(*(run(pid) for pid in work_ids), desc="judge")
    finally:
        output_fh.close()
        await client.close()

    log.info(f"Wrote DPO pairs to {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completions", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    p.add_argument("--judge-concurrency", type=int, default=256)
    p.add_argument("--judge-max-tokens", type=int, default=8)
    p.add_argument("--judge-timeout-s", type=float, default=120.0)
    p.add_argument("--judge-retry-attempts", type=int, default=4)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main(args))
