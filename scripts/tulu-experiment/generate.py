"""Generate completions for Tulu IF prompts across multiple models via OpenRouter.

Output JSONL — one row per (prompt, model):
    {id, row_id, model, raw_prompt, constraints, completion, error}

Resumable: re-running skips (id, model) pairs already in the output file.

Usage:
    uv run scripts/tulu-experiment/generate.py \\
        --input  datasets/tulu_if_train.jsonl \\
        --output datasets/dpo/train_completions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm as atqdm

MODELS = [
    "qwen/qwen3-14b",
    "google/gemma-4-31b-it",
    "google/gemini-3-flash-preview",
    "openai/gpt-5.1",
]
# The two base/pretrained slots (meta-llama/Llama-3.1-8B,
# deepseek-ai/DeepSeek-V3.1-Base) are handled by generate_tinker.py — OpenRouter
# does not expose pretrained-only base models.

log = logging.getLogger("generate")


async def generate_one(
    client: AsyncOpenAI,
    *,
    model: str,
    raw_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    retry_attempts: int,
    global_sem: asyncio.Semaphore,
    model_sem: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Returns (completion_text_or_none, error_repr_or_none)."""

    async def _call() -> str | None:
        async with global_sem, model_sem:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": raw_prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout_s,
            )
        text = (resp.choices[0].message.content or "").strip()
        return text or None

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=before_sleep_log(log, logging.WARNING),
        ):
            with attempt:
                text = await _call()
        return text, None
    except Exception as exc:
        return None, repr(exc)


def _load_existing(path: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    if not path.exists():
        return seen
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                seen.add((row["id"], row["model"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


async def main(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    with args.input.open() as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    log.info(f"Loaded {len(samples)} samples from {args.input}")

    existing = _load_existing(args.output)
    log.info(f"Found {len(existing)} existing completions; will skip those")

    work: list[tuple[dict, str]] = [
        (s, m) for s in samples for m in MODELS if (s["id"], m) not in existing
    ]
    if not work:
        log.info("Nothing to do.")
        return
    log.info(
        f"Generating {len(work)} completions "
        f"({len(samples)} prompts × {len(MODELS)} models, minus skips)"
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=0,  # we run our own retry loop
    )
    global_sem = asyncio.Semaphore(args.global_concurrency)
    model_sems = {m: asyncio.Semaphore(args.per_model_concurrency) for m in MODELS}
    write_lock = asyncio.Lock()
    output_fh = args.output.open("a")

    try:

        async def run_one(sample: dict, model: str) -> None:
            text, err = await generate_one(
                client,
                model=model,
                raw_prompt=sample["raw_prompt"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                retry_attempts=args.retry_attempts,
                global_sem=global_sem,
                model_sem=model_sems[model],
            )
            record = {
                "id": sample["id"],
                "row_id": sample["row_id"],
                "model": model,
                "raw_prompt": sample["raw_prompt"],
                "constraints": sample["constraints"],
                "completion": text,
                "error": err,
            }
            async with write_lock:
                output_fh.write(json.dumps(record) + "\n")
                output_fh.flush()

        await atqdm.gather(
            *(run_one(s, m) for s, m in work),
            desc="generate",
        )
    finally:
        output_fh.close()
        await client.close()

    log.info(f"Wrote completions to {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--retry-attempts", type=int, default=4)
    p.add_argument("--global-concurrency", type=int, default=256)
    p.add_argument("--per-model-concurrency", type=int, default=64)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main(args))
