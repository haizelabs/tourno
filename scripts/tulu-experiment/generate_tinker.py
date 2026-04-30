"""Generate completions from pretrained-only base models via Tinker.

Writes to the SAME JSONL output that generate.py uses, so build_dpo.py picks
these up alongside the OpenRouter completions.

Usage:
    uv run scripts/tulu-experiment/generate_tinker.py \\
        --input  datasets/tulu_if_train.jsonl \\
        --output datasets/dpo/train_completions.jsonl \\
        --models meta-llama/Llama-3.1-8B deepseek-ai/DeepSeek-V3.1-Base
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm.asyncio import tqdm as atqdm

from tourno.training.models import get_sampling_client

log = logging.getLogger("generate_tinker")


async def generate_for_model(
    *,
    model: str,
    samples: list[dict],
    existing: set[tuple[str, str]],
    output_path: Path,
    write_lock: asyncio.Lock,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    base_url: str | None,
) -> None:
    work = [s for s in samples if (s["id"], model) not in existing]
    if not work:
        log.info(f"{model}: nothing to do")
        return
    log.info(f"{model}: generating {len(work)} completions")

    client = await get_sampling_client(base_model=model, base_url=base_url)
    tokenizer = get_tokenizer(model)
    sem = asyncio.Semaphore(concurrency)

    output_fh = output_path.open("a")

    async def one(sample: dict) -> None:
        completion: str | None
        err: str | None
        async with sem:
            try:
                prompt_tokens = tokenizer.encode(sample["raw_prompt"])
                model_input = tinker.ModelInput.from_ints(prompt_tokens)
                result = await client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                tokens = list(result.sequences[0].tokens)
                completion = tokenizer.decode(tokens, skip_special_tokens=True).strip() or None
                err = None
            except Exception as exc:
                completion = None
                err = repr(exc)

        record = {
            "id": sample["id"],
            "row_id": sample["row_id"],
            "model": model,
            "raw_prompt": sample["raw_prompt"],
            "constraints": sample["constraints"],
            "completion": completion,
            "error": err,
        }
        async with write_lock:
            output_fh.write(json.dumps(record) + "\n")
            output_fh.flush()

    try:
        await atqdm.gather(*(one(s) for s in work), desc=f"tinker({model})")
    finally:
        output_fh.close()


async def main(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    with args.input.open() as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    log.info(f"Loaded {len(samples)} samples from {args.input}")

    existing: set[tuple[str, str]] = set()
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    existing.add((row["id"], row["model"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    log.info(f"Found {len(existing)} existing completions across all models")

    write_lock = asyncio.Lock()
    for model in args.models:
        try:
            await generate_for_model(
                model=model,
                samples=samples,
                existing=existing,
                output_path=args.output,
                write_lock=write_lock,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                concurrency=args.concurrency,
                base_url=args.base_url,
            )
        except Exception:
            log.exception(f"{model}: failed; continuing to next model")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(main(args))
