import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from statistics import mean

import tinker
from data import RewardBenchEvalSample, load_eval_samples
from paths import POLICY_POINTWISE_PROMPT_PATH
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tourno.logger import get_logger, setup
from tourno.training.models import get_sampling_client

_NUMBER_RE = re.compile(r"[-]?\d+\.?\d*")


def _stop_token_ids(renderer: Renderer) -> list[int]:
    ids: set[int] = set()
    for stop in renderer.get_stop_sequences():
        if isinstance(stop, int):
            ids.add(stop)
        elif isinstance(stop, str):
            encoded = renderer.tokenizer.encode(stop)
            if encoded:
                ids.add(encoded[-1])

    return sorted(ids)


def parse_policy_score(text: str) -> float:
    stripped = text.strip()
    last_line = next((line.strip() for line in reversed(stripped.splitlines()) if line.strip()), "")
    try:
        score = float(last_line)
    except ValueError:
        numbers = _NUMBER_RE.findall(last_line) or _NUMBER_RE.findall(stripped)
        if not numbers:
            raise ValueError(f"Could not parse score from: {text!r}")
        score = float(numbers[-1])

    if not 0.0 <= score <= 100.0:
        raise ValueError(f"Expected score in [0, 100], got {score}")

    return score


async def generate_judge_outputs(
    samples: list[RewardBenchEvalSample],
    *,
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    policy_template: str,
    max_tokens: int,
    temperature: float,
    gen_concurrency: int,
) -> list[list[dict]]:
    log = get_logger("eval.gen")
    sem = asyncio.Semaphore(gen_concurrency)
    stop_ids = _stop_token_ids(renderer)

    async def _gen(prompt: str, response: str) -> dict:
        policy_msg = policy_template.format(prompt=prompt, completion=response)
        obs = renderer.build_generation_prompt([{"role": "user", "content": policy_msg}])
        async with sem:
            result = await sampling_client.sample_async(
                prompt=obs,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_ids,
                ),
            )

        seq = result.sequences[0]
        tokens = list(seq.tokens)
        if tokens and tokens[-1] in stop_ids:
            tokens = tokens[:-1]
        completion = renderer.tokenizer.decode(tokens, skip_special_tokens=True)

        try:
            score = parse_policy_score(completion)
            error = None
        except ValueError as exc:
            score = None
            error = repr(exc)

        return {
            "response": response,
            "judge_output": completion,
            "score": score,
            "error": error,
        }

    log.info(f"Scoring {len(samples)} RewardBench2 row(s)")

    async def _score_sample(sample: RewardBenchEvalSample) -> list[dict]:
        candidates = [*sample.chosen, *sample.rejected]
        return await asyncio.gather(*[_gen(sample.prompt, response) for response in candidates])

    return await asyncio.gather(*[_score_sample(sample) for sample in samples])


def build_rows(
    samples: list[RewardBenchEvalSample],
    scored_per_sample: list[list[dict]],
) -> list[dict]:
    rows: list[dict] = []
    for sample, scored in zip(samples, scored_per_sample):
        scores = [item["score"] for item in scored]
        valid_scores = [s for s in scores if s is not None]
        if len(valid_scores) != len(scores):
            best_idx = None
            correct = None
        else:
            best_idx = max(range(len(scores)), key=lambda idx: float(scores[idx]))
            correct = best_idx < len(sample.chosen)

        rows.append(
            {
                "source_id": sample.source_id,
                "subset": sample.subset,
                "prompt": sample.prompt,
                "num_chosen": len(sample.chosen),
                "num_rejected": len(sample.rejected),
                "best_idx": best_idx,
                "correct": correct,
                "candidates": [
                    {
                        **item,
                        "is_chosen": idx < len(sample.chosen),
                    }
                    for idx, item in enumerate(scored)
                ],
            }
        )

    return rows


def summarize(rows: list[dict]) -> dict:
    per_subset: dict[str, list[bool]] = {}
    n_errors = 0
    for row in rows:
        correct = row["correct"]
        if correct is None:
            n_errors += 1
            continue
        per_subset.setdefault(row["subset"], []).append(bool(correct))

    all_correct = [val for vals in per_subset.values() for val in vals]
    return {
        "n_rows": len(rows),
        "n_errors": n_errors,
        "n_scored": len(all_correct),
        "accuracy": mean(all_correct) if all_correct else None,
        "per_subset": {
            subset: {
                "n": len(vals),
                "accuracy": mean(vals) if vals else None,
            }
            for subset, vals in sorted(per_subset.items())
        },
    }


async def main(
    *,
    label: str,
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    sampler_path: str | None,
    base_url: str | None,
    renderer_name: str | None,
    max_samples: int | None,
    max_tokens: int,
    temperature: float,
    gen_concurrency: int,
) -> dict:
    log = get_logger("eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_eval_samples(dataset_path, max_samples=max_samples)
    log.info(f"Loaded {len(samples)} samples from {dataset_path}")

    tokenizer = get_tokenizer(base_model)
    renderer_name = renderer_name or get_recommended_renderer_name(base_model)
    renderer = get_renderer(renderer_name, tokenizer)
    log.info(f"Model: {base_model} | Renderer: {renderer_name}")

    sampling_client = await get_sampling_client(
        base_model=base_model if sampler_path is None else None,
        load_checkpoint_path=sampler_path,
        base_url=base_url,
    )
    policy_template = POLICY_POINTWISE_PROMPT_PATH.read_text()
    scored_per_sample = await generate_judge_outputs(
        samples,
        sampling_client=sampling_client,
        renderer=renderer,
        policy_template=policy_template,
        max_tokens=max_tokens,
        temperature=temperature,
        gen_concurrency=gen_concurrency,
    )

    rows = build_rows(samples, scored_per_sample)
    summary = summarize(rows)
    summary.update(
        {
            "label": label,
            "dataset": str(dataset_path),
            "base_model": base_model,
            "sampler_path": sampler_path,
            "renderer": renderer_name,
            "config": {
                "max_samples": max_samples,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }
    )

    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (label_dir / "rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info(
        f"{label}: accuracy={summary['accuracy']} | "
        f"scored={summary['n_scored']}/{summary['n_rows']} | errors={summary['n_errors']}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RewardBench2 pointwise judge eval")

    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--sampler-path", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--renderer", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gen-concurrency", type=int, default=32)

    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-filter", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(
        level=getattr(logging, args.log_level.upper()),
        filter_pattern=args.log_filter,
    )
    asyncio.run(
        main(
            label=args.label,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            base_model=args.base_model,
            sampler_path=args.sampler_path,
            base_url=args.base_url,
            renderer_name=args.renderer,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gen_concurrency=args.gen_concurrency,
        )
    )
