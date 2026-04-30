import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from statistics import mean, stdev

import tinker
from data import TuluSample
from openai import AsyncOpenAI
from paths import EVAL_PROMPT_PATH
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tourno.eval.core import SweepResult, sweep_judges
from tourno.eval.judges import PointwiseJudge
from tourno.logger import get_logger, setup
from tourno.training.models import get_sampling_client

POINTWISE_MIN_SCORE = 1.0
POINTWISE_MAX_SCORE = 5.0


def load_samples(path: Path, max_samples: int | None = None) -> list[TuluSample]:
    samples: list[TuluSample] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                samples.append(TuluSample.model_validate(json.loads(line)))

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def normalize_score(raw: float) -> float:
    return max(
        0.0,
        min(1.0, (raw - POINTWISE_MIN_SCORE) / (POINTWISE_MAX_SCORE - POINTWISE_MIN_SCORE)),
    )


async def generate_completions(
    samples: list[TuluSample],
    *,
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    num_completions: int = 1,
    max_tokens: int | None = None,
    temperature: float = 1.0,
    gen_concurrency: int = 128,
) -> list[list[str]]:
    log = get_logger("eval.gen")
    sem = asyncio.Semaphore(gen_concurrency)
    stop_ids: set[int] = set()
    for stop in renderer.get_stop_sequences():
        if isinstance(stop, int):
            stop_ids.add(stop)
        elif isinstance(stop, str):
            ids = renderer.tokenizer.encode(stop)
            if ids:
                stop_ids.add(ids[-1])

    async def _gen(sample: TuluSample) -> list[str]:
        async with sem:
            result = await sampling_client.sample_async(
                prompt=renderer.build_generation_prompt(
                    [{"role": "user", "content": sample.prompt}]
                ),
                num_samples=num_completions,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

        completions: list[str] = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            if tokens and tokens[-1] in stop_ids:
                tokens = tokens[:-1]

            completions.append(renderer.tokenizer.decode(tokens, skip_special_tokens=True))

        return completions

    log.info(f"Generating {num_completions} completion(s) per sample for {len(samples)} samples")
    return await asyncio.gather(*[_gen(s) for s in samples])


def build_rows(
    samples: list[TuluSample],
    completions_per_sample: list[list[str]],
) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    sample_idx_per_row: list[int] = []
    for i, (sample, completions) in enumerate(zip(samples, completions_per_sample)):
        constraints = (
            "\n".join(f"- {c}" for c in sample.constraints) if sample.constraints else "(none)"
        )
        for completion in completions:
            rows.append(
                {
                    "id": sample.id,
                    "prompt": sample.prompt,
                    "constraints": constraints,
                    "completion": completion,
                }
            )
            sample_idx_per_row.append(i)

    return rows, sample_idx_per_row


def summarize(
    samples: list[TuluSample],
    sweep_results: list[SweepResult],
    sample_idx_per_row: list[int],
    judge_names: list[str],
) -> dict:
    n_samples = len(samples)
    per_judge: dict[str, dict] = {}

    for name in judge_names:
        scores_per_sample: list[list[float]] = [[] for _ in range(n_samples)]
        n_errors = 0

        for sw, sample_idx in zip(sweep_results, sample_idx_per_row):
            val = sw.outputs.get(name)
            if val is None:
                n_errors += 1
                continue

            scores_per_sample[sample_idx].append(float(val))

        sample_means = [mean(ss) for ss in scores_per_sample if ss]

        per_judge[name] = {
            "n_completions_total": len(sweep_results),
            "n_errors": n_errors,
            "n_samples_with_score": len(sample_means),
            "mean_score": mean(sample_means) if sample_means else None,
            "std_score": stdev(sample_means) if len(sample_means) > 1 else 0.0,
        }

    return {
        "n_samples": n_samples,
        "n_completions_total": len(sweep_results),
        "judges": per_judge,
    }


async def main(
    *,
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    sampler_path: str | None,
    base_url: str | None,
    judge_client: AsyncOpenAI,
    judge_models: list[str],
    renderer_name: str | None,
    max_samples: int | None,
    num_completions: int,
    max_tokens: int,
    temperature: float,
    gen_concurrency: int,
    judge_concurrency: int,
):
    log = get_logger("eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    ### Load samples ###
    samples = load_samples(dataset_path, max_samples=max_samples)
    log.info(f"Loaded {len(samples)} samples from {dataset_path}")

    ### Build renderer ###
    tokenizer = get_tokenizer(base_model)
    renderer_name = renderer_name or get_recommended_renderer_name(base_model)
    renderer = get_renderer(renderer_name, tokenizer)
    log.info(f"Model: {base_model} | Renderer: {renderer_name}")

    ### Build sampling client ###
    sampling_client = await get_sampling_client(
        base_model=base_model if sampler_path is None else None,
        load_checkpoint_path=sampler_path,
        base_url=base_url,
    )

    ### Generate completions ###
    completions_per_sample = await generate_completions(
        samples,
        sampling_client=sampling_client,
        renderer=renderer,
        num_completions=num_completions,
        max_tokens=max_tokens,
        temperature=temperature,
        gen_concurrency=gen_concurrency,
    )

    ### Build judges and rows ###
    judge_template = EVAL_PROMPT_PATH.read_text()
    judges = {
        model: PointwiseJudge(
            client=judge_client,
            model=model,
            judge_template=judge_template,
            max_concurrency=judge_concurrency,
        )
        for model in judge_models
    }
    rows, sample_idx_per_row = build_rows(samples, completions_per_sample)
    log.info(f"Running {len(judges)} judge(s) over {len(rows)} (sample, completion) pair(s)")

    ### Run judge sweep ###
    sweep_results = await sweep_judges(judges, rows, output_dir=output_dir)

    ### Summarize ###
    summary = summarize(samples, sweep_results, sample_idx_per_row, list(judges))
    summary["config"] = {
        "dataset": str(dataset_path),
        "base_model": base_model,
        "sampler_path": sampler_path,
        "renderer": renderer_name,
        "num_completions": num_completions,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "judge_models": judge_models,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info(f"Wrote summary to {summary_path}")

    log.info("=== Per-judge summary ===")
    for name, stats in summary["judges"].items():
        mean_score = stats["mean_score"]
        log.info(
            f"{name}: mean_score={mean_score:.4f}"
            f" | std_score={stats['std_score']:.4f}"
            f" | errors={stats['n_errors']}/{stats['n_completions_total']}"
            if mean_score is not None
            else f"{name}: all errors ({stats['n_errors']}/{stats['n_completions_total']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tulu multi-judge eval")

    ### Dataset settings ###
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)

    ### Model / generation settings ###
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--sampler-path", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--renderer", type=str, default=None)
    parser.add_argument("--num-completions", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gen-concurrency", type=int, default=32)

    ### Judge settings ###
    parser.add_argument(
        "--judge-models",
        type=str,
        nargs="+",
        required=True,
        help="One or more judge model identifiers to compare side-by-side",
    )
    parser.add_argument("--judge-concurrency", type=int, default=128)

    ### Output / logging settings ###
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-filter", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(
        level=getattr(logging, args.log_level.upper()),
        filter_pattern=args.log_filter,
    )

    if os.getenv("OPENAI_API_KEY") is not None:
        judge_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif os.getenv("OPENROUTER_API_KEY") is not None:
        judge_client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError("No LLM Provider API key found")

    asyncio.run(
        main(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            base_model=args.base_model,
            sampler_path=args.sampler_path,
            base_url=args.base_url,
            judge_client=judge_client,
            judge_models=args.judge_models,
            renderer_name=args.renderer,
            max_samples=args.max_samples,
            num_completions=args.num_completions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gen_concurrency=args.gen_concurrency,
            judge_concurrency=args.judge_concurrency,
        )
    )
