import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from statistics import mean, stdev

import tinker
from data import CreativeBenchSample
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tourno.logger import get_logger, setup
from tourno.training.models import get_sampling_client

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
JUDGE_TEMPLATE_PATH = PROMPTS_DIR / "cwbench_judge_official.txt"
CRITERIA_PATH = PROMPTS_DIR / "cwbench_criteria.txt"
NEGATIVE_CRITERIA_PATH = PROMPTS_DIR / "cwbench_negative_criteria.txt"

ALL_CRITERIA = CRITERIA_PATH.read_text().strip().splitlines()
NEGATIVE_CRITERIA = NEGATIVE_CRITERIA_PATH.read_text().strip().splitlines()
NEGATIVE_SET = set(NEGATIVE_CRITERIA)
SCORE_RANGE_MAX = 20.0

_SCORE_PAT_1 = re.compile(r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)")
_SCORE_PAT_2 = re.compile(r"(.*?):\s*\[(-?\d+(?:\.\d+)?)\]")
_CANON = set(ALL_CRITERIA)


def parse_judge_scores(judge_response: str) -> dict[str, float]:
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
    return {k: v for k, v in scores.items() if k in _CANON}


def piece_score_0_20(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    vals: list[float] = []
    for metric, val in scores.items():
        adj = (SCORE_RANGE_MAX - val) if metric in NEGATIVE_SET else val
        vals.append(adj)
    return sum(vals) / len(vals)


def load_samples(path: Path, max_samples: int | None = None) -> list[CreativeBenchSample]:
    samples: list[CreativeBenchSample] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                samples.append(CreativeBenchSample.model_validate(json.loads(line)))

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def _stop_token_ids(renderer: Renderer) -> set[int]:
    stop_ids: set[int] = set()
    for stop in renderer.get_stop_sequences():
        if isinstance(stop, int):
            stop_ids.add(stop)
        elif isinstance(stop, str):
            ids = renderer.tokenizer.encode(stop)
            if ids:
                stop_ids.add(ids[-1])

    return stop_ids


async def generate_completions(
    samples: list[CreativeBenchSample],
    *,
    sampling_client: tinker.SamplingClient,
    renderer: Renderer,
    num_completions: int = 1,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    gen_concurrency: int = 32,
) -> list[list[tuple[str, int]]]:
    log = get_logger("eval.gen")
    sem = asyncio.Semaphore(gen_concurrency)
    stop_ids = _stop_token_ids(renderer)

    async def _gen(sample: CreativeBenchSample) -> list[tuple[str, int]]:
        async with sem:
            result = await sampling_client.sample_async(
                prompt=renderer.build_generation_prompt(
                    [{"role": "user", "content": sample.writing_prompt}]
                ),
                num_samples=num_completions,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=sorted(stop_ids),
                ),
            )

        out: list[tuple[str, int]] = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            if tokens and tokens[-1] in stop_ids:
                tokens = tokens[:-1]

            out.append((renderer.tokenizer.decode(tokens, skip_special_tokens=True), len(tokens)))

        return out

    log.info(f"Generating {num_completions} completion(s) per sample for {len(samples)} samples")
    return await asyncio.gather(*[_gen(s) for s in samples])


def _build_judge_prompt(judge_template: str, writing_prompt: str, completion: str) -> str:
    return judge_template.format(
        writing_prompt=writing_prompt,
        test_model_response=completion,
        lower_is_better_criteria="\n".join(NEGATIVE_CRITERIA),
        creative_writing_criteria="\n".join(ALL_CRITERIA),
    )


async def score_completions(
    samples: list[CreativeBenchSample],
    completions_per_sample: list[list[tuple[str, int]]],
    *,
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_template: str,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 2048,
    judge_concurrency: int = 128,
    request_timeout_s: float = 120.0,
) -> list[list[dict]]:
    log = get_logger("eval.judge")
    sem = asyncio.Semaphore(judge_concurrency)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(log, logging.WARNING),
        retry_error_callback=lambda _state: None,
    )
    async def _judge(writing_prompt: str, completion: str) -> str | None:
        async with sem:
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": _build_judge_prompt(judge_template, writing_prompt, completion),
                    }
                ],
                timeout=request_timeout_s,
                temperature=judge_temperature,
                max_tokens=judge_max_tokens,
                stream=False,
            )
            content = response.choices[0].message.content
            if not content:
                finish = response.choices[0].finish_reason
                raise RuntimeError(f"Empty judge content (finish_reason={finish!r})")
            return content

    async def _score_iter(sample: CreativeBenchSample, completion: str, n_tok: int) -> dict:
        judge_response = await _judge(sample.writing_prompt, completion)
        if judge_response is None:
            return {
                "completion": completion,
                "token_len": n_tok,
                "judge_scores": {},
                "judge_response": None,
                "piece_score_0_20": None,
            }
        scores = parse_judge_scores(judge_response)
        return {
            "completion": completion,
            "token_len": n_tok,
            "judge_scores": scores,
            "judge_response": judge_response,
            "piece_score_0_20": piece_score_0_20(scores),
        }

    log.info(
        f"Scoring {sum(len(c) for c in completions_per_sample)} (sample, completion) pair(s)"
        f" with judge {judge_model}"
    )

    async def _score_sample(
        sample: CreativeBenchSample, completions: list[tuple[str, int]]
    ) -> list[dict]:
        return await asyncio.gather(
            *[_score_iter(sample, text, ntok) for text, ntok in completions]
        )

    return await asyncio.gather(
        *[_score_sample(s, c) for s, c in zip(samples, completions_per_sample)]
    )


def build_rows(
    samples: list[CreativeBenchSample],
    scored_per_sample: list[list[dict]],
) -> list[dict]:
    rows: list[dict] = []
    for sample, iter_blocks in zip(samples, scored_per_sample):
        rows.append(
            {
                "prompt_id": sample.prompt_id,
                "scenario_id": sample.scenario_id,
                "category": sample.category,
                "writing_prompt": sample.writing_prompt,
                "iterations": [{"iteration": i, **blk} for i, blk in enumerate(iter_blocks)],
            }
        )

    return rows


def summarize(rows: list[dict]) -> dict:
    piece_scores: list[float] = []
    per_criterion: dict[str, list[float]] = {c: [] for c in ALL_CRITERIA}
    per_category: dict[str, list[float]] = {}
    n_judge_errors = 0

    for row in rows:
        cat = row.get("category") or "unknown"
        for blk in row.get("iterations", []):
            ps = blk.get("piece_score_0_20")
            if ps is None:
                n_judge_errors += 1
                continue
            piece_scores.append(float(ps))
            per_category.setdefault(cat, []).append(float(ps))
            for crit, val in (blk.get("judge_scores") or {}).items():
                if crit in NEGATIVE_SET:
                    val = SCORE_RANGE_MAX - val
                per_criterion.setdefault(crit, []).append(val)

    overall_0_20 = mean(piece_scores) if piece_scores else 0.0

    return {
        "n_pieces_scored": len(piece_scores),
        "n_judge_errors": n_judge_errors,
        "overall_score_0_20": round(overall_0_20, 4),
        "eqbench_creative_score_0_100": round(overall_0_20 * 5.0, 2),
        "std_piece_score_0_20": round(stdev(piece_scores), 4) if len(piece_scores) > 1 else 0.0,
        "per_criterion_0_20": {
            c: (round(mean(vs), 3) if vs else None) for c, vs in per_criterion.items()
        },
        "per_category_0_20": {c: round(mean(vs), 3) for c, vs in per_category.items()},
    }


async def main(
    *,
    label: str,
    dataset_path: Path,
    output_dir: Path,
    base_model: str,
    sampler_path: str | None,
    base_url: str | None,
    judge_client: AsyncOpenAI,
    judge_model: str,
    renderer_name: str | None,
    max_samples: int | None,
    num_completions: int,
    max_tokens: int,
    temperature: float,
    gen_concurrency: int,
    judge_temperature: float,
    judge_max_tokens: int,
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

    ### Score completions ###
    judge_template = JUDGE_TEMPLATE_PATH.read_text()
    scored_per_sample = await score_completions(
        samples,
        completions_per_sample,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_template=judge_template,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
        judge_concurrency=judge_concurrency,
    )

    ### Aggregate ###
    rows = build_rows(samples, scored_per_sample)
    summary = summarize(rows)
    summary.update(
        {
            "label": label,
            "model_path": sampler_path,
            "base_model": base_model,
            "renderer": renderer_name,
            "judge_model": judge_model,
            "n_prompts": len(samples),
            "n_iterations": num_completions,
            "config": {
                "dataset": str(dataset_path),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "judge_temperature": judge_temperature,
                "judge_max_tokens": judge_max_tokens,
            },
        }
    )

    output_path = output_dir / f"{label}.json"
    with output_path.open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False)

    log.info(f"Wrote eval to {output_path}")
    log.info(
        f"{label}: overall_0_20={summary['overall_score_0_20']:.4f}"
        f" | eqbench_0_100={summary['eqbench_creative_score_0_100']:.2f}"
        f" | pieces={summary['n_pieces_scored']} | errors={summary['n_judge_errors']}"
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Creative Writing Bench v3 eval")

    ### Run identity ###
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/cwbench_test.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    ### Model / generation settings ###
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--sampler-path", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--renderer", type=str, default="qwen3_disable_thinking")
    parser.add_argument("--num-completions", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gen-concurrency", type=int, default=32)

    ### Judge settings ###
    parser.add_argument("--judge-model", type=str, default="anthropic/claude-sonnet-4.5")
    parser.add_argument("--judge-base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--judge-concurrency", type=int, default=128)

    ### Logging settings ###
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-filter", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(
        level=getattr(logging, args.log_level.upper()),
        filter_pattern=args.log_filter,
    )

    api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise ValueError(f"Missing judge API key in env var {args.judge_api_key_env}")
    judge_client = AsyncOpenAI(api_key=api_key, base_url=args.judge_base_url or None)

    asyncio.run(
        main(
            label=args.label,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            base_model=args.base_model,
            sampler_path=args.sampler_path,
            base_url=args.base_url,
            judge_client=judge_client,
            judge_model=args.judge_model,
            renderer_name=args.renderer,
            max_samples=args.max_samples,
            num_completions=args.num_completions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            gen_concurrency=args.gen_concurrency,
            judge_temperature=args.judge_temperature,
            judge_max_tokens=args.judge_max_tokens,
            judge_concurrency=args.judge_concurrency,
        )
    )
