from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import tinker
from gold_judges import GoldJudge
from openai import AsyncOpenAI
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from truthfulqa_types import TruthfulQASample

TRUTHFULQA_DIR = Path(__file__).resolve().parent.parent.parent / "truthfulqa-rl"
DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

log = logging.getLogger(__name__)

EvalTarget = tuple[str, int]


def make_openai_client(provider: str) -> AsyncOpenAI:
    if provider == "openrouter":
        return AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


def load_samples(path: Path, max_samples: int | None = None) -> list[TruthfulQASample]:
    samples = [
        TruthfulQASample.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    return samples[:max_samples] if max_samples else samples


def get_base_model(model_name: str) -> str:
    return json.loads((TRUTHFULQA_DIR / model_name / "base_model.json").read_text())["base_model"]


def get_sampler_path(model_name: str, step: int) -> str:
    path = TRUTHFULQA_DIR / model_name / "checkpoints.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry["step"] == step:
                return entry["sampler_path"]

    raise ValueError(f"Step {step} not found in {path}")


def get_available_steps(model_name: str) -> set[int]:
    path = TRUTHFULQA_DIR / model_name / "checkpoints.jsonl"
    if not path.exists():
        return set()

    return {json.loads(line)["step"] for line in path.read_text().splitlines() if line.strip()}


def discover_run(
    base_model_short: str,
    method: str,
    judge: str,
    mixture_alpha: float = 3.0,
    truthfulqa_dir: Path | None = None,
) -> str | None:
    truthfulqa_dir = truthfulqa_dir or TRUTHFULQA_DIR
    if not truthfulqa_dir.exists():
        return None
    prefix = f"{base_model_short}_lr"
    candidates: list[str] = []
    for ckpt in truthfulqa_dir.rglob("checkpoints.jsonl"):
        rel = ckpt.parent.relative_to(truthfulqa_dir).as_posix()
        if not rel.startswith(prefix):
            continue
        if f"_{method}_judge{judge}_" not in rel:
            continue
        if method == "mixture" and f"_alpha{mixture_alpha}" not in rel:
            continue
        candidates.append(rel)

    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning(f"Multiple runs for {method}/{judge}: {candidates}; using {candidates[0]}")

    return candidates[0]


def discover_runs(
    *,
    base_model_short: str,
    methods: list[str],
    training_judge: str,
    mixture_alpha: float,
) -> dict[str, str]:
    runs_by_method: dict[str, str] = {}
    for method in methods:
        run_name = discover_run(base_model_short, method, training_judge, mixture_alpha)
        if run_name is None:
            log.warning(f"No run found for {method}/{training_judge}")
            continue
        runs_by_method[method] = run_name
        log.info(f"{method:10s} -> {run_name}")

    return runs_by_method


def _parse_eval_target_spec(spec: str) -> tuple[str, int]:
    method, step_str = spec.split(":", 1)
    return method, int(step_str)


def resolve_eval_targets(
    *,
    runs_by_method: dict[str, str],
    steps: list[int],
    explicit_targets: list[str] | None,
) -> list[EvalTarget]:
    if explicit_targets:
        eval_targets: list[EvalTarget] = []
        for spec in explicit_targets:
            method, step = _parse_eval_target_spec(spec)
            if method not in runs_by_method:
                log.warning(f"Skipping {spec}: no run discovered for method {method!r}")
                continue
            eval_targets.append((runs_by_method[method], step))
        if eval_targets:
            log.info(f"Evaluating {len(eval_targets)} explicit eval targets")
        return eval_targets

    runs = list(runs_by_method.values())
    valid_steps = sorted({s for r in runs for s in steps if s == 0 or s in get_available_steps(r)})
    log.info(f"Will evaluate steps: {valid_steps}")
    return [(run_name, step) for run_name in runs for step in valid_steps]


def serialize_conversation(prompt: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in prompt)


def _split_stop_sequences(renderer) -> tuple[set[int], list[str]]:
    stops = renderer.get_stop_sequences()
    return (
        {s for s in stops if isinstance(s, int)},
        [s for s in stops if isinstance(s, str)],
    )


@dataclass
class JudgeScore:
    raw: float
    normalized: float
    judge_model: str


@dataclass
class CompletionRecord:
    sample: TruthfulQASample
    model_name: str
    step: int
    sample_idx: int
    completion: str
    judges: dict[str, JudgeScore] = field(default_factory=dict)


def per_prompt_means(
    records: list[CompletionRecord],
    judge_kind: str,
    model_name: str,
    step: int,
) -> dict[str, float]:
    """Mean normalized score per prompt for the (judge_kind, model, step) cell."""
    buckets: dict[str, list[float]] = {}
    for r in records:
        if r.model_name == model_name and r.step == step and judge_kind in r.judges:
            buckets.setdefault(r.sample.prompt_id, []).append(r.judges[judge_kind].normalized)

    return {pid: mean(vs) for pid, vs in buckets.items() if vs}


def mean_score_summary(
    records: list[CompletionRecord],
    judge_kinds: list[str],
    eval_targets: list[EvalTarget],
) -> dict[tuple[str, int, str], tuple[float | None, int]]:
    summary: dict[tuple[str, int, str], tuple[float | None, int]] = {}
    for run_name, step in eval_targets:
        for kind in judge_kinds:
            prompt_means = per_prompt_means(records, kind, run_name, step)
            score = mean(prompt_means.values()) if prompt_means else None
            summary[(run_name, step, kind)] = (score, len(prompt_means))

    return summary


def prefixed_mean_metrics(
    records: list[CompletionRecord],
    judge_kinds: list[str],
    eval_target: EvalTarget,
    prefix: str,
) -> dict[str, float | int]:
    run_name, step = eval_target
    metrics: dict[str, float | int] = {}
    for kind in judge_kinds:
        prompt_means = per_prompt_means(records, kind, run_name, step)
        if prompt_means:
            metrics[f"{prefix}/{kind}_mean"] = float(mean(prompt_means.values()))
            metrics[f"{prefix}/{kind}_n_prompts"] = len(prompt_means)

    return metrics


class EvalPipeline:
    """Generate completions, judge them, and optionally write JSONL results."""

    def __init__(
        self,
        samples: list[TruthfulQASample],
        judges: list[GoldJudge],
        service: tinker.ServiceClient,
        results_path: Path | None = None,
        n_samples_per_prompt: int = 1,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        gen_concurrency: int = 32,
        judge_concurrency: int = 128,
        client_overrides: dict[EvalTarget, tinker.SamplingClient] | None = None,
    ):
        self.samples = samples
        self.judges = judges
        self.service = service
        self.results_path = results_path
        self.n_samples_per_prompt = n_samples_per_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.gen_sem = asyncio.Semaphore(gen_concurrency)
        self.judge_sem = asyncio.Semaphore(judge_concurrency)

        self._clients: dict[EvalTarget, tinker.SamplingClient | None] = dict(client_overrides or {})
        self._renderers: dict[str, tuple] = {}

    def _renderer(self, model_name: str):
        base = get_base_model(model_name)
        if base not in self._renderers:
            tokenizer = get_tokenizer(base)
            r = get_renderer(get_recommended_renderer_name(base), tokenizer)
            stop_ids, stop_strs = _split_stop_sequences(r)
            self._renderers[base] = (r, stop_ids, stop_strs)
        return self._renderers[base]

    async def _create_client(self, model_name: str, step: int):
        try:
            if step == 0:
                return await self.service.create_sampling_client_async(
                    base_model=get_base_model(model_name)
                )
            return await self.service.create_sampling_client_async(
                model_path=get_sampler_path(model_name, step)
            )
        except Exception as exc:
            log.warning(f"Cannot load {model_name} step {step}, skipping: {exc!r}")
            return None

    async def _gen_one(
        self,
        sample: TruthfulQASample,
        model_name: str,
        step: int,
        sample_idx: int,
    ) -> CompletionRecord | None:
        client = self._clients.get((model_name, step))
        if client is None:
            return None
        renderer, stop_ids, stop_strs = self._renderer(model_name)
        try:
            async with self.gen_sem:
                obs = renderer.build_generation_prompt(sample.prompt)
                result = await client.sample_async(
                    prompt=obs,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        stop=renderer.get_stop_sequences(),
                    ),
                )
                tokens = list(result.sequences[0].tokens)
                if tokens and tokens[-1] in stop_ids:
                    tokens = tokens[:-1]
                completion = renderer.tokenizer.decode(tokens, skip_special_tokens=True)
                for s in stop_strs:
                    if completion.endswith(s):
                        completion = completion[: -len(s)]
                        break

            return CompletionRecord(sample, model_name, step, sample_idx, completion)
        except Exception:
            log.exception(
                f"Gen failed: {model_name} step {step} prompt {sample.prompt_id} idx {sample_idx}"
            )
            return None

    async def _judge_one(self, rec: CompletionRecord, judge: GoldJudge) -> None:
        try:
            async with self.judge_sem:
                conversation = serialize_conversation(rec.sample.prompt)
                raw_scores = await judge(conversation, [rec.completion], rec.sample)
                raw = raw_scores[0]
            rec.judges[judge.name] = JudgeScore(
                raw=raw, normalized=judge.normalize(raw), judge_model=judge.model
            )
        except Exception:
            log.exception(
                f"Judge {judge.name} failed: {rec.model_name} step {rec.step} "
                f"prompt {rec.sample.prompt_id}"
            )

    def _write(self, records: list[CompletionRecord]) -> None:
        if self.results_path is None:
            return
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w") as f:
            for r in records:
                row = {
                    **r.sample.model_dump(),
                    "model": r.model_name,
                    "step": r.step,
                    "sample_idx": r.sample_idx,
                    "completion": r.completion,
                    "judges": {
                        name: {
                            "raw_score": s.raw,
                            "normalized_score": s.normalized,
                            "judge_model": s.judge_model,
                        }
                        for name, s in r.judges.items()
                    },
                }
                f.write(json.dumps(row) + "\n")
        log.info(f"Wrote {len(records)} records to {self.results_path}")

    async def _prepare_clients(self, eval_targets: list[EvalTarget]) -> None:
        for run_name, step in eval_targets:
            target = (run_name, step)
            if target not in self._clients:
                self._clients[target] = await self._create_client(run_name, step)
                log.info(f"Created sampling client for {run_name} step {step}")

    async def _generate(self, eval_targets: list[EvalTarget]) -> list[CompletionRecord]:
        n_total = len(eval_targets) * len(self.samples) * self.n_samples_per_prompt
        log.info(f"Phase 1 (gen): {n_total} completions to generate")
        gen_tasks = [
            self._gen_one(sample, run_name, step, sample_idx)
            for run_name, step in eval_targets
            for sample in self.samples
            for sample_idx in range(self.n_samples_per_prompt)
        ]
        records = [r for r in await asyncio.gather(*gen_tasks) if r is not None]
        log.info(f"Phase 1 done: {len(records)}/{n_total} succeeded")
        return records

    async def _judge(self, records: list[CompletionRecord]) -> None:
        if records and self.judges:
            log.info(f"Phase 2 (judge): {len(records) * len(self.judges)} judgments")
            await asyncio.gather(*(self._judge_one(r, j) for r in records for j in self.judges))

    async def run(self, eval_targets: list[EvalTarget]) -> list[CompletionRecord]:
        await self._prepare_clients(eval_targets)
        records = await self._generate(eval_targets)
        await self._judge(records)

        self._write(records)
        return records
