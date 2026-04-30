from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tinker
from healthbench_types import HealthBenchSample, Rubric
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tourno.eval import evaluate_judge
from tourno.eval.judges import PointwiseJudge

HEALTHBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "healthbench-rl"
DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

POINTWISE_PROMPT = (PROMPTS_DIR / "healthbench_pointwise_judge.txt").read_text()
PAIRWISE_PROMPT = (PROMPTS_DIR / "healthbench_pairwise_judge.txt").read_text()

log = logging.getLogger(__name__)


def load_samples(path: Path, max_samples: int | None = None) -> list[HealthBenchSample]:
    samples: list[HealthBenchSample] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(HealthBenchSample.model_validate(json.loads(line)))
    if max_samples:
        samples = samples[:max_samples]
    return samples


def get_base_model(model_name: str) -> str:
    path = HEALTHBENCH_DIR / model_name / "base_model.json"
    with open(path) as f:
        return json.load(f)["base_model"]


def get_sampler_path(model_name: str, step: int) -> str:
    path = HEALTHBENCH_DIR / model_name / "checkpoints.jsonl"
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry["step"] == step:
                return entry["sampler_path"]

    raise ValueError(f"Step {step} not found in {path}")


def get_available_steps(model_name: str) -> set[int]:
    path = HEALTHBENCH_DIR / model_name / "checkpoints.jsonl"
    steps: set[int] = set()
    if not path.exists():
        return steps
    with open(path) as f:
        for line in f:
            if line.strip():
                steps.add(json.loads(line.strip())["step"])

    return steps


def infer_base_model(model_name: str) -> str:
    return f"Qwen/{re.split(r'_lr', model_name, maxsplit=1)[0]}"


def short_model_name(model_name: str) -> str:
    return re.split(r"_lr", model_name, maxsplit=1)[0]


def get_stop_token_ids(renderer: Renderer) -> set[int]:
    stop_ids: set[int] = set()
    for s in renderer.get_stop_sequences():
        if isinstance(s, int):
            stop_ids.add(s)
        elif isinstance(s, str):
            ids = renderer.tokenizer.encode(s)
            if ids:
                stop_ids.add(ids[-1])
    return stop_ids


def serialize_conversation(prompt: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in prompt)


def serialize_rubric(rubric: list[Rubric]) -> str:
    return "\n".join(f"{i + 1}. {r.criterion} Weight: {r.points}" for i, r in enumerate(rubric))


def normalize_score(raw: float, rubrics: list[Rubric]) -> float:
    pos = sum(r.points for r in rubrics if r.points > 0)
    neg = sum(r.points for r in rubrics if r.points < 0)
    return (raw - neg) / max(1e-4, pos - neg)


def bootstrap_se(scores: np.ndarray, n_bootstrap: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n < 2:
        return 0.0
    means = np.array([rng.choice(scores, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    return float(np.std(means, ddof=1))


def discover_run(
    base_model_short: str,
    method: str,
    judge: str,
    mixture_alpha: float = 3.0,
    healthbench_dir: Path | None = None,
) -> str | None:
    if healthbench_dir is None:
        healthbench_dir = HEALTHBENCH_DIR
    prefix = f"{base_model_short}_lr"
    candidates: list[str] = []
    for d in healthbench_dir.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        name = d.name
        if f"_{method}_judge{judge}_" not in name:
            continue
        if method == "mixture" and f"_alpha{mixture_alpha}" not in name:
            continue
        candidates.append(name)

    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning(f"Multiple runs for {method}/{judge}: {candidates}; using {candidates[0]}")
    return candidates[0]


CacheKey = tuple[str, str, int]


class ResultCache:
    def __init__(self, cache_path: Path):
        self._path = cache_path
        self._data: dict[CacheKey, dict] = {}

    def _key(self, prompt_id: str, model: str, step: int) -> CacheKey:
        return (prompt_id, model, step)

    def load(self, legacy_dirs: list[Path] | None = None) -> int:
        loaded = 0
        if self._path.exists():
            with open(self._path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    self._data[self._key(entry["prompt_id"], entry["model"], entry["step"])] = entry
                    loaded += 1
            log.info(f"Loaded {loaded} entries from {self._path}")

        migrated = self._load_legacy(legacy_dirs or [])
        if migrated:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                for entry in migrated:
                    f.write(json.dumps(entry) + "\n")
            log.info(f"Migrated {len(migrated)} legacy entries into {self._path}")

        return loaded + len(migrated)

    def _load_legacy(self, legacy_dirs: list[Path]) -> list[dict]:
        migrated: list[dict] = []
        for legacy_dir in legacy_dirs:
            if not legacy_dir.is_dir():
                continue
            model_name = legacy_dir.name
            for step_file in sorted(legacy_dir.glob("step*.jsonl")):
                match = re.match(r"step(\d+)\.jsonl", step_file.name)
                if not match:
                    continue
                step = int(match.group(1))
                with open(step_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        key = self._key(entry["prompt_id"], model_name, step)
                        if key in self._data:
                            continue
                        full_entry = {
                            "prompt_id": entry["prompt_id"],
                            "model": model_name,
                            "step": step,
                            "completion": entry.get("completion", ""),
                            "raw_score": entry["raw_score"],
                            "normalized_score": entry["normalized_score"],
                        }
                        self._data[key] = full_entry
                        migrated.append(full_entry)
        return migrated

    def get(self, prompt_id: str, model: str, step: int) -> dict | None:
        return self._data.get(self._key(prompt_id, model, step))

    def put(self, prompt_id: str, model: str, step: int, entry: dict) -> None:
        full_entry = {
            "prompt_id": prompt_id,
            "model": model,
            "step": step,
            **entry,
        }
        self._data[self._key(prompt_id, model, step)] = full_entry
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(full_entry) + "\n")

    def get_scores(self, model: str, step: int, prompt_ids: set[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for pid in prompt_ids:
            entry = self.get(pid, model, step)
            if entry is not None:
                out[pid] = entry["normalized_score"]
        return out

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class CheckpointResult:
    sample: HealthBenchSample
    completions: list[str]
    raw_scores: list[float]
    normalized_scores: list[float]
    mean_normalized: float


async def evaluate_checkpoint(
    samples: list[HealthBenchSample],
    *,
    sampler_path: str | None,
    base_model: str,
    judge: PointwiseJudge,
    service: tinker.ServiceClient,
    num_completions: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    gen_concurrency: int = 32,
) -> list[CheckpointResult]:
    if num_completions < 1:
        raise ValueError("num_completions must be >= 1")

    client = await (
        service.create_sampling_client_async(model_path=sampler_path)
        if sampler_path
        else service.create_sampling_client_async(base_model=base_model)
    )
    tokenizer = get_tokenizer(base_model)
    renderer = get_renderer(get_recommended_renderer_name(base_model), tokenizer)
    stop_ids = get_stop_token_ids(renderer)

    sem = asyncio.Semaphore(gen_concurrency)

    async def _gen(sample: HealthBenchSample) -> list[str]:
        async with sem:
            result = await client.sample_async(
                prompt=renderer.build_generation_prompt(sample.prompt),
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
            completions.append(renderer.tokenizer.decode(tokens))
        return completions

    completions_per_sample = await asyncio.gather(*[_gen(sample) for sample in samples])

    flat_rows: list[dict] = []
    flat_index: list[int] = []
    for sample_idx, (sample, completions) in enumerate(zip(samples, completions_per_sample)):
        for completion in completions:
            flat_rows.append(
                {
                    "prompt": serialize_conversation(sample.prompt),
                    "completion": completion,
                    "rubric": serialize_rubric(sample.rubrics),
                }
            )
            flat_index.append(sample_idx)

    judge_results = await evaluate_judge(judge, flat_rows)
    raw_scores: list[list[float]] = [[] for _ in samples]
    normalized_scores: list[list[float]] = [[] for _ in samples]

    for sample_idx, result in zip(flat_index, judge_results):
        if result.error is not None:
            raw_scores[sample_idx].append(float("nan"))
            normalized_scores[sample_idx].append(float("nan"))
            continue

        raw = float(result.output)
        raw_scores[sample_idx].append(raw)
        normalized_scores[sample_idx].append(normalize_score(raw, samples[sample_idx].rubrics))

    return [
        CheckpointResult(
            sample=sample,
            completions=completions,
            raw_scores=raw_scores[i],
            normalized_scores=normalized_scores[i],
            mean_normalized=_nanmean_or_nan(normalized_scores[i]),
        )
        for i, (sample, completions) in enumerate(zip(samples, completions_per_sample))
    ]


def _nanmean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))
