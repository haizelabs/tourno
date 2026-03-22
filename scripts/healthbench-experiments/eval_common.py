from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tinker
from healthbench_types import HealthBenchSample, Rubric
from judges import HealthBenchPointwiseJudge
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

HEALTHBENCH_DIR = Path(__file__).resolve().parent.parent.parent / "healthbench-rl"
DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_samples(path: Path, max_samples: int | None = None) -> list[HealthBenchSample]:
    samples: list[HealthBenchSample] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(HealthBenchSample.model_validate(json.loads(line)))
    if max_samples:
        samples = samples[:max_samples]
    return samples


# ---------------------------------------------------------------------------
# Model / checkpoint helpers
# ---------------------------------------------------------------------------


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


def get_stop_token_ids(renderer) -> set[int]:
    stop_ids: set[int] = set()
    for s in renderer.get_stop_sequences():
        if isinstance(s, int):
            stop_ids.add(s)
        elif isinstance(s, str):
            ids = renderer.tokenizer.encode(s)
            if ids:
                stop_ids.add(ids[-1])
    return stop_ids


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def serialize_conversation(prompt: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in prompt)


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


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


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
        if method == "mixture":
            if f"_alpha{mixture_alpha}" not in name:
                continue
        candidates.append(name)

    if not candidates:
        return None
    if len(candidates) > 1:
        log.warning(f"Multiple runs for {method}/{judge}: {candidates}; using {candidates[0]}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Result cache (single global cache.jsonl)
# ---------------------------------------------------------------------------

CacheKey = tuple[str, str, int]  # (prompt_id, model, step)


class ResultCache:
    def __init__(self, cache_path: Path):
        self._path = cache_path
        self._data: dict[CacheKey, dict] = {}
        self._file = None

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
                    key = self._key(entry["prompt_id"], entry["model"], entry["step"])
                    self._data[key] = entry
                    loaded += 1
            log.info(f"Loaded {loaded} entries from {self._path}")

        migrated = 0
        for legacy_dir in legacy_dirs or []:
            if not legacy_dir.is_dir():
                continue
            model_name = legacy_dir.name
            for step_file in sorted(legacy_dir.glob("step*.jsonl")):
                m = re.match(r"step(\d+)\.jsonl", step_file.name)
                if not m:
                    continue
                step = int(m.group(1))
                with open(step_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        key = self._key(entry["prompt_id"], model_name, step)
                        if key not in self._data:
                            full_entry = {
                                "prompt_id": entry["prompt_id"],
                                "model": model_name,
                                "step": step,
                                "completion": entry.get("completion", ""),
                                "raw_score": entry["raw_score"],
                                "normalized_score": entry["normalized_score"],
                            }
                            self._data[key] = full_entry
                            migrated += 1

        if migrated:
            log.info(f"Migrated {migrated} legacy entries; writing to {self._path}")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                for key, entry in self._data.items():
                    if entry.get("_legacy_written"):
                        continue
                    f.write(json.dumps(entry) + "\n")

        return loaded + migrated

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


# ---------------------------------------------------------------------------
# Eval pipeline (gen + judge, writes to ResultCache)
# ---------------------------------------------------------------------------


@dataclass
class GenJob:
    sample: HealthBenchSample
    model_name: str
    step: int


@dataclass
class JudgeJob:
    sample: HealthBenchSample
    completion: str
    model_name: str
    step: int


class EvalPipeline:
    def __init__(
        self,
        samples: list[HealthBenchSample],
        judge: HealthBenchPointwiseJudge,
        cache: ResultCache,
        service: tinker.ServiceClient,
        max_tokens: int = 4096,
        temperature: float = 0.6,
        gen_concurrency: int = 32,
        judge_concurrency: int = 128,
    ):
        self.samples = samples
        self.judge = judge
        self.cache = cache
        self.service = service
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.gen_queue: asyncio.Queue[GenJob | None] = asyncio.Queue()
        self.judge_queue: asyncio.Queue[JudgeJob | None] = asyncio.Queue()
        self.gen_sem = asyncio.Semaphore(gen_concurrency)
        self.judge_sem = asyncio.Semaphore(judge_concurrency)

        self._client_cache: dict[tuple[str, int], tinker.SamplingClient] = {}
        self._client_lock = asyncio.Lock()
        self._renderer_cache: dict[str, tuple] = {}

        self._scores: dict[str, dict[int, list[float]]] = {}
        self._gen_done = 0
        self._judge_done = 0
        self._total_jobs = 0

    def _get_renderer(self, model_name: str):
        base_model = infer_base_model(model_name)
        if base_model not in self._renderer_cache:
            tokenizer = get_tokenizer(base_model)
            name = get_recommended_renderer_name(base_model)
            renderer = get_renderer(name, tokenizer)
            self._renderer_cache[base_model] = (renderer, get_stop_token_ids(renderer))
            log.info(f"Loaded renderer {name} for {base_model}")
        return self._renderer_cache[base_model]

    async def _get_sampling_client(
        self, model_name: str, step: int
    ) -> tinker.SamplingClient | None:
        key = (model_name, step)
        if key not in self._client_cache:
            async with self._client_lock:
                if key not in self._client_cache:
                    try:
                        if step == 0:
                            base_model = get_base_model(model_name)
                            self._client_cache[key] = (
                                await self.service.create_sampling_client_async(
                                    base_model=base_model
                                )
                            )
                        else:
                            sampler_path = get_sampler_path(model_name, step)
                            self._client_cache[key] = (
                                await self.service.create_sampling_client_async(
                                    model_path=sampler_path
                                )
                            )
                        log.info(f"Created sampling client for {model_name} step {step}")
                    except Exception as exc:
                        log.warning(f"Cannot load {model_name} step {step}, skipping: {exc!r}")
                        self._client_cache[key] = None
        return self._client_cache[key]

    def _record_score(self, model_name: str, step: int, score: float) -> None:
        self._scores.setdefault(model_name, {}).setdefault(step, []).append(score)

    async def _gen_one(self, job: GenJob) -> None:
        try:
            async with self.gen_sem:
                client = await self._get_sampling_client(job.model_name, job.step)
                if client is None:
                    return
                renderer, stop_ids = self._get_renderer(job.model_name)
                obs = renderer.build_generation_prompt(job.sample.prompt)
                result = await client.sample_async(
                    prompt=obs,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    ),
                )
                tokens = list(result.sequences[0].tokens)
                if tokens and tokens[-1] in stop_ids:
                    tokens = tokens[:-1]
                completion = renderer.tokenizer.decode(tokens)

            self._gen_done += 1
            if self._gen_done % 50 == 0:
                log.info(f"[gen] {self._gen_done}/{self._total_jobs} completions")

            await self.judge_queue.put(
                JudgeJob(
                    sample=job.sample,
                    completion=completion,
                    model_name=job.model_name,
                    step=job.step,
                )
            )
        except Exception:
            log.exception(
                f"Generation failed: {job.model_name} step {job.step} prompt {job.sample.prompt_id}"
            )

    async def _judge_one(self, job: JudgeJob) -> None:
        try:
            async with self.judge_sem:
                conversation = serialize_conversation(job.sample.prompt)
                raw_scores = await self.judge(conversation, [job.completion], job.sample.rubrics)
                raw = raw_scores[0]
                normalized = normalize_score(raw, job.sample.rubrics)

            self.cache.put(
                job.sample.prompt_id,
                job.model_name,
                job.step,
                {
                    "completion": job.completion,
                    "raw_score": raw,
                    "normalized_score": normalized,
                },
            )
            self._record_score(job.model_name, job.step, normalized)

            self._judge_done += 1
            if self._judge_done % 50 == 0:
                log.info(f"[judge] {self._judge_done}/{self._total_jobs} scored")
        except Exception:
            log.exception(
                f"Judging failed: {job.model_name} step {job.step} prompt {job.sample.prompt_id}"
            )

    async def _gen_consumer(self) -> None:
        tasks: list[asyncio.Task] = []
        while True:
            job = await self.gen_queue.get()
            if job is None:
                break
            tasks.append(asyncio.create_task(self._gen_one(job)))
        await asyncio.gather(*tasks)
        await self.judge_queue.put(None)

    async def _judge_consumer(self) -> None:
        tasks: list[asyncio.Task] = []
        while True:
            job = await self.judge_queue.get()
            if job is None:
                break
            tasks.append(asyncio.create_task(self._judge_one(job)))
        await asyncio.gather(*tasks)

    async def run(self, models: list[str], steps: list[int]) -> dict[str, dict[int, list[float]]]:
        prompt_ids = {s.prompt_id for s in self.samples}
        for model_name in models:
            for step in steps:
                cached = self.cache.get_scores(model_name, step, prompt_ids)
                for pid, score in cached.items():
                    self._record_score(model_name, step, score)

                missing = [s for s in self.samples if s.prompt_id not in cached]
                if not missing:
                    log.info(f"Skipping {model_name} step {step} (fully cached)")
                    continue

                log.info(
                    f"{model_name} step {step}: {len(cached)} cached, {len(missing)} to evaluate"
                )
                for sample in missing:
                    self.gen_queue.put_nowait(
                        GenJob(sample=sample, model_name=model_name, step=step)
                    )
                    self._total_jobs += 1

        if self._total_jobs == 0:
            log.info("All evaluations cached, nothing to run")
            return self._scores

        self.gen_queue.put_nowait(None)
        log.info(f"Pipeline starting: {self._total_jobs} jobs to process")

        await asyncio.gather(
            self._gen_consumer(),
            self._judge_consumer(),
        )

        log.info(f"Pipeline done: {self._gen_done} generated, {self._judge_done} judged")
        return self._scores
