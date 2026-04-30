"""Train Qwen/Qwen3-8B-Base on Tulu IF — single method per process.

Methods (`--method`):
  tourno       GRPO + mixture reward (pointwise + pairwise tournament, α=3)
  grpo-point   GRPO + pointwise reward
  grpo-pair    GRPO + pairwise reward (batched ELO, ~n log n judge calls per group)
  dpo          Offline DPO on prepared Tulu pairs (datasets/dpo/train.jsonl)
  dpo-online   Online DPO: sample, tournament-judge, DPO update on derived pairs

Reward signal: Anthropic Claude Sonnet 4.5 via OpenRouter, using
prompts/tulu_*_judge.jinja templates.

W&B logging covers per-step scalars (reward / advantage / KL / judge stats),
distributional histograms every 10 steps, rolling sample tables (rollouts,
judge_traces, dpo_pairs), and validation pointwise scoring every save_every
steps. See docstrings below for the complete schema.

Single method per process — wandb is process-global. Use train_all.py to fan
out all 5 methods as parallel subprocesses.

Usage:
    uv run --env-file .env scripts/tulu-experiment/train.py \\
        --method tourno --base-model Qwen/Qwen3-8B-Base --n-steps 400 \\
        --wandb-project tulu-tourno
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import random
import re
import statistics
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pydantic
import tinker
import wandb
from jinja2 import Template
from openai import AsyncOpenAI
from tinker import types as tinker_types
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tourno.data import DataLoader
from tourno.logger import get_logger, setup, trace
from tourno.tournament import batched_elo_reward_fn, compute_round_robin_wins
from tourno.training.dpo import dpo_train_step
from tourno.training.dpo import training_loop as dpo_training_loop
from tourno.training.grpo import training_loop as grpo_training_loop
from tourno.training.models import get_sampling_client, get_training_client
from tourno.training.types import (
    DPOConfig,
    DPOPair,
    GRPOConfig,
    TrainingQueue,
    TrajectoryGroup,
    TrajectoryTurn,
)
from tourno.training.utils import (
    get_learning_rate,
    save_checkpoint_and_get_sampling_client,
)
from tourno.types import PairwiseComparison

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3-8B-Base"
JUDGE_MODEL = "anthropic/claude-sonnet-4-5"
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
POINTWISE_TEMPLATE_PATH = PROMPTS_DIR / "tulu_pointwise_judge.jinja"
PAIRWISE_TEMPLATE_PATH = PROMPTS_DIR / "tulu_pairwise_judge.jinja"

ALL_METHODS = ("tourno", "grpo-point", "grpo-pair", "dpo", "dpo-online")
GRPO_METHODS = {"tourno", "grpo-point", "grpo-pair"}

# Fixed display color per method. W&B's SDK doesn't let you set chart line
# color directly, but we log the hex into config + tags so you can pin the
# colors once in your W&B workspace and they'll persist across runs.
METHOD_COLORS = {
    "tourno":     "#E63946",  # red — TournO (the headline method)
    "grpo-point": "#1D3557",  # navy — pointwise GRPO baseline
    "grpo-pair":  "#06A77D",  # emerald — pairwise GRPO baseline
    "dpo":        "#F4A261",  # warm orange — offline DPO
    "dpo-online": "#9D4EDD",  # purple — online DPO
}

DEFAULT_TRAIN_PATH = Path("datasets/tulu_if_train.jsonl")
DEFAULT_VAL_PATH = Path("datasets/tulu_if_val.jsonl")
DEFAULT_DPO_TRAIN_PATH = Path("datasets/dpo/train.jsonl")

# Logging cadences
HISTOGRAM_EVERY_N_STEPS = 10
ROLLOUTS_TABLE_EVERY_N_STEPS = 20
JUDGE_TRACES_EVERY_N_STEPS = 10
DPO_PAIRS_TABLE_EVERY_N_STEPS = 20
VAL_NUM_PROMPTS = 64

# Trace storage caps (rolling tables hold this many rows max)
ROLLOUTS_MAX_ROWS = 400
JUDGE_TRACES_MAX_ROWS = 400
DPO_PAIRS_MAX_ROWS = 400
VAL_EXAMPLES_PER_EVAL = 10

# Truncation for table cells
PROMPT_TRUNC = 800
COMPLETION_TRUNC = 2000
JUDGE_INPUT_TRUNC = 1200

log = get_logger("train")

# ---------------------------------------------------------------------------
# Sample / data types
# ---------------------------------------------------------------------------


class TuluIFSample(pydantic.BaseModel):
    id: str
    row_id: int | None = None
    prompt: list[dict[str, str]]
    raw_prompt: str
    constraints: list[str]


class TuluIFDataLoader(DataLoader[TuluIFSample]):
    def __init__(self, path: str, **kwargs: Any) -> None:
        super().__init__(path, TuluIFSample, **kwargs)


class TuluPreferenceSample(pydantic.BaseModel):
    id: str | None = None
    row_id: int | None = None
    prompt: list[dict[str, str]]
    chosen: str
    rejected: str
    constraints: list[str] | None = None
    raw_prompt: str | None = None
    chosen_model: str | None = None
    rejected_model: str | None = None


class TuluPreferenceLoader(DataLoader[TuluPreferenceSample]):
    def __init__(self, path: str, **kwargs: Any) -> None:
        super().__init__(path, TuluPreferenceSample, **kwargs)


# ---------------------------------------------------------------------------
# Judge stats collector — drained once per training step
# ---------------------------------------------------------------------------


class JudgeStats:
    """Accumulates judge call outcomes between drains.

    A drain emits scalar metrics + histograms + a sample of raw judge traces
    and clears state. Called once per training step from extra_metrics_fn.
    """

    def __init__(self, *, max_traces: int = 5) -> None:
        # Pointwise
        self.pointwise_scores: list[float] = []
        self.pointwise_latencies_ms: list[float] = []
        self.pointwise_errors: int = 0
        # Pairwise (outcome ∈ {1.0=picked first, 0.0=picked second, 0.5=tie/parse-fail})
        self.pairwise_outcomes: list[float] = []
        self.pairwise_latencies_ms: list[float] = []
        self.pairwise_errors: int = 0
        # Rolling sample of raw judge traces for table
        self.recent_traces: deque[dict[str, Any]] = deque(maxlen=max_traces * 4)
        self._max_traces_per_drain = max_traces

    def record_pointwise(
        self,
        *,
        score: float,
        latency_ms: float,
        error: str | None,
        rendered_input: str,
        raw_response: str,
    ) -> None:
        if error is not None:
            self.pointwise_errors += 1
            return
        self.pointwise_scores.append(score)
        self.pointwise_latencies_ms.append(latency_ms)
        if random.random() < 0.05:  # ~5% sampling rate for traces
            self.recent_traces.append(
                {
                    "judge_type": "pointwise",
                    "judge_input_truncated": rendered_input[-JUDGE_INPUT_TRUNC:],
                    "judge_raw_response": raw_response[:200],
                    "parsed_score": score,
                    "latency_ms": latency_ms,
                    "error": error,
                }
            )

    def record_pairwise(
        self,
        *,
        outcome: float,
        latency_ms: float,
        error: str | None,
        rendered_input: str,
        raw_response: str,
    ) -> None:
        if error is not None:
            self.pairwise_errors += 1
            return
        self.pairwise_outcomes.append(outcome)
        self.pairwise_latencies_ms.append(latency_ms)
        if random.random() < 0.02:
            self.recent_traces.append(
                {
                    "judge_type": "pairwise",
                    "judge_input_truncated": rendered_input[-JUDGE_INPUT_TRUNC:],
                    "judge_raw_response": raw_response[:50],
                    "parsed_score": outcome,
                    "latency_ms": latency_ms,
                    "error": error,
                }
            )

    def drain_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        n_pt = len(self.pointwise_scores)
        n_pw = len(self.pairwise_outcomes)

        if n_pt:
            scores = self.pointwise_scores
            out["judge/pointwise/calls"] = n_pt
            out["judge/pointwise/mean"] = statistics.fmean(scores)
            out["judge/pointwise/std"] = statistics.pstdev(scores) if n_pt > 1 else 0.0
            out["judge/pointwise/frac_zero"] = sum(1 for s in scores if s <= 1e-6) / n_pt
            out["judge/pointwise/frac_one"] = sum(1 for s in scores if s >= 1 - 1e-6) / n_pt
            out["judge/pointwise/latency_ms_p50"] = _quantile(self.pointwise_latencies_ms, 0.5)
            out["judge/pointwise/latency_ms_p95"] = _quantile(self.pointwise_latencies_ms, 0.95)
        if self.pointwise_errors:
            denom = max(n_pt + self.pointwise_errors, 1)
            out["judge/pointwise/error_rate"] = self.pointwise_errors / denom

        if n_pw:
            outs = self.pairwise_outcomes
            chose_first = sum(1 for o in outs if o == 1.0)
            chose_second = sum(1 for o in outs if o == 0.0)
            ties = sum(1 for o in outs if o == 0.5)
            out["judge/pairwise/calls"] = n_pw
            out["judge/pairwise/frac_choose_first"] = chose_first / n_pw
            out["judge/pairwise/frac_choose_second"] = chose_second / n_pw
            out["judge/pairwise/frac_decisive"] = (chose_first + chose_second) / n_pw
            out["judge/pairwise/frac_tie_or_parse_fail"] = ties / n_pw
            # Position bias: 0 ↔ ideal (50/50 first/second); 1 ↔ totally biased one direction.
            decisive = chose_first + chose_second
            if decisive:
                out["judge/pairwise/position_bias"] = abs(chose_first / decisive - 0.5) * 2
            out["judge/pairwise/latency_ms_p50"] = _quantile(self.pairwise_latencies_ms, 0.5)
            out["judge/pairwise/latency_ms_p95"] = _quantile(self.pairwise_latencies_ms, 0.95)
        if self.pairwise_errors:
            denom = max(n_pw + self.pairwise_errors, 1)
            out["judge/pairwise/error_rate"] = self.pairwise_errors / denom

        out["judge/calls_step"] = n_pt + n_pw
        return out

    def drain_histograms(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.pointwise_scores:
            out["judge/pointwise_score_distribution"] = wandb.Histogram(self.pointwise_scores)
        if self.pairwise_outcomes:
            out["judge/pairwise_decision_distribution"] = wandb.Histogram(self.pairwise_outcomes)
        return out

    def drain_traces(self) -> list[dict[str, Any]]:
        traces = list(self.recent_traces)[: self._max_traces_per_drain]
        self.recent_traces.clear()
        return traces

    def reset_step_state(self) -> None:
        self.pointwise_scores.clear()
        self.pointwise_latencies_ms.clear()
        self.pointwise_errors = 0
        self.pairwise_outcomes.clear()
        self.pairwise_latencies_ms.clear()
        self.pairwise_errors = 0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = max(0, min(len(arr) - 1, int(q * (len(arr) - 1))))
    return arr[idx]


# ---------------------------------------------------------------------------
# Sonnet 4.5 judges (instrumented with JudgeStats)
# ---------------------------------------------------------------------------


def _build_openrouter_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        max_retries=2,
    )


def _parse_pointwise(text: str) -> float:
    text = text.strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        pass
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return max(0.0, min(1.0, float(nums[-1])))
    return 0.0


class SonnetPointwiseJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        template: Template,
        *,
        stats: JudgeStats,
        sem: asyncio.Semaphore,
        model: str = JUDGE_MODEL,
        max_tokens: int = 8,
    ) -> None:
        self.client = client
        self.template = template
        self.stats = stats
        self.sem = sem
        self.model = model
        self.max_tokens = max_tokens

    async def score(self, *, prompt: str, completion: str, constraints: list[str]) -> float:
        message = self.template.render(prompt=prompt, completion=completion, constraints=constraints)
        text = ""
        err: str | None = None
        async with self.sem:
            t0 = time.monotonic()
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": message}],
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    timeout=60.0,
                )
                text = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                err = repr(exc)
            latency_ms = (time.monotonic() - t0) * 1000
        score = _parse_pointwise(text) if err is None else 0.0
        self.stats.record_pointwise(
            score=score, latency_ms=latency_ms, error=err,
            rendered_input=message, raw_response=text,
        )
        return score


class SonnetPairwiseJudge:
    """Implements `tourno.types.PairwiseJudge` for one prompt's constraints."""

    def __init__(
        self,
        client: AsyncOpenAI,
        template: Template,
        constraints: list[str],
        *,
        stats: JudgeStats,
        sem: asyncio.Semaphore,
        model: str = JUDGE_MODEL,
    ) -> None:
        self.client = client
        self.template = template
        self.constraints = constraints
        self.stats = stats
        self.sem = sem
        self.model = model

    async def __call__(self, samples: list[PairwiseComparison]) -> list[float]:
        return await asyncio.gather(*(self._one(s) for s in samples))

    async def _one(self, s: PairwiseComparison) -> float:
        message = self.template.render(
            prompt=s["prompt"],
            completion1=s["completion1"],
            completion2=s["completion2"],
            constraints=self.constraints,
        )
        text = ""
        err: str | None = None
        async with self.sem:
            t0 = time.monotonic()
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": message}],
                    max_tokens=4,
                    temperature=0.0,
                    timeout=60.0,
                )
                text = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                err = repr(exc)
            latency_ms = (time.monotonic() - t0) * 1000

        if err is not None:
            outcome = 0.5
        elif not text:
            outcome = 0.5
        elif text[0] == "0":
            outcome = 1.0
        elif text[0] == "1":
            outcome = 0.0
        else:
            outcome = 0.5
        self.stats.record_pairwise(
            outcome=outcome, latency_ms=latency_ms, error=err,
            rendered_input=message, raw_response=text,
        )
        return outcome


# ---------------------------------------------------------------------------
# Rolling tables
# ---------------------------------------------------------------------------


class RollingTable:
    def __init__(self, columns: list[str], max_rows: int) -> None:
        self.columns = columns
        self.rows: deque[list[Any]] = deque(maxlen=max_rows)

    def add(self, row: list[Any]) -> None:
        self.rows.append(row)

    def to_wandb(self) -> wandb.Table:
        return wandb.Table(columns=list(self.columns), data=[list(r) for r in self.rows])

    def __len__(self) -> int:
        return len(self.rows)


def _truncate(text: str, n: int) -> str:
    if text is None:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


# ---------------------------------------------------------------------------
# Reward functions for GRPO methods
# ---------------------------------------------------------------------------


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-8:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def make_get_rewards(
    method: str,
    *,
    judge_client: AsyncOpenAI,
    pointwise_template: Template,
    pairwise_template: Template,
    pairwise_alpha: float,
    pointwise_sem: asyncio.Semaphore,
    pairwise_sem: asyncio.Semaphore,
    judge_stats: JudgeStats,
    rollouts_table: RollingTable,
    sampling_step_ref: list[int],
    rollouts_sample_rate: float = 0.125,
) -> Callable[[TuluIFSample, list[str], list[str]], Awaitable[tuple[list[float], int]]]:
    pointwise_judge = SonnetPointwiseJudge(
        judge_client, pointwise_template, stats=judge_stats, sem=pointwise_sem
    )

    async def get_rewards(
        sample: TuluIFSample,
        completions: list[str],
        _ids: list[str],
    ) -> tuple[list[float], int]:
        n = len(completions)
        do_pointwise = method in ("tourno", "grpo-point")
        do_pairwise = method in ("tourno", "grpo-pair")

        point: list[float] = [0.0] * n
        pair: list[float] = [0.0] * n
        judge_calls = 0

        if do_pointwise:
            point = list(
                await asyncio.gather(
                    *(
                        pointwise_judge.score(
                            prompt=sample.raw_prompt,
                            completion=c,
                            constraints=sample.constraints,
                        )
                        for c in completions
                    )
                )
            )
            judge_calls += n

        if do_pairwise:
            pj = SonnetPairwiseJudge(
                judge_client, pairwise_template, sample.constraints,
                stats=judge_stats, sem=pairwise_sem,
            )
            if method == "grpo-pair":
                pair, calls = await batched_elo_reward_fn(sample.raw_prompt, completions, pj)
                judge_calls += calls
            else:
                wins = await compute_round_robin_wins(sample.raw_prompt, completions, pj)
                judge_calls += int(n * (n - 1) / 2)
                win_rates = wins.sum(axis=1) / max(n - 1, 1)
                pair = _normalize(win_rates.tolist())

        if method == "tourno":
            r_bar = sum(point) / max(len(point), 1)
            mix_w = math.exp(-pairwise_alpha * r_bar)
            final = [p + mix_w * q for p, q in zip(point, pair)]
        elif method == "grpo-point":
            mix_w = 0.0
            final = point
        elif method == "grpo-pair":
            mix_w = 0.0
            final = pair
        else:
            raise ValueError(method)

        # Probabilistically log this prompt's group into the rollouts table.
        # Decoupled from training-loop step boundaries — workers run async ahead
        # of the trainer. Use sampling_step_ref[0] as best-effort step label.
        if random.random() < rollouts_sample_rate:
            cur_step = sampling_step_ref[0]
            for i, comp in enumerate(completions):
                rollouts_table.add(
                    [
                        cur_step,
                        sample.id,
                        _truncate(sample.raw_prompt, PROMPT_TRUNC),
                        ", ".join(sample.constraints),
                        i,
                        _truncate(comp, COMPLETION_TRUNC),
                        len(comp),
                        round(point[i], 4) if do_pointwise else None,
                        round(pair[i], 4) if do_pairwise else None,
                        round(mix_w, 4),
                        round(final[i], 4),
                    ]
                )

        return final, judge_calls

    return get_rewards


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


async def make_validation_fn(
    *,
    val_path: Path,
    judge_client: AsyncOpenAI,
    pointwise_template: Template,
    pointwise_sem: asyncio.Semaphore,
    n_prompts: int,
    sampling_client_ref: list[tinker.SamplingClient | None],
    renderer: Renderer,
    max_tokens: int,
    val_examples_table: RollingTable,
    judge_stats: JudgeStats,
) -> Callable[[int], Awaitable[dict[str, Any]]]:
    samples: list[TuluIFSample] = []
    if n_prompts > 0 and val_path.exists():
        loader = TuluIFDataLoader(str(val_path), batch_size=1, max_length=n_prompts, shuffle=True)
        for _epoch, batch in loader:
            if not batch:
                break
            samples.append(batch[0])
            if len(samples) >= n_prompts:
                break
    if not samples:
        log.info(f"Validation: skipping (n_prompts={n_prompts}, val_path={val_path})")

    pointwise_judge = SonnetPointwiseJudge(
        judge_client, pointwise_template, stats=judge_stats, sem=pointwise_sem
    )

    async def run(_step: int) -> dict[str, Any]:
        sampling_client = sampling_client_ref[0]
        if sampling_client is None or not samples:
            return {}

        async def _gen_and_score(s: TuluIFSample) -> tuple[str, float]:
            obs = renderer.build_generation_prompt(s.prompt)
            res = await sampling_client.sample_async(
                prompt=obs,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=0.7),
            )
            tokens = list(res.sequences[0].tokens)
            text = renderer.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            score = await pointwise_judge.score(
                prompt=s.raw_prompt, completion=text, constraints=s.constraints
            )
            return text, score

        results = await asyncio.gather(*(_gen_and_score(s) for s in samples))
        scores = [r[1] for r in results]
        for s, (text, score) in list(zip(samples, results))[:VAL_EXAMPLES_PER_EVAL]:
            val_examples_table.add(
                [
                    _step,
                    s.id,
                    _truncate(s.raw_prompt, PROMPT_TRUNC),
                    ", ".join(s.constraints),
                    _truncate(text, COMPLETION_TRUNC),
                    round(score, 4),
                ]
            )
        out: dict[str, Any] = {
            "val/pointwise_mean": statistics.fmean(scores) if scores else 0.0,
            "val/pointwise_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "val/n_prompts": len(scores),
        }
        if scores:
            out["val/pointwise_distribution"] = wandb.Histogram(scores)
            out["val/examples_table"] = val_examples_table.to_wandb()
        return out

    return run


# ---------------------------------------------------------------------------
# GRPO trainer
# ---------------------------------------------------------------------------


def _decode_trajectories(trajectories: list[list[TrajectoryTurn]], renderer: Renderer) -> list[str]:
    stop_ids: set[int] = set()
    for stop in renderer.get_stop_sequences():
        if isinstance(stop, int):
            stop_ids.add(stop)
        elif isinstance(stop, str):
            ids = renderer.tokenizer.encode(stop)
            if ids:
                stop_ids.add(ids[-1])

    out: list[str] = []
    for traj in trajectories:
        tokens = list(traj[0].ac.tokens)
        if tokens and tokens[-1] in stop_ids:
            tokens = tokens[:-1]
        out.append(renderer.tokenizer.decode(tokens, skip_special_tokens=True))
    return out


@trace
async def _rollout(
    *,
    sample: TuluIFSample,
    renderer: Renderer,
    sampling_client_with_step: tuple[tinker.SamplingClient, int],
    training_queue: TrainingQueue,
    get_rewards: Callable[[TuluIFSample, list[str], list[str]], Awaitable[tuple[list[float], int]]],
    group_size: int,
    max_tokens: int,
    temperature: float,
) -> None:
    sampling_client, step = sampling_client_with_step
    obs = renderer.build_generation_prompt(sample.prompt)
    completions = await sampling_client.sample_async(
        prompt=obs,
        num_samples=group_size,
        sampling_params=tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature),
    )
    trajectories = [[TrajectoryTurn(obs=obs, ac=seq)] for seq in completions.sequences]
    texts = _decode_trajectories(trajectories, renderer)
    rollout_ids = [f"{sample.row_id}_{i}" for i in range(len(texts))]
    rewards, judge_calls = await get_rewards(sample, texts, rollout_ids)
    await training_queue.put(
        (
            step,
            TrajectoryGroup(
                group_size=group_size,
                trajectories=trajectories,
                rewards=rewards,
                judge_calls=judge_calls,
            ),
        )
    )


async def run_grpo(
    *,
    method: str,
    args: argparse.Namespace,
    judge_client: AsyncOpenAI,
    pointwise_template: Template,
    pairwise_template: Template,
) -> None:
    config = GRPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        judge_type=method,
        judge_model=JUDGE_MODEL,
        base_url=args.base_url,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        group_size=args.group_size,
        n_steps=args.n_steps,
        save_every=args.save_every,
        log_path=str(args.log_root / method),
        run_name=f"tulu_{method}",
        wandb_project=None,  # we handle wandb ourselves below
    )
    Path(config.log_path).mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(config.base_model)
    renderer_name = args.renderer or get_recommended_renderer_name(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer)

    pointwise_sem = asyncio.Semaphore(args.judge_concurrency)
    pairwise_sem = asyncio.Semaphore(args.judge_concurrency)
    judge_stats = JudgeStats()
    rollouts_table = RollingTable(
        columns=[
            "step", "prompt_id", "prompt", "constraints", "comp_idx", "completion",
            "comp_len_chars", "pointwise_reward", "pairwise_reward", "mixture_weight",
            "final_reward",
        ],
        max_rows=ROLLOUTS_MAX_ROWS,
    )
    judge_traces_table = RollingTable(
        columns=[
            "step", "judge_type", "judge_input_truncated", "judge_raw_response",
            "parsed_score", "latency_ms", "error",
        ],
        max_rows=JUDGE_TRACES_MAX_ROWS,
    )
    val_examples_table = RollingTable(
        columns=["step", "prompt_id", "prompt", "constraints", "completion", "pointwise_score"],
        max_rows=200,
    )

    sampling_step_ref = [0]  # updated by update_sampling_client; read by get_rewards

    get_rewards = make_get_rewards(
        method,
        judge_client=judge_client,
        pointwise_template=pointwise_template,
        pairwise_template=pairwise_template,
        pairwise_alpha=args.pairwise_alpha,
        pointwise_sem=pointwise_sem,
        pairwise_sem=pairwise_sem,
        judge_stats=judge_stats,
        rollouts_table=rollouts_table,
        sampling_step_ref=sampling_step_ref,
    )

    dataloader = TuluIFDataLoader(
        str(args.train_path), batch_size=1, max_length=args.max_samples
    )

    sampling_client_ref: list[tinker.SamplingClient | None] = [None]
    sampling_client_with_step: tuple[tinker.SamplingClient, int] | None = None
    sampling_client_ready = asyncio.Event()

    def update_sampling_client(client: tinker.SamplingClient, step: int) -> None:
        nonlocal sampling_client_with_step
        sampling_client_with_step = (client, step)
        sampling_client_ref[0] = client
        sampling_step_ref[0] = max(step, 0)
        if not sampling_client_ready.is_set():
            sampling_client_ready.set()

    validation_fn = await make_validation_fn(
        val_path=args.val_path,
        judge_client=judge_client,
        pointwise_template=pointwise_template,
        pointwise_sem=pointwise_sem,
        n_prompts=args.val_n_prompts,
        sampling_client_ref=sampling_client_ref,
        renderer=renderer,
        max_tokens=args.max_tokens,
        val_examples_table=val_examples_table,
        judge_stats=judge_stats,
    )

    def extra_metrics_fn(step: int, _trajectory_groups: list[TrajectoryGroup]) -> dict[str, Any]:
        out = judge_stats.drain_metrics()
        # Histograms
        if step % HISTOGRAM_EVERY_N_STEPS == 0:
            out.update(judge_stats.drain_histograms())
        # Judge traces table
        if step % JUDGE_TRACES_EVERY_N_STEPS == 0:
            for tr in judge_stats.drain_traces():
                judge_traces_table.add(
                    [step, tr["judge_type"], tr["judge_input_truncated"], tr["judge_raw_response"],
                     tr["parsed_score"], round(tr["latency_ms"], 1), tr["error"]]
                )
            if len(judge_traces_table) > 0:
                out["judge_traces_table"] = judge_traces_table.to_wandb()
        # Rollouts table push
        if step % ROLLOUTS_TABLE_EVERY_N_STEPS == 0 and len(rollouts_table) > 0:
            out["rollouts_table"] = rollouts_table.to_wandb()
        judge_stats.reset_step_state()
        return out

    training_queue: TrainingQueue = asyncio.Queue()
    training_loop_task = asyncio.create_task(
        grpo_training_loop(
            config,
            training_queue,
            update_sampling_client,
            extra_metrics_fn=extra_metrics_fn,
            validation_fn=validation_fn,
        )
    )

    await asyncio.wait(
        [asyncio.create_task(sampling_client_ready.wait()), training_loop_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if training_loop_task.done():
        training_loop_task.result()
        return

    async def worker(_worker_id: int) -> None:
        async for _, batch in dataloader:
            sample = batch[0]
            assert sampling_client_with_step is not None
            await _rollout(
                sample=sample,
                renderer=renderer,
                sampling_client_with_step=sampling_client_with_step,
                training_queue=training_queue,
                get_rewards=get_rewards,
                group_size=args.group_size,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

    worker_tasks = [asyncio.create_task(worker(i)) for i in range(args.num_workers)]
    all_tasks = [training_loop_task, *worker_tasks]
    try:
        done, _ = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception():
                raise t.exception()  # type: ignore[misc]
    finally:
        for t in all_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Offline DPO
# ---------------------------------------------------------------------------


def _build_dpo_pair(sample: TuluPreferenceSample, renderer: Renderer) -> DPOPair:
    obs = renderer.build_generation_prompt(sample.prompt)

    def _resp_tokens(text: str) -> list[int]:
        messages = [*sample.prompt, {"role": "assistant", "content": text}]
        tokens, weights = renderer.build_supervised_example(messages)
        return [t for t, w in zip(tokens.tolist(), weights.tolist()) if w > 0]

    return DPOPair(
        obs=obs,
        chosen=tinker_types.SampledSequence(stop_reason="stop", tokens=_resp_tokens(sample.chosen)),
        rejected=tinker_types.SampledSequence(
            stop_reason="stop", tokens=_resp_tokens(sample.rejected)
        ),
    )


def _dpo_pair_batches_with_meta(
    loader: TuluPreferenceLoader,
    renderer: Renderer,
    meta_buf: dict[int, list[TuluPreferenceSample]],
) -> Iterable[tuple[int, Sequence[DPOPair]]]:
    """Yields DPO batches, also stashes the source samples into meta_buf[step]
    so the extra_metrics_fn can read them when logging the dpo_pairs table."""
    step = 0
    for epoch, samples in loader:
        meta_buf[step] = list(samples)
        yield epoch, [_build_dpo_pair(s, renderer) for s in samples]
        step += 1


async def run_dpo_offline(
    *,
    args: argparse.Namespace,
    judge_client: AsyncOpenAI,
    pointwise_template: Template,
) -> None:
    config = DPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        base_url=args.base_url,
        learning_rate=args.dpo_learning_rate,
        beta=args.dpo_beta,
        batch_size=args.dpo_batch_size,
        n_steps=args.n_steps,
        save_every=args.save_every,
        log_path=str(args.log_root / "dpo"),
        run_name="tulu_dpo",
        wandb_project=None,
    )
    Path(config.log_path).mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(config.base_model)
    renderer_name = args.renderer or get_recommended_renderer_name(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer)

    loader = TuluPreferenceLoader(
        str(args.dpo_train_path), batch_size=args.dpo_batch_size, max_length=args.max_samples
    )

    judge_stats = JudgeStats()
    pointwise_sem = asyncio.Semaphore(args.judge_concurrency)
    val_examples_table = RollingTable(
        columns=["step", "prompt_id", "prompt", "constraints", "completion", "pointwise_score"],
        max_rows=200,
    )
    dpo_pairs_table = RollingTable(
        columns=[
            "step", "id", "prompt", "chosen", "rejected", "chosen_model", "rejected_model",
        ],
        max_rows=DPO_PAIRS_MAX_ROWS,
    )

    meta_buf: dict[int, list[TuluPreferenceSample]] = {}
    train_data = _dpo_pair_batches_with_meta(loader, renderer, meta_buf)

    # Validation is skipped in the offline DPO path — the underlying
    # `dpo.training_loop` doesn't expose its sampling client to us, and rerolling
    # one per validation cycle would be wasteful. Use --method dpo-online if you
    # want validation curves on a DPO method.

    def extra_metrics_fn(step: int, _batch: Sequence[DPOPair]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if step % DPO_PAIRS_TABLE_EVERY_N_STEPS == 0:
            samples = meta_buf.pop(step, [])
            for s in samples[:3]:
                dpo_pairs_table.add(
                    [
                        step,
                        s.id or "",
                        _truncate(s.raw_prompt or s.prompt[0]["content"], PROMPT_TRUNC),
                        _truncate(s.chosen, COMPLETION_TRUNC),
                        _truncate(s.rejected, COMPLETION_TRUNC),
                        s.chosen_model or "",
                        s.rejected_model or "",
                    ]
                )
            if len(dpo_pairs_table) > 0:
                out["dpo_pairs_table"] = dpo_pairs_table.to_wandb()
        # Drop unused step state in meta_buf (long-running keep-clean)
        for old in [k for k in meta_buf if k < step - 5]:
            meta_buf.pop(old, None)
        return out

    await dpo_training_loop(config, train_data, extra_metrics_fn=extra_metrics_fn)


# ---------------------------------------------------------------------------
# Online DPO
# ---------------------------------------------------------------------------


async def _build_online_pairs(
    *,
    sample: TuluIFSample,
    completions: tinker.SampleResult,
    renderer: Renderer,
    judge_factory: Callable[[list[str]], SonnetPairwiseJudge],
) -> tuple[list[DPOPair], list[float], list[str], int, int]:
    n = len(completions.sequences)
    if n < 2:
        return [], [], [], -1, -1

    obs = renderer.build_generation_prompt(sample.prompt)
    decoded = _decode_trajectories(
        [[TrajectoryTurn(obs=obs, ac=seq)] for seq in completions.sequences], renderer,
    )
    judge = judge_factory(sample.constraints)
    wins = await compute_round_robin_wins(sample.raw_prompt, decoded, judge)
    win_rates = (wins.sum(axis=1) / max(n - 1, 1)).tolist()

    pairs: list[DPOPair] = []
    for i in range(n):
        for j in range(i + 1, n):
            if wins[i, j] > wins[j, i]:
                ci, ri = i, j
            elif wins[j, i] > wins[i, j]:
                ci, ri = j, i
            else:
                continue
            pairs.append(
                DPOPair(
                    obs=obs,
                    chosen=completions.sequences[ci],
                    rejected=completions.sequences[ri],
                )
            )
    best = max(range(n), key=lambda i: win_rates[i])
    worst = min(range(n), key=lambda i: win_rates[i])
    return pairs, win_rates, decoded, best, worst


async def run_dpo_online(
    *,
    args: argparse.Namespace,
    judge_client: AsyncOpenAI,
    pairwise_template: Template,
    pointwise_template: Template,
) -> None:
    log_path = args.log_root / "dpo-online"
    log_path.mkdir(parents=True, exist_ok=True)

    config = DPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        base_url=args.base_url,
        learning_rate=args.dpo_learning_rate,
        beta=args.dpo_beta,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        save_every=args.save_every,
        log_path=str(log_path),
        run_name="tulu_dpo_online",
        wandb_project=None,
    )

    tokenizer = get_tokenizer(config.base_model)
    renderer_name = args.renderer or get_recommended_renderer_name(config.base_model)
    renderer = get_renderer(renderer_name, tokenizer)

    training_client = await get_training_client(
        config.base_model, lora_rank=config.lora_rank, base_url=config.base_url
    )
    reference_client = await get_sampling_client(
        base_model=config.base_model, base_url=config.base_url
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    sampling_client_ref: list[tinker.SamplingClient | None] = [sampling_client]

    dataloader = TuluIFDataLoader(
        str(args.train_path), batch_size=args.batch_size, max_length=args.max_samples
    )
    judge_stats = JudgeStats()
    pointwise_sem = asyncio.Semaphore(args.judge_concurrency)
    pairwise_sem = asyncio.Semaphore(args.judge_concurrency)
    val_examples_table = RollingTable(
        columns=["step", "prompt_id", "prompt", "constraints", "completion", "pointwise_score"],
        max_rows=200,
    )
    dpo_pairs_table = RollingTable(
        columns=[
            "step", "prompt_id", "prompt", "chosen_idx", "rejected_idx",
            "chosen_winrate", "rejected_winrate", "n_pairs_in_group",
        ],
        max_rows=DPO_PAIRS_MAX_ROWS,
    )

    def judge_factory(constraints: list[str]) -> SonnetPairwiseJudge:
        return SonnetPairwiseJudge(
            judge_client, pairwise_template, constraints,
            stats=judge_stats, sem=pairwise_sem,
        )

    validation_fn = await make_validation_fn(
        val_path=args.val_path,
        judge_client=judge_client,
        pointwise_template=pointwise_template,
        pointwise_sem=pointwise_sem,
        n_prompts=args.val_n_prompts,
        sampling_client_ref=sampling_client_ref,
        renderer=renderer,
        max_tokens=args.max_tokens,
        val_examples_table=val_examples_table,
        judge_stats=judge_stats,
    )

    for step, (epoch, samples) in enumerate(dataloader):
        if step >= config.n_steps:
            break
        t_start = time.time()

        sample_results = await asyncio.gather(
            *(
                sampling_client_ref[0].sample_async(  # type: ignore[union-attr]
                    prompt=renderer.build_generation_prompt(s.prompt),
                    num_samples=args.group_size,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=args.max_tokens, temperature=args.temperature
                    ),
                )
                for s in samples
            )
        )

        per_prompt = await asyncio.gather(
            *(
                _build_online_pairs(
                    sample=s, completions=r, renderer=renderer, judge_factory=judge_factory,
                )
                for s, r in zip(samples, sample_results)
            )
        )
        pairs: list[DPOPair] = [p for grp in per_prompt for p in grp[0]]

        if step % DPO_PAIRS_TABLE_EVERY_N_STEPS == 0:
            for s, (group_pairs, win_rates, _decoded, best, worst) in list(zip(samples, per_prompt))[:3]:
                if best < 0:
                    continue
                dpo_pairs_table.add(
                    [
                        step, s.id,
                        _truncate(s.raw_prompt, PROMPT_TRUNC),
                        best, worst,
                        round(win_rates[best], 4) if win_rates else None,
                        round(win_rates[worst], 4) if win_rates else None,
                        len(group_pairs),
                    ]
                )

        if not pairs:
            log.warning(f"step {step}: no valid pairs from tournaments; skipping")
            continue

        lr = get_learning_rate(step, config)
        train_metrics = await dpo_train_step(
            pairs, training_client, reference_client,
            tokenizer=tokenizer, beta=config.beta, learning_rate=lr,
        )

        completed = step + 1
        should_save = (
            config.save_every > 0 and completed > 0 and completed % config.save_every == 0
        )
        if should_save:
            sampling_client_ref[0], ckpt_metrics = await save_checkpoint_and_get_sampling_client(
                training_client, completed, config.log_path, config.ttl_seconds
            )
        else:
            sampling_client_ref[0] = await training_client.save_weights_and_get_sampling_client_async()
            ckpt_metrics = {}

        metrics: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "optim/lr": lr,
            "progress/done_frac": (step + 1) / max(config.n_steps, 1),
            "online/n_completions_total": sum(len(r.sequences) for r in sample_results),
            "online/n_pairs_built": len(pairs),
            "online/pairs_per_prompt": len(pairs) / max(len(samples), 1),
            "time/total": time.time() - t_start,
        }
        metrics.update(train_metrics)
        metrics.update(judge_stats.drain_metrics())
        if step % HISTOGRAM_EVERY_N_STEPS == 0:
            metrics.update(judge_stats.drain_histograms())
        if step % DPO_PAIRS_TABLE_EVERY_N_STEPS == 0 and len(dpo_pairs_table) > 0:
            metrics["dpo_pairs_table"] = dpo_pairs_table.to_wandb()
        if should_save:
            val_metrics = await validation_fn(completed)
            metrics.update(val_metrics)
        metrics.update(ckpt_metrics)
        judge_stats.reset_step_state()

        wandb.log(metrics, step=step)
        log.info(
            f"[dpo-online] step={step} pairs={len(pairs)} "
            f"loss={train_metrics.get('dpo/loss', float('nan')):.4f}"
        )


# ---------------------------------------------------------------------------
# Main / dispatch
# ---------------------------------------------------------------------------


async def main(args: argparse.Namespace) -> None:
    if args.method not in ALL_METHODS:
        raise SystemExit(f"--method must be one of {ALL_METHODS}; got {args.method!r}")

    args.log_root.mkdir(parents=True, exist_ok=True)

    pointwise_template = Template(POINTWISE_TEMPLATE_PATH.read_text())
    pairwise_template = Template(PAIRWISE_TEMPLATE_PATH.read_text())
    judge_client = _build_openrouter_client()

    # Initialize wandb upfront so all subsequent wandb.log calls (from ourselves
    # AND from the underlying training loops) use the same run.
    model_short = args.base_model.split("/")[-1]
    run_name = args.run_name or f"tulu_{args.method}_{model_short}"
    color = METHOD_COLORS[args.method]
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=[
                f"method={args.method}",
                f"base={model_short}",
                f"color={color}",
            ],
            config={
                "method": args.method,
                "color": color,  # so you can color the workspace by config.color
                "base_model": args.base_model,
                "judge_model": JUDGE_MODEL,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "lora_rank": args.lora_rank,
                "learning_rate": args.learning_rate,
                "pairwise_alpha": args.pairwise_alpha,
                "dpo_learning_rate": args.dpo_learning_rate,
                "dpo_beta": args.dpo_beta,
                "save_every": args.save_every,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "val_n_prompts": args.val_n_prompts,
            },
        )

    log.info(f"=== running {args.method} on {args.base_model} ===")
    if args.method in GRPO_METHODS:
        await run_grpo(
            method=args.method, args=args, judge_client=judge_client,
            pointwise_template=pointwise_template, pairwise_template=pairwise_template,
        )
    elif args.method == "dpo":
        await run_dpo_offline(
            args=args, judge_client=judge_client, pointwise_template=pointwise_template,
        )
    elif args.method == "dpo-online":
        await run_dpo_online(
            args=args, judge_client=judge_client,
            pairwise_template=pairwise_template, pointwise_template=pointwise_template,
        )
    log.info(f"=== finished {args.method} ===")
    if args.wandb_project:
        wandb.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method", type=str, required=True,
        choices=list(ALL_METHODS),
        help=f"One of: {', '.join(ALL_METHODS)}",
    )
    p.add_argument("--base-model", type=str, default=BASE_MODEL)
    p.add_argument("--renderer", type=str, default=None)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    p.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH)
    p.add_argument("--dpo-train-path", type=Path, default=DEFAULT_DPO_TRAIN_PATH)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)

    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--pairwise-alpha", type=float, default=3.0)

    p.add_argument("--dpo-learning-rate", type=float, default=1e-5)
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--dpo-batch-size", type=int, default=8)

    p.add_argument("--judge-concurrency", type=int, default=128)
    p.add_argument("--val-n-prompts", type=int, default=VAL_NUM_PROMPTS)

    p.add_argument("--log-root", type=Path, default=Path("./tulu-runs"))
    p.add_argument("--wandb-project", type=str, default="tourno-tulu")
    p.add_argument("--wandb-entity", type=str, default="Haize-Research")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-filter", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(level=getattr(logging, args.log_level.upper()), filter_pattern=args.log_filter)
    asyncio.run(main(args))
