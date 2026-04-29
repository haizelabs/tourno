import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path

import tracelog
from jinja2 import Template
from openai import AsyncOpenAI
from pioneer.logger import get_logger, log_agent_run, trace
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tourno.types import PairwiseComparison

PROMPTS_DIR = Path("./prompts")
POINTWISE_TEMPLATE = Template((PROMPTS_DIR / "tulu_pointwise.jinja").read_text())
PAIRWISE_TEMPLATE = Template((PROMPTS_DIR / "tulu_pairwise.jinja").read_text())

# Chat / instruction-following rubric — 5 positive criteria, 0-10 each.
# Constraint Satisfaction is dominant when constraints are present.
POSITIVE_CRITERIA = [
    "Constraint Satisfaction",
    "Helpfulness",
    "Accuracy",
    "Completeness",
    "Clarity",
]
NEGATIVE_CRITERIA: list[str] = []
ALL_CRITERIA = POSITIVE_CRITERIA + NEGATIVE_CRITERIA
NEGATIVE_SET: set[str] = set(NEGATIVE_CRITERIA)
_NORMALIZED_TO_CANONICAL = {
    re.sub(r"[^a-z0-9]", "", c.lower()): c for c in ALL_CRITERIA
}

PAIRWISE_CRITERIA = [
    "constraint_satisfaction",
    "helpfulness",
    "accuracy",
    "completeness",
    "clarity",
    "instruction_following",
    "avoids_sycophancy",
]

MARGIN_TABLE = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
RESPONDENT_A = "A0493"
RESPONDENT_B = "A0488"
SCORE_RANGE_MAX = 10.0  # 0-10 scale (vs cwb's 0-20)

_log = get_logger("cwbench_judges")


class EmptyJudgeResponse(RuntimeError):
    """Raised when the judge returns an empty/None content (Anthropic safety filter, etc.).

    Caught by the tenacity retry decorator; after retry exhaustion the wrapped fn falls back
    to its `retry_error_callback` reward (0.0 for pointwise, random 0/1 for pairwise).
    """
_RETRY_KWARGS = dict(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_log, logging.WARNING),
)
_pointwise_retry = retry(**_RETRY_KWARGS, retry_error_callback=lambda _: 0.0)
_pairwise_retry = retry(
    **_RETRY_KWARGS, retry_error_callback=lambda _: float(random.randint(0, 1))
)

_SCORE_LINE_RE = re.compile(
    r"^([^\n:]+?):\s*\[?\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)\s*\]?\s*$",
    re.MULTILINE,
)


def _canonicalize(metric: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", metric.lower())
    return _NORMALIZED_TO_CANONICAL.get(key)


_SINGLE_SCORE_RE = re.compile(r"(?im)^\s*Score\s*:\s*\[?\s*(-?\d+(?:\.\d+)?)\s*\]?\s*$")


def _parse_pointwise_scores(text: str) -> dict[str, float]:
    """Tulu-3 IF judge emits a single 'Score: N' line on a 0-10 scale (proportional to
    fraction of constraints satisfied)."""
    scores: dict[str, float] = {}
    m = _SINGLE_SCORE_RE.search(text)
    if m:
        try:
            val = float(m.group(1))
        except ValueError:
            return {}
        if 0.0 <= val <= SCORE_RANGE_MAX:
            scores["Score"] = val
    return scores


def _pointwise_reward(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    val = scores.get("Score")
    if val is None:
        adjusted: list[float] = []
        for metric, s in scores.items():
            adj = (SCORE_RANGE_MAX - s) if metric in NEGATIVE_SET else s
            adjusted.append(adj)
        if not adjusted:
            return 0.0
        val = sum(adjusted) / len(adjusted)
    return max(0.0, min(1.0, val / SCORE_RANGE_MAX))


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _robust_json_loads(text: str) -> dict:
    text = text.strip()
    # Strip raw control chars (excluding tab/newline/CR) which Sonnet sometimes emits inside
    # JSON string values and which json.loads rejects with "Invalid control character".
    text = _CONTROL_CHAR_RE.sub(" ", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object in: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in: {text[:200]!r}")


def _parse_margin(text: str) -> float:
    t = text.strip()
    if t.startswith(RESPONDENT_A):
        sign = +1
        rest = t[len(RESPONDENT_A) :]
    elif t.startswith(RESPONDENT_B):
        sign = -1
        rest = t[len(RESPONDENT_B) :]
    else:
        raise ValueError(f"No respondent code in: {t!r}")
    plus_count = min(rest.count("+"), 5)
    return sign * MARGIN_TABLE[plus_count]


def _parse_pairwise_response(text: str) -> float:
    data = _robust_json_loads(text)
    margins: list[float] = []
    for c in PAIRWISE_CRITERIA:
        v = data.get(c)
        if isinstance(v, str):
            try:
                margins.append(_parse_margin(v))
            except ValueError:
                continue
    if not margins:
        raise ValueError(f"No parsable criteria in: {text[:200]!r}")
    avg = sum(margins) / len(margins)
    return (avg + 1.0) / 2.0


def _wandb_log_safe(payload: dict) -> None:
    """Log to wandb if a run is active; no-op otherwise. Never let logging crash training."""
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log(payload)
    except Exception:
        pass


class _JudgeStats:
    """Counters/latencies shared across calls of a single judge instance."""

    __slots__ = ("success", "failure", "total_latency_s", "total_tokens_in", "total_tokens_out")

    def __init__(self) -> None:
        self.success = 0
        self.failure = 0
        self.total_latency_s = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0


class CreativeBenchPointwiseJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 120.0,
        concurrency: int = 128,
    ):
        self.client = client
        self.model = model
        self.judge_sampling_args = judge_sampling_args or {}
        self.request_timeout_s = request_timeout_s
        self._global_request_sem = asyncio.Semaphore(concurrency)
        self.stats = _JudgeStats()

    @property
    def call_count(self) -> int:
        return self.stats.success + self.stats.failure

    @trace
    @_pointwise_retry
    async def _judge_one(
        self,
        writing_prompt: str,
        completion: str,
        constraints: list[str] | None = None,
    ) -> float:
        async with self._global_request_sem:
            user_prompt = POINTWISE_TEMPLATE.render(
                prompt=writing_prompt,
                completion=completion,
                constraints=constraints or [],
            )
            messages = [{"role": "user", "content": user_prompt}]
            t0 = time.monotonic()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=self.request_timeout_s,
                    stream=False,
                    **self.judge_sampling_args,
                )
                content = response.choices[0].message.content
                if not content:
                    finish_reason = response.choices[0].finish_reason
                    raise EmptyJudgeResponse(
                        f"Empty judge content (finish_reason={finish_reason!r}, model={self.model!r})"
                    )
                self.stats.success += 1
                if response.usage is not None:
                    self.stats.total_tokens_in += response.usage.prompt_tokens or 0
                    self.stats.total_tokens_out += response.usage.completion_tokens or 0
            except Exception:
                self.stats.failure += 1
                raise
            finally:
                self.stats.total_latency_s += time.monotonic() - t0

            scores = _parse_pointwise_scores(content)
            reward = _pointwise_reward(scores)

            payload: dict = {
                "judge/pointwise/n_criteria_parsed": len(scores),
                "judge/pointwise/reward": reward,
                "judge/pointwise/parse_failed": int(len(scores) == 0),
            }
            # Per-criterion scores so the W&B run shows the full 22-criterion breakdown.
            for crit, s in scores.items():
                payload[f"judge/pointwise/criterion/{crit}"] = float(s)
            _wandb_log_safe(payload)
            tracelog.add_pointwise_judge(
                judge_model=self.model,
                reward=reward,
                n_criteria_parsed=len(scores),
                scores_json=json.dumps(scores, ensure_ascii=False),
                judge_prompt=user_prompt,
                judge_response=content,
            )
            log_agent_run(
                messages + [{"role": "assistant", "content": content}],
                {
                    "type": "pointwise_judge",
                    "model": self.model,
                    "scores": scores,
                    "n_criteria_parsed": len(scores),
                    "reward": reward,
                },
            )
            return reward

    @trace
    async def __call__(
        self,
        writing_prompt: str,
        completions: list[str],
        constraints: list[str] | None = None,
    ) -> list[float]:
        async def _wrapped(idx: int, comp: str) -> tuple[int, float]:
            return idx, await self._judge_one(writing_prompt, comp, constraints)

        tasks = [asyncio.create_task(_wrapped(i, c)) for i, c in enumerate(completions)]
        results: list[float] = [0.0] * len(completions)
        for fut in asyncio.as_completed(tasks):
            idx, result = await fut
            results[idx] = result
        return results


class CreativeBenchPairwiseJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 120.0,
        concurrency: int = 128,
    ):
        self.client = client
        self.model = model
        self.judge_sampling_args = judge_sampling_args or {}
        self.request_timeout_s = request_timeout_s
        self._global_request_sem = asyncio.Semaphore(concurrency)
        self.stats = _JudgeStats()

    @property
    def call_count(self) -> int:
        return self.stats.success + self.stats.failure

    @trace
    @_pairwise_retry
    async def _judge_one(self, sample: PairwiseComparison) -> float:
        async with self._global_request_sem:
            user_prompt = PAIRWISE_TEMPLATE.render(
                prompt=sample["prompt"],
                completion1=sample["completion1"],
                completion2=sample["completion2"],
                constraints=sample.get("constraints", []) if isinstance(sample, dict) else [],
            )
            messages = [{"role": "user", "content": user_prompt}]
            t0 = time.monotonic()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=self.request_timeout_s,
                    stream=False,
                    **self.judge_sampling_args,
                )
                content = response.choices[0].message.content
                if not content:
                    finish_reason = response.choices[0].finish_reason
                    raise EmptyJudgeResponse(
                        f"Empty judge content (finish_reason={finish_reason!r}, model={self.model!r})"
                    )
                self.stats.success += 1
                if response.usage is not None:
                    self.stats.total_tokens_in += response.usage.prompt_tokens or 0
                    self.stats.total_tokens_out += response.usage.completion_tokens or 0
            except Exception:
                self.stats.failure += 1
                raise
            finally:
                self.stats.total_latency_s += time.monotonic() - t0

            prob = _parse_pairwise_response(content)

            _wandb_log_safe({"judge/pairwise/prob_c1_wins": prob})
            tracelog.add_pairwise_judge(
                judge_model=self.model,
                prob_completion1_wins=prob,
                prompt=sample["prompt"],
                completion1=sample["completion1"],
                completion2=sample["completion2"],
                judge_response=content,
            )
            log_agent_run(
                messages + [{"role": "assistant", "content": content}],
                {
                    "type": "pairwise_judge",
                    "model": self.model,
                    "prob_completion1_wins": prob,
                },
            )
            return prob

    @trace
    async def __call__(self, samples: list[PairwiseComparison]) -> list[float]:
        async def _wrapped(idx: int, s: PairwiseComparison) -> tuple[int, float]:
            return idx, await self._judge_one(s)

        tasks = [asyncio.create_task(_wrapped(i, s)) for i, s in enumerate(samples)]
        results: list[float] = [0.0] * len(samples)
        for fut in asyncio.as_completed(tasks):
            idx, result = await fut
            results[idx] = result
        return results
