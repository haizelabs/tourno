import asyncio
import json
import logging
import random
import re
from pathlib import Path

from eqbench_types import TaskType
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
POINTWISE_PROMPTS = {
    "standard": (PROMPTS_DIR / "eqbench3_pointwise_standard.txt").read_text(),
    "analysis": (PROMPTS_DIR / "eqbench3_pointwise_analysis.txt").read_text(),
}
PAIRWISE_PROMPTS = {
    "standard": (PROMPTS_DIR / "eqbench3_pairwise_standard.txt").read_text(),
    "analysis": (PROMPTS_DIR / "eqbench3_pairwise_analysis.txt").read_text(),
}

# Scoring criteria kept per task_type; `overall_eq` gets weight 3 for standard tasks
# to mirror EQ-Bench 3's canonical scoring (analysis has no overall_eq, equal weight).
STANDARD_ALLOWED = {
    "demonstrated_empathy",
    "pragmatic_ei",
    "depth_of_insight",
    "social_dexterity",
    "emotional_reasoning",
    "message_tailoring",
    "theory_of_mind",
    "subtext_identification",
    "intellectual_grounding",
    "correctness",
    "overall_eq",
}
ANALYSIS_ALLOWED = {
    "depth_of_insight",
    "emotional_reasoning",
    "theory_of_mind",
    "subtext_identification",
    "intellectual_grounding",
    "correctness",
}
ALLOWED_BY_TYPE = {"standard": STANDARD_ALLOWED, "analysis": ANALYSIS_ALLOWED}

# Margin tokens "A0493+++" where plus-count in [1..5]. 0 plus = tie (which the
# prompt forbids, but we still handle defensively).
MARGIN_TABLE = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
RESPONDENT_A = "A0493"
RESPONDENT_B = "A0488"

_log = get_logger("eqbench_judges")
_RETRY_KWARGS = dict(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_log, logging.WARNING),
)
# Point failures → reward 0 (neutral-bad). Pair failures → coin flip.
_pointwise_retry = retry(**_RETRY_KWARGS, retry_error_callback=lambda _: 0.0)
_pairwise_retry = retry(
    **_RETRY_KWARGS, retry_error_callback=lambda _: float(random.randint(0, 1))
)


def _robust_json_loads(text: str) -> dict:
    """Extract the first balanced JSON object. Tolerates prose prefix/suffix."""
    text = text.strip()
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


def _parse_rubric_scores(text: str, allowed: set[str]) -> dict[str, float]:
    data = _robust_json_loads(text)
    scores: dict[str, float] = {}
    for k, v in data.items():
        if k not in allowed:
            continue
        if isinstance(v, (int, float)):
            scores[k] = float(v)
        else:
            m = re.search(r"\b(\d+(?:\.\d+)?)\b", str(v))
            if m:
                scores[k] = float(m.group(1))
    return scores


def _weighted_mean_0_to_20(scores: dict[str, float], task_type: TaskType) -> float:
    if not scores:
        return 0.0
    values: list[float] = []
    for k, s in scores.items():
        weight = 3 if (task_type == "standard" and k == "overall_eq") else 1
        values.extend([s] * weight)
    return sum(values) / len(values)


def _parse_margin(text: str) -> float:
    """Map 'A0493+++' → +0.6, 'A0488+' → -0.2. Signed toward completion1 (A0493)."""
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


def _parse_pairwise_response(text: str, criteria: list[str]) -> float:
    """Return P(completion1 > completion2) ∈ [0,1] by averaging signed margins."""
    data = _robust_json_loads(text)
    margins: list[float] = []
    for c in criteria:
        v = data.get(c)
        if isinstance(v, str):
            try:
                margins.append(_parse_margin(v))
            except ValueError:
                continue
    if not margins:
        raise ValueError(f"No parsable criteria in: {text[:200]!r}")
    avg = sum(margins) / len(margins)  # in [-1, +1]
    return (avg + 1.0) / 2.0  # in [0, 1]


# Criterion keys expected in pairwise judge JSON (excluding chain_of_thought_reasoning).
PAIRWISE_CRITERIA = {
    "standard": [
        "demonstrated_empathy",
        "pragmatic_ei",
        "depth_of_insight",
        "social_dexterity",
        "emotional_reasoning",
        "appropriate_validating_challenging",
        "message_tailoring",
        "overall_eq",
    ],
    "analysis": [
        "depth_of_insight",
        "authentic_eu",
        "causal_attribution",
        "theory_of_mind",
        "incisiveness",
        "reading_between_lines",
        "correctness",
        "overall_eq",
    ],
}


class EQBenchPointwiseJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 60.0,
    ):
        self.client = client
        self.model = model
        self.judge_sampling_args = judge_sampling_args or {}
        self.request_timeout_s = request_timeout_s

        self._global_request_sem = asyncio.Semaphore(128)
        self.call_count = 0

    @trace
    @_pointwise_retry
    async def _judge_one(
        self,
        scenario_text: str,
        completion: str,
        task_type: TaskType,
    ) -> float:
        async with self._global_request_sem:
            user_prompt = POINTWISE_PROMPTS[task_type].format(
                prompt=scenario_text, completion=completion
            )
            messages = [{"role": "user", "content": user_prompt}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=self.request_timeout_s,
                stream=False,
                **self.judge_sampling_args,
            )
            content = response.choices[0].message.content
            assert content is not None
            self.call_count += 1

            scores = _parse_rubric_scores(content, ALLOWED_BY_TYPE[task_type])
            raw = _weighted_mean_0_to_20(scores, task_type)
            normalized = max(0.0, min(1.0, raw / 20.0))

            log_agent_run(
                messages + [{"role": "assistant", "content": content}],
                {
                    "type": "pointwise_judge",
                    "model": self.model,
                    "task_type": task_type,
                    "scores": scores,
                    "raw": raw,
                    "normalized": normalized,
                },
            )
            return normalized

    @trace
    async def __call__(
        self,
        scenario_text: str,
        completions: list[str],
        task_type: TaskType,
    ) -> list[float]:
        async def _wrapped(idx: int, comp: str) -> tuple[int, float]:
            return idx, await self._judge_one(scenario_text, comp, task_type)

        tasks = [asyncio.create_task(_wrapped(i, c)) for i, c in enumerate(completions)]
        results: list[float] = [0.0] * len(completions)
        for fut in asyncio.as_completed(tasks):
            idx, result = await fut
            results[idx] = result
        return results


class EQBenchPairwiseJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 60.0,
    ):
        self.client = client
        self.model = model
        self.judge_sampling_args = judge_sampling_args or {}
        self.request_timeout_s = request_timeout_s

        self._global_request_sem = asyncio.Semaphore(128)
        self.call_count = 0

    @trace
    @_pairwise_retry
    async def _judge_one(
        self,
        sample: PairwiseComparison,
        task_type: TaskType,
    ) -> float:
        async with self._global_request_sem:
            user_prompt = PAIRWISE_PROMPTS[task_type].format(
                prompt=sample["prompt"],
                completion1=sample["completion1"],
                completion2=sample["completion2"],
            )
            messages = [{"role": "user", "content": user_prompt}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=self.request_timeout_s,
                stream=False,
                **self.judge_sampling_args,
            )
            content = response.choices[0].message.content
            assert content is not None
            self.call_count += 1

            prob = _parse_pairwise_response(content, PAIRWISE_CRITERIA[task_type])

            log_agent_run(
                messages + [{"role": "assistant", "content": content}],
                {
                    "type": "pairwise_judge",
                    "model": self.model,
                    "task_type": task_type,
                    "prob_completion1_wins": prob,
                },
            )
            return prob

    @trace
    async def __call__(
        self,
        samples: list[PairwiseComparison],
        task_type: TaskType,
    ) -> list[float]:
        async def _wrapped(idx: int, s: PairwiseComparison) -> tuple[int, float]:
            return idx, await self._judge_one(s, task_type)

        tasks = [asyncio.create_task(_wrapped(i, s)) for i, s in enumerate(samples)]
        results: list[float] = [0.0] * len(samples)
        for fut in asyncio.as_completed(tasks):
            idx, result = await fut
            results[idx] = result
        return results
