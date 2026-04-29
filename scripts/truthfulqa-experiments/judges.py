import asyncio
import logging
import random
import re
from pathlib import Path
from typing import Callable

from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from truthfulqa_types import TruthfulQASample

from pioneer.logger import current_step, get_logger, log_trace, trace
from tourno.types import PairwiseComparison

_log = get_logger("judges")

DEFAULT_POINTWISE_PROMPT = Path("./prompts/truthfulqa_pointwise_judge.txt").read_text()
DEFAULT_PAIRWISE_PROMPT = Path("./prompts/truthfulqa_pairwise_judge.txt").read_text()


def _retry_with_default(default_factory: Callable[[], float]):
    """Tenacity retry policy: 6 attempts, exp backoff, return ``default_factory()`` on exhaustion."""
    return retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        retry_error_callback=lambda _state: default_factory(),
    )


class _BaseJudge:
    """Shared LLM-call plumbing: rate-limited chat completion + call counter."""

    use_throughput_routing: bool = False

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_prompt: str,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 45.0,
    ):
        self.client = client
        self.model = model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = dict(judge_sampling_args or {})
        self.request_timeout_s = request_timeout_s
        self._is_openrouter = "openrouter" in str(client.base_url or "").lower()
        self._sem = asyncio.Semaphore(64)
        self.call_count = 0

    async def _chat(self, user_prompt: str) -> str:
        kwargs = dict(self.judge_sampling_args)
        if self.use_throughput_routing and self._is_openrouter:
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            extra_body.setdefault("provider", {"sort": "throughput"})
            kwargs["extra_body"] = extra_body
        async with self._sem:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=self.request_timeout_s,
                stream=False,
                **kwargs,
            )
            content = response.choices[0].message.content
            assert content is not None
            self.call_count += 1
            return content


class TruthfulQAPointwiseJudge(_BaseJudge):
    name: str = "pointwise"
    max_score: float = 100.0
    default_judge_prompt: str = DEFAULT_POINTWISE_PROMPT
    default_sampling_args: dict = {}
    # Set to ``None`` in subclasses (e.g. gold judges) to skip trace logging.
    trace_table: str | None = "traces/pointwise_judge"

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_prompt: str | None = None,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 45.0,
    ):
        super().__init__(
            client,
            model,
            judge_prompt or self.default_judge_prompt,
            {**self.default_sampling_args, **(judge_sampling_args or {})},
            request_timeout_s,
        )

    def _parse_score(self, response: str) -> float:
        text = response.strip()
        try:
            return float(text)
        except ValueError:
            pass
        numbers = re.findall(r"[-]?\d+\.?\d*", text)
        if numbers:
            return float(numbers[-1])
        raise ValueError(f"Could not parse score from: {text!r}")

    def normalize(self, raw: float) -> float:
        return max(0.0, min(1.0, raw / self.max_score))

    def _format_user_prompt(self, prompt: str, completion: str, sample: TruthfulQASample) -> str:
        return self.judge_prompt.format(prompt=prompt, completion=completion)

    @trace
    async def __call__(
        self, prompt: str, completions: list[str], sample: TruthfulQASample
    ) -> list[float]:
        @_retry_with_default(lambda: 0.0)
        async def score_one(completion: str) -> float:
            user_prompt = self._format_user_prompt(prompt, completion, sample)
            content = await self._chat(user_prompt)
            score = self._parse_score(content)
            if self.trace_table:
                log_trace(
                    self.trace_table,
                    step=current_step.get(),
                    judge_model=self.model,
                    prompt_id=sample.prompt_id,
                    score=score,
                    judge_prompt=user_prompt,
                    judge_response=content,
                )
            return score

        return list(await asyncio.gather(*(score_one(c) for c in completions)))


class TruthfulQAPairwiseJudge(_BaseJudge):
    trace_table: str | None = "traces/pairwise_judge"

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_prompt: str | None = None,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 45.0,
    ):
        super().__init__(
            client,
            model,
            judge_prompt or DEFAULT_PAIRWISE_PROMPT,
            {"temperature": 0, **(judge_sampling_args or {})},
            request_timeout_s,
        )

    @staticmethod
    def _parse_winner(response: str) -> float:
        text = response.strip().lower()
        if text.startswith("0"):
            return 1.0
        if text.startswith("1"):
            return 0.0
        raise ValueError(f"Could not parse winner from: {text!r}")

    def _format_user_prompt(self, pairwise_sample: PairwiseComparison) -> str:
        return self.judge_prompt.format(
            prompt=pairwise_sample["prompt"],
            completion1=pairwise_sample["completion1"],
            completion2=pairwise_sample["completion2"],
        )

    @trace
    async def __call__(
        self,
        samples: list[PairwiseComparison],
        truthfulqa_sample: TruthfulQASample,
    ) -> list[float]:
        @_retry_with_default(lambda: float(random.randint(0, 1)))
        async def score_one(s: PairwiseComparison) -> float:
            user_prompt = self._format_user_prompt(s)
            content = await self._chat(user_prompt)
            p_a_wins = self._parse_winner(content)
            if self.trace_table:
                log_trace(
                    self.trace_table,
                    step=current_step.get(),
                    judge_model=self.model,
                    prompt_id=truthfulqa_sample.prompt_id,
                    p_a_wins=p_a_wins,
                    completion_a=s["completion1"],
                    completion_b=s["completion2"],
                    judge_prompt=user_prompt,
                    judge_response=content,
                )
            return p_a_wins

        return list(await asyncio.gather(*(score_one(s) for s in samples)))
