import asyncio
import logging
import random
import re
from pathlib import Path

from healthbench_types import Rubric
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pioneer.logger import get_logger, trace
from tourno.types import PairwiseComparison

DEFAULT_POINTWISE_PROMPT = Path("./prompts/healthbench_pointwise_judge.txt").read_text()
DEFAULT_PAIRWISE_PROMPT = Path("./prompts/healthbench_pairwise_judge.txt").read_text()

_log = get_logger("judges")
_RETRY_KWARGS = dict(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_log, logging.WARNING),
)
_pointwise_retry = retry(**_RETRY_KWARGS, retry_error_callback=lambda _: 0)
_pairwise_retry = retry(**_RETRY_KWARGS, retry_error_callback=lambda _: float(random.randint(0, 1)))


def serialize_rubric(rubric: list[Rubric]) -> str:
    rules: list[str] = []
    for i, r in enumerate(rubric):
        rules.append(f"{i+1}. {r.criterion} Weight: {r.points}")

    return "\n".join(rules)


class HealthBenchPointwiseJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_prompt: str = DEFAULT_POINTWISE_PROMPT,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 45.0,
    ):
        self.client = client
        self.model = model
        self.judge_prompt = judge_prompt or DEFAULT_POINTWISE_PROMPT
        self.judge_sampling_args = judge_sampling_args or {}
        self.request_timeout_s = request_timeout_s

        self._global_request_sem = asyncio.Semaphore(128)
        self.call_count = 0

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

    @trace
    @_pointwise_retry
    async def _judge_one(
        self,
        prompt: str,
        completion: str,
        rubric: str,
    ) -> float:
        async with self._global_request_sem:
            user_prompt = self.judge_prompt.format(
                prompt=prompt,
                rubric=rubric,
                completion=completion,
            )
            messages = [{"role": "user", "content": user_prompt}]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # max_tokens=5,
                # temperature=0,
                timeout=self.request_timeout_s,
                stream=False,
                **self.judge_sampling_args,
            )

            content = response.choices[0].message.content
            assert content is not None
            self.call_count += 1
            return self._parse_score(content)

    @trace
    async def __call__(
        self,
        prompt: str,
        completions: list[str],
        rubric: list[Rubric],
    ) -> list[float]:
        rubric_str = serialize_rubric(rubric)

        async def _wrapped(idx: int, completion: str) -> tuple[int, float]:
            return idx, await self._judge_one(prompt, completion, rubric_str)

        tasks = [asyncio.create_task(_wrapped(idx, comp)) for idx, comp in enumerate(completions)]
        results: list[float] = [0.0] * len(completions)

        for future in asyncio.as_completed(tasks):
            idx, result = await future
            results[idx] = result

        return results


class HealthBenchPairwiseJudge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_prompt: str = DEFAULT_PAIRWISE_PROMPT,
        judge_sampling_args: dict | None = None,
        request_timeout_s: float = 45.0,
    ):
        self.client = client
        self.model = model
        self.judge_prompt = judge_prompt or DEFAULT_PAIRWISE_PROMPT
        self.judge_sampling_args = judge_sampling_args or {}
        self.request_timeout_s = request_timeout_s

        self._global_request_sem = asyncio.Semaphore(128)
        self.call_count = 0

    def _parse_winner(self, response: str) -> float:
        text = response.strip().lower()

        if text.startswith("0"):
            return 1.0
        if text.startswith("1"):
            return 0.0

        raise ValueError(f"Could not parse winner from: {text!r}")

    @trace
    @_pairwise_retry
    async def _judge_one(
        self,
        sample: PairwiseComparison,
        rubric: str,
    ) -> float:
        async with self._global_request_sem:
            user_prompt = self.judge_prompt.format(
                prompt=sample["prompt"],
                rubric=rubric,
                completion1=sample["completion1"],
                completion2=sample["completion2"],
            )
            messages = [{"role": "user", "content": user_prompt}]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=5,
                temperature=0,
                timeout=self.request_timeout_s,
                stream=False,
                **self.judge_sampling_args,
            )

            content = response.choices[0].message.content
            assert content is not None
            self.call_count += 1
            return self._parse_winner(content)

    @trace
    async def __call__(
        self,
        samples: list[PairwiseComparison],
        rubric: list[Rubric],
    ) -> list[float]:
        rubric_str = serialize_rubric(rubric)

        async def _wrapped(idx: int, sample: PairwiseComparison) -> tuple[int, float]:
            return idx, await self._judge_one(sample, rubric_str)

        tasks = [asyncio.create_task(_wrapped(idx, s)) for idx, s in enumerate(samples)]
        results: list[float] = [0.0] * len(samples)

        for future in asyncio.as_completed(tasks):
            idx, result = await future
            results[idx] = result

        return results


### Qwen3-4B-Instruct-2507 ###
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type mixture --pairwise-alpha 3.0 --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-4B-Instruct-2507 --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type pointwise --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-4B-Instruct-2507 --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type pairwise --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-4B-Instruct-2507 --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final

### Qwen3-8B ###
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type mixture --pairwise-alpha 3.0 --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-8B --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type pointwise --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-8B --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type pairwise --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-8B --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final

### Qwen3-8B-Base ###
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type mixture --pairwise-alpha 3.0 --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-8B-Base --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type pointwise --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-8B-Base --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final
# uv run scripts/healthbench-experiments/train_no_reasoning_grade.py --num-workers 16 --max-tokens 1024 --judge-type pairwise --judge-model gpt-4.1-mini --log-level DEBUG --base-model Qwen/Qwen3-8B-Base --n-steps 400 --wandb-project healthbench-final --log-path ./healthbench-rl-final


### Plotting ###
# uv run scripts/healthbench-experiments/plot_paper_bar_chart.py --judges gpt-4.1-mini --candidate-steps 0 60 120 180 240 300 360 400 --base-model Qwen/Qwen3-4B-Instruct-2507 --output figures/out/Qwen3-4B-mini/bar_chart.pdf --output-dir figures/out/Qwen3-4B-mini/ --eval-judge-model openai/gpt-5.2 --gen-concurrency 1024 --judge-concurrency 512 --healthbench-dir healthbench-rl-special
# uv run scripts/healthbench-experiments/plot_paper_line_chart.py --judge gpt-4.1-mini --steps 0 60 120 180 240 300 360 400 --base-model Qwen/Qwen3-8B --dataset test --output figures/out/Qwen3-8B/line_chart.pdf --output-dir figures/out/Qwen3-8B/ --eval-judge-model openai/gpt-5.2 --max-tokens 1024 --gen-concurrency 1024 --judge-concurrency 512 --healthbench-dir healthbench-rl-final/
