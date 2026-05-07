import asyncio
import logging
import random
import re
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tourno.logger import get_logger, trace

_log = get_logger("judges")
# _FINAL_SCORE_RE = re.compile(r"final\s*score[\s:=*]*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
# _FINAL_CHOICE_RE = re.compile(r"final\s*choice[\s:=\-*]*([ab])\b", re.IGNORECASE)
_RETRY_KWARGS = dict(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_log, logging.WARNING),
)
_pointwise_retry = retry(**_RETRY_KWARGS, retry_error_callback=lambda _: 0)
_pairwise_retry = retry(**_RETRY_KWARGS, retry_error_callback=lambda _: float(random.randint(0, 1)))


def _get_msg_key(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)

    value = getattr(message, key, None)
    if value is not None:
        return value

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(key)

    return None


def _extract_reasoning(message: Any) -> str | None:
    for key in ("reasoning", "reasoning_content", "reasoning_details"):
        value = _get_msg_key(message, key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            chunks = [
                item.get("text") or item.get("content") for item in value if isinstance(item, dict)
            ]
            return (
                "\n".join(chunk for chunk in chunks if isinstance(chunk, str) and chunk)
                if chunks
                else None
            )

    return None


class Judge:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_template: str | list[dict],
        judge_sampling_kwargs: dict | None = None,
        request_timeout_s: float = 45.0,
        max_concurrency: int = 128,
    ):
        self.client = client
        self.model = model
        self.judge_template = judge_template
        self.judge_sampling_kwargs = judge_sampling_kwargs or {}
        self.request_timeout_s = request_timeout_s

        self._request_sem = asyncio.Semaphore(max_concurrency)

    async def __call__(self, **kwargs: Any) -> float:
        raise NotImplementedError()

    def construct_input(self, **template_kwargs: Any) -> list[dict]:
        messages: list[dict] = []
        if isinstance(self.judge_template, str):
            messages.append(
                {
                    "role": "user",
                    "content": self.judge_template.format(**template_kwargs),
                }
            )
        elif isinstance(self.judge_template, list):
            for message in self.judge_template:
                messages.append(
                    {
                        **message,
                        "content": message["content"].format(**template_kwargs),
                    }
                )
        else:
            raise ValueError(f"Invalid judge template type: {type(self.judge_template)}")

        return messages

    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        text = content.strip()

        last_line = next(
            (line.strip() for line in reversed(text.splitlines()) if line.strip()),
            "",
        )

        try:
            return float(last_line)
        except ValueError:
            pass

        numbers = re.findall(r"[-]?\d+\.?\d*", last_line)
        if numbers:
            return float(numbers[-1])

        # Fall back to scanning the whole output (lenient: catches malformed CoT
        # outputs that put the score somewhere other than the last line).
        numbers = re.findall(r"[-]?\d+\.?\d*", text)
        if numbers:
            return float(numbers[-1])

        raise ValueError(f"Could not parse score from: {text!r}")


class PointwiseJudge(Judge):
    @trace
    @_pointwise_retry
    async def __call__(self, *, prompt: str, completion: str, **template_kwargs: Any) -> float:
        async with self._request_sem:
            judge_input = self.construct_input(
                prompt=prompt,
                completion=completion,
                **template_kwargs,
            )
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=judge_input,
                timeout=self.request_timeout_s,
                stream=False,
                **self.judge_sampling_kwargs,
            )

        message = response.choices[0].message
        content = message.content
        reasoning = _extract_reasoning(message)
        inputs = {"prompt": prompt, "completion": completion, **template_kwargs}
        assert content is not None

        return self.parse_output(inputs, content, reasoning)


class PairwiseJudge(Judge):
    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        score = super().parse_output(inputs, content, reasoning)
        if score in (0.0, 1.0):
            return score

        raise ValueError(f"Expected pairwise choice 0 or 1, got {score} from: {content!r}")

    @trace
    @_pairwise_retry
    async def __call__(
        self,
        *,
        prompt: str,
        completion1: str,
        completion2: str,
        **template_kwargs: Any,
    ) -> float:
        async with self._request_sem:
            judge_input = self.construct_input(
                prompt=prompt,
                completion1=completion1,
                completion2=completion2,
                **template_kwargs,
            )
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=judge_input,
                timeout=self.request_timeout_s,
                stream=False,
                **self.judge_sampling_kwargs,
            )

        message = response.choices[0].message
        content = message.content
        reasoning = _extract_reasoning(message)
        inputs = {
            "prompt": prompt,
            "completion1": completion1,
            "completion2": completion2,
            **template_kwargs,
        }

        assert content is not None

        return self.parse_output(inputs, content, reasoning)
