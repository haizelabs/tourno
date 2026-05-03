import asyncio
import logging
import random
import re
from typing import Any, Callable

import tinker
from openai import AsyncOpenAI
from paths import META_PAIRWISE_PROMPT_PATH, META_POINTWISE_PROMPT_PATH
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tinker_cookbook.renderers import Renderer

from tourno.eval.judges import PairwiseJudge, PointwiseJudge
from tourno.logger import get_logger

_log = get_logger("judges")


def render_responses_section(response: str) -> str:
    return response


_PAIR_BRACKET_RE = re.compile(r"\[\[(.+?)\]\]", re.DOTALL)
_PAIR_SINGLE_BRACKET_RE = re.compile(r"\[([^\[\]]+?)\]", re.DOTALL)
_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_SCORE_HEADER_RE = re.compile(r"#\s*SCORE[S]?\b", re.IGNORECASE)


def _parse_pairwise_verdict(content: str) -> float:
    """Parse meta-pairwise output of the form `# SCORES\\n[[score_a, score_b]]`
    (each 0-100) and return P(B wins) ∈ [0, 1].

    Anchors on the last `# SCORES` header (or `# SCORE` for tolerance), then
    extracts the LAST [[...]] block in the suffix and takes the first two numbers
    inside it. Permissive: handles "[[80, 60]]", "[[score_a: 80, score_b: 60]]",
    "[[A=85, B=92]]". Falls back to last-[[...]]-anywhere if no header is found.

    Linear mapping: a 100-point gap is a full win, equal scores are a tie (0.5).
    Tourno convention: 0 = A wins (so a_won = 1.0), 1 = B wins (a_won = 0.0).
    """
    headers = list(_SCORE_HEADER_RE.finditer(content))
    suffix = content[headers[-1].end():] if headers else content
    matches = _PAIR_BRACKET_RE.findall(suffix)
    if not matches:
        # Sonnet often drops to single brackets ("# SCORES\n[80, 60]").
        matches = _PAIR_SINGLE_BRACKET_RE.findall(suffix)

    score_a: float | None = None
    score_b: float | None = None
    if matches:
        nums = _NUMBER_RE.findall(matches[-1])
        if len(nums) >= 2:
            score_a, score_b = float(nums[0]), float(nums[1])

    if score_a is None and headers:
        # Last fallback: bare numbers after the # SCORES header ("# SCORES\n78, 72").
        nums = _NUMBER_RE.findall(suffix)
        if len(nums) >= 2:
            score_a, score_b = float(nums[0]), float(nums[1])

    if score_a is None:
        # Pre-header fallback: any [[...]] or [...] anywhere in content.
        matches = _PAIR_BRACKET_RE.findall(content) or _PAIR_SINGLE_BRACKET_RE.findall(content)
        if matches:
            nums = _NUMBER_RE.findall(matches[-1])
            if len(nums) >= 2:
                score_a, score_b = float(nums[0]), float(nums[1])

    if score_a is None:
        raise ValueError(f"No pair-of-scores verdict in: {content[-300:]!r}")
    if not (0.0 <= score_a <= 100.0 and 0.0 <= score_b <= 100.0):
        raise ValueError(
            f"Pairwise scores out of [0, 100]: a={score_a} b={score_b}"
        )
    prob_b_wins = 0.5 + 0.005 * (score_b - score_a)
    return max(0.0, min(1.0, prob_b_wins))


class MetaPointwiseJudge(PointwiseJudge):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_sampling_kwargs: dict | None = None,
        request_timeout_s: float = 120.0,
        max_concurrency: int = 128,
    ):
        super().__init__(
            client=client,
            model=model,
            judge_template=META_POINTWISE_PROMPT_PATH.read_text(),
            judge_sampling_kwargs=judge_sampling_kwargs,
            request_timeout_s=request_timeout_s,
            max_concurrency=max_concurrency,
        )

    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        # Anchor on `# SCORE` header so explanation-prose numbers don't fool us.
        headers = list(_SCORE_HEADER_RE.finditer(content))
        if headers:
            suffix = content[headers[-1].end():]
            nums = _NUMBER_RE.findall(suffix)
            if nums:
                score = float(nums[0])
                if 0.0 <= score <= 100.0:
                    return score
        # Fallback: parent's "extract last number" behavior
        score = super().parse_output(inputs, content, reasoning)
        if 0.0 <= score <= 100.0:
            return score
        raise ValueError(f"Expected meta pointwise score in [0, 100], got {score}")


class MetaPairwiseJudge(PairwiseJudge):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        judge_sampling_kwargs: dict | None = None,
        request_timeout_s: float = 120.0,
        max_concurrency: int = 128,
    ):
        super().__init__(
            client=client,
            model=model,
            judge_template=META_PAIRWISE_PROMPT_PATH.read_text(),
            judge_sampling_kwargs=judge_sampling_kwargs,
            request_timeout_s=request_timeout_s,
            max_concurrency=max_concurrency,
        )

    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        return _parse_pairwise_verdict(content)


# --- Self-play meta-judge: uses the LIVE training policy as the meta-judge ---


def _parse_pointwise_verdict(content: str) -> float:
    """Same anchor-on-`# SCORE` parser as MetaPointwiseJudge, factored out
    so the self-play judge can reuse it without inheriting from PointwiseJudge.
    """
    headers = list(_SCORE_HEADER_RE.finditer(content))
    if headers:
        suffix = content[headers[-1].end():]
        nums = _NUMBER_RE.findall(suffix)
        if nums:
            score = float(nums[0])
            if 0.0 <= score <= 100.0:
                return score
    # Fallback: last number anywhere in the text, if it's in range.
    nums = _NUMBER_RE.findall(content)
    if nums:
        score = float(nums[-1])
        if 0.0 <= score <= 100.0:
            return score
    raise ValueError(f"Could not parse meta pointwise score from: {content[-300:]!r}")


def _self_play_stop_token_ids(renderer: Renderer) -> list[int]:
    ids: set[int] = set()
    for stop in renderer.get_stop_sequences():
        if isinstance(stop, int):
            ids.add(stop)
        elif isinstance(stop, str):
            encoded = renderer.tokenizer.encode(stop)
            if encoded:
                ids.add(encoded[-1])
    return sorted(ids)


_SELF_PLAY_RETRY_KWARGS = dict(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(_log, logging.WARNING),
)


class _SelfPlayJudgeBase:
    """Shared plumbing for self-play meta-judges. The judge is the LIVE training
    policy: every call resolves the latest sampling client via the holder lambda,
    so as the policy improves, the meta-judge improves with it.
    """

    def __init__(
        self,
        get_sampling_client: Callable[[], tinker.SamplingClient],
        renderer: Renderer,
        template: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_concurrency: int = 64,
    ):
        self._get_client = get_sampling_client
        self._renderer = renderer
        self._template = template
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._stop_ids = _self_play_stop_token_ids(renderer)
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _generate(self, **template_kwargs: Any) -> str:
        text = self._template.format(**template_kwargs)
        obs = self._renderer.build_generation_prompt([{"role": "user", "content": text}])
        async with self._sem:
            result = await self._get_client().sample_async(
                prompt=obs,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    stop=self._stop_ids,
                ),
            )
        seq = result.sequences[0]
        tokens = list(seq.tokens)
        if tokens and tokens[-1] in self._stop_ids:
            tokens = tokens[:-1]
        return self._renderer.tokenizer.decode(tokens, skip_special_tokens=True)


class SelfPlayMetaPointwiseJudge(_SelfPlayJudgeBase):
    def __init__(
        self,
        get_sampling_client: Callable[[], tinker.SamplingClient],
        renderer: Renderer,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_concurrency: int = 64,
    ):
        super().__init__(
            get_sampling_client=get_sampling_client,
            renderer=renderer,
            template=META_POINTWISE_PROMPT_PATH.read_text(),
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrency=max_concurrency,
        )

    @retry(**_SELF_PLAY_RETRY_KWARGS, retry_error_callback=lambda _: 50.0)
    async def __call__(
        self, *, prompt: str, completion: str, **template_kwargs: Any
    ) -> float:
        content = await self._generate(
            prompt=prompt, completion=completion, **template_kwargs
        )
        return _parse_pointwise_verdict(content)


class SelfPlayMetaPairwiseJudge(_SelfPlayJudgeBase):
    def __init__(
        self,
        get_sampling_client: Callable[[], tinker.SamplingClient],
        renderer: Renderer,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_concurrency: int = 64,
    ):
        super().__init__(
            get_sampling_client=get_sampling_client,
            renderer=renderer,
            template=META_PAIRWISE_PROMPT_PATH.read_text(),
            max_tokens=max_tokens,
            temperature=temperature,
            max_concurrency=max_concurrency,
        )

    @retry(
        **_SELF_PLAY_RETRY_KWARGS,
        retry_error_callback=lambda _: float(random.randint(0, 1)),
    )
    async def __call__(
        self,
        *,
        prompt: str,
        completion1: str,
        completion2: str,
        **template_kwargs: Any,
    ) -> float:
        content = await self._generate(
            prompt=prompt,
            completion1=completion1,
            completion2=completion2,
            **template_kwargs,
        )
        return _parse_pairwise_verdict(content)
