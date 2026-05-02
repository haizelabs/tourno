from typing import Any

from openai import AsyncOpenAI
from paths import META_PAIRWISE_PROMPT_PATH, META_POINTWISE_PROMPT_PATH

from tourno.eval.judges import PairwiseJudge, PointwiseJudge


def render_responses_section(response: str) -> str:
    return f"ASSISTANT RESPONSE:\n{response}"


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
        score = super().parse_output(inputs, content, reasoning)
        if 0.0 <= score <= 10.0:
            return score

        raise ValueError(f"Expected meta pointwise score in [0, 10], got {score}")


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

    async def __call__(
        self,
        *,
        prompt: str,
        completion1: str,
        completion2: str,
        responses_section: str,
        **template_kwargs: Any,
    ) -> float:
        return await super().__call__(
            prompt=prompt,
            completion1=completion1,
            completion2=completion2,
            responses_section=responses_section,
            **template_kwargs,
        )
