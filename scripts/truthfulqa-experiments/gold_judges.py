import re
from pathlib import Path

from judges import TruthfulQAPointwiseJudge
from openai import AsyncOpenAI
from truthfulqa_types import TruthfulQASample


def serialize_reference_answers(sample: TruthfulQASample) -> tuple[str, str]:
    acceptable = "\n".join(f"- {a}" for a in sample.correct_answers)
    unacceptable = "\n".join(f"- {a}" for a in sample.incorrect_answers)
    return acceptable, unacceptable


DEFAULT_STRICT_PROMPT = Path("./prompts/truthfulqa_strict_judge.txt").read_text()
DEFAULT_LENIENT_PROMPT = Path("./prompts/truthfulqa_lenient_judge.txt").read_text()


class _ReferenceAwareJudge(TruthfulQAPointwiseJudge):
    trace_table = None

    def _format_user_prompt(self, prompt: str, completion: str, sample: TruthfulQASample) -> str:
        acceptable, unacceptable = serialize_reference_answers(sample)
        return self.judge_prompt.format(
            prompt=prompt,
            completion=completion,
            acceptable_answers=acceptable,
            unacceptable_answers=unacceptable,
        )


class TruthfulQAStrictJudge(_ReferenceAwareJudge):
    name = "strict"
    max_score = 1.0
    default_judge_prompt = DEFAULT_STRICT_PROMPT
    default_sampling_args = {"max_tokens": 128, "temperature": 0}

    def _parse_score(self, response: str) -> float:
        text = response.strip().lower()
        if text.startswith("1"):
            return 1.0
        if text.startswith("0"):
            return 0.0

        numbers = re.findall(r"[01]", text)
        if numbers:
            return float(numbers[0])

        raise ValueError(f"Could not parse strict label from: {response!r}")


class TruthfulQALenientJudge(_ReferenceAwareJudge):
    name = "lenient"
    max_score = 100.0
    default_judge_prompt = DEFAULT_LENIENT_PROMPT


GoldJudge = TruthfulQAStrictJudge | TruthfulQALenientJudge


def get_gold_judge(
    kind: str,
    client: AsyncOpenAI,
    model: str,
    **kwargs,
) -> GoldJudge:
    if kind == "strict":
        return TruthfulQAStrictJudge(client=client, model=model, **kwargs)
    if kind == "lenient":
        return TruthfulQALenientJudge(client=client, model=model, **kwargs)

    raise ValueError(f"Unknown gold judge kind: {kind!r} (expected 'strict' or 'lenient')")
