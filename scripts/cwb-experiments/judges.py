import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Template

from tourno.eval.judges import PairwiseJudge, PointwiseJudge

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
CRITERIA_PATH = PROMPTS_DIR / "cwbench_criteria.txt"
NEGATIVE_CRITERIA_PATH = PROMPTS_DIR / "cwbench_negative_criteria.txt"

ALL_CRITERIA = CRITERIA_PATH.read_text().strip().splitlines()
NEGATIVE_CRITERIA = NEGATIVE_CRITERIA_PATH.read_text().strip().splitlines()
NEGATIVE_SET = set(NEGATIVE_CRITERIA)
POSITIVE_CRITERIA = [c for c in ALL_CRITERIA if c not in NEGATIVE_SET]
SCORE_RANGE_MAX = 20.0

PAIRWISE_CRITERIA = [
    "character_authenticity_insight",
    "interesting_original",
    "writing_quality",
    "coherence",
    "instruction_following",
    "world_and_atmosphere",
    "avoids_cliches",
    "avoids_verbosity",
    "avoids_poetic_overload",
]
RESPONDENT_A = "A0493"
RESPONDENT_B = "A0488"
MARGIN_TABLE = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

_NORMALIZED_TO_CANONICAL = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in ALL_CRITERIA}
_SCORE_LINE_RE = re.compile(
    r"^([^\n:]+?):\s*\[?\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)\s*\]?\s*$",
    re.MULTILINE,
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _canonicalize(metric: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", metric.lower())
    return _NORMALIZED_TO_CANONICAL.get(key)


def _parse_pointwise_scores(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for m in _SCORE_LINE_RE.finditer(text):
        canonical = _canonicalize(m.group(1).strip())
        if canonical is None:
            continue
        val = float(m.group(2))
        if 0.0 <= val <= SCORE_RANGE_MAX:
            scores[canonical] = val
    return scores


def _pointwise_reward(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    adjusted = [(SCORE_RANGE_MAX - s) if m in NEGATIVE_SET else s for m, s in scores.items()]
    return max(0.0, min(1.0, (sum(adjusted) / len(adjusted)) / SCORE_RANGE_MAX))


def _robust_json_loads(text: str) -> dict:
    text = _CONTROL_CHAR_RE.sub(" ", text.strip())
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
    # Cwbench convention: avg=+1 → A0493 (= completion_a) wins fully.
    # Tourno's batched_elo_rewards expects 0=a_wins / 1=b_wins, so flip here.
    return 1.0 - (avg + 1.0) / 2.0


class CreativeBenchPointwiseJudge(PointwiseJudge):
    def __init__(
        self,
        client: Any,
        model: str,
        judge_template: str,
        judge_sampling_kwargs: dict | None = None,
        request_timeout_s: float = 120.0,
        max_concurrency: int = 128,
    ):
        super().__init__(
            client=client,
            model=model,
            judge_template=judge_template,
            judge_sampling_kwargs=judge_sampling_kwargs,
            request_timeout_s=request_timeout_s,
            max_concurrency=max_concurrency,
        )
        self._jinja = Template(judge_template)

    def construct_input(self, **template_kwargs: Any) -> list[dict]:
        return [{"role": "user", "content": self._jinja.render(**template_kwargs)}]

    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        scores = _parse_pointwise_scores(content)
        if not scores:
            raise ValueError(f"No parsable scores in: {content[:200]!r}")
        return _pointwise_reward(scores)


class CreativeBenchPairwiseJudge(PairwiseJudge):
    def __init__(
        self,
        client: Any,
        model: str,
        judge_template: str,
        judge_sampling_kwargs: dict | None = None,
        request_timeout_s: float = 120.0,
        max_concurrency: int = 128,
    ):
        super().__init__(
            client=client,
            model=model,
            judge_template=judge_template,
            judge_sampling_kwargs=judge_sampling_kwargs,
            request_timeout_s=request_timeout_s,
            max_concurrency=max_concurrency,
        )
        self._jinja = Template(judge_template)

    def construct_input(self, **template_kwargs: Any) -> list[dict]:
        rendered = dict(template_kwargs)
        if "completion_a" in rendered:
            rendered["completion1"] = rendered.pop("completion_a")
        if "completion_b" in rendered:
            rendered["completion2"] = rendered.pop("completion_b")
        return [{"role": "user", "content": self._jinja.render(**rendered)}]

    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        return _parse_pairwise_response(content)
