import json
import random
import re
from typing import Any

from tourno.eval.judges import PairwiseJudge, PointwiseJudge


class TuluPairwiseJudge(PairwiseJudge):
    @staticmethod
    def _winner_prob(n_plus: int) -> float:
        return 0.5 + 0.1 * min(max(n_plus, 0), 5)

    @staticmethod
    def _new_id(rng: random.Random) -> str:
        return f"A{rng.randint(0, 9999):04d}"

    def _extra_template_kwargs(self) -> dict[str, Any]:
        rng = random.Random()
        id_a = self._new_id(rng)
        id_b = self._new_id(rng)
        while id_b == id_a:
            id_b = self._new_id(rng)

        return {"id_a": id_a, "id_b": id_b}

    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        id_a = inputs["id_a"]
        id_b = inputs["id_b"]

        verdict: str | None = None
        try:
            data = json.loads(content)
            cs = data.get("constraint_satisfaction")
            if isinstance(cs, str):
                verdict = cs
        except (json.JSONDecodeError, TypeError):
            pass

        haystack = verdict if verdict is not None else content
        m_a = re.search(rf"{re.escape(id_a)}\s*(\++)", haystack)
        m_b = re.search(rf"{re.escape(id_b)}\s*(\++)", haystack)

        if m_a and not m_b:
            return self._winner_prob(len(m_a.group(1)))
        if m_b and not m_a:
            return 1.0 - self._winner_prob(len(m_b.group(1)))
        if m_a and m_b:
            last = max(m_a, m_b, key=lambda m: m.start())
            prob_winner = self._winner_prob(len(last.group(1)))
            return prob_winner if last is m_a else 1.0 - prob_winner

        raise ValueError(
            f"Could not parse winner+disparity (expected '{id_a}+...' or '{id_b}+...') "
            f"from: {content!r}"
        )


class TuluPointwiseJudge(PointwiseJudge):
    _SCORE_RE = re.compile(r"score\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        matches = self._SCORE_RE.findall(content)
        if matches:
            return float(matches[-1])

        numbers = re.findall(r"\d+(?:\.\d+)?", content)
        if numbers:
            return float(numbers[-1])

        raise ValueError(f"Could not parse pointwise score from: {content!r}")
