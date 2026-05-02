import re

from tourno.eval.judges import PairwiseJudge

RESPONDENT_A = "H0427"
RESPONDENT_B = "H0813"
MARGIN_TABLE = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

_MARGIN_RE = re.compile(rf"({re.escape(RESPONDENT_A)}|{re.escape(RESPONDENT_B)})(\++)?")


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
    last_line = next(
        (line.strip() for line in reversed(text.splitlines()) if line.strip()),
        "",
    )
    try:
        margin = _parse_margin(last_line)
    except ValueError:
        # Fallback: scan whole output for the last respondent code with optional pluses.
        matches = _MARGIN_RE.findall(text)
        if not matches:
            raise ValueError(f"No respondent code in: {text[-300:]!r}") from None
        respondent, pluses = matches[-1]
        margin = _parse_margin(f"{respondent}{pluses}")
    # Convention: margin=+1 -> H0427 (= completion_a) wins fully.
    # Tourno's batched_elo_rewards expects 0=a_wins / 1=b_wins, so flip here.
    return 1.0 - (margin + 1.0) / 2.0


class HealthBenchPairwiseJudge(PairwiseJudge):
    def parse_output(self, inputs: dict, content: str, reasoning: str | None) -> float:
        return _parse_pairwise_response(content)
