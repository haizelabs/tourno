"""Full-text trace logging to W&B.

`pioneer.logger.log_agent_run` writes traces to Docent. For the cwbench paper we
also want every model completion and every judge prompt/response visible inside
the W&B run UI, so figures and qualitative analysis can be done without leaving
W&B. This module maintains accumulating `wandb.Table`s for rollouts and judge
calls, and re-logs them at a configurable cadence.

Memory-bounded: rows above `max_rows` are dropped (FIFO) so a long run cannot
OOM the trainer.
"""

from __future__ import annotations

import os
import threading
from typing import Any

_LOCK = threading.Lock()

# NOTE: wandb stores every logged Table version on disk. With FLUSH_EVERY=8 and a long run we
# saw ~9 GB per run, blowing the local disk. Flushing every ~500 rows + capping the table at
# 1000 rows holds local cache to <1 GB per run for full 138-step training. Tune via env vars.
_FLUSH_EVERY = int(os.environ.get("TOURNO_TRACE_FLUSH_EVERY", "500"))
_MAX_ROWS = int(os.environ.get("TOURNO_TRACE_MAX_ROWS", "1000"))

_ROLLOUT_COLS = [
    "row_id",
    "rollout_id",
    "prompt_id",
    "scenario_id",
    "category",
    "reward",
    "token_len",
    "prompt",
    "completion",
]
_POINTWISE_COLS = [
    "judge_model",
    "reward",
    "n_criteria_parsed",
    "scores_json",
    "judge_prompt",
    "judge_response",
]
_PAIRWISE_COLS = [
    "judge_model",
    "prob_completion1_wins",
    "prompt",
    "completion1",
    "completion2",
    "judge_response",
]

_rollouts: list[list[Any]] = []
_pointwise: list[list[Any]] = []
_pairwise: list[list[Any]] = []
_rollouts_since_flush = 0
_pointwise_since_flush = 0
_pairwise_since_flush = 0


def _flush(key: str, columns: list[str], rows: list[list[Any]]) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        wandb.log({key: wandb.Table(columns=columns, data=list(rows))})
    except Exception:
        pass


def _trim(rows: list[list[Any]]) -> None:
    overflow = len(rows) - _MAX_ROWS
    if overflow > 0:
        del rows[:overflow]


def add_rollout(
    *,
    row_id: int | None,
    rollout_id: str,
    prompt_id: str,
    scenario_id: str,
    category: str,
    reward: float,
    token_len: int,
    prompt: str,
    completion: str,
) -> None:
    global _rollouts_since_flush
    with _LOCK:
        _rollouts.append(
            [
                row_id,
                rollout_id,
                prompt_id,
                scenario_id,
                category,
                float(reward),
                int(token_len),
                prompt,
                completion,
            ]
        )
        _trim(_rollouts)
        _rollouts_since_flush += 1
        if _rollouts_since_flush >= _FLUSH_EVERY:
            _rollouts_since_flush = 0
            _flush("traces/rollouts", _ROLLOUT_COLS, _rollouts)


def add_pointwise_judge(
    *,
    judge_model: str,
    reward: float,
    n_criteria_parsed: int,
    scores_json: str,
    judge_prompt: str,
    judge_response: str,
) -> None:
    global _pointwise_since_flush
    with _LOCK:
        _pointwise.append(
            [
                judge_model,
                float(reward),
                int(n_criteria_parsed),
                scores_json,
                judge_prompt,
                judge_response,
            ]
        )
        _trim(_pointwise)
        _pointwise_since_flush += 1
        if _pointwise_since_flush >= _FLUSH_EVERY:
            _pointwise_since_flush = 0
            _flush("traces/pointwise_judge", _POINTWISE_COLS, _pointwise)


def add_pairwise_judge(
    *,
    judge_model: str,
    prob_completion1_wins: float,
    prompt: str,
    completion1: str,
    completion2: str,
    judge_response: str,
) -> None:
    global _pairwise_since_flush
    with _LOCK:
        _pairwise.append(
            [
                judge_model,
                float(prob_completion1_wins),
                prompt,
                completion1,
                completion2,
                judge_response,
            ]
        )
        _trim(_pairwise)
        _pairwise_since_flush += 1
        if _pairwise_since_flush >= _FLUSH_EVERY:
            _pairwise_since_flush = 0
            _flush("traces/pairwise_judge", _PAIRWISE_COLS, _pairwise)


def flush_all() -> None:
    """Force-flush all tables. Call at end of training to capture trailing rows."""
    with _LOCK:
        _flush("traces/rollouts", _ROLLOUT_COLS, _rollouts)
        _flush("traces/pointwise_judge", _POINTWISE_COLS, _pointwise)
        _flush("traces/pairwise_judge", _PAIRWISE_COLS, _pairwise)
