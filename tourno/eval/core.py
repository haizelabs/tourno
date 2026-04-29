import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tourno.eval.judges import Judge


@dataclass
class EvalResult:
    """Result of evaluating one row with a single judge."""

    inputs: dict
    output: Any | None = None
    error: str | None = None


@dataclass
class SweepResult:
    """Result of evaluating one row with multiple judges."""

    inputs: dict
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    aggregated: Any | None = None
    aggregated_error: str | None = None


async def _call_judge(judge: Judge, inputs: dict) -> Any:
    return await judge(**inputs)


async def evaluate_judge(
    judge: Judge,
    rows: list[dict],
    template_kwargs: dict[str, Any] | None = None,
) -> list[EvalResult]:
    """Run a single judge over every row, preserving row order."""
    extras = template_kwargs or {}
    inputs_per_row = [{**extras, **row} for row in rows]

    raw_outputs = await asyncio.gather(
        *[_call_judge(judge, inputs) for inputs in inputs_per_row],
        return_exceptions=True,
    )

    results: list[EvalResult] = []
    for inputs, output in zip(inputs_per_row, raw_outputs):
        if isinstance(output, BaseException):
            results.append(EvalResult(inputs=inputs, error=repr(output)))
        else:
            results.append(EvalResult(inputs=inputs, output=output))

    return results


async def sweep_judges(
    judges: dict[str, Judge],
    rows: list[dict],
    aggregator: Callable[[dict[str, Any]], Any] | None = None,
    template_kwargs: dict[str, Any] | None = None,
) -> list[SweepResult]:
    """Run multiple judges over every row, preserving row order."""
    extras = template_kwargs or {}
    inputs_per_row = [{**extras, **row} for row in rows]
    judge_names = list(judges)

    per_judge_outputs = await asyncio.gather(
        *[
            asyncio.gather(
                *[_call_judge(judge, inputs) for inputs in inputs_per_row],
                return_exceptions=True,
            )
            for judge in judges.values()
        ]
    )

    results: list[SweepResult] = []
    for row_idx, inputs in enumerate(inputs_per_row):
        outputs: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for judge_idx, name in enumerate(judge_names):
            output = per_judge_outputs[judge_idx][row_idx]
            if isinstance(output, BaseException):
                errors[name] = repr(output)
            else:
                outputs[name] = output

        result = SweepResult(inputs=inputs, outputs=outputs, errors=errors)
        if aggregator is not None:
            if errors:
                result.aggregated_error = "skipped: judge errors"
            else:
                try:
                    result.aggregated = aggregator(outputs)
                except BaseException as exc:
                    result.aggregated_error = repr(exc)

        results.append(result)

    return results
