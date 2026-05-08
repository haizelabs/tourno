import asyncio
import dataclasses
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm.asyncio import tqdm as atqdm

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


async def _gather_judge(
    judge: Judge,
    inputs_per_row: list[dict],
    *,
    desc: str,
    show_progress: bool,
    position: int = 0,
    leave: bool = True,
) -> list[Any]:
    async def call_judge(inputs: dict) -> float | BaseException:
        try:
            return await judge(**inputs)
        except BaseException as exc:
            return exc

    coros = [call_judge(inputs) for inputs in inputs_per_row]

    return await atqdm.gather(
        *coros,
        desc=desc,
        disable=not show_progress,
        position=position,
        leave=leave,
    )


def _to_jsonable(item: Any) -> Any:
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return asdict(item)

    return item


def _write_jsonl(path: Path, items: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(_to_jsonable(item), default=str) + "\n")


async def evaluate_judge(
    judge: Judge,
    rows: list[dict],
    template_kwargs: dict[str, Any] | None = None,
    *,
    name: str = "judge",
    output_dir: str | Path | None = None,
    show_progress: bool = True,
    position: int = 0,
    leave: bool = True,
) -> list[EvalResult]:
    """Run a single judge over every row, preserving row order."""
    extras = template_kwargs or {}
    inputs_per_row = [{**extras, **row} for row in rows]

    raw = await _gather_judge(
        judge,
        inputs_per_row,
        desc=name,
        show_progress=show_progress,
        position=position,
        leave=leave,
    )

    results: list[EvalResult] = [
        (
            EvalResult(inputs=inputs, error=repr(val))
            if isinstance(val, BaseException)
            else EvalResult(inputs=inputs, output=val)
        )
        for inputs, val in zip(inputs_per_row, raw)
    ]

    if output_dir is not None:
        _write_jsonl(Path(output_dir) / f"{name}.jsonl", results)

    return results


async def sweep_judges(
    judges: dict[str, Judge],
    rows: list[dict],
    aggregator: Callable[[dict[str, Any]], Any] | None = None,
    template_kwargs: dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
    show_progress: bool = True,
    leave: bool = True,
) -> list[SweepResult]:
    """Run multiple judges over every row, preserving row order."""
    extras = template_kwargs or {}
    inputs_per_row = [{**extras, **row} for row in rows]
    judge_names = list(judges)

    per_judge_outputs = await asyncio.gather(
        *[
            _gather_judge(
                judge,
                inputs_per_row,
                desc=name,
                show_progress=show_progress,
                position=idx,
                leave=leave,
            )
            for idx, (name, judge) in enumerate(judges.items())
        ]
    )

    results: list[SweepResult] = []
    for row_idx, inputs in enumerate(inputs_per_row):
        outputs: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for judge_idx, name in enumerate(judge_names):
            val = per_judge_outputs[judge_idx][row_idx]
            if isinstance(val, BaseException):
                errors[name] = repr(val)
            else:
                outputs[name] = val

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

    if output_dir is not None:
        flat_rows = [
            {
                "inputs": r.inputs,
                **{name: r.outputs.get(name) for name in judge_names},
                "errors": r.errors,
                "aggregated": r.aggregated,
                "aggregated_error": r.aggregated_error,
            }
            for r in results
        ]

        _write_jsonl(Path(output_dir) / "sweep.jsonl", flat_rows)

    return results
