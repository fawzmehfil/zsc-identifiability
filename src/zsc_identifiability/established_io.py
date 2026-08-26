"""Versioned trace interchange helpers shared by the isolated runtimes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from zsc_identifiability.established_models import CommitmentTraceStep


def load_trace_jsonl(path: str | Path) -> tuple[CommitmentTraceStep, ...]:
    source = Path(path)
    result: list[CommitmentTraceStep] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload: Any = json.loads(line)
            try:
                result.append(CommitmentTraceStep.model_validate(payload))
            except ValueError as exc:
                raise ValueError(f"invalid trace at {source}:{line_number}: {exc}") from exc
    _validate_trace_order(result)
    return tuple(result)


def write_trace_jsonl(path: str | Path, steps: Iterable[CommitmentTraceStep]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    items = tuple(steps)
    _validate_trace_order(items)
    payload = "".join(
        json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for item in items
    )
    destination.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def trace_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_trace_order(steps: Iterable[CommitmentTraceStep]) -> None:
    previous: tuple[str, int, int] | None = None
    identifiers: set[tuple[str, int, int]] = set()
    for step in steps:
        key = (step.episode_id, step.work_unit, step.step)
        if key in identifiers:
            raise ValueError(f"duplicate trace record {key!r}")
        identifiers.add(key)
        order_key = key
        if previous is not None and order_key < previous:
            raise ValueError("trace records must be sorted by episode, work unit, and step")
        previous = order_key
