"""Mode-conditioned ego-visible prefix divergence for established traces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from zsc_identifiability.established_commitment import (
    WorkUnitHistory,
    extract_work_units,
    prefix_history,
)
from zsc_identifiability.established_io import load_trace_jsonl
from zsc_identifiability.established_models import CommitmentTraceStep


def estimate_prefix_tv_curves(
    trace_path: str | Path,
    *,
    prefixes: Sequence[int | str] = (0, 8, 16, 32, "pre_commitment"),
    response_signatures: Mapping[str, Hashable] | None = None,
) -> dict[str, object]:
    histories = tuple(
        item
        for item in extract_work_units(load_trace_jsonl(trace_path))
        if item.work_unit == 0
    )
    by_partner: dict[str, list[WorkUnitHistory]] = {}
    for history in histories:
        by_partner.setdefault(history.partner_id, []).append(history)
    if len(by_partner) < 2:
        raise ValueError("prefix-TV audit requires at least two partner modes")
    if response_signatures is not None and set(response_signatures) != set(by_partner):
        raise ValueError("response signatures must cover exactly the traced partners")
    pair_curves: list[dict[str, Any]] = []
    partner_totals: dict[str, list[float]] = {partner: [] for partner in by_partner}
    for left, right in combinations(sorted(by_partner), 2):
        values: dict[str, float] = {}
        for prefix in prefixes:
            left_distribution = _distribution(by_partner[left], prefix)
            right_distribution = _distribution(by_partner[right], prefix)
            tv = _total_variation(left_distribution, right_distribution)
            values[str(prefix)] = tv
        conflicting = (
            None
            if response_signatures is None
            else response_signatures[left] != response_signatures[right]
        )
        pair_curves.append(
            {
                "left_partner_id": left,
                "right_partner_id": right,
                "response_conflicting": conflicting,
                "prefix_tv": values,
            }
        )
        partner_totals[left].append(values["pre_commitment"])
        partner_totals[right].append(values["pre_commitment"])
    prefix_means = {
        str(prefix): sum(float(row["prefix_tv"][str(prefix)]) for row in pair_curves)
        / len(pair_curves)
        for prefix in prefixes
    }
    conflicting_rows = [row for row in pair_curves if row["response_conflicting"] is True]
    conflicting_means = {
        str(prefix): (
            sum(float(row["prefix_tv"][str(prefix)]) for row in conflicting_rows)
            / len(conflicting_rows)
            if conflicting_rows
            else None
        )
        for prefix in prefixes
    }
    return {
        "estimator": "binned_visible_prefix_tv",
        "partner_count": len(by_partner),
        "episode_count": len(histories),
        "pair_curves": pair_curves,
        "mean_prefix_tv": prefix_means,
        "mean_conflicting_prefix_tv": conflicting_means,
        "partner_precommitment_tv": {
            partner: sum(values) / len(values) for partner, values in partner_totals.items()
        },
    }


def _distribution(
    histories: Sequence[WorkUnitHistory], prefix: int | str
) -> dict[tuple[str, ...], float]:
    counts = Counter(_signature(prefix_history(history, prefix)) for history in histories)
    total = sum(counts.values())
    return {signature: count / total for signature, count in counts.items()}


def _signature(steps: Sequence[CommitmentTraceStep]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for step in steps:
        counts[f"ego_action:{step.ego_action}"] += 1
        if step.visible_partner_action is not None:
            counts[f"partner_action:{step.visible_partner_action}"] += 1
        counts.update(step.high_level_events)
        reward_token = (
            "reward:positive"
            if step.reward > 0
            else "reward:negative"
            if step.reward < 0
            else "reward:zero"
        )
        counts[reward_token] += 1
    result = [f"length:{_bin(len(steps))}"]
    result.extend(f"{token}:{_bin(count)}" for token, count in sorted(counts.items()))
    return tuple(result)


def _bin(value: int) -> str:
    if value <= 2:
        return str(value)
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    if value <= 16:
        return "9-16"
    return "17+"


def _total_variation(
    left: dict[tuple[str, ...], float], right: dict[tuple[str, ...], float]
) -> float:
    support = set(left) | set(right)
    return 0.5 * sum(abs(left.get(item, 0.0) - right.get(item, 0.0)) for item in support)
