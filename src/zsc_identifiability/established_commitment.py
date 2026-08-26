"""Commitment-boundary extraction for versioned OvercookedV2 traces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from zsc_identifiability.established_models import CommitmentRuleSpec, CommitmentTraceStep


@dataclass(frozen=True)
class WorkUnitHistory:
    episode_id: str
    partner_id: str
    work_unit: int
    pre_commitment: tuple[CommitmentTraceStep, ...]
    commitment_step: CommitmentTraceStep | None
    eventual: tuple[CommitmentTraceStep, ...]
    commitment_reached: bool
    delivery_reached: bool


def is_commitment_step(step: CommitmentTraceStep, rule: CommitmentRuleSpec) -> bool:
    """Return true only for the declared successful irreversible task event."""

    return any(event.name == rule.event_name and event.success for event in step.events)


def is_successful_delivery(step: CommitmentTraceStep) -> bool:
    return any(event.name == "successful_delivery" and event.success for event in step.events)


def extract_work_units(
    steps: Iterable[CommitmentTraceStep],
    rule: CommitmentRuleSpec | None = None,
) -> tuple[WorkUnitHistory, ...]:
    """Extract histories without allowing commitment or delivery leakage.

    `pre_commitment` ends immediately before the first pot modification.
    `eventual` includes the first delivery event and no later evidence.
    """

    rule = rule or CommitmentRuleSpec()
    grouped: dict[tuple[str, str, int], list[CommitmentTraceStep]] = {}
    for step in steps:
        grouped.setdefault((step.episode_id, step.partner_id, step.work_unit), []).append(step)
    histories: list[WorkUnitHistory] = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda item: item.step)
        if len({item.step for item in ordered}) != len(ordered):
            raise ValueError(f"duplicate trace step in work unit {key!r}")
        commitment_index = next(
            (index for index, item in enumerate(ordered) if is_commitment_step(item, rule)),
            None,
        )
        delivery_index = next(
            (index for index, item in enumerate(ordered) if is_successful_delivery(item)),
            None,
        )
        if (
            commitment_index is not None
            and delivery_index is not None
            and delivery_index < commitment_index
        ):
            raise ValueError("delivery cannot precede the first commitment in a work unit")
        pre = tuple(ordered if commitment_index is None else ordered[:commitment_index])
        eventual_end = len(ordered) if delivery_index is None else delivery_index + 1
        eventual = tuple(ordered[:eventual_end])
        histories.append(
            WorkUnitHistory(
                episode_id=key[0],
                partner_id=key[1],
                work_unit=key[2],
                pre_commitment=pre,
                commitment_step=(
                    None if commitment_index is None else ordered[commitment_index]
                ),
                eventual=eventual,
                commitment_reached=commitment_index is not None,
                delivery_reached=delivery_index is not None,
            )
        )
    return tuple(histories)


def prefix_history(
    history: WorkUnitHistory,
    prefix: int | str,
) -> tuple[CommitmentTraceStep, ...]:
    if isinstance(prefix, int):
        if prefix < 0:
            raise ValueError("prefix step cannot be negative")
        return history.pre_commitment[:prefix]
    if prefix == "pre_commitment":
        return history.pre_commitment
    if prefix == "eventual":
        return history.eventual
    raise ValueError(f"unknown trace prefix: {prefix!r}")
