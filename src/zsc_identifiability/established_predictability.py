"""Cross-fitted visible partner-action predictability for Stage 6 controls."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from zsc_identifiability.established_commitment import WorkUnitHistory, extract_work_units
from zsc_identifiability.established_io import load_trace_jsonl
from zsc_identifiability.established_models import CommitmentTraceStep


def estimate_lobp_action_oracle_from_trace_files(
    calibration_path: str | Path,
    confirmatory_path: str | Path,
    *,
    smoothing: float = 1.0,
) -> dict[str, float | int | str]:
    """Score a discrete action oracle on disjoint ego-visible trajectories.

    This is deliberately named a LoBP-style *action oracle*: it estimates
    visible partner-action cross-entropy, not latent intentions and not the
    original learned LoBP observer.
    """

    calibration = _first_units(calibration_path)
    confirmatory = _first_units(confirmatory_path)
    calibration_ids = {item.episode_id for item in calibration}
    confirmatory_ids = {item.episode_id for item in confirmatory}
    if calibration_ids & confirmatory_ids:
        raise ValueError("predictability calibration and confirmatory episodes overlap")
    precommit = _score_scope(calibration, confirmatory, "pre_commitment", smoothing)
    full = _score_scope(calibration, confirmatory, "eventual", smoothing)
    return {
        "estimator": "discrete_visible_action_oracle",
        "precommit_score_nats": -precommit[0],
        "precommit_target_count": precommit[1],
        "full_episode_score_nats": -full[0],
        "full_episode_target_count": full[1],
    }


def _score_scope(
    calibration: Sequence[WorkUnitHistory],
    confirmatory: Sequence[WorkUnitHistory],
    scope: str,
    smoothing: float,
) -> tuple[float, int]:
    calibration_samples = _samples(calibration, scope)
    confirmatory_samples = _samples(confirmatory, scope)
    if not calibration_samples or not confirmatory_samples:
        raise ValueError("predictability audit requires visible partner-action targets")
    actions = sorted({target for _, target in calibration_samples})
    if any(target not in actions for _, target in confirmatory_samples):
        raise ValueError("confirmatory trace contains an unseen partner action")
    global_counts = Counter(target for _, target in calibration_samples)
    context_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for context, target in calibration_samples:
        context_counts[context][target] += 1
    loss = 0.0
    for context, target in confirmatory_samples:
        counts = context_counts.get(context, global_counts)
        denominator = sum(counts.values()) + smoothing * len(actions)
        probability = (counts.get(target, 0) + smoothing) / denominator
        loss -= math.log(probability)
    return loss / len(confirmatory_samples), len(confirmatory_samples)


def _samples(
    histories: Sequence[WorkUnitHistory], scope: str
) -> list[tuple[tuple[str, ...], str]]:
    result: list[tuple[tuple[str, ...], str]] = []
    for history in histories:
        steps = history.pre_commitment if scope == "pre_commitment" else history.eventual
        previous: CommitmentTraceStep | None = None
        for step in steps:
            if step.visible_partner_action is not None:
                result.append((_context(previous, step), str(step.visible_partner_action)))
            previous = step
    return result


def _context(
    previous: CommitmentTraceStep | None,
    current: CommitmentTraceStep,
) -> tuple[str, ...]:
    time_bin = min(current.step // 8, 50)
    if previous is None:
        return (f"time:{time_bin}", "start")
    partner_action = (
        "hidden"
        if previous.visible_partner_action is None
        else str(previous.visible_partner_action)
    )
    event_signature = "+".join(sorted(previous.high_level_events)) or "none"
    return (
        f"time:{time_bin}",
        f"ego:{previous.ego_action}",
        f"partner:{partner_action}",
        f"events:{event_signature}",
    )


def _first_units(path: str | Path) -> tuple[WorkUnitHistory, ...]:
    return tuple(
        item for item in extract_work_units(load_trace_jsonl(path)) if item.work_unit == 0
    )
