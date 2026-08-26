"""Trace-to-DRI curve estimation with strict calibration/evaluation separation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from zsc_identifiability.established_commitment import (
    WorkUnitHistory,
    extract_work_units,
    prefix_history,
)
from zsc_identifiability.established_dri import (
    fit_event_posterior,
    predict_event_posteriors,
    summarize_posteriors,
)
from zsc_identifiability.established_gru import fit_cross_fitted_gru_posterior
from zsc_identifiability.established_io import load_trace_jsonl
from zsc_identifiability.established_models import (
    CommitmentTraceStep,
    DRIEstimate,
    DRIEstimatorSpec,
    DRIPoint,
    ResponseLibrary,
)


def estimate_dri_curve_from_trace_files(
    calibration_path: str | Path,
    validation_path: str | Path,
    confirmatory_path: str | Path,
    library: ResponseLibrary,
    config: DRIEstimatorSpec,
    *,
    estimator: str,
) -> DRIEstimate:
    if estimator not in {"event", "gru"}:
        raise ValueError(f"unknown established DRI estimator: {estimator!r}")
    calibration = _first_work_units(calibration_path)
    validation = _first_work_units(validation_path)
    confirmatory = _first_work_units(confirmatory_path)
    _require_disjoint(calibration, validation, confirmatory)
    partner_index = {partner: index for index, partner in enumerate(library.partner_ids)}
    for history in (*calibration, *validation, *confirmatory):
        if history.partner_id not in partner_index:
            raise ValueError(
                f"trace includes partner outside response library: {history.partner_id}"
            )
    prior = tuple(1.0 / len(library.partner_ids) for _ in library.partner_ids)
    signatures = tuple(
        str(library.response_clusters[partner]) for partner in library.partner_ids
    )
    prefixes: tuple[int | str, ...] = (*config.prefix_steps, "pre_commitment", "eventual")
    points: list[DRIPoint] = []
    temperature: float | None = None
    for prefix in prefixes:
        calibration_sequences, calibration_labels, _ = _sequences(
            calibration, prefix, partner_index, estimator
        )
        validation_sequences, validation_labels, _ = _sequences(
            validation, prefix, partner_index, estimator
        )
        confirmatory_sequences, confirmatory_labels, censored = _sequences(
            confirmatory, prefix, partner_index, estimator
        )
        if estimator == "event":
            table = fit_event_posterior(
                calibration_sequences,  # type: ignore[arg-type]
                calibration_labels,
                len(prior),
            )
            posteriors = predict_event_posteriors(
                table,
                confirmatory_sequences,  # type: ignore[arg-type]
                prior,
            )
        else:
            result = fit_cross_fitted_gru_posterior(
                calibration_sequences,  # type: ignore[arg-type]
                calibration_labels,
                validation_sequences,  # type: ignore[arg-type]
                validation_labels,
                confirmatory_sequences,  # type: ignore[arg-type]
                confirmatory_labels,
                prior,
                library.loss_matrix,
                response_signatures=signatures,
                hidden_size=config.gru_hidden_size,
                learning_rate=config.gru_learning_rate,
                max_epochs=config.gru_max_epochs,
                patience=config.gru_patience,
                seed=config.random_seed,
            )
            posteriors = result.posteriors
            temperature = result.temperature
        # No-commitment episodes are retained but get exactly prior residual
        # risk at the pre-commitment endpoint, as preregistered.
        if prefix == "pre_commitment" and censored:
            for index in censored:
                posteriors[index] = np.asarray(prior)
        summary = summarize_posteriors(
            prior,
            library.loss_matrix,
            posteriors.tolist(),
            response_signatures=signatures,
            true_modes=confirmatory_labels,
        )
        points.append(
            DRIPoint(
                prefix=str(prefix),
                prior_risk=summary.prior_risk,
                residual_risk=summary.residual_risk,
                dri=summary.dri,
                identity_mutual_information_nats=summary.identity_mutual_information_nats,
                response_signature_mutual_information_nats=(
                    summary.response_signature_mutual_information_nats
                ),
                episode_count=len(confirmatory),
                censored_count=len(censored) if prefix == "pre_commitment" else 0,
            )
        )
    return DRIEstimate(
        estimator=estimator,  # type: ignore[arg-type]
        split="evaluation",
        evidence_policy="ordinary_progress",
        points=tuple(points),
        temperature=temperature,
        cross_fitted=True,
        calibration_trajectory_hashes=(_hash_file(calibration_path),),
        confirmatory_trajectory_hashes=(_hash_file(confirmatory_path),),
        leakage_checks_passed=True,
    )


def treatment_effect_agreement(
    low_event: DRIEstimate,
    high_event: DRIEstimate,
    low_gru: DRIEstimate,
    high_gru: DRIEstimate,
    *,
    prefix: str = "pre_commitment",
    tolerance: float = 0.05,
) -> dict[str, float | bool]:
    event_effect = _point(high_event, prefix) - _point(low_event, prefix)
    gru_effect = _point(high_gru, prefix) - _point(low_gru, prefix)
    difference = abs(event_effect - gru_effect)
    return {
        "event_treatment_effect": event_effect,
        "gru_treatment_effect": gru_effect,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def _first_work_units(path: str | Path) -> tuple[WorkUnitHistory, ...]:
    return tuple(
        history
        for history in extract_work_units(load_trace_jsonl(path))
        if history.work_unit == 0
    )


def _sequences(
    histories: Sequence[WorkUnitHistory],
    prefix: int | str,
    partner_index: dict[str, int],
    estimator: str,
) -> tuple[list[tuple[str, ...] | np.ndarray], list[int], list[int]]:
    sequence_values: list[tuple[str, ...] | np.ndarray] = []
    labels: list[int] = []
    censored: list[int] = []
    for index, history in enumerate(histories):
        steps = prefix_history(history, prefix)
        if prefix == "pre_commitment" and not history.commitment_reached:
            censored.append(index)
        if estimator == "event":
            sequence_values.append(_event_tokens(steps))
        else:
            sequence_values.append(_gru_features(steps, history))
        labels.append(partner_index[history.partner_id])
    return sequence_values, labels, censored


def _event_tokens(steps: Sequence[CommitmentTraceStep]) -> tuple[str, ...]:
    tokens: list[str] = []
    for step in steps:
        tokens.append(f"ego_action:{step.ego_action}")
        partner_action = step.visible_partner_action
        if partner_action is not None:
            tokens.append(f"partner_action:{partner_action}")
        tokens.extend(step.high_level_events)
        reward = float(step.reward)
        reward_token = (
            "reward:positive"
            if reward > 0
            else "reward:negative"
            if reward < 0
            else "reward:zero"
        )
        tokens.append(reward_token)
    return tuple(tokens) or ("zero_step",)


def _gru_features(
    steps: Sequence[CommitmentTraceStep], history: WorkUnitHistory
) -> np.ndarray:
    observation_width = next(
        (
            len(step.ego_observation)
            for step in (*history.pre_commitment, *history.eventual)
            if step.ego_observation
        ),
        1,
    )
    rows: list[np.ndarray] = []
    for step in steps:
        observation = np.asarray(step.ego_observation, dtype=np.float32)
        if len(observation) == 0:
            observation = np.zeros(observation_width, dtype=np.float32)
        if len(observation) != observation_width:
            raise ValueError("observation width changes inside a fixed-layout trace")
        partner_action = step.visible_partner_action
        extras = np.asarray(
            [
                float(step.ego_action),
                -1.0 if partner_action is None else float(partner_action),
                float(step.reward),
                float(step.step) / 400.0,
                1.0,
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate((observation, extras)))
    if not rows:
        return np.zeros((1, observation_width + 5), dtype=np.float32)
    return np.stack(rows)


def _require_disjoint(
    calibration: Sequence[WorkUnitHistory],
    validation: Sequence[WorkUnitHistory],
    confirmatory: Sequence[WorkUnitHistory],
) -> None:
    groups = [
        {item.episode_id for item in calibration},
        {item.episode_id for item in validation},
        {item.episode_id for item in confirmatory},
    ]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("calibration, validation, and confirmatory episode IDs must be disjoint")


def _point(estimate: DRIEstimate, prefix: str) -> float:
    value = next(item.dri for item in estimate.points if item.prefix == prefix)
    if value is None:
        raise ValueError("treatment-effect agreement requires non-null DRI")
    return value


def _hash_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
