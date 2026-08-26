"""Cross-fitted decision-relevant identifiability estimators."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from zsc_identifiability.established_models import DRIEstimate, DRIPoint


@dataclass(frozen=True)
class PosteriorRiskSummary:
    prior_risk: float
    residual_risk: float
    dri: float | None
    identity_mutual_information_nats: float
    response_signature_mutual_information_nats: float | None


@dataclass(frozen=True)
class EventPosteriorModel:
    mode_count: int
    token_log_likelihoods: dict[str, tuple[float, ...]]
    unknown_token_log_likelihood: tuple[float, ...]


def bayes_risk(prior: Sequence[float], loss_matrix: Sequence[Sequence[float]]) -> float:
    weights, losses = _validate_inputs(prior, loss_matrix)
    return float(np.min(weights @ losses))


def summarize_posteriors(
    prior: Sequence[float],
    loss_matrix: Sequence[Sequence[float]],
    posteriors: Sequence[Sequence[float]],
    *,
    sample_weights: Sequence[float] | None = None,
    response_signatures: Sequence[Hashable] | None = None,
    true_modes: Sequence[int] | None = None,
) -> PosteriorRiskSummary:
    """Convert held-out partner posteriors into residual task-decision risk."""

    prior_array, losses = _validate_inputs(prior, loss_matrix)
    posterior_array = np.asarray(posteriors, dtype=np.float64)
    if posterior_array.ndim != 2 or posterior_array.shape[1] != len(prior_array):
        raise ValueError("posterior matrix has the wrong shape")
    if np.any(posterior_array < -1e-12):
        raise ValueError("posteriors cannot be negative")
    row_sums = posterior_array.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-9):
        raise ValueError("each posterior row must sum to one")
    weights = (
        np.full(len(posterior_array), 1.0 / len(posterior_array))
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if weights.shape != (len(posterior_array),) or np.any(weights < 0):
        raise ValueError("invalid posterior sample weights")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("posterior sample weights must sum to one")
    posterior_decisions = np.argmin(posterior_array @ losses, axis=1)
    if true_modes is None:
        conditional_risks = np.min(posterior_array @ losses, axis=1)
    else:
        true_mode_array = np.asarray(true_modes, dtype=np.int64)
        if true_mode_array.shape != (len(posterior_array),):
            raise ValueError("one held-out true mode is required per posterior")
        if np.any(true_mode_array < 0) or np.any(true_mode_array >= len(prior_array)):
            raise ValueError("held-out true mode is outside prior support")
        conditional_risks = losses[true_mode_array, posterior_decisions]
    prior_risk = float(np.min(prior_array @ losses))
    residual = float(weights @ conditional_risks)
    dri = None if prior_risk <= 1e-15 else (prior_risk - residual) / prior_risk
    identity_mi = _posterior_information(prior_array, posterior_array, weights)
    signature_mi = None
    if response_signatures is not None:
        if len(response_signatures) != len(prior_array):
            raise ValueError("one response signature is required per partner mode")
        unique = tuple(dict.fromkeys(response_signatures))
        signature_prior = np.asarray(
            [
                sum(
                    prior_array[i]
                    for i, item in enumerate(response_signatures)
                    if item == value
                )
                for value in unique
            ]
        )
        signature_posteriors = np.asarray(
            [
                [
                    sum(row[i] for i, item in enumerate(response_signatures) if item == value)
                    for value in unique
                ]
                for row in posterior_array
            ]
        )
        signature_mi = _posterior_information(
            signature_prior, signature_posteriors, weights
        )
    return PosteriorRiskSummary(
        prior_risk=prior_risk,
        residual_risk=residual,
        dri=dri,
        identity_mutual_information_nats=identity_mi,
        response_signature_mutual_information_nats=signature_mi,
    )


def fit_event_posterior(
    calibration_histories: Sequence[Sequence[str]],
    calibration_labels: Sequence[int],
    mode_count: int,
    *,
    smoothing: float = 1.0,
) -> EventPosteriorModel:
    """Fit a Laplace-smoothed multinomial event posterior on calibration only."""

    if len(calibration_histories) != len(calibration_labels) or not calibration_histories:
        raise ValueError("calibration histories and labels must be nonempty and aligned")
    if mode_count < 2 or smoothing <= 0:
        raise ValueError("mode_count must be >=2 and smoothing must be positive")
    token_counts: list[Counter[str]] = [Counter() for _ in range(mode_count)]
    vocabulary: set[str] = set()
    for history, label in zip(calibration_histories, calibration_labels, strict=True):
        if label < 0 or label >= mode_count:
            raise ValueError("calibration label out of range")
        tokens = tuple(history) or ("zero_step",)
        token_counts[label].update(tokens)
        vocabulary.update(tokens)
    vocabulary_size = len(vocabulary) + 1
    denominators = np.asarray(
        [sum(row.values()) + smoothing * vocabulary_size for row in token_counts]
    )
    likelihoods = {
        token: tuple(
            math.log((token_counts[mode].get(token, 0) + smoothing) / denominators[mode])
            for mode in range(mode_count)
        )
        for token in sorted(vocabulary)
    }
    return EventPosteriorModel(
        mode_count=mode_count,
        token_log_likelihoods=likelihoods,
        unknown_token_log_likelihood=tuple(
            float(math.log(smoothing / denominators[mode])) for mode in range(mode_count)
        ),
    )


def predict_event_posteriors(
    model: EventPosteriorModel,
    histories: Sequence[Sequence[str]],
    prior: Sequence[float],
) -> np.ndarray:
    prior_array = np.asarray(prior, dtype=np.float64)
    if prior_array.shape != (model.mode_count,):
        raise ValueError("event posterior prior has the wrong mode count")
    if np.any(prior_array <= 0) or not np.isclose(prior_array.sum(), 1.0):
        raise ValueError("event posterior requires a normalized positive prior")
    rows: list[np.ndarray] = []
    for history in histories:
        counts = Counter(tuple(history) or ("zero_step",))
        logits = np.log(prior_array)
        for token, count in counts.items():
            token_values = model.token_log_likelihoods.get(
                token, model.unknown_token_log_likelihood
            )
            logits += count * np.asarray(token_values)
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        total = float(probabilities.sum())
        rows.append(prior_array if total <= 0 else probabilities / total)
    return np.asarray(rows, dtype=np.float64)


def estimate_event_dri(
    calibration_histories: Sequence[Sequence[str]],
    calibration_labels: Sequence[int],
    confirmatory_histories: Sequence[Sequence[str]],
    prior: Sequence[float],
    loss_matrix: Sequence[Sequence[float]],
    *,
    response_signatures: Sequence[Hashable] | None = None,
    confirmatory_labels: Sequence[int] | None = None,
    prefix: str = "pre_commitment",
    censored_count: int = 0,
) -> DRIEstimate:
    table = fit_event_posterior(
        calibration_histories,
        calibration_labels,
        len(prior),
    )
    posteriors = predict_event_posteriors(table, confirmatory_histories, prior)
    if censored_count:
        prior_rows = np.repeat(
            np.asarray(prior, dtype=np.float64)[None, :], censored_count, axis=0
        )
        posteriors = np.concatenate((posteriors, prior_rows), axis=0)
    summary = summarize_posteriors(
        prior,
        loss_matrix,
        posteriors.tolist(),
        response_signatures=response_signatures,
        true_modes=confirmatory_labels,
    )
    return DRIEstimate(
        estimator="event",
        split="evaluation",
        evidence_policy="ordinary_progress",
        points=(
            DRIPoint(
                prefix=prefix,
                prior_risk=summary.prior_risk,
                residual_risk=summary.residual_risk,
                dri=summary.dri,
                identity_mutual_information_nats=summary.identity_mutual_information_nats,
                response_signature_mutual_information_nats=(
                    summary.response_signature_mutual_information_nats
                ),
                episode_count=len(posteriors),
                censored_count=censored_count,
            ),
        ),
        temperature=None,
        cross_fitted=True,
        calibration_trajectory_hashes=(),
        confirmatory_trajectory_hashes=(),
        leakage_checks_passed=True,
    )


def synthetic_dri_calibration() -> dict[str, float | bool]:
    """Executable Stage 3 bridge used before any Overcooked claim is allowed."""

    binary_loss = ((0.0, 40.0), (40.0, 0.0))
    informative = summarize_posteriors(
        (0.5, 0.5),
        binary_loss,
        ((0.8, 0.2), (0.2, 0.8)),
    )
    late_pre = summarize_posteriors(
        (0.5, 0.5), binary_loss, ((0.5, 0.5), (0.5, 0.5))
    )
    late_eventual = summarize_posteriors(
        (0.5, 0.5), binary_loss, ((1.0, 0.0), (0.0, 1.0))
    )
    four_loss = (
        (0.0, 40.0),
        (0.0, 40.0),
        (40.0, 0.0),
        (40.0, 0.0),
    )
    subtype = summarize_posteriors(
        (0.25, 0.25, 0.25, 0.25),
        four_loss,
        ((0.5, 0.0, 0.5, 0.0), (0.0, 0.5, 0.0, 0.5)),
        response_signatures=("A", "A", "B", "B"),
    )
    informative_calibration = (
        (("signal_a",),) * 800
        + (("signal_b",),) * 200
        + (("signal_a",),) * 200
        + (("signal_b",),) * 800
    )
    informative_labels = (0,) * 1_000 + (1,) * 1_000
    informative_event = estimate_event_dri(
        informative_calibration,
        informative_labels,
        informative_calibration,
        (0.5, 0.5),
        binary_loss,
        response_signatures=("A", "B"),
        confirmatory_labels=informative_labels,
    )
    subtype_histories = (
        (("subtype_0",),) * 500
        + (("subtype_1",),) * 500
        + (("subtype_0",),) * 500
        + (("subtype_1",),) * 500
    )
    subtype_labels = (0,) * 500 + (1,) * 500 + (2,) * 500 + (3,) * 500
    subtype_event = estimate_event_dri(
        subtype_histories,
        subtype_labels,
        subtype_histories,
        (0.25, 0.25, 0.25, 0.25),
        four_loss,
        response_signatures=("A", "A", "B", "B"),
        confirmatory_labels=subtype_labels,
    )
    shuffled_histories = (("signal_a",),) * 1_000 + (("signal_b",),) * 1_000
    shuffled_labels = (0, 1) * 500 + (0, 1) * 500
    shuffled_event = estimate_event_dri(
        shuffled_histories,
        shuffled_labels,
        informative_calibration,
        (0.5, 0.5),
        binary_loss,
        response_signatures=("A", "B"),
        confirmatory_labels=informative_labels,
    )
    event_informative_dri = informative_event.points[0].dri
    event_subtype_dri = subtype_event.points[0].dri
    shuffled_dri = shuffled_event.points[0].dri
    values = {
        "informative_dri": float(informative.dri or 0.0),
        "informative_residual_risk": informative.residual_risk,
        "identity_only_dri": float(subtype.dri or 0.0),
        "identity_only_identity_mi_nats": subtype.identity_mutual_information_nats,
        "identity_only_response_mi_nats": float(
            subtype.response_signature_mutual_information_nats or 0.0
        ),
        "late_precommitment_dri": float(late_pre.dri or 0.0),
        "late_eventual_dri": float(late_eventual.dri or 0.0),
        "event_estimator_informative_dri": float(event_informative_dri or 0.0),
        "event_estimator_identity_only_dri": float(event_subtype_dri or 0.0),
        "event_estimator_label_shuffle_dri": float(shuffled_dri or 0.0),
    }
    values["passed"] = bool(
        abs(values["informative_dri"] - 0.6) <= 0.03
        and abs(values["informative_residual_risk"] - 8.0) <= 1e-12
        and abs(values["identity_only_dri"]) <= 1e-12
        and values["identity_only_identity_mi_nats"] > 0
        and abs(values["identity_only_response_mi_nats"]) <= 1e-12
        and abs(values["late_precommitment_dri"]) <= 1e-12
        and abs(values["late_eventual_dri"] - 1.0) <= 1e-12
        and abs(values["event_estimator_informative_dri"] - 0.6) <= 0.03
        and abs(values["event_estimator_identity_only_dri"]) <= 0.03
        and abs(values["event_estimator_label_shuffle_dri"]) <= 0.03
    )
    return values


def _posterior_information(
    prior: np.ndarray, posteriors: np.ndarray, weights: np.ndarray
) -> float:
    total = 0.0
    for weight, posterior in zip(weights, posteriors, strict=True):
        for posterior_mass, prior_mass in zip(posterior, prior, strict=True):
            if posterior_mass > 0:
                if prior_mass <= 0:
                    raise ValueError("posterior assigns mass outside prior support")
                total += float(weight * posterior_mass * math.log(posterior_mass / prior_mass))
    return total


def _validate_inputs(
    prior: Sequence[float], loss_matrix: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(prior, dtype=np.float64)
    losses = np.asarray(loss_matrix, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < 1 or np.any(weights < 0):
        raise ValueError("invalid partner prior")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("partner prior must sum to one")
    if losses.ndim != 2 or losses.shape[0] != len(weights) or losses.shape[1] < 1:
        raise ValueError("loss matrix has the wrong shape")
    if np.any(losses < 0) or not np.all(np.isfinite(losses)):
        raise ValueError("loss matrix must be finite and nonnegative")
    return weights, losses
