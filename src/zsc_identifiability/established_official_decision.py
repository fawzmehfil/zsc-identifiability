"""Decision-risk estimators used by the Stage 6 v3 measurement redesign."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from zsc_identifiability.established_official_redesign_models import (
    PairwiseDecisionDecoder,
)
from zsc_identifiability.learning_statistics import holm_adjust

_EVENT_SCALAR_COUNT = 3


@dataclass(frozen=True)
class PairwiseDecoderDataset:
    seed: int | None
    left_partner_id: str
    right_partner_id: str
    calibration_features: np.ndarray
    calibration_labels: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    loss_matrix: np.ndarray
    validation_censored: np.ndarray | None = None


@dataclass(frozen=True)
class DecoderConfigurationSelection:
    configuration_id: str
    ridge_strength: float
    temperature: float
    prior_shrinkage: float
    mean_validation_loss: float
    worst_pair_validation_loss: float
    decoders: tuple[PairwiseDecisionDecoder, ...]


@dataclass(frozen=True)
class PairwiseDecisionEvaluation:
    prior_risk: float
    residual_risk: float
    fixed_response_risk: float
    dri: float | None
    brier_score: float
    uniform_brier_score: float
    probabilities_right: np.ndarray
    decisions: np.ndarray


def signed_hash_event_features(
    timed_tokens: Sequence[tuple[int, str]],
    *,
    observed_length: int,
    cumulative_reward: float,
    partner_visibility_rate: float,
    width: int = 512,
    salt: str = "zsc-dri-v3-event-features",
) -> np.ndarray:
    """Create a deterministic signed-hash history vector with three scalar slots."""

    if width <= _EVENT_SCALAR_COUNT:
        raise ValueError("event feature width must reserve space for the registered scalars")
    if observed_length < 0:
        raise ValueError("observed history length cannot be negative")
    if partner_visibility_rate < 0 or partner_visibility_rate > 1:
        raise ValueError("partner visibility rate must lie in [0, 1]")
    hashed_width = width - _EVENT_SCALAR_COUNT
    result = np.zeros(width, dtype=np.float64)
    for step, token in timed_tokens:
        if step < 0:
            raise ValueError("event token steps cannot be negative")
        temporal_bin = _temporal_bin(step)
        digest = hashlib.sha256(f"{salt}:{temporal_bin}:{token}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % hashed_width
        sign = 1.0 if digest[8] & 1 else -1.0
        result[index] += sign
    result[-3:] = (float(observed_length), float(cumulative_reward), partner_visibility_rate)
    return result


def fit_ridge_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_strength: float,
    maximum_iterations: int = 100,
    convergence_tolerance: float = 1e-9,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit deterministic binary ridge logistic regression using damped Newton steps."""

    x, y = _validate_binary_data(features, labels)
    if ridge_strength <= 0:
        raise ValueError("ridge strength must be positive")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack((np.ones(len(x)), standardized))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_strength
    penalty[0, 0] = 0.0
    sample_count = float(len(x))
    for _ in range(maximum_iterations):
        logits = design @ parameters
        probabilities = _sigmoid(logits)
        gradient = design.T @ (probabilities - y) / sample_count + penalty @ parameters
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-9, None)
        hessian = (design.T * weights) @ design / sample_count + penalty
        step = np.linalg.pinv(hessian) @ gradient
        parameters -= step
        if float(np.max(np.abs(step))) <= convergence_tolerance:
            break
    return parameters[1:], float(parameters[0]), mean, scale


def decoder_probabilities(
    decoder: PairwiseDecisionDecoder,
    features: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(decoder.coefficient):
        raise ValueError("decoder feature matrix has the wrong shape")
    mean = np.asarray(decoder.feature_mean, dtype=np.float64)
    scale = np.asarray(decoder.feature_scale, dtype=np.float64)
    coefficient = np.asarray(decoder.coefficient, dtype=np.float64)
    logits = ((x - mean) / scale) @ coefficient + decoder.intercept
    probabilities = _sigmoid(logits / decoder.temperature)
    return (1.0 - decoder.prior_shrinkage) * probabilities + (
        decoder.prior_shrinkage * 0.5
    )


def evaluate_pairwise_decision(
    probabilities_right: npt.ArrayLike,
    true_modes: npt.ArrayLike,
    loss_matrix: npt.ArrayLike,
    *,
    censored: npt.ArrayLike | None = None,
) -> PairwiseDecisionEvaluation:
    probabilities = np.asarray(probabilities_right, dtype=np.float64)
    labels = np.asarray(true_modes, dtype=np.int64)
    losses = np.asarray(loss_matrix, dtype=np.float64)
    if probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("one probability and true binary mode are required per history")
    if losses.ndim != 2 or losses.shape[0] != 2 or losses.shape[1] < 1:
        raise ValueError("pairwise loss matrix must have two modes and at least one response")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("pair probabilities must lie in [0, 1]")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("pairwise true modes must be binary")
    if censored is not None:
        mask = np.asarray(censored, dtype=bool)
        if mask.shape != probabilities.shape:
            raise ValueError("censoring mask must align with pair histories")
        probabilities = probabilities.copy()
        probabilities[mask] = 0.5
    posteriors = np.column_stack((1.0 - probabilities, probabilities))
    decisions = np.argmin(posteriors @ losses, axis=1)
    actual_losses = losses[labels, decisions]
    prior_risk = float(np.min(np.mean(losses, axis=0)))
    residual_risk = float(actual_losses.mean())
    fixed_response_risk = prior_risk
    dri = None if prior_risk <= 1e-15 else (prior_risk - residual_risk) / prior_risk
    return PairwiseDecisionEvaluation(
        prior_risk=prior_risk,
        residual_risk=residual_risk,
        fixed_response_risk=fixed_response_risk,
        dri=dri,
        brier_score=float(np.mean((probabilities - labels) ** 2)),
        uniform_brier_score=float(np.mean((0.5 - labels) ** 2)),
        probabilities_right=probabilities,
        decisions=decisions,
    )


def select_global_decoder_configuration(
    datasets: Sequence[PairwiseDecoderDataset],
    *,
    layout_id: str,
    evidence_policy: str,
    prefix: int | str,
    representation: str,
    seed: int | None,
    ridge_strengths: Sequence[float],
    temperatures: Sequence[float],
    prior_shrinkages: Sequence[float],
    maximum_iterations: int = 100,
    convergence_tolerance: float = 1e-9,
) -> DecoderConfigurationSelection:
    """Select one validation-only decoder configuration for a complete unit."""

    if not datasets:
        raise ValueError("decoder selection requires at least one response-conflicting pair")
    fitted: dict[
        tuple[float, int | None, str, str],
        tuple[np.ndarray, float, np.ndarray, np.ndarray],
    ] = {}
    for ridge in ridge_strengths:
        for dataset in datasets:
            fitted[
                (
                    float(ridge),
                    dataset.seed,
                    dataset.left_partner_id,
                    dataset.right_partner_id,
                )
            ] = (
                fit_ridge_logistic(
                    dataset.calibration_features,
                    dataset.calibration_labels,
                    ridge_strength=float(ridge),
                    maximum_iterations=maximum_iterations,
                    convergence_tolerance=convergence_tolerance,
                )
            )
    candidates: list[tuple[tuple[float, float, float, float, str], str]] = []
    for ridge in ridge_strengths:
        for temperature in temperatures:
            for shrinkage in prior_shrinkages:
                configuration_id = _configuration_id(ridge, temperature, shrinkage)
                pair_losses: list[float] = []
                for dataset in datasets:
                    coefficient, intercept, mean, scale = fitted[
                        (
                            float(ridge),
                            dataset.seed,
                            dataset.left_partner_id,
                            dataset.right_partner_id,
                        )
                    ]
                    probabilities = _predict(
                        dataset.validation_features,
                        coefficient,
                        intercept,
                        mean,
                        scale,
                        float(temperature),
                        float(shrinkage),
                    )
                    evaluation = evaluate_pairwise_decision(
                        probabilities,
                        dataset.validation_labels,
                        dataset.loss_matrix,
                        censored=dataset.validation_censored,
                    )
                    pair_losses.append(evaluation.residual_risk)
                mean_loss = float(np.mean(pair_losses))
                worst_loss = float(np.max(pair_losses))
                key = (
                    mean_loss,
                    worst_loss,
                    -float(shrinkage),
                    -float(ridge),
                    configuration_id,
                )
                candidates.append((key, configuration_id))
    selection_key, configuration_id = min(candidates, key=lambda item: item[0])
    ridge, temperature, shrinkage = _parse_configuration_id(configuration_id)
    decoders: list[PairwiseDecisionDecoder] = []
    for dataset in datasets:
        coefficient, intercept, mean, scale = fitted[
            (ridge, dataset.seed, dataset.left_partner_id, dataset.right_partner_id)
        ]
        decoder_id = hashlib.sha256(
            (
                f"{layout_id}:{evidence_policy}:{prefix}:{representation}:"
                f"{dataset.seed if dataset.seed is not None else seed}:"
                f"{dataset.left_partner_id}:{dataset.right_partner_id}:{configuration_id}"
            ).encode()
        ).hexdigest()
        decoders.append(
            PairwiseDecisionDecoder(
                decoder_id=decoder_id,
                layout_id=layout_id,  # type: ignore[arg-type]
                evidence_policy=evidence_policy,
                prefix=prefix,  # type: ignore[arg-type]
                representation=representation,  # type: ignore[arg-type]
                seed=dataset.seed if dataset.seed is not None else seed,
                left_partner_id=dataset.left_partner_id,
                right_partner_id=dataset.right_partner_id,
                ridge_strength=ridge,
                temperature=temperature,
                prior_shrinkage=shrinkage,
                coefficient=tuple(float(value) for value in coefficient),
                intercept=intercept,
                feature_mean=tuple(float(value) for value in mean),
                feature_scale=tuple(float(value) for value in scale),
                configuration_id=configuration_id,
                calibration_examples=len(dataset.calibration_labels),
                validation_examples=len(dataset.validation_labels),
            )
        )
    return DecoderConfigurationSelection(
        configuration_id=configuration_id,
        ridge_strength=ridge,
        temperature=temperature,
        prior_shrinkage=shrinkage,
        mean_validation_loss=selection_key[0],
        worst_pair_validation_loss=selection_key[1],
        decoders=tuple(decoders),
    )


def one_sided_permutation_p_value(observed: float, null_values: Sequence[float]) -> float:
    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("permutation null must be a nonempty finite vector")
    return float((1 + np.count_nonzero(values >= observed)) / (len(values) + 1))


def holm_correct_permutation_tests(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    if not raw_p_values:
        raise ValueError("Holm correction requires registered permutation tests")
    if any(value < 0 or value > 1 for value in raw_p_values.values()):
        raise ValueError("permutation p-values must lie in [0, 1]")
    return holm_adjust(dict(raw_p_values))


def binary_posterior_information(probabilities_right: Sequence[float]) -> float:
    probabilities = np.asarray(probabilities_right, dtype=np.float64)
    if probabilities.ndim != 1 or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("binary posterior probabilities must lie in [0, 1]")
    entropy = -(
        _xlogx(probabilities) + _xlogx(1.0 - probabilities)
    )
    return max(0.0, math.log(2.0) - float(np.mean(entropy)))


def synthetic_decision_estimator_controls(*, tolerance: float = 0.03) -> dict[str, object]:
    """Executable controls for the v3 decision-value definition and censoring rule."""

    symmetric_losses = np.asarray(((0.0, 40.0), (40.0, 0.0)))
    labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
    informative = np.asarray(
        [0.2] * 8 + [0.8] * 2 + [0.8] * 8 + [0.2] * 2,
        dtype=np.float64,
    )
    informative_result = evaluate_pairwise_decision(informative, labels, symmetric_losses)
    identity_only = evaluate_pairwise_decision(
        np.asarray([0.5] * len(labels)), labels, symmetric_losses
    )
    late_pre = identity_only
    late_eventual = evaluate_pairwise_decision(labels.astype(np.float64), labels, symmetric_losses)
    inseparable = identity_only
    asymmetric_losses = np.asarray(((0.0, 1.0, 4.0), (5.0, 0.0, 1.0)))
    asymmetric_probability = np.asarray([0.1, 0.55])
    asymmetric_labels = np.asarray([0, 1])
    asymmetric = evaluate_pairwise_decision(
        asymmetric_probability, asymmetric_labels, asymmetric_losses
    )
    censored = evaluate_pairwise_decision(
        np.asarray([0.01, 0.99]),
        np.asarray([1, 0]),
        symmetric_losses,
        censored=np.asarray([True, True]),
    )
    checks = {
        "binary_q08": bool(
            informative_result.dri is not None
            and abs(informative_result.dri - 0.6) <= tolerance
        ),
        "identity_only_zero": identity_only.dri == 0.0,
        "late_precommitment_zero": late_pre.dri == 0.0,
        "late_eventual_one": late_eventual.dri == 1.0,
        "inseparable_zero": inseparable.dri == 0.0,
        "asymmetric_loss_decision": asymmetric.decisions.tolist() == [0, 1],
        "censored_prior_fallback": censored.dri == 0.0,
    }
    return {
        "tolerance": tolerance,
        "informative_dri": informative_result.dri,
        "identity_only_dri": identity_only.dri,
        "late_precommitment_dri": late_pre.dri,
        "late_eventual_dri": late_eventual.dri,
        "inseparable_dri": inseparable.dri,
        "asymmetric_decisions": asymmetric.decisions.tolist(),
        "censored_dri": censored.dri,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _predict(
    features: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    mean: np.ndarray,
    scale: np.ndarray,
    temperature: float,
    shrinkage: float,
) -> np.ndarray:
    logits = ((np.asarray(features, dtype=np.float64) - mean) / scale) @ coefficient + intercept
    probability = _sigmoid(logits / temperature)
    return (1.0 - shrinkage) * probability + shrinkage * 0.5


def _validate_binary_data(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("binary decoder features must be a nonempty matrix")
    if y.shape != (len(x),) or np.any((y != 0) & (y != 1)):
        raise ValueError("binary decoder labels must align and contain only 0/1")
    if len(np.unique(y)) != 2:
        raise ValueError("binary decoder fitting requires both partner modes")
    if not np.all(np.isfinite(x)):
        raise ValueError("binary decoder features must be finite")
    return x, y


def _configuration_id(ridge: float, temperature: float, shrinkage: float) -> str:
    return f"r={float(ridge):.12g}|t={float(temperature):.12g}|s={float(shrinkage):.12g}"


def _parse_configuration_id(value: str) -> tuple[float, float, float]:
    pieces = dict(part.split("=", 1) for part in value.split("|"))
    return float(pieces["r"]), float(pieces["t"]), float(pieces["s"])


def _temporal_bin(step: int) -> str:
    if step <= 7:
        return "0-7"
    if step <= 15:
        return "8-15"
    if step <= 31:
        return "16-31"
    return "32+"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _xlogx(values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values)
    positive = values > 0
    output[positive] = values[positive] * np.log(values[positive])
    return output
