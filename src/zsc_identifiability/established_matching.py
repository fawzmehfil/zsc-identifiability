"""Frozen mixed-integer selection and confirmatory matching audits."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from zsc_identifiability.established_models import (
    CandidatePartnerMetrics,
    MatchedPopulationAudit,
    MatchingMetricAudit,
    MatchingSpec,
)


def select_matched_population_pair(
    candidates: Sequence[CandidatePartnerMetrics],
    contract: MatchingSpec,
    contrast: str,
) -> MatchedPopulationAudit:
    """Select two disjoint populations with a frozen MILP contract.

    Linear nuisance controls are enforced inside the MILP.  Population-level
    BR-Div is audited afterward without jitter; failure is returned as a failed
    construction and never triggers relaxed margins.
    """

    if contrast not in {"passive_dri", "active_dri"}:
        raise ValueError(f"unknown matched contrast: {contrast!r}")
    if len(candidates) < 2 * contract.subset_size:
        raise ValueError("not enough candidates for two disjoint matched populations")
    if len({item.partner_id for item in candidates}) != len(candidates):
        raise ValueError("duplicate candidate partner identifier")
    feature_widths = {len(item.br_event_features) for item in candidates}
    if len(feature_widths) != 1:
        raise ValueError("best-response event feature widths must match")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional established extra
        raise RuntimeError(
            "matched selection requires the established extra: uv sync --extra established"
        ) from exc

    ordered = tuple(sorted(candidates, key=lambda item: item.partner_id))
    count = len(ordered)
    variable_count = 2 * count
    objective = np.zeros(variable_count)
    treatment = np.asarray([getattr(item, contrast) for item in ordered])
    objective[:count] = -treatment
    objective[count:] = treatment
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: np.ndarray, low: float, high: float) -> None:
        rows.append(coefficients)
        lower.append(low)
        upper.append(high)

    size_row = np.zeros(variable_count)
    size_row[:count] = 1
    add(size_row, contract.subset_size, contract.subset_size)
    size_row = np.zeros(variable_count)
    size_row[count:] = 1
    add(size_row, contract.subset_size, contract.subset_size)
    for index in range(count):
        row = np.zeros(variable_count)
        row[index] = 1
        row[count + index] = 1
        add(row, 0, 1)
    for cluster in sorted({item.response_cluster for item in ordered}):
        row = np.zeros(variable_count)
        for index, item in enumerate(ordered):
            if item.response_cluster == cluster:
                row[index] = 1
                row[count + index] = -1
        add(row, 0, 0)
    controls: tuple[tuple[str, float], ...] = (
        ("competence", contract.competence_margin),
        ("best_fixed_response_value", contract.fixed_response_margin),
        ("br_prox", contract.br_prox_margin),
        ("lobp_score_nats", contract.predictability_margin_nats),
        ("trajectory_divergence", contract.trajectory_divergence_margin),
        ("commitment_reached_rate", contract.discovery_commitment_rate_margin),
    )
    if contrast == "active_dri":
        controls += (("passive_dri", contract.active_contrast_passive_dri_margin),)
    for field, margin in controls:
        values = np.asarray([float(getattr(item, field)) for item in ordered])
        row = np.concatenate((values, -values))
        scaled = contract.subset_size * margin
        add(row, -scaled, scaled)
    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(np.stack(rows), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        return _failed_selection(contrast, str(result.message))
    left = tuple(ordered[index] for index in range(count) if result.x[index] >= 0.5)
    right = tuple(ordered[index] for index in range(count) if result.x[count + index] >= 0.5)
    audits = _audit_selected(left, right, contract, contrast)
    passed = all(item.passed for item in audits)
    selection_payload = {
        "contrast": contrast,
        "left": [item.partner_id for item in left],
        "right": [item.partner_id for item in right],
        "contract": contract.model_dump(mode="json"),
    }
    selection_hash = hashlib.sha256(
        json.dumps(selection_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return MatchedPopulationAudit(
        contrast=contrast,  # type: ignore[arg-type]
        left_partner_ids=tuple(item.partner_id for item in left),
        right_partner_ids=tuple(item.partner_id for item in right),
        frozen=True,
        discovery_metrics=audits,
        discovery_passed=passed,
        solver_status=str(result.message),
        selection_hash=selection_hash,
    )


def audit_confirmatory_population_pair(
    frozen: MatchedPopulationAudit,
    candidates: Sequence[CandidatePartnerMetrics],
    contract: MatchingSpec,
    commitment_outcomes: dict[str, Sequence[bool]] | None = None,
) -> MatchedPopulationAudit:
    by_id = {item.partner_id: item for item in candidates}
    try:
        left = tuple(by_id[item] for item in frozen.left_partner_ids)
        right = tuple(by_id[item] for item in frozen.right_partner_ids)
    except KeyError as exc:
        raise ValueError(f"confirmatory metrics omit frozen partner {exc.args[0]!r}") from exc
    audits = list(_audit_selected(left, right, contract, frozen.contrast))
    if contract.require_commitment_rate_nonsignificant:
        audits.append(_commitment_significance(left, right, commitment_outcomes))
    return frozen.model_copy(
        update={
            "confirmatory_metrics": tuple(audits),
            "confirmatory_passed": all(item.passed for item in audits),
        }
    )


def _audit_selected(
    left: Sequence[CandidatePartnerMetrics],
    right: Sequence[CandidatePartnerMetrics],
    contract: MatchingSpec,
    contrast: str,
) -> tuple[MatchingMetricAudit, ...]:
    results: list[MatchingMetricAudit] = []
    controls: tuple[tuple[str, float], ...] = (
        ("competence", contract.competence_margin),
        ("best_fixed_response_value", contract.fixed_response_margin),
        ("br_prox", contract.br_prox_margin),
        ("lobp_score_nats", contract.predictability_margin_nats),
        ("trajectory_divergence", contract.trajectory_divergence_margin),
        ("commitment_reached_rate", contract.discovery_commitment_rate_margin),
    )
    if contrast == "active_dri":
        controls += (("passive_dri", contract.active_contrast_passive_dri_margin),)
    for field, margin in controls:
        left_value = float(np.mean([float(getattr(item, field)) for item in left]))
        right_value = float(np.mean([float(getattr(item, field)) for item in right]))
        difference = abs(left_value - right_value)
        results.append(
            MatchingMetricAudit(
                metric=field,
                left_value=left_value,
                right_value=right_value,
                difference=difference,
                margin=margin,
                role="control",
                passed=difference <= margin + 1e-12,
                reason=f"absolute mean difference {difference:.6g} <= {margin:.6g}",
            )
        )
    left_clusters = Counter(item.response_cluster for item in left)
    right_clusters = Counter(item.response_cluster for item in right)
    cluster_pass = left_clusters == right_clusters
    results.append(
        MatchingMetricAudit(
            metric="response_cluster_counts",
            left_value=json.dumps(dict(sorted(left_clusters.items()))),
            right_value=json.dumps(dict(sorted(right_clusters.items()))),
            difference=None,
            margin=None,
            role="control",
            passed=cluster_pass,
            reason="response-cluster counts must be identical",
        )
    )
    left_det = _brdiv_logdet(left)
    right_det = _brdiv_logdet(right)
    if math.isinf(left_det) and math.isinf(right_det) and left_det == right_det:
        det_difference = 0.0
    else:
        det_difference = abs(left_det - right_det)
    results.append(
        MatchingMetricAudit(
            metric="normalized_br_div_logdet",
            left_value=left_det,
            right_value=right_det,
            difference=det_difference,
            margin=contract.br_div_logdet_margin,
            role="control",
            passed=det_difference <= contract.br_div_logdet_margin + 1e-12,
            reason="no diagonal jitter; equally singular populations match at -infinity",
        )
    )
    left_treatment = float(np.mean([float(getattr(item, contrast)) for item in left]))
    right_treatment = float(np.mean([float(getattr(item, contrast)) for item in right]))
    separation = abs(left_treatment - right_treatment)
    results.append(
        MatchingMetricAudit(
            metric=contrast,
            left_value=left_treatment,
            right_value=right_treatment,
            difference=separation,
            margin=contract.minimum_dri_separation,
            role="treatment",
            passed=separation + 1e-12 >= contract.minimum_dri_separation,
            reason=(
                f"absolute DRI separation {separation:.6g} >= "
                f"{contract.minimum_dri_separation:.6g}"
            ),
        )
    )
    treatment_interval = _treatment_interval(left, right, contrast)
    results.append(treatment_interval)
    return tuple(results)


def _brdiv_logdet(items: Sequence[CandidatePartnerMetrics]) -> float:
    features = np.asarray([item.br_event_features for item in items], dtype=np.float64)
    gram = features @ features.T
    sign, value = np.linalg.slogdet(gram)
    return float(value) if sign > 0 else -math.inf


def _treatment_interval(
    left: Sequence[CandidatePartnerMetrics],
    right: Sequence[CandidatePartnerMetrics],
    contrast: str,
) -> MatchingMetricAudit:
    left_values = np.asarray([float(getattr(item, contrast)) for item in left])
    right_values = np.asarray([float(getattr(item, contrast)) for item in right])
    difference = float(left_values.mean() - right_values.mean())
    left_variance = float(left_values.var(ddof=1)) if len(left_values) > 1 else 0.0
    right_variance = float(right_values.var(ddof=1)) if len(right_values) > 1 else 0.0
    standard_error = math.sqrt(
        left_variance / len(left_values) + right_variance / len(right_values)
    )
    if standard_error <= 1e-15:
        lower = upper = difference
    else:
        numerator = (left_variance / len(left_values) + right_variance / len(right_values)) ** 2
        denominator = 0.0
        if len(left_values) > 1:
            denominator += (left_variance / len(left_values)) ** 2 / (len(left_values) - 1)
        if len(right_values) > 1:
            denominator += (right_variance / len(right_values)) ** 2 / (len(right_values) - 1)
        degrees_freedom = numerator / denominator if denominator > 0 else math.inf
        try:
            from scipy.stats import t  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional established extra
            raise RuntimeError("DRI matching interval requires the established extra") from exc
        critical = float(t.ppf(0.975, degrees_freedom))
        lower = difference - critical * standard_error
        upper = difference + critical * standard_error
    passed = lower > 0 or upper < 0
    return MatchingMetricAudit(
        metric=f"{contrast}_discovery_interval",
        left_value=lower,
        right_value=upper,
        difference=abs(difference),
        margin=0.0,
        role="treatment",
        passed=passed,
        reason=(
            f"Welch 95% interval [{lower:.6g}, {upper:.6g}] "
            "must exclude zero"
        ),
    )


def _failed_selection(contrast: str, status: str) -> MatchedPopulationAudit:
    payload = f"{contrast}:{status}".encode()
    return MatchedPopulationAudit(
        contrast=contrast,  # type: ignore[arg-type]
        left_partner_ids=(),
        right_partner_ids=(),
        frozen=False,
        discovery_metrics=(),
        discovery_passed=False,
        solver_status=status,
        selection_hash=hashlib.sha256(payload).hexdigest(),
    )


def _commitment_significance(
    left: Sequence[CandidatePartnerMetrics],
    right: Sequence[CandidatePartnerMetrics],
    outcomes: dict[str, Sequence[bool]] | None,
) -> MatchingMetricAudit:
    if outcomes is None:
        return MatchingMetricAudit(
            metric="commitment_reached_rate_significance",
            left_value="missing",
            right_value="missing",
            difference=None,
            margin=0.05,
            role="control",
            passed=False,
            reason="confirmatory episode-level commitment outcomes are required",
        )
    missing = [item.partner_id for item in (*left, *right) if item.partner_id not in outcomes]
    if missing:
        raise ValueError(f"missing commitment outcomes for partners: {missing}")
    left_values = [bool(value) for item in left for value in outcomes[item.partner_id]]
    right_values = [bool(value) for item in right for value in outcomes[item.partner_id]]
    if not left_values or not right_values:
        raise ValueError("commitment significance audit needs episode-level outcomes")
    try:
        from scipy.stats import fisher_exact
    except ImportError as exc:  # pragma: no cover - optional established extra
        raise RuntimeError("commitment matching requires the established extra") from exc
    table = np.asarray(
        [
            [sum(left_values), len(left_values) - sum(left_values)],
            [sum(right_values), len(right_values) - sum(right_values)],
        ]
    )
    p_value = float(fisher_exact(table).pvalue)
    left_rate = float(np.mean(left_values))
    right_rate = float(np.mean(right_values))
    return MatchingMetricAudit(
        metric="commitment_reached_rate_significance",
        left_value=left_rate,
        right_value=right_rate,
        difference=abs(left_rate - right_rate),
        margin=0.05,
        role="control",
        passed=p_value >= 0.05,
        reason=f"two-sided Fisher exact p={p_value:.6g}; preregistered alpha=0.05",
    )
