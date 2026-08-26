"""Approximate response-library construction and response-conflict audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from zsc_identifiability.established_models import ResponseLibrary


def build_response_library_from_values(
    values: Mapping[str, Mapping[str, float]],
    *,
    adequacy_margin: float = 0.02,
    response_clusters: Mapping[str, int] | None = None,
) -> ResponseLibrary:
    """Build the fixed empirical loss matrix used by all DRI estimators.

    Rows are frozen partners and columns are response-library policies.  The
    resulting oracle is deliberately labelled `response_library_oracle` rather
    than a globally optimal oracle.
    """

    if not values:
        raise ValueError("response-library values cannot be empty")
    partner_ids = tuple(sorted(values))
    response_ids = tuple(sorted(next(iter(values.values()))))
    if not response_ids:
        raise ValueError("response library must include at least one response")
    for partner_id, row in values.items():
        if tuple(sorted(row)) != response_ids:
            raise ValueError(f"response columns differ for partner {partner_id!r}")
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError("response values must be finite")
    matrix = np.asarray(
        [[float(values[partner][response]) for response in response_ids] for partner in partner_ids]
    )
    maxima = matrix.max(axis=1, keepdims=True)
    losses = maxima - matrix
    adequate: dict[str, tuple[str, ...]] = {}
    for index, partner in enumerate(partner_ids):
        adequate[partner] = tuple(
            response_ids[column]
            for column in range(len(response_ids))
            if losses[index, column] <= adequacy_margin + 1e-12
        )
    conflicts = tuple(
        (left, right)
        for left_index, left in enumerate(partner_ids)
        for right in partner_ids[left_index + 1 :]
        if not set(adequate[left]) & set(adequate[right])
    )
    clusters = (
        {partner: index for index, partner in enumerate(partner_ids)}
        if response_clusters is None
        else {partner: int(response_clusters[partner]) for partner in partner_ids}
    )
    return ResponseLibrary(
        partner_ids=partner_ids,
        response_ids=response_ids,
        normalized_values=tuple(tuple(float(value) for value in row) for row in matrix),
        loss_matrix=tuple(tuple(float(value) for value in row) for row in losses),
        adequate_responses=adequate,
        response_clusters=clusters,
        response_conflicts=conflicts,
        adequacy_margin=adequacy_margin,
    )


def known_partner_competence(library: ResponseLibrary) -> dict[str, float]:
    return {
        partner: max(library.normalized_values[index])
        for index, partner in enumerate(library.partner_ids)
    }


def best_fixed_response_value(
    library: ResponseLibrary,
    prior: Sequence[float] | None = None,
) -> float:
    weights = _prior(len(library.partner_ids), prior)
    values = np.asarray(library.normalized_values, dtype=np.float64)
    return float(np.max(weights @ values))


def br_prox(
    library: ResponseLibrary,
    response_id: str,
    prior: Sequence[float] | None = None,
) -> float:
    weights = _prior(len(library.partner_ids), prior)
    column = library.response_ids.index(response_id)
    values = np.asarray(library.normalized_values, dtype=np.float64)
    denominators = values.max(axis=1)
    ratios = np.divide(
        values[:, column],
        denominators,
        out=np.zeros_like(denominators),
        where=np.abs(denominators) > 1e-12,
    )
    return float(weights @ ratios)


def _prior(size: int, prior: Sequence[float] | None) -> np.ndarray:
    result = np.full(size, 1.0 / size) if prior is None else np.asarray(prior, dtype=np.float64)
    if result.shape != (size,) or np.any(result < 0) or not np.isclose(result.sum(), 1.0):
        raise ValueError("invalid partner prior")
    return result
