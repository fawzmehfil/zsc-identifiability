"""Restricted empirical active-identifiability option audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from zsc_identifiability.established_models import (
    DiagnosticActionAudit,
    DiagnosticOptionResult,
    EmpiricalFrontierPoint,
)


def audit_diagnostic_options(
    layout_id: str,
    measurements: Sequence[Mapping[str, object]],
) -> DiagnosticActionAudit:
    """Classify ordinary task options without treating recipe information as partner evidence."""

    results: list[DiagnosticOptionResult] = []
    for item in measurements:
        option = str(item["option_id"])
        passive_dri = _optional_float(item.get("passive_dri"))
        option_dri = _optional_float(item.get("option_dri"))
        cost = _required_float(item, "expected_cost")
        risk_reduction = _required_float(item, "expected_risk_reduction")
        recipe_only = bool(item.get("recipe_prediction_only", False))
        before = bool(item["completes_before_commitment"])
        response_tv = _required_float(item, "conflicting_mode_response_tv")
        universal = bool(item["universal_response_succeeds"])
        dri_gain = (
            option_dri is not None
            and passive_dri is not None
            and option_dri > passive_dri + 1e-12
        )
        qualifies = bool(
            option != "recipe_button_control"
            and before
            and response_tv > 0
            and dri_gain
            and not recipe_only
            and cost > 0
            and not universal
        )
        results.append(
            DiagnosticOptionResult(
                option_id=option,  # type: ignore[arg-type]
                completes_before_commitment=before,
                conflicting_mode_response_tv=response_tv,
                passive_dri=passive_dri,
                option_dri=option_dri,
                recipe_prediction_only=recipe_only,
                expected_cost=cost,
                expected_risk_reduction=risk_reduction,
                universal_response_succeeds=universal,
                qualifying_partner_diagnostic=qualifies,
                net_value=risk_reduction - cost,
            )
        )
    qualifying = [item for item in results if item.qualifying_partner_diagnostic]
    selected = (
        max(qualifying, key=lambda item: (item.net_value, -item.expected_cost, item.option_id))
        if qualifying
        else None
    )
    deterministic = _deterministic_frontier(results)
    return DiagnosticActionAudit(
        layout_id=layout_id,
        results=tuple(results),
        selected_option=None if selected is None else selected.option_id,
        deterministic_frontier=deterministic,
        convexified_frontier=_convexified_frontier(deterministic),
        passed=selected is not None,
        verdict="qualifying_option_found" if selected is not None else "redesign",
    )


def _deterministic_frontier(
    results: Sequence[DiagnosticOptionResult],
) -> tuple[EmpiricalFrontierPoint, ...]:
    candidates = [
        (item.option_id, item.expected_cost, item.option_dri)
        for item in results
        if item.option_dri is not None and not item.recipe_prediction_only
    ]
    frontier: list[EmpiricalFrontierPoint] = []
    for option, cost, dri in sorted(candidates, key=lambda item: (item[1], -float(item[2]))):
        assert dri is not None
        dominated = any(
            other_cost <= cost and float(other_dri) >= dri
            for _, other_cost, other_dri in candidates
            if (other_cost, other_dri) != (cost, dri)
        )
        if not dominated:
            frontier.append(
                EmpiricalFrontierPoint(
                    source=option,
                    expected_cost=cost,
                    dri=dri,
                    deterministic=True,
                )
            )
    return tuple(frontier)


def _convexified_frontier(
    points: Sequence[EmpiricalFrontierPoint],
) -> tuple[EmpiricalFrontierPoint, ...]:
    """Return hull vertices; linear interpolation represents episode-level mixtures."""

    ordered = sorted(points, key=lambda item: (item.expected_cost, item.dri))
    hull: list[EmpiricalFrontierPoint] = []
    for point in ordered:
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            left_slope = (second.dri - first.dri) / max(
                second.expected_cost - first.expected_cost, 1e-15
            )
            right_slope = (point.dri - second.dri) / max(
                point.expected_cost - second.expected_cost, 1e-15
            )
            if right_slope >= left_slope - 1e-12:
                hull.pop()
            else:
                break
        hull.append(point)
    mixtures = [
        EmpiricalFrontierPoint(
            source=f"mixture:{left.source}:{right.source}:0.5",
            expected_cost=0.5 * (left.expected_cost + right.expected_cost),
            dri=0.5 * (left.dri + right.dri),
            deterministic=False,
            mixture=(left.source, right.source, 0.5),
        )
        for left, right in zip(hull, hull[1:], strict=False)
    ]
    return tuple(sorted((*hull, *mixtures), key=lambda item: item.expected_cost))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        raise ValueError("expected a numeric value")
    return float(value)


def _required_float(item: Mapping[str, object], key: str) -> float:
    value = item[key]
    if not isinstance(value, str | int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)
