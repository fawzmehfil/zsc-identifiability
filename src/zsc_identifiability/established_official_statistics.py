"""Preregistered scheme-level predictive analysis for Stage 6 v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_CONTROLS = (
    "partner_competence",
    "prior_confusion_risk",
    "conflict_coefficient",
    "rahman_brdiv_return",
    "zsceval_br_div_raw",
    "visible_action_predictability",
    "prefix_tv",
)


def nested_leave_one_scheme_out_regression(
    rows: Sequence[Mapping[str, Any]],
    *,
    ridge_strengths: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 10.0),
) -> dict[str, Any]:
    """Compare controls with controls+DRI using fully nested scheme folds."""

    return nested_leave_one_scheme_out_feature_regression(
        rows,
        incremental_feature="precommitment_dri",
        ridge_strengths=ridge_strengths,
    )


def nested_leave_one_scheme_out_feature_regression(
    rows: Sequence[Mapping[str, Any]],
    *,
    incremental_feature: str,
    ridge_strengths: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 10.0),
) -> dict[str, Any]:
    """Compare registered controls with one candidate feature on held-out schemes.

    If the candidate is already a registered control (for example prefix TV), it is
    removed from the reduced model. This makes the sensitivity analysis a genuine
    replacement for DRI instead of adding a duplicate, collinear column.
    """

    if not rows:
        raise ValueError("official regression requires pairwise method rows")
    controls = tuple(field for field in _CONTROLS if field != incremental_feature)
    required = {
        "normalized_response_library_regret",
        "left_scheme_id",
        "right_scheme_id",
        "method_id",
        "layout_id",
        incremental_feature,
        *controls,
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("official regression rows omit preregistered fields")
    schemes = tuple(
        sorted({str(row[field]) for row in rows for field in ("left_scheme_id", "right_scheme_id")})
    )
    methods = tuple(sorted({str(row["method_id"]) for row in rows}))
    layouts = tuple(sorted({str(row["layout_id"]) for row in rows}))
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for scheme in schemes:
        test_indices = [
            index
            for index, row in enumerate(rows)
            if scheme in {str(row["left_scheme_id"]), str(row["right_scheme_id"])}
        ]
        train_indices = [index for index in range(len(rows)) if index not in test_indices]
        if not test_indices or not train_indices:
            continue
        baseline_alpha = _inner_alpha(
            rows,
            train_indices,
            schemes,
            methods,
            layouts,
            ridge_strengths,
            controls,
            incremental_feature,
            include_dri=False,
        )
        full_alpha = _inner_alpha(
            rows,
            train_indices,
            schemes,
            methods,
            layouts,
            ridge_strengths,
            controls,
            incremental_feature,
            include_dri=True,
        )
        baseline = _fit_predict(
            rows,
            train_indices,
            test_indices,
            methods,
            layouts,
            baseline_alpha,
            controls,
            incremental_feature,
            False,
        )
        full = _fit_predict(
            rows,
            train_indices,
            test_indices,
            methods,
            layouts,
            full_alpha,
            controls,
            incremental_feature,
            True,
        )
        truth = np.asarray(
            [float(rows[index]["normalized_response_library_regret"]) for index in test_indices]
        )
        prediction_rows.extend(
            {
                "held_out_scheme": scheme,
                "row_index": index,
                "truth": float(truth[position]),
                "baseline_prediction": float(baseline[position]),
                "full_prediction": float(full[position]),
            }
            for position, index in enumerate(test_indices)
        )
        baseline_metrics = _metrics(truth, baseline)
        full_metrics = _metrics(truth, full)
        fold_rows.append(
            {
                "held_out_scheme": scheme,
                "row_count": len(test_indices),
                "baseline_alpha": baseline_alpha,
                "full_alpha": full_alpha,
                "baseline": baseline_metrics,
                "full": full_metrics,
                "delta_mae": full_metrics["mae"] - baseline_metrics["mae"],
                "delta_mse": full_metrics["mse"] - baseline_metrics["mse"],
                "delta_r2": full_metrics["r2"] - baseline_metrics["r2"],
            }
        )
    if not prediction_rows:
        raise ValueError("leave-one-scheme-out analysis produced no valid folds")
    truth = np.asarray([float(row["truth"]) for row in prediction_rows])
    baseline_predictions = np.asarray(
        [float(row["baseline_prediction"]) for row in prediction_rows]
    )
    full_predictions = np.asarray([float(row["full_prediction"]) for row in prediction_rows])
    baseline_metrics = _metrics(truth, baseline_predictions)
    full_metrics = _metrics(truth, full_predictions)
    return {
        "schema_version": 1,
        "held_out_unit": "hsp_scheme",
        "incremental_feature": incremental_feature,
        "baseline_features": list(controls) + ["method", "layout"],
        "full_features": list(controls)
        + ["method", "layout", incremental_feature, f"method_x_{incremental_feature}"],
        "br_prox_used_as_predictor": False,
        "baseline": baseline_metrics,
        "full": full_metrics,
        "delta_mae": full_metrics["mae"] - baseline_metrics["mae"],
        "delta_mse": full_metrics["mse"] - baseline_metrics["mse"],
        "delta_r2": full_metrics["r2"] - baseline_metrics["r2"],
        "incremental_value": (
            full_metrics["mae"] < baseline_metrics["mae"]
            and full_metrics["mse"] < baseline_metrics["mse"]
        ),
        "folds": fold_rows,
        "negative_folds": [
            row["held_out_scheme"]
            for row in fold_rows
            if row["delta_mae"] >= 0 or row["delta_mse"] >= 0
        ],
        "predictions": prediction_rows,
    }


def clustered_dri_coefficient_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    resamples: int = 10_000,
    seed: int = 6173,
) -> dict[str, Any]:
    """Adjusted three-way cluster bootstrap over method seed, scheme, and episode key."""

    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    required = {
        "method_seed",
        "left_scheme_id",
        "right_scheme_id",
        "environment_key",
        "normalized_response_library_regret",
        "precommitment_dri",
        "method_id",
        "layout_id",
        *_CONTROLS,
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("clustered bootstrap rows omit required cluster identifiers")
    rng = np.random.default_rng(seed)
    method_seeds = tuple(sorted({f"{row['method_id']}:{row['method_seed']}" for row in rows}))
    schemes = tuple(
        sorted({str(row[field]) for row in rows for field in ("left_scheme_id", "right_scheme_id")})
    )
    episode_keys = tuple(sorted({str(row["environment_key"]) for row in rows}))
    methods = tuple(sorted({str(row["method_id"]) for row in rows}))
    layouts = tuple(sorted({str(row["layout_id"]) for row in rows}))
    continuous = np.asarray(
        [[float(row[field]) for field in _CONTROLS] for row in rows], dtype=np.float64
    )
    scales = continuous.std(axis=0)
    scales[scales < 1e-12] = 1.0
    continuous = (continuous - continuous.mean(axis=0)) / scales
    dri = np.asarray([float(row["precommitment_dri"]) for row in rows])
    method_dummies = np.asarray(
        [[float(row["method_id"] == method) for method in methods[1:]] for row in rows]
    )
    layout_dummies = np.asarray(
        [[float(row["layout_id"] == layout) for layout in layouts[1:]] for row in rows]
    )
    interactions = method_dummies * dri[:, None]
    nuisance = np.column_stack(
        (
            np.ones(len(rows)),
            continuous,
            method_dummies,
            layout_dummies,
            interactions,
        )
    )
    outcome = np.asarray([float(row["normalized_response_library_regret"]) for row in rows])
    dri_residual = dri - nuisance @ np.linalg.lstsq(nuisance, dri, rcond=None)[0]
    outcome_residual = outcome - nuisance @ np.linalg.lstsq(nuisance, outcome, rcond=None)[0]
    denominator = float(dri_residual @ dri_residual)
    if denominator <= 1e-15:
        raise ValueError("adjusted DRI coefficient is not identifiable")
    point = float(dri_residual @ outcome_residual / denominator)
    method_index = {value: index for index, value in enumerate(method_seeds)}
    scheme_index = {value: index for index, value in enumerate(schemes)}
    episode_index = {value: index for index, value in enumerate(episode_keys)}
    numerator_tensor = np.zeros(
        (len(method_seeds), len(schemes), len(episode_keys)), dtype=np.float64
    )
    denominator_tensor = np.zeros_like(numerator_tensor)
    for index, row in enumerate(rows):
        method_key = f"{row['method_id']}:{row['method_seed']}"
        episode_key = str(row["environment_key"])
        for scheme in (str(row["left_scheme_id"]), str(row["right_scheme_id"])):
            location = (
                method_index[method_key],
                scheme_index[scheme],
                episode_index[episode_key],
            )
            numerator_tensor[location] += 0.5 * dri_residual[index] * outcome_residual[index]
            denominator_tensor[location] += 0.5 * dri_residual[index] ** 2
    coefficients: list[float] = []
    probabilities = (
        np.full(len(method_seeds), 1.0 / len(method_seeds)),
        np.full(len(schemes), 1.0 / len(schemes)),
        np.full(len(episode_keys), 1.0 / len(episode_keys)),
    )
    completed = 0
    batch_size = 128
    while completed < resamples:
        batch = min(batch_size, resamples - completed)
        method_weights = rng.multinomial(len(method_seeds), probabilities[0], size=batch)
        scheme_weights = rng.multinomial(len(schemes), probabilities[1], size=batch)
        episode_weights = rng.multinomial(len(episode_keys), probabilities[2], size=batch)
        numerators = np.einsum(
            "bi,bj,bk,ijk->b",
            method_weights,
            scheme_weights,
            episode_weights,
            numerator_tensor,
            optimize=True,
        )
        denominators = np.einsum(
            "bi,bj,bk,ijk->b",
            method_weights,
            scheme_weights,
            episode_weights,
            denominator_tensor,
            optimize=True,
        )
        coefficients.extend(
            float(numerator / weighted_denominator)
            for numerator, weighted_denominator in zip(numerators, denominators, strict=True)
            if weighted_denominator > 1e-15
        )
        completed += batch
    if not coefficients:
        raise ValueError("cluster bootstrap produced no estimable samples")
    values = np.asarray(coefficients)
    return {
        "resamples_requested": resamples,
        "resamples_completed": len(coefficients),
        "coefficient_point": point,
        "coefficient_mean": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "adjusted_for_registered_controls": True,
        "method_by_dri_interactions_in_nuisance_model": True,
        "cluster_dimensions": ["method_seed", "hsp_scheme", "episode_key"],
    }


def _inner_alpha(
    rows: Sequence[Mapping[str, Any]],
    outer_train: Sequence[int],
    schemes: Sequence[str],
    methods: Sequence[str],
    layouts: Sequence[str],
    strengths: Sequence[float],
    controls: Sequence[str],
    incremental_feature: str,
    *,
    include_dri: bool,
) -> float:
    candidates: list[tuple[float, float]] = []
    for alpha in strengths:
        errors: list[float] = []
        for scheme in schemes:
            validation = [
                index
                for index in outer_train
                if scheme
                in {
                    str(rows[index]["left_scheme_id"]),
                    str(rows[index]["right_scheme_id"]),
                }
            ]
            training = [index for index in outer_train if index not in validation]
            if not validation or not training:
                continue
            prediction = _fit_predict(
                rows,
                training,
                validation,
                methods,
                layouts,
                float(alpha),
                controls,
                incremental_feature,
                include_dri,
            )
            truth = np.asarray(
                [float(rows[index]["normalized_response_library_regret"]) for index in validation]
            )
            errors.append(float(np.mean((truth - prediction) ** 2)))
        candidates.append((float(np.mean(errors)) if errors else float("inf"), float(alpha)))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _fit_predict(
    rows: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    methods: Sequence[str],
    layouts: Sequence[str],
    alpha: float,
    controls: Sequence[str],
    incremental_feature: str,
    include_dri: bool,
) -> np.ndarray:
    train_continuous = np.asarray(
        [[float(rows[index][field]) for field in controls] for index in train_indices]
    )
    means = train_continuous.mean(axis=0)
    scales = train_continuous.std(axis=0)
    scales[scales < 1e-12] = 1.0
    train_x = _design(
        rows,
        train_indices,
        methods,
        layouts,
        means,
        scales,
        controls,
        incremental_feature,
        include_dri,
    )
    test_x = _design(
        rows,
        test_indices,
        methods,
        layouts,
        means,
        scales,
        controls,
        incremental_feature,
        include_dri,
    )
    train_y = np.asarray(
        [float(rows[index]["normalized_response_library_regret"]) for index in train_indices]
    )
    penalty = np.eye(train_x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.pinv(train_x.T @ train_x + penalty) @ train_x.T @ train_y
    return np.asarray(test_x @ coefficient, dtype=np.float64)


def _design(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    methods: Sequence[str],
    layouts: Sequence[str],
    means: np.ndarray,
    scales: np.ndarray,
    controls: Sequence[str],
    incremental_feature: str,
    include_dri: bool,
) -> np.ndarray:
    values: list[list[float]] = []
    for index in indices:
        row = rows[index]
        continuous = (np.asarray([float(row[field]) for field in controls]) - means) / scales
        method_values = [float(row["method_id"] == method) for method in methods[1:]]
        layout_values = [float(row["layout_id"] == layout) for layout in layouts[1:]]
        result = [1.0, *continuous.tolist(), *method_values, *layout_values]
        if include_dri:
            feature = float(row[incremental_feature])
            result.extend((feature, *(feature * value for value in method_values)))
        values.append(result)
    return np.asarray(values, dtype=np.float64)


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = truth - prediction
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": float(np.mean(residual**2)),
        "r2": 0.0 if denominator <= 1e-15 else 1.0 - float(np.sum(residual**2)) / denominator,
    }
