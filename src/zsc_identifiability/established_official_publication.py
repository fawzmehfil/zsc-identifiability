"""Export compact, path-safe publication artifacts from the completed Stage 6 v3 audit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SOURCE_FILES = {
    "manifest": "official-measurement-audit-manifest-v3.json",
    "regression": "held-out-regression-report-v3.json",
    "intervention": "fresh-intervention-audit-v3.json",
    "permutation": "permutation-report-v3.json",
    "calibration": "measurement-calibration-report-v3.json",
    "sensitivity": "measurement-sensitivity-report-v3.json",
}


def export_official_measurement_publication(
    input_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create compact tables, figures, and a provenance manifest from final v3 outputs."""

    source = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    payloads = {
        key: _read_object(source / filename) for key, filename in _SOURCE_FILES.items()
    }
    _validate_completed_audit(payloads)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    regression_rows = _regression_rows(payloads["regression"])
    intervention_rows = _intervention_rows(payloads["intervention"], payloads["permutation"])
    permutation_rows = _permutation_rows(payloads["permutation"])
    _write_csv(output / "predictive-value.csv", regression_rows)
    _write_csv(output / "intervention-audit.csv", intervention_rows)
    _write_csv(output / "permutation-tests.csv", permutation_rows)

    summary = _summary(payloads, regression_rows, intervention_rows, permutation_rows)
    _write_json(output / "stage-6-summary.json", summary)
    figure_paths = _write_figures(payloads, figure_dir)

    generated = sorted(
        [
            output / "predictive-value.csv",
            output / "intervention-audit.csv",
            output / "permutation-tests.csv",
            output / "stage-6-summary.json",
            *figure_paths,
        ],
        key=lambda path: path.relative_to(output).as_posix(),
    )
    artifact_manifest = {
        "schema_version": 1,
        "source_audit": {
            "status": payloads["manifest"]["status"],
            "verdict": payloads["manifest"]["verdict"],
            "source_hashes": payloads["manifest"]["source_hashes"],
        },
        "source_file_hashes": {
            filename: _sha256(source / filename) for filename in _SOURCE_FILES.values()
        },
        "generated_file_hashes": {
            path.relative_to(output).as_posix(): _sha256(path) for path in generated
        },
        "contains_raw_traces": False,
        "contains_checkpoints": False,
        "contains_machine_specific_paths": False,
    }
    _write_json(output / "artifact-manifest.json", artifact_manifest)
    return artifact_manifest


def _validate_completed_audit(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    manifest = payloads["manifest"]
    if manifest.get("status") != "complete":
        raise ValueError("publication export requires a completed Stage 6 v3 audit")
    if manifest.get("verdict") != "complete_measurement_only":
        raise ValueError("publication export is frozen to the measurement-only verdict")
    if manifest.get("calibration_passed") is not True:
        raise ValueError("publication export requires passed measurement calibration")
    if payloads["calibration"].get("passed") is not True:
        raise ValueError("calibration report does not pass")
    if payloads["regression"].get("event_same_effect_direction") is not True:
        raise ValueError("event sensitivity does not agree with the primary regression direction")


def _regression_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in ("gru", "event"):
        result = _mapping(report[representation])
        scopes = {"overall": result["overall"], **_mapping(result["by_layout"])}
        for scope, metrics_raw in scopes.items():
            metrics = _mapping(metrics_raw)
            folds = [_mapping(item) for item in _sequence(metrics["folds"])]
            rows.append(
                {
                    "representation": representation,
                    "scope": scope,
                    "delta_r2": metrics["delta_r2"],
                    "delta_mae": metrics["delta_mae"],
                    "delta_mse": metrics["delta_mse"],
                    "improved_mae_folds": sum(float(item["delta_mae"]) < 0 for item in folds),
                    "improved_mse_folds": sum(float(item["delta_mse"]) < 0 for item in folds),
                    "positive_delta_r2_folds": sum(
                        float(item["delta_r2"]) > 0 for item in folds
                    ),
                    "fold_count": len(folds),
                }
            )
    return rows


def _intervention_rows(
    intervention: Mapping[str, Any],
    permutation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tests = _mapping(permutation["tests"])
    rows: list[dict[str, Any]] = []
    for item_raw in _sequence(intervention["layouts"]):
        item = _mapping(item_raw)
        layout = str(item["layout_id"])
        test = _mapping(tests[f"{layout}:selected_intervention"])
        rows.append(
            {
                "layout": layout,
                "intervention": item["selected_intervention"],
                "conflicting_pair_count": item["pair_count"],
                "completion_rate": item["completion_rate"],
                "partner_response_tv": item["partner_response_tv"],
                "gru_risk_reduction": item["gru_decision_risk_reduction"],
                "ci_low": item["gru_corrected_ci_low"],
                "ci_high": item["gru_corrected_ci_high"],
                "event_risk_reduction": item["event_decision_risk_reduction"],
                "normalized_task_cost": item["normalized_task_cost"],
                "raw_permutation_p": test["raw_p"],
                "holm_adjusted_p": test["holm_adjusted_p"],
                "confirmed": False,
            }
        )
    return rows


def _permutation_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_id, item_raw in sorted(_mapping(report["tests"]).items()):
        item = _mapping(item_raw)
        null_values = [float(value) for value in _sequence(item["null_values"])]
        ordered = sorted(null_values)
        rows.append(
            {
                "test": test_id,
                "observed": item["observed"],
                "null_mean": sum(null_values) / len(null_values),
                "null_95th_percentile": _quantile(ordered, 0.95),
                "raw_p": item["raw_p"],
                "holm_adjusted_p": item["holm_adjusted_p"],
                "passed": item["passed"],
                "permutations": len(null_values),
            }
        )
    return rows


def _summary(
    payloads: Mapping[str, Mapping[str, Any]],
    regression_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = payloads["manifest"]
    regression = payloads["regression"]
    calibration = payloads["calibration"]
    return {
        "schema_version": 1,
        "working_title": (
            "Can Your Teammate Be Identified in Time? "
            "Decision-Sufficient Identifiability in Zero-Shot Coordination"
        ),
        "verdict": manifest["verdict"],
        "supported_result": (
            "Pre-commitment DRI adds held-out predictive value for normalized "
            "response-library regret in both official layouts."
        ),
        "unsupported_result": (
            "Neither preregistered natural task intervention produced a confirmed "
            "decision-risk improvement."
        ),
        "audit_scale": {
            "official_partners": 50,
            "layouts": 2,
            "official_zsc_method_families": 6,
            "fresh_confirmation_episodes": manifest["total_fresh_episodes"],
            "fresh_confirmation_environment_steps": manifest[
                "total_fresh_environment_steps"
            ],
            "policy_training_performed": False,
        },
        "regression": list(regression_rows),
        "dri_coefficient_intervals": {
            representation: _mapping(regression[representation])["dri_coefficient_interval"]
            for representation in ("gru", "event")
        },
        "permutation_tests": list(permutation_rows),
        "interventions": list(intervention_rows),
        "calibration": {
            "passed": calibration["passed"],
            "synthetic_controls": calibration["synthetic_controls"],
            "brier_checks": calibration["brier_checks"],
            "fixed_response_checks": calibration["fixed_response_checks"],
            "leakage_checks": calibration["leakage_checks"],
            "seed_stability": calibration["seed_stability"],
            "direct_binary_gru_diagnostic": {
                key: calibration["direct_binary_gru_diagnostic"][key]
                for key in (
                    "pair_count",
                    "dri_correlation",
                    "mean_decision_loss_difference",
                    "mean_seed_dispersion",
                    "status",
                )
            },
        },
        "scientific_gates": manifest["scientific_gates"],
        "source_hashes": manifest["source_hashes"],
    }


def _write_figures(
    payloads: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
            "figure.dpi": 120,
        }
    )
    colors = {"gru": "#2563eb", "event": "#f59e0b"}
    regression = payloads["regression"]
    generated: list[Path] = []

    scopes = ("overall", "random3_m", "small_corridor")
    labels = ("Overall", "Random3-m", "Small corridor")
    x = np.arange(len(scopes))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for offset, representation in zip((-width / 2, width / 2), ("gru", "event"), strict=True):
        result = _mapping(regression[representation])
        values = [
            float(
                _mapping(result["overall"] if scope == "overall" else result["by_layout"][scope])[
                    "delta_r2"
                ]
            )
            for scope in scopes
        ]
        axes[0].bar(
            x + offset,
            values,
            width,
            label=representation.upper(),
            color=colors[representation],
        )
        interval = _mapping(result["dri_coefficient_interval"])
        point = float(interval["coefficient_point"])
        axes[1].errorbar(
            point,
            representation.upper(),
            xerr=[[point - float(interval["ci_low"])], [float(interval["ci_high"]) - point]],
            fmt="o",
            capsize=4,
            color=colors[representation],
        )
    axes[0].axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Held-out $\\Delta R^2$")
    axes[0].set_title("Predictive gain from adding DRI")
    axes[0].legend(frameon=False)
    axes[1].axvline(0, color="#111827", linewidth=0.8)
    axes[1].set_xlabel("DRI coefficient for regret (95% CI)")
    axes[1].set_title("Higher DRI predicts lower regret")
    figure.tight_layout()
    generated.extend(_save_figure_pair(figure, output / "dri-predictive-value"))
    plt.close(figure)

    tests = _mapping(payloads["permutation"]["tests"])
    passive_ids = ("random3_m:passive_dri", "small_corridor:passive_dri")
    figure, axis = plt.subplots(figsize=(6.8, 3.8))
    null_values = [
        [float(value) for value in _sequence(_mapping(tests[test_id])["null_values"])]
        for test_id in passive_ids
    ]
    observed = [float(_mapping(tests[test_id])["observed"]) for test_id in passive_ids]
    axis.boxplot(null_values, tick_labels=("Random3-m", "Small corridor"), showfliers=False)
    axis.scatter((1, 2), observed, color="#dc2626", zorder=3, label="Observed DRI")
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.set_ylabel("Mean pairwise pre-commitment DRI")
    axis.set_title("Passive DRI exceeds the registered permutation null")
    axis.legend(frameon=False)
    figure.tight_layout()
    generated.extend(_save_figure_pair(figure, output / "passive-dri-permutation"))
    plt.close(figure)

    layouts = [_mapping(item) for item in _sequence(payloads["intervention"]["layouts"])]
    points = np.asarray([float(item["gru_decision_risk_reduction"]) for item in layouts])
    lower = np.asarray([float(item["gru_corrected_ci_low"]) for item in layouts])
    upper = np.asarray([float(item["gru_corrected_ci_high"]) for item in layouts])
    costs = np.asarray([float(item["normalized_task_cost"]) for item in layouts])
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(layouts))
    axis.errorbar(
        positions,
        points,
        yerr=np.vstack((points - lower, upper - points)),
        fmt="o",
        capsize=5,
        color="#2563eb",
        label="GRU risk reduction (corrected 95% CI)",
    )
    axis.scatter(positions, costs, marker="D", color="#f59e0b", label="Normalized task cost")
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.set_xticks(positions, ("Random3-m", "Small corridor"))
    axis.set_ylabel("Normalized decision value")
    axis.set_title("Preregistered natural interventions did not confirm")
    axis.legend(frameon=False)
    figure.tight_layout()
    generated.extend(_save_figure_pair(figure, output / "natural-intervention-audit"))
    plt.close(figure)
    return tuple(generated)


def _save_figure_pair(figure: Any, stem: Path) -> tuple[Path, Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "zsc-identifiability"},
    )
    figure.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "zsc-identifiability"},
    )
    return pdf, png


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty publication table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required Stage 6 output is missing: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path.name}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping in Stage 6 publication source")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("expected sequence in Stage 6 publication source")
    return value


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
