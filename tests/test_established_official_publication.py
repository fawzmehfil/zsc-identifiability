from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zsc_identifiability.established_official_publication import (
    export_official_measurement_publication,
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sources(root: Path) -> None:
    folds = [
        {"delta_mae": -0.1, "delta_mse": -0.05, "delta_r2": 0.2},
        {"delta_mae": 0.01, "delta_mse": -0.01, "delta_r2": 0.01},
    ]
    metric = {"delta_r2": 0.1, "delta_mae": -0.02, "delta_mse": -0.01, "folds": folds}
    interval = {
        "coefficient_point": -0.2,
        "coefficient_mean": -0.21,
        "ci_low": -0.3,
        "ci_high": -0.1,
    }
    regression = {
        "schema_version": 3,
        "event_same_effect_direction": True,
        "gru": {
            "overall": metric,
            "by_layout": {"random3_m": metric, "small_corridor": metric},
            "dri_coefficient_interval": interval,
        },
        "event": {
            "overall": metric,
            "by_layout": {"random3_m": metric, "small_corridor": metric},
            "dri_coefficient_interval": interval,
        },
    }
    intervention = {
        "schema_version": 3,
        "layouts": [
            {
                "layout_id": layout,
                "selected_intervention": option,
                "pair_count": 10,
                "completion_rate": 1.0,
                "partner_response_tv": 0.1,
                "gru_decision_risk_reduction": effect,
                "gru_corrected_ci_low": -0.01,
                "gru_corrected_ci_high": 0.02,
                "event_decision_risk_reduction": effect,
                "normalized_task_cost": 0.01,
            }
            for layout, option, effect in (
                ("random3_m", "temporary_role_takeover", 0.005),
                ("small_corridor", "corridor_yield", -0.005),
            )
        ],
    }
    tests = {}
    for layout in ("random3_m", "small_corridor"):
        tests[f"{layout}:passive_dri"] = {
            "observed": 0.6,
            "null_values": [-0.1, -0.08, -0.12],
            "raw_p": 0.01,
            "holm_adjusted_p": 0.04,
            "passed": True,
        }
        tests[f"{layout}:selected_intervention"] = {
            "observed": 0.0,
            "null_values": [-0.01, 0.0, 0.01],
            "raw_p": 0.5,
            "holm_adjusted_p": 1.0,
            "passed": False,
        }
    calibration = {
        "schema_version": 3,
        "passed": True,
        "synthetic_controls": {"passed": True},
        "brier_checks": {"random3_m": True, "small_corridor": True},
        "fixed_response_checks": {"random3_m": True, "small_corridor": True},
        "leakage_checks": {"fresh_confirmation_not_used_for_tuning": True},
        "seed_stability": {"random3_m": {"passed": True}},
        "direct_binary_gru_diagnostic": {
            "pair_count": 10,
            "dri_correlation": 0.98,
            "mean_decision_loss_difference": -0.01,
            "mean_seed_dispersion": 0.03,
            "status": "complete",
        },
    }
    manifest = {
        "schema_version": 3,
        "status": "complete",
        "verdict": "complete_measurement_only",
        "calibration_passed": True,
        "scientific_gates": {"primary_intervention_confirmed": False},
        "source_hashes": {"suite": "abc"},
        "total_fresh_episodes": 9600,
        "total_fresh_environment_steps": 3840000,
    }
    _write(root / "official-measurement-audit-manifest-v3.json", manifest)
    _write(root / "held-out-regression-report-v3.json", regression)
    _write(root / "fresh-intervention-audit-v3.json", intervention)
    _write(root / "permutation-report-v3.json", {"schema_version": 3, "tests": tests})
    _write(root / "measurement-calibration-report-v3.json", calibration)
    _write(root / "measurement-sensitivity-report-v3.json", {"schema_version": 3})


def test_publication_export_is_compact_path_safe_and_claim_faithful(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "publication"
    source.mkdir()
    _sources(source)

    manifest = export_official_measurement_publication(source, output)
    summary = json.loads((output / "stage-6-summary.json").read_text(encoding="utf-8"))

    assert summary["verdict"] == "complete_measurement_only"
    assert summary["scientific_gates"]["primary_intervention_confirmed"] is False
    assert len(summary["regression"]) == 6
    assert all(row["confirmed"] is False for row in summary["interventions"])
    assert manifest["contains_machine_specific_paths"] is False
    assert not any(
        str(tmp_path) in path.read_text(errors="ignore") for path in output.glob("*.json")
    )
    assert (output / "figures" / "dri-predictive-value.pdf").is_file()
    assert (output / "figures" / "natural-intervention-audit.png").is_file()
