from __future__ import annotations

import json
from pathlib import Path

from zsc_identifiability.cli import main
from zsc_identifiability.established_runner import execute_established_audit

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "phase-6-established-validation/suites/full-scale-overcookedv2.json"


def test_established_validate_distinguishes_valid_schema_from_missing_runtime(
    capsys, tmp_path: Path
) -> None:
    code = main(
        (
            "established",
            "validate",
            "--suite",
            str(SUITE),
            "--project-root",
            str(tmp_path),
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 4
    assert payload["valid"] is True
    assert payload["metric_calibration"]["passed"] is True
    assert payload["runtime_ready"] is False


def test_exit_audit_is_incomplete_before_confirmatory_artifacts(tmp_path: Path) -> None:
    manifest = execute_established_audit(
        SUITE,
        tmp_path / "artifacts",
        state_dir=tmp_path / "state",
        project_root=tmp_path,
    )
    assert manifest.status == "incomplete"
    assert manifest.scientific_verdict == "pending"
    assert manifest.metric_calibration_passed
    assert manifest.missing_assets
    assert (tmp_path / "artifacts/dri-estimator-calibration.json").is_file()
