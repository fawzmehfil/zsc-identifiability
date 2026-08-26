"""Stage 6 validation, audit assembly, and explicit verdict logic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from zsc_identifiability.established_dri import synthetic_dri_calibration
from zsc_identifiability.established_models import (
    DiagnosticActionAudit,
    EstablishedAuditManifest,
    Stage6Verdict,
    load_established_suite_file,
)
from zsc_identifiability.established_runtime import validate_upstreams


def validate_established_configuration(
    suite_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(suite_path).resolve()
    suite = load_established_suite_file(source)
    root = _project_root(project_root)
    upstream = validate_upstreams(suite, root)
    calibration = synthetic_dri_calibration()
    return {
        "valid": True,
        "suite_id": suite.suite_id,
        "schema_version": suite.schema_version,
        "upstream_audit": upstream.to_dict(),
        "metric_calibration": calibration,
        "runtime_ready": upstream.passed,
        "scientific_analysis_ready": bool(calibration["passed"]),
    }


def execute_established_audit(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    state_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> EstablishedAuditManifest:
    source = Path(suite_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = load_established_suite_file(source)
    root = _project_root(project_root)
    state = (
        (root / suite.state_directory).resolve()
        if state_dir is None
        else Path(state_dir).resolve()
    )
    upstream = validate_upstreams(suite, root)
    calibration = synthetic_dri_calibration()
    _write_json(output / "upstream-audit.json", upstream.to_dict())
    _write_json(output / "dri-estimator-calibration.json", calibration)

    missing: list[str] = list(upstream.missing_required_assets)
    matched_payload = _read_optional(state / "matched-population-audit.json", missing)
    diagnostic_payload = _read_optional(state / "diagnostic-action-audit.json", missing)
    replication_payload = _read_optional(state / "established-replication-audit.json", missing)
    incremental_payload = _read_optional(state / "incremental-dri-regression.json", missing)
    active_gap_payload = _read_optional(state / "active-gap-audit.json", missing)
    secondary_payload = _read_optional(state / "secondary-zsceval-audit.json", None)

    matched_passed = _matched_passed(matched_payload)
    diagnostic_passed = _diagnostic_passed(diagnostic_payload)
    replication_passed = _bool_field(replication_payload, "passed")
    incremental_passed = _bool_field(incremental_payload, "incremental_value")
    active_gap = _bool_field(active_gap_payload, "robust_active_gap")
    secondary_status = (
        "secondary_unavailable"
        if secondary_payload is None
        else str(secondary_payload.get("status", "complete"))
    )
    if secondary_status not in {"complete", "secondary_unavailable", "pending"}:
        raise ValueError("invalid secondary ZSC-Eval status")

    core_values = (
        matched_passed,
        diagnostic_passed,
        replication_passed,
        incremental_passed,
        active_gap,
    )
    complete = upstream.passed and bool(calibration["passed"]) and all(
        value is not None for value in core_values
    )
    verdict: Stage6Verdict = "pending"
    status = "incomplete"
    if complete:
        status = "complete"
        if not matched_passed or not diagnostic_passed or not replication_passed:
            verdict = "redesign"
        elif not incremental_passed:
            verdict = "stop"
        elif active_gap:
            verdict = "reopen_phase5"
        else:
            verdict = "complete_evaluation_only"
    manifest = EstablishedAuditManifest(
        suite_id=suite.suite_id,
        status=status,  # type: ignore[arg-type]
        scientific_verdict=verdict,
        upstream_audit_passed=upstream.passed,
        metric_calibration_passed=bool(calibration["passed"]),
        matched_population_passed=matched_passed,
        diagnostic_action_passed=diagnostic_passed,
        established_replication_passed=replication_passed,
        incremental_dri_value_passed=incremental_passed,
        secondary_status=secondary_status,  # type: ignore[arg-type]
        generated_files=(),
        missing_assets=tuple(sorted(set(missing))),
        configuration_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_tree_hash=_source_hash(root),
        invoked_command=(
            f"zsc-identifiability established audit --suite {source} --output {output}",
        ),
    )
    _write_json(output / "manifest.json", manifest.to_dict())
    generated = tuple(
        str(path.relative_to(output))
        for path in sorted(output.rglob("*"))
        if path.is_file()
    )
    manifest = manifest.model_copy(update={"generated_files": generated})
    _write_json(output / "manifest.json", manifest.to_dict())
    return manifest


def _read_optional(path: Path, missing: list[str] | None) -> dict[str, Any] | None:
    if not path.is_file():
        if missing is not None:
            missing.append(str(path))
        return None
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"audit artifact must be a JSON object: {path}")
    return payload


def _matched_passed(payload: dict[str, Any] | None) -> bool | None:
    if payload is None:
        return None
    if "contrasts" in payload:
        values = payload["contrasts"]
        if not isinstance(values, list):
            raise ValueError("matched-population contrasts must be a list")
        return any(bool(item.get("confirmatory_passed")) for item in values)
    return bool(payload.get("confirmatory_passed"))


def _diagnostic_passed(payload: dict[str, Any] | None) -> bool | None:
    if payload is None:
        return None
    audit = DiagnosticActionAudit.model_validate(payload)
    return audit.passed


def _bool_field(payload: dict[str, Any] | None, key: str) -> bool | None:
    if payload is None:
        return None
    if key not in payload:
        raise ValueError(f"audit artifact omits required field {key!r}")
    return bool(payload[key])


def _project_root(value: str | Path | None) -> Path:
    return Path(value).resolve() if value is not None else Path(__file__).resolve().parents[2]


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    source_roots = (
        root / "src",
        root / "phase-6-established-validation/runtime-overcookedv2/src",
        root / "phase-6-established-validation/runtime-legacy/src",
    )
    paths = sorted(
        path
        for source_root in source_roots
        if source_root.is_dir()
        for path in source_root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in paths:
        relative = path.relative_to(root)
        digest.update(len(str(relative)).to_bytes(8, "big"))
        digest.update(str(relative).encode())
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
