"""Pinned-upstream bootstrap and isolated runtime dispatch for Stage 6."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from zsc_identifiability.established_models import (
    EstablishedValidationSuite,
    RuntimeKind,
    UpstreamAudit,
    UpstreamRepositoryAudit,
)


def validate_upstreams(
    suite: EstablishedValidationSuite,
    project_root: str | Path,
) -> UpstreamAudit:
    root = Path(project_root).resolve()
    repositories: list[UpstreamRepositoryAudit] = []
    missing: list[str] = []
    for spec in suite.upstreams:
        target = _safe_project_path(root, spec.local_directory)
        exists = (target / ".git").is_dir()
        observed_commit = _git_output(target, ("rev-parse", "HEAD")) if exists else None
        remote = _git_output(target, ("config", "--get", "remote.origin.url")) if exists else None
        commit_matches = observed_commit == spec.commit
        remote_matches = remote is not None and _normalize_url(remote) == _normalize_url(spec.url)
        issues: list[str] = []
        if not exists:
            issues.append("repository is not bootstrapped")
        if exists and not commit_matches:
            issues.append("checked-out commit does not match the full pinned hash")
        if exists and not remote_matches:
            issues.append("origin URL does not match the declared upstream")
        passed = exists and commit_matches and remote_matches
        if spec.required and not passed:
            missing.append(spec.repository_id)
        repositories.append(
            UpstreamRepositoryAudit(
                repository_id=spec.repository_id,
                path=str(target),
                expected_commit=spec.commit,
                observed_commit=observed_commit,
                remote_url=remote,
                exists=exists,
                commit_matches=commit_matches,
                remote_matches=remote_matches,
                passed=passed,
                issues=tuple(issues),
            )
        )
    runtimes = {
        runtime.runtime_id: _runtime_ready(
            _safe_project_path(root, runtime.project_directory), runtime.python_version
        )
        for runtime in suite.runtimes
    }
    for runtime_id, present in runtimes.items():
        if not present:
            missing.append(f"runtime:{runtime_id}")
    return UpstreamAudit(
        repositories=tuple(repositories),
        runtime_projects_present={str(key): value for key, value in runtimes.items()},
        passed=all(item.passed for item in repositories) and all(runtimes.values()),
        missing_required_assets=tuple(sorted(missing)),
    )


def bootstrap_upstreams(
    suite: EstablishedValidationSuite,
    project_root: str | Path,
) -> UpstreamAudit:
    """Clone and detach-checkout only inside the ignored upstream directory."""

    root = Path(project_root).resolve()
    for spec in suite.upstreams:
        target = _safe_project_path(root, spec.local_directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            _run(("git", "clone", "--filter=blob:none", spec.url, str(target)), cwd=root)
        if not (target / ".git").is_dir():
            raise RuntimeError(f"refusing to bootstrap over non-git path: {target}")
        remote = _git_output(target, ("config", "--get", "remote.origin.url"))
        if remote is None or _normalize_url(remote) != _normalize_url(spec.url):
            raise RuntimeError(f"upstream origin mismatch at {target}")
        _run(("git", "fetch", "origin", spec.commit), cwd=target)
        _run(("git", "checkout", "--detach", spec.commit), cwd=target)
        observed = _git_output(target, ("rev-parse", "HEAD"))
        if observed != spec.commit:
            raise RuntimeError(f"failed to establish pinned commit for {spec.repository_id}")
    return validate_upstreams(suite, root)


def bootstrap_isolated_runtimes(
    suite: EstablishedValidationSuite,
    project_root: str | Path,
) -> UpstreamAudit:
    """Synchronize each declared runtime only after all upstream pins match."""

    root = Path(project_root).resolve()
    upstream_audit = validate_upstreams(suite, root)
    if any(not item.passed for item in upstream_audit.repositories):
        raise RuntimeError("cannot install isolated runtimes before upstream pins validate")
    installed_projects: set[Path] = set()
    for runtime in suite.runtimes:
        project = _safe_project_path(root, runtime.project_directory)
        if project in installed_projects:
            continue
        installed_projects.add(project)
        _run(
            (
                "uv",
                "sync",
                "--project",
                str(project),
                "--python",
                runtime.python_version,
            ),
            cwd=root,
        )
    audit = validate_upstreams(suite, root)
    if not audit.passed:
        raise RuntimeError(
            f"runtime bootstrap incomplete: {', '.join(audit.missing_required_assets)}"
        )
    return audit


def write_runtime_request(
    suite: EstablishedValidationSuite,
    runtime: RuntimeKind,
    operation: str,
    payload: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    upstreams = {
        item.repository_id: {"url": item.url, "commit": item.commit, "path": item.local_directory}
        for item in suite.upstreams
        if item.runtime == runtime
    }
    request = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "runtime": runtime,
        "operation": operation,
        "upstreams": upstreams,
        "payload": dict(payload),
    }
    request["request_hash"] = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(path, request)
    return path


def dispatch_runtime_request(
    suite: EstablishedValidationSuite,
    runtime: RuntimeKind,
    request_path: str | Path,
    result_path: str | Path,
    project_root: str | Path,
    *,
    extra_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute an isolated request without importing upstream packages here."""

    root = Path(project_root).resolve()
    runtime_spec = next(item for item in suite.runtimes if item.runtime_id == runtime)
    runtime_project = _safe_project_path(root, runtime_spec.project_directory)
    request = Path(request_path).resolve()
    result = Path(result_path).resolve()
    request_payload: Any = json.loads(request.read_text(encoding="utf-8"))
    if not isinstance(request_payload, dict):
        raise RuntimeError("isolated runtime request must be a JSON object")
    if request_payload.get("runtime") != runtime:
        raise RuntimeError("isolated runtime request was dispatched to the wrong boundary")
    request_hash = request_payload.get("request_hash")
    if not isinstance(request_hash, str):
        raise RuntimeError("isolated runtime request omits its content hash")
    result.parent.mkdir(parents=True, exist_ok=True)
    module = (
        "stage6_overcooked_runtime"
        if runtime == "overcookedv2_py310"
        else "stage6_legacy_runtime"
    )
    command = (
        "uv",
        "run",
        "--project",
        str(runtime_project),
        "python",
        "-m",
        module,
        "--request",
        str(request),
        "--result",
        str(result),
    )
    environment = dict(os.environ)
    environment.update(extra_environment or {})
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"isolated runtime failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    if not result.is_file():
        raise RuntimeError("isolated runtime completed without a result manifest")
    payload: Any = json.loads(result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("isolated runtime result must be a JSON object")
    if payload.get("schema_version") != 1:
        raise RuntimeError("isolated runtime result has an unsupported schema version")
    if payload.get("request_hash") != request_hash:
        raise RuntimeError("isolated runtime result does not correspond to the request")
    if payload.get("operation") != request_payload.get("operation"):
        raise RuntimeError("isolated runtime result operation does not match the request")
    if payload.get("status") not in {"complete", "secondary_unavailable"}:
        raise RuntimeError("isolated runtime returned an invalid completion status")
    return payload


def _safe_project_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"path escapes project root: {relative!r}")
    return candidate


def _git_output(path: Path, arguments: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ("git", "-C", str(path), *arguments),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _runtime_ready(project: Path, expected_version: str) -> bool:
    interpreter = project / ".venv/bin/python"
    if not interpreter.is_file():
        return False
    completed = subprocess.run(
        (
            str(interpreter),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == expected_version


def _normalize_url(url: str) -> str:
    return url.rstrip("/").removesuffix(".git")


def _run(command: Sequence[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
