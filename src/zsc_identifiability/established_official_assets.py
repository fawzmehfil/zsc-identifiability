"""Content-addressed acquisition of the minimal official ZSC-Eval asset set."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from zsc_identifiability.established_official_models import (
    OfficialAssetInventory,
    OfficialAssetLock,
    OfficialAssetLockEntry,
    OfficialAssetRecord,
    OfficialCheckpointAuditSuiteV2,
    OfficialMethodAsset,
    OfficialPartnerAsset,
    load_official_checkpoint_suite,
)

_USER_AGENT = "zsc-identifiability-official-audit/1"


def prepare_official_asset_lock(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    workspace: str | Path,
) -> OfficialAssetLock:
    """Resolve the exact official population without downloading checkpoint weights."""

    spec, suite_path = _resolve_suite(suite)
    target = Path(workspace).resolve()
    target.mkdir(parents=True, exist_ok=True)
    suite_hash = _sha256_path(suite_path) if suite_path is not None else _hash_json(spec.to_dict())

    entries: dict[str, OfficialAssetLockEntry] = {}
    for layout in spec.layouts:
        benchmark_bytes = _download_bytes(_pool_url(spec, layout.benchmark_yaml_path))
        parsed = _parse_benchmark_yaml(benchmark_bytes, layout.layout_id)
        _add_entry(
            entries,
            relative_path=layout.benchmark_yaml_path,
            revision=spec.upstream.policy_pool_revision,
            role="benchmark",
            expected_size=len(benchmark_bytes),
            expected_sha256=hashlib.sha256(benchmark_bytes).hexdigest(),
        )
        for config in sorted({*spec.required_policy_configs, *(row["config"] for row in parsed)}):
            path = config.format(layout=layout.layout_id)
            _add_entry(
                entries,
                relative_path=path,
                revision=spec.upstream.policy_pool_revision,
                role="config",
            )
        for row in parsed:
            _add_entry(
                entries,
                relative_path=row["partner"],
                revision=spec.upstream.policy_pool_revision,
                role="partner",
            )
            _add_entry(
                entries,
                relative_path=row["response"],
                revision=spec.upstream.policy_pool_revision,
                role="response",
            )
        for method in spec.methods:
            for seed in method.seeds:
                _add_entry(
                    entries,
                    relative_path=(
                        f"{layout.layout_id}/"
                        + method.asset_path_template.format(seed=seed).lstrip("/")
                    ),
                    revision=spec.upstream.policy_pool_revision,
                    role="method",
                )
    ordered = tuple(sorted(entries.values(), key=lambda item: (item.role, item.relative_path)))
    lock_payload = {
        "schema_version": 1,
        "suite_id": spec.suite_id,
        "suite_hash": suite_hash,
        "repository_commit": spec.upstream.repository_commit,
        "policy_pool_revision": spec.upstream.policy_pool_revision,
        "workspace": str(target),
        "entries": [item.to_dict() for item in ordered],
    }
    lock = OfficialAssetLock.model_validate({**lock_payload, "lock_hash": _hash_json(lock_payload)})
    _atomic_json(target / "official-asset-lock.json", lock.to_dict())
    return lock


def sync_official_assets(
    lock: OfficialAssetLock | str | Path,
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> OfficialAssetInventory:
    """Download only locked assets and establish the pinned source checkout."""

    asset_lock = _load_lock(lock)
    spec, _ = _resolve_suite(suite)
    if asset_lock.suite_id != spec.suite_id:
        raise ValueError("asset lock belongs to a different suite")
    if asset_lock.policy_pool_revision != spec.upstream.policy_pool_revision:
        raise ValueError("asset lock policy-pool revision does not match the suite")
    workspace = Path(asset_lock.workspace).resolve()
    source_root = _safe_child(workspace, spec.runtime.upstream_directory)
    pool_root = _safe_child(workspace, spec.runtime.asset_directory)
    _sync_repository(spec, source_root)

    records: list[OfficialAssetRecord] = []
    missing: list[str] = []
    for entry in asset_lock.entries:
        destination = _safe_child(pool_root, entry.relative_path)
        try:
            _sync_one_pool_asset(spec, entry, destination)
            file_hash = _sha256_path(destination)
            if entry.expected_sha256 is not None and file_hash != entry.expected_sha256:
                raise RuntimeError(f"download hash mismatch for {entry.relative_path}")
            tensor_hash = (
                _normalized_tensor_hash(destination) if destination.suffix == ".pt" else None
            )
            records.append(
                OfficialAssetRecord(
                    asset_id=entry.asset_id,
                    local_path=str(destination),
                    repository_path=entry.relative_path,
                    revision=entry.source_revision,
                    size=destination.stat().st_size,
                    file_hash=file_hash,
                    normalized_tensor_hash=tensor_hash,
                    **_asset_provenance(spec, entry),
                )
            )
        except (OSError, RuntimeError, ValueError, urllib.error.URLError):
            missing.append(entry.asset_id)

    record_by_id = {record.asset_id: record for record in records}
    partners = _inventory_partners(spec, asset_lock, record_by_id, pool_root)
    methods = _inventory_methods(spec, asset_lock, record_by_id, pool_root)
    duplicate_groups = _duplicate_tensor_groups(records)
    payload = {
        "schema_version": 1,
        "suite_id": spec.suite_id,
        "lock_hash": asset_lock.lock_hash,
        "partners": [item.to_dict() for item in partners],
        "methods": [item.to_dict() for item in methods],
        "assets": [item.to_dict() for item in records],
        "duplicate_tensor_groups": [list(group) for group in duplicate_groups],
        "complete": not missing and len(records) == len(asset_lock.entries),
        "missing_asset_ids": sorted(missing),
    }
    inventory = OfficialAssetInventory.model_validate(
        {**payload, "inventory_hash": _hash_json(payload)}
    )
    _atomic_json(workspace / "official-asset-inventory.json", inventory.to_dict())
    return inventory


def load_official_asset_lock(path: str | Path) -> OfficialAssetLock:
    return _load_lock(path)


def load_official_asset_inventory(path: str | Path) -> OfficialAssetInventory:
    return OfficialAssetInventory.model_validate(_read_json(Path(path)))


def _resolve_suite(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> tuple[OfficialCheckpointAuditSuiteV2, Path | None]:
    if isinstance(suite, OfficialCheckpointAuditSuiteV2):
        return suite, None
    path = Path(suite).resolve()
    return load_official_checkpoint_suite(path), path


def _load_lock(lock: OfficialAssetLock | str | Path) -> OfficialAssetLock:
    if isinstance(lock, OfficialAssetLock):
        return lock
    return OfficialAssetLock.model_validate(_read_json(Path(lock)))


def _parse_benchmark_yaml(data: bytes, layout_id: str) -> list[dict[str, str]]:
    parsed: Any = yaml.safe_load(io.BytesIO(data))
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"benchmark YAML for {layout_id} is empty or invalid")
    rows: list[dict[str, str]] = []
    schemes: set[tuple[str, str]] = set()
    for benchmark_id, raw in parsed.items():
        if not isinstance(benchmark_id, str) or not isinstance(raw, dict):
            raise ValueError("benchmark entries must map names to policy records")
        # The official YAML appends an `agent_name` template used by evaluator
        # scripts. It is not one of the BR-Div-selected benchmark partners.
        if benchmark_id == "agent_name":
            continue
        if not benchmark_id.startswith("bias"):
            raise ValueError(f"unexpected official benchmark entry: {benchmark_id!r}")
        model_path = raw.get("model_path")
        config = raw.get("policy_config_path")
        if not isinstance(model_path, dict) or not isinstance(model_path.get("actor"), str):
            raise ValueError(f"benchmark {benchmark_id} omits its actor checkpoint")
        if not isinstance(config, str):
            raise ValueError(f"benchmark {benchmark_id} omits its policy configuration")
        partner = model_path["actor"]
        if "_w0_actor.pt" not in partner:
            raise ValueError(f"benchmark partner is not a w0 actor: {partner}")
        stage = "final" if "_final_" in partner else "mid" if "_mid_" in partner else None
        scheme_match = re.search(r"/(hsp\d+)_(?:mid|final)_w0_actor\.pt$", partner)
        if stage is None or scheme_match is None:
            raise ValueError(f"cannot derive scheme and stage from {partner}")
        scheme = scheme_match.group(1)
        key = (scheme, stage)
        if key in schemes:
            raise ValueError(f"duplicate scheme-stage benchmark entry: {key}")
        schemes.add(key)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "partner": partner,
                "response": partner.replace("_w0_actor.pt", "_w1_actor.pt"),
                "config": config,
                "scheme": scheme,
                "stage": stage,
            }
        )
    return rows


def _add_entry(
    entries: dict[str, OfficialAssetLockEntry],
    *,
    relative_path: str,
    revision: str,
    role: str,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    normalized = _safe_relative(relative_path)
    asset_id = hashlib.sha256(f"policy_pool:{revision}:{normalized}".encode()).hexdigest()
    candidate = OfficialAssetLockEntry(
        asset_id=asset_id,
        relative_path=normalized,
        source="policy_pool",
        source_revision=revision,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        role=role,  # type: ignore[arg-type]
    )
    existing = entries.get(asset_id)
    if existing is not None and existing.relative_path != normalized:
        raise ValueError("official asset identifier collision")
    entries[asset_id] = candidate


def _asset_provenance(
    suite: OfficialCheckpointAuditSuiteV2,
    entry: OfficialAssetLockEntry,
) -> dict[str, Any]:
    first = entry.relative_path.split("/", 1)[0]
    layout_id = first if first in {item.layout_id for item in suite.layouts} else None
    if entry.role == "partner":
        return {
            "layout_id": layout_id,
            "algorithm": "hsp",
            "seed": 1,
            "policy_architecture": "mlp",
            "provenance": "official_benchmark_yaml",
        }
    if entry.role == "response":
        return {
            "layout_id": layout_id,
            "algorithm": "hsp",
            "seed": 1,
            "policy_architecture": "mlp",
            "provenance": "official_response_counterpart",
        }
    if entry.role == "method":
        matching = [
            (method.method_id, seed)
            for method in suite.methods
            for seed in method.seeds
            if entry.relative_path
            == f"{layout_id}/{method.asset_path_template.format(seed=seed).lstrip('/')}"
        ]
        if len(matching) != 1:
            raise ValueError(f"cannot resolve official method provenance: {entry.relative_path}")
        method_id, seed = matching[0]
        method = next(item for item in suite.methods if item.method_id == method_id)
        if layout_id is None:
            raise ValueError(f"official method asset has no layout: {entry.relative_path}")
        return {
            "layout_id": layout_id,
            "algorithm": method_id,
            "seed": seed,
            "policy_architecture": method.architecture_by_layout[layout_id],
            "provenance": "official_method_spec",
        }
    if entry.role == "config":
        return {
            "layout_id": layout_id,
            "algorithm": None,
            "seed": None,
            "policy_architecture": "configuration",
            "provenance": "required_policy_config",
        }
    return {
        "layout_id": layout_id,
        "algorithm": None,
        "seed": None,
        "policy_architecture": "metadata",
        "provenance": "official_metadata",
    }


def _inventory_partners(
    suite: OfficialCheckpointAuditSuiteV2,
    lock: OfficialAssetLock,
    records: Mapping[str, OfficialAssetRecord],
    pool_root: Path,
) -> tuple[OfficialPartnerAsset, ...]:
    by_path = {entry.relative_path: entry for entry in lock.entries}
    partners: list[OfficialPartnerAsset] = []
    for layout in suite.layouts:
        benchmark = _safe_child(pool_root, layout.benchmark_yaml_path)
        if not benchmark.exists():
            continue
        rows = _parse_benchmark_yaml(benchmark.read_bytes(), layout.layout_id)
        if len(rows) != layout.expected_partner_count:
            raise ValueError(
                f"{layout.layout_id} benchmark contains {len(rows)} partners; "
                f"expected {layout.expected_partner_count}"
            )
        for row in rows:
            partner_entry = by_path[row["partner"]]
            response_entry = by_path[row["response"]]
            if partner_entry.asset_id not in records or response_entry.asset_id not in records:
                continue
            partners.append(
                OfficialPartnerAsset(
                    partner_id=f"{layout.layout_id}:{row['scheme']}:{row['stage']}",
                    layout_id=layout.layout_id,
                    scheme_id=row["scheme"],
                    training_stage=row["stage"],  # type: ignore[arg-type]
                    partner_checkpoint_path=str(_safe_child(pool_root, row["partner"])),
                    response_checkpoint_path=str(_safe_child(pool_root, row["response"])),
                    partner_asset_id=partner_entry.asset_id,
                    response_asset_id=response_entry.asset_id,
                )
            )
    return tuple(partners)


def _inventory_methods(
    suite: OfficialCheckpointAuditSuiteV2,
    lock: OfficialAssetLock,
    records: Mapping[str, OfficialAssetRecord],
    pool_root: Path,
) -> tuple[OfficialMethodAsset, ...]:
    by_path = {entry.relative_path: entry for entry in lock.entries}
    methods: list[OfficialMethodAsset] = []
    for layout in suite.layouts:
        for method in suite.methods:
            for seed in method.seeds:
                relative = f"{layout.layout_id}/" + method.asset_path_template.format(
                    seed=seed
                ).lstrip("/")
                entry = by_path[relative]
                if entry.asset_id not in records:
                    continue
                methods.append(
                    # The official evaluator uses MLP configs for E3T on both
                    # layouts and COLE on random3_m; this cannot be inferred
                    # from the generic numeric checkpoint filename.
                    OfficialMethodAsset(
                        method_id=method.method_id,
                        layout_id=layout.layout_id,
                        seed=seed,
                        checkpoint_path=str(_safe_child(pool_root, relative)),
                        asset_id=entry.asset_id,
                        policy_architecture=method.architecture_by_layout[layout.layout_id],
                        recurrent=method.architecture_by_layout[layout.layout_id] == "rnn",
                    )
                )
    return tuple(methods)


def _sync_repository(suite: OfficialCheckpointAuditSuiteV2, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        _run(
            (
                "git",
                "clone",
                "--filter=blob:none",
                suite.upstream.repository_url,
                str(destination),
            ),
            destination.parent,
        )
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git source directory: {destination}")
    _run(("git", "fetch", "origin", suite.upstream.repository_commit), destination)
    _run(("git", "checkout", "--detach", suite.upstream.repository_commit), destination)
    observed = _run(("git", "rev-parse", "HEAD"), destination).strip()
    if observed != suite.upstream.repository_commit:
        raise RuntimeError("ZSC-Eval checkout does not match the pinned commit")


def _sync_one_pool_asset(
    suite: OfficialCheckpointAuditSuiteV2,
    entry: OfficialAssetLockEntry,
    destination: Path,
) -> None:
    marker = destination.with_name(destination.name + ".asset-lock.json")
    if destination.exists():
        wrong_size = (
            entry.expected_size is not None and destination.stat().st_size != entry.expected_size
        )
        wrong_hash = (
            entry.expected_sha256 is not None and _sha256_path(destination) != entry.expected_sha256
        )
        trusted_expected = entry.expected_size is not None and entry.expected_sha256 is not None
        trusted_marker = _valid_asset_marker(marker, destination, entry)
        if wrong_size or wrong_hash or (not trusted_expected and not trusted_marker):
            destination.unlink()
            marker.unlink(missing_ok=True)
        else:
            if not marker.exists():
                _write_asset_marker(marker, destination, entry)
            return
    data = _download_bytes(_pool_url(suite, entry.relative_path))
    if entry.expected_size is not None and len(data) != entry.expected_size:
        raise RuntimeError(f"asset size mismatch for {entry.relative_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(destination)
        _write_asset_marker(marker, destination, entry)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _valid_asset_marker(marker: Path, destination: Path, entry: OfficialAssetLockEntry) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = _read_json(marker)
        return bool(
            payload.get("asset_id") == entry.asset_id
            and payload.get("source_revision") == entry.source_revision
            and payload.get("size") == destination.stat().st_size
            and payload.get("file_hash") == _sha256_path(destination)
        )
    except (OSError, ValueError):
        return False


def _write_asset_marker(marker: Path, destination: Path, entry: OfficialAssetLockEntry) -> None:
    _atomic_json(
        marker,
        {
            "schema_version": 1,
            "asset_id": entry.asset_id,
            "relative_path": entry.relative_path,
            "source_revision": entry.source_revision,
            "size": destination.stat().st_size,
            "file_hash": _sha256_path(destination),
        },
    )


def _normalized_tensor_hash(path: Path) -> str:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by minimal installs
        raise RuntimeError("checkpoint synchronization requires the established extra") from error
    value: Any = torch.load(path, map_location="cpu", weights_only=False)
    state = _find_state_dict(value)
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not hasattr(tensor, "detach"):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _find_state_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        if value and all(isinstance(key, str) for key in value):
            tensor_like = {key: item for key, item in value.items() if hasattr(item, "detach")}
            if tensor_like:
                return tensor_like
        for key in ("state_dict", "model", "actor", "policy"):
            child = value.get(key)
            if isinstance(child, Mapping):
                found = _find_state_dict(child)
                if found:
                    return found
    raise ValueError("checkpoint contains no recognizable tensor state dictionary")


def _duplicate_tensor_groups(
    records: list[OfficialAssetRecord],
) -> tuple[tuple[str, ...], ...]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.normalized_tensor_hash is not None:
            grouped[record.normalized_tensor_hash].append(record.asset_id)
    return tuple(tuple(sorted(group)) for _, group in sorted(grouped.items()) if len(group) > 1)


def _pool_url(suite: OfficialCheckpointAuditSuiteV2, relative_path: str) -> str:
    base = suite.upstream.policy_pool_url.rstrip("/")
    revision = suite.upstream.policy_pool_revision
    return f"{base}/resolve/{revision}/{_safe_relative(relative_path)}?download=true"


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return bytes(response.read())


def _safe_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe official asset path: {value!r}")
    return path.as_posix()


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / _safe_relative(relative)).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("official asset path escapes its workspace")
    return candidate


def _run(command: tuple[str, ...], cwd: Path) -> str:
    process = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or f"command failed: {' '.join(command)}")
    return process.stdout


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
