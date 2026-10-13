"""Integrity and semantic validation for deployable BeforeCommit data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from zsc_identifiability.demo.models import (
    BenchmarkSummary,
    DiagnosticPoint,
    ReplayCatalog,
    ReplayEvent,
    ReplayFrame,
    ReplayManifest,
)


def validate_site_data(site_directory: str | Path) -> dict[str, Any]:
    root = Path(site_directory).resolve()
    catalog = ReplayCatalog.model_validate(_read(root / "catalog.json"))
    BenchmarkSummary.model_validate(_read(root / "benchmark-summary.json"))
    if catalog.total != len(catalog.items):
        raise ValueError("replay catalog total is inconsistent")
    for catalog_item in catalog.items:
        package = root / "replays" / catalog_item.replay_id
        manifest = ReplayManifest.model_validate(_read(package / "manifest.json"))
        frames = tuple(
            ReplayFrame.model_validate(item) for item in _read(package / "frames.json")
        )
        diagnostics = tuple(
            DiagnosticPoint.model_validate(item)
            for item in _read(package / "diagnostics.json")
        )
        events = tuple(
            ReplayEvent.model_validate(item) for item in _read(package / "events.json")
        )
        if len(frames) != manifest.frame_count or len(diagnostics) != manifest.diagnostic_count:
            raise ValueError(f"replay counts are inconsistent for {manifest.replay_id}")
        if len(events) != manifest.event_count:
            raise ValueError(f"event count is inconsistent for {manifest.replay_id}")
        if [frame.frame for frame in frames] != list(range(len(frames))):
            raise ValueError(f"frame indices are not contiguous for {manifest.replay_id}")
        for filename, expected in manifest.artifacts.items():
            path = package / filename
            if _sha256(path) != expected:
                raise ValueError(f"artifact hash mismatch for {path}")
        if any("ego_observation" in json.dumps(frame.to_dict()) for frame in frames):
            raise ValueError(f"raw model observation leaked into {manifest.replay_id}")
    return {
        "valid": True,
        "collection_id": catalog.collection_id,
        "replay_count": catalog.total,
    }


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
