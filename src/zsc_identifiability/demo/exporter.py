"""Deterministic capture and static-site export for BeforeCommit."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from zsc_identifiability.demo.curation import load_demo_curation
from zsc_identifiability.demo.diagnostics import FrozenDiagnosticSources
from zsc_identifiability.demo.models import (
    BenchmarkSummary,
    CommitmentRecord,
    DemoCurationEntry,
    ReplayCatalog,
    ReplayCatalogEntry,
    ReplayEvent,
    ReplayFrame,
    ReplayManifest,
    ReplayResponse,
    ReplaySource,
    ReplaySummary,
)
from zsc_identifiability.demo.state_adapter import normalize_state, normalize_terrain


def capture_collection(
    curation_path: str | Path,
    output_directory: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Re-run the six selected inference episodes and emit normalized packages."""

    root = _project_root(project_root)
    curation = load_demo_curation(curation_path)
    output = Path(output_directory).resolve()
    capture_directory = output.parent / "captures"
    output.mkdir(parents=True, exist_ok=True)
    capture_directory.mkdir(parents=True, exist_ok=True)
    diagnostics = FrozenDiagnosticSources(root)
    produced: list[Path] = []
    for entry in curation.entries:
        source_request_path, source_result_path, source_episode = _source_episode(root, entry)
        source_request = _read_json(source_request_path)
        request = _demo_request(source_request, entry)
        request_path = capture_directory / f"{entry.replay_id}.request.json"
        result_path = capture_directory / f"{entry.replay_id}.result.json.gz"
        _write_json(request_path, request, pretty=True)
        _run_demo_capture(root, request_path, result_path)
        with gzip.open(result_path, "rt", encoding="utf-8") as handle:
            result: dict[str, Any] = json.load(handle)
        if result.get("status") != "complete" or len(result.get("episodes", ())) != 1:
            raise RuntimeError(f"demo capture did not complete for {entry.replay_id}")
        captured_episode = result["episodes"][0]
        _verify_trace_parity(entry, source_episode, captured_episode)
        package = _emit_package(
            root=root,
            collection_id=curation.collection_id,
            entry=entry,
            source_request=source_request,
            source_request_path=source_request_path,
            source_result_path=source_result_path,
            episode=captured_episode,
            diagnostics=diagnostics,
            output=output,
        )
        produced.append(package)
    return tuple(produced)


def export_site_data(
    replay_directory: str | Path,
    output_directory: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Copy validated replay packages and frozen aggregate results to the web tree."""

    root = _project_root(project_root)
    source = Path(replay_directory).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_replays = output / "replays"
    if target_replays.exists():
        shutil.rmtree(target_replays)
    target_replays.mkdir(parents=True)
    manifests: list[ReplayManifest] = []
    written: list[Path] = []
    for package in sorted(path for path in source.iterdir() if path.is_dir()):
        manifest_path = package / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = ReplayManifest.model_validate(_read_json(manifest_path))
        _verify_package_hashes(package, manifest)
        destination = target_replays / manifest.replay_id
        shutil.copytree(package, destination)
        manifests.append(manifest)
        written.extend(sorted(destination.glob("*.json")))
    if not manifests:
        raise ValueError("replay export found no valid replay packages")
    collection_ids = {item.collection_id for item in manifests}
    if len(collection_ids) != 1:
        raise ValueError("site export requires one replay collection")
    catalog = ReplayCatalog(
        collection_id=next(iter(collection_ids)),
        items=tuple(
            ReplayCatalogEntry(
                replay_id=item.replay_id,
                title=item.title,
                short_title=item.short_title,
                category=item.category,
                caption=item.caption,
                layout_id=item.source.layout_id,
                evidence_policy=item.source.evidence_policy,
                featured=item.featured,
                representative=item.representative,
                sparse_return=item.summary.sparse_return,
                commitment_step=item.commitment.step,
                aggregate_dri_primary=item.summary.aggregate_dri_primary,
                eventual_dri_primary=item.summary.eventual_dri_primary,
                frame_count=item.frame_count,
            )
            for item in manifests
        ),
        total=len(manifests),
    )
    catalog_path = output / "catalog.json"
    _write_json(catalog_path, catalog.to_dict())
    written.append(catalog_path)
    summary_path = output / "benchmark-summary.json"
    _write_json(summary_path, _benchmark_summary(root).to_dict())
    written.append(summary_path)
    schema_directory = root / "demo/schemas"
    schema_directory.mkdir(parents=True, exist_ok=True)
    schema_path = schema_directory / "beforecommit-replay-v1.schema.json"
    _write_json(schema_path, ReplayManifest.model_json_schema(), pretty=True)
    written.append(schema_path)
    return tuple(written)


def _emit_package(
    *,
    root: Path,
    collection_id: str,
    entry: DemoCurationEntry,
    source_request: Mapping[str, Any],
    source_request_path: Path,
    source_result_path: Path,
    episode: Mapping[str, Any],
    diagnostics: FrozenDiagnosticSources,
    output: Path,
) -> Path:
    package = output / entry.replay_id
    package.mkdir(parents=True, exist_ok=True)
    terrain = normalize_terrain(episode["terrain"])
    initial_state = normalize_state(episode["initial_state"], ego_seat=entry.ego_seat)
    commitment_step_raw = episode.get("commitment_step")
    commitment_step = None if commitment_step_raw is None else int(commitment_step_raw)
    commitment_frame = None if commitment_step is None else commitment_step + 1
    frames: list[ReplayFrame] = [
        ReplayFrame(frame=0, state=initial_state, phase="before")
    ]
    events: list[ReplayEvent] = []
    for frame_index, row in enumerate(episode["steps"], start=1):
        step = int(row["step"])
        phase = _phase(step, commitment_step)
        event_kinds = tuple(str(value) for value in row.get("events", ()))
        if commitment_step == step:
            event_kinds += ("commitment",)
        if float(row.get("reward", 0.0)) > 0:
            event_kinds += ("delivery",)
        frames.append(
            ReplayFrame(
                frame=frame_index,
                action_step=step,
                state=normalize_state(row["world_state"], ego_seat=entry.ego_seat),
                ego_action=int(row["ego_action"]),
                partner_action=(
                    None
                    if row.get("visible_partner_action") is None
                    else int(row["visible_partner_action"])
                ),
                sparse_reward=float(row.get("reward", 0.0)),
                event_kinds=event_kinds,
                phase=phase,
            )
        )
        for kind in event_kinds:
            events.append(
                ReplayEvent(
                    frame=frame_index,
                    step=step,
                    kind=kind,
                    label=_event_label(kind),
                    phase=phase,
                )
            )
    points = diagnostics.diagnostics(entry, episode)
    precommitment = next(point for point in points if point.prefix == "pre_commitment")
    primary = diagnostics.decision_row(entry, "gru", "pre_commitment")
    event = diagnostics.decision_row(entry, "event", "pre_commitment")
    eventual = diagnostics.decision_row(entry, "gru", "eventual")
    if primary is None or event is None or eventual is None:
        raise ValueError(f"frozen decision-value rows omit {entry.replay_id}")
    response_contract = diagnostics.response_contract(entry)
    responses = tuple(
        ReplayResponse(
            response_id=response_id,
            alias=alias,
            optimal_for=optimal_for,  # type: ignore[arg-type]
        )
        for response_id, alias, optimal_for in response_contract
    )
    initial_path = package / "initial.json"
    frames_path = package / "frames.json"
    diagnostics_path = package / "diagnostics.json"
    events_path = package / "events.json"
    _write_json(initial_path, initial_state.to_dict())
    _write_json(frames_path, [item.to_dict() for item in frames])
    _write_json(diagnostics_path, [item.to_dict() for item in points])
    _write_json(events_path, [item.to_dict() for item in events])
    source_payload = source_request["payload"]
    checkpoint_paths = {
        "ego": Path(str(source_payload["reference_checkpoint_path"])),
        "partner": Path(str(source_payload["partner_checkpoint_path"])),
    }
    manifest = ReplayManifest(
        replay_id=entry.replay_id,
        collection_id=collection_id,
        title=entry.title,
        short_title=entry.short_title,
        category=entry.category,
        caption=entry.caption,
        selection_note=entry.selection_note,
        representative=entry.representative,
        featured=entry.featured,
        source=ReplaySource(
            suite_id=str(source_request["suite_id"]),
            layout_id=entry.layout_id,
            environment_key=entry.environment_key,
            ego_seat=entry.ego_seat,
            ego_policy_id=checkpoint_paths["ego"].stem,
            partner_id=entry.partner_id,
            evidence_policy=entry.evidence_policy,
            checkpoint_hashes={
                key: _sha256(path) for key, path in checkpoint_paths.items()
            },
            source_artifact_hashes={
                "request": _sha256(source_request_path),
                "stored_trace": _sha256(source_result_path),
                "visual_capture": _json_sha256(
                    {
                        "terrain": episode["terrain"],
                        "initial_state": episode["initial_state"],
                        "episode": {
                            key: value
                            for key, value in episode.items()
                            if key not in {"terrain", "initial_state"}
                        },
                    }
                ),
            },
        ),
        commitment=CommitmentRecord(
            definition="first_successful_ingredient_placement",
            step=commitment_step,
            frame=commitment_frame,
            reached=commitment_step is not None,
        ),
        terrain=terrain,
        partner_modes={
            "left": entry.left_partner_id,
            "right": entry.right_partner_id,
        },
        responses=responses,
        summary=ReplaySummary(
            sparse_return=float(episode["sparse_return"]),
            first_delivery_step=episode.get("first_delivery_step"),
            steps_elapsed=int(episode["steps_elapsed"]),
            prior_risk=float(primary["prior_risk"]),
            aggregate_dri_primary=primary.get("dri"),
            aggregate_dri_event=event.get("dri"),
            aggregate_residual_risk_primary=float(primary["residual_risk"]),
            selected_response_loss_at_commitment=(
                precommitment.selected_response_loss_for_true_mode
            ),
            eventual_dri_primary=eventual.get("dri"),
        ),
        frame_count=len(frames),
        diagnostic_count=len(points),
        event_count=len(events),
        artifacts={
            path.name: _sha256(path)
            for path in (initial_path, frames_path, diagnostics_path, events_path)
        },
    )
    _write_json(package / "manifest.json", manifest.to_dict(), pretty=True)
    return package


def _source_episode(
    root: Path,
    entry: DemoCurationEntry,
) -> tuple[Path, Path, dict[str, Any]]:
    base = root / "phase-6-established-validation/runs/official-measurement-v3/confirmation"
    partner_slug = entry.partner_id.replace("_", "-").replace(":", "-")
    evidence_slug = entry.evidence_policy.replace("_", "-")
    pattern = f"*--trace--{partner_slug}--{evidence_slug}--confirmatory.json"
    request_paths = sorted((base / "requests").glob(pattern))
    if len(request_paths) != 1:
        raise ValueError(f"could not resolve one source request for {entry.replay_id}")
    request_path = request_paths[0]
    result_path = base / "results/trace" / f"{request_path.name}.gz"
    if not result_path.is_file():
        raise ValueError(f"source trace is absent for {entry.replay_id}")
    with gzip.open(result_path, "rt", encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    matches = [
        episode
        for episode in result["episodes"]
        if int(episode["environment_key"]) == entry.environment_key
        and int(episode["ego_seat"]) == entry.ego_seat
    ]
    if len(matches) != 1:
        raise ValueError(f"source trace does not resolve one episode for {entry.replay_id}")
    return request_path, result_path, matches[0]


def _demo_request(
    source_request: Mapping[str, Any],
    entry: DemoCurationEntry,
) -> dict[str, Any]:
    request = {
        key: value
        for key, value in source_request.items()
        if key != "request_hash"
    }
    request["operation"] = "official_demo_replay"
    payload = dict(request["payload"])
    payload.update(
        {
            "episode_keys": [entry.environment_key],
            "ego_seat": entry.ego_seat,
            "balanced_seats": False,
            "split": "demo",
        }
    )
    request["payload"] = payload
    request["request_hash"] = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return request


def _run_demo_capture(root: Path, request: Path, result: Path) -> None:
    result.unlink(missing_ok=True)
    environment = os.environ.copy()
    external = root / "phase-6-established-validation/runs/official-checkpoints/.external"
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "ZSC_EVAL_SOURCE": str(external / "zsc-eval"),
            "ZSC_EVAL_POLICY_POOL": str(external / "zsc-eval-policy-pool"),
            "ZSC_IDENTIFIABILITY_PROJECT_ROOT": str(root),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    command = (
        "uv",
        "run",
        "--project",
        str(root / "phase-6-established-validation/runtime-zsceval"),
        "python",
        "-m",
        "stage6_legacy_runtime",
        "--request",
        str(request),
        "--result",
        str(result),
    )
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"visual replay capture failed: {message}")


def _verify_trace_parity(
    entry: DemoCurationEntry,
    source: Mapping[str, Any],
    captured: Mapping[str, Any],
) -> None:
    scalar_fields = (
        "environment_key",
        "ego_seat",
        "sparse_return",
        "commitment_reached",
        "commitment_step",
        "first_delivery_step",
        "steps_elapsed",
    )
    for field in scalar_fields:
        if captured.get(field) != source.get(field):
            raise ValueError(f"{entry.replay_id} changed stored trace field {field!r}")
    source_actions = [
        (item["ego_action"], item.get("visible_partner_action"))
        for item in source["steps"]
    ]
    captured_actions = [
        (item["ego_action"], item.get("visible_partner_action"))
        for item in captured["steps"]
    ]
    if source_actions != captured_actions:
        raise ValueError(f"{entry.replay_id} visual rerun changed the visible action trace")
    if captured.get("visual_state_schema") != "official-overcooked-state-v1":
        raise ValueError(f"{entry.replay_id} omitted the visual state contract")


def _benchmark_summary(root: Path) -> BenchmarkSummary:
    path = (
        root
        / "phase-6-established-validation/artifacts/official-measurement-v3/"
        "held-out-regression-report-v3.json"
    )
    report = _read_json(path)
    primary = report["gru"]
    interval = primary["dri_coefficient_interval"]
    return BenchmarkSummary(
        title="Official ZSC-Eval measurement audit",
        held_out_delta_r2=float(primary["overall"]["delta_r2"]),
        adjusted_dri_coefficient=float(interval["coefficient_point"]),
        coefficient_interval=(float(interval["ci_low"]), float(interval["ci_high"])),
        verdict="complete_measurement_only",
        intervention_result="neither_confirmed",
        source_hash=_sha256(path),
    )


def _verify_package_hashes(package: Path, manifest: ReplayManifest) -> None:
    for filename, expected in manifest.artifacts.items():
        path = package / filename
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"replay artifact integrity failure: {path}")


def _phase(
    step: int, commitment: int | None
) -> Literal["before", "commitment", "after"]:
    if commitment is None or step < commitment:
        return "before"
    if step == commitment:
        return "commitment"
    return "after"


def _event_label(kind: str) -> str:
    if kind == "commitment":
        return "First ingredient placed"
    if kind == "delivery":
        return "Soup delivered"
    return kind.removeprefix("ego_event:").replace("_", " ").title()


def _project_root(value: str | Path | None) -> Path:
    root = Path.cwd() if value is None else Path(value)
    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("BeforeCommit commands must run from the project root")
    return root


def _write_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
        if pretty
        else json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
