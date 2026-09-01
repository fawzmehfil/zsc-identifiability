"""Resumable orchestration for the Stage 6 v3 decision-risk redesign."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from zsc_identifiability.established_official_assets import load_official_asset_inventory
from zsc_identifiability.established_official_decision import (
    PairwiseDecoderDataset,
    select_global_decoder_configuration,
    synthetic_decision_estimator_controls,
)
from zsc_identifiability.established_official_models import (
    OfficialResponseValueMatrix,
    OfficialRolloutLedger,
    OfficialRolloutLedgerEntry,
    OfficialRolloutPlan,
    OfficialRolloutShard,
    OfficialTraceIndex,
)
from zsc_identifiability.established_official_redesign_models import (
    MeasurementFitManifest,
    MeasurementPrefix,
    MeasurementRepresentationArtifact,
    MeasurementRepresentationManifest,
    OfficialConfirmationLedger,
    OfficialConfirmationPlan,
    OfficialMeasurementAuditSuiteV3,
    PairwiseDecisionDecoder,
    load_official_measurement_suite,
)
from zsc_identifiability.established_official_representation import (
    EncodedHistories,
    deterministic_stratified_calibration_split,
    encode_with_frozen_identity_representation,
    event_representation_matrix,
    fit_gru_identity_representation,
    load_encoded_histories,
    save_encoded_histories,
)
from zsc_identifiability.established_official_rollouts import (
    get_official_rollout_status,
    run_official_rollouts,
)
from zsc_identifiability.established_official_trace_store import OfficialCompactTraceStore


def prepare_measurement_redesign(
    suite: OfficialMeasurementAuditSuiteV3 | str | Path,
    workspace: str | Path,
) -> dict[str, Any]:
    """Validate and record the frozen v2-to-v3 data-use boundary without inference."""

    spec, suite_path, _project_root = _suite_context(suite)
    payload = validate_measurement_redesign(suite)
    target = Path(workspace).resolve()
    target.mkdir(parents=True, exist_ok=True)
    payload["suite_path"] = None if suite_path is None else str(suite_path)
    payload["preflight_hash"] = _hash_json(payload)
    _atomic_json(target / "measurement-redesign-preflight.json", payload)
    return payload


def validate_measurement_redesign(
    suite: OfficialMeasurementAuditSuiteV3 | str | Path,
) -> dict[str, Any]:
    """Validate v3 schema, source locks, and synthetic controls without writing files."""

    spec, suite_path, project_root = _suite_context(suite)
    source = _validate_v2_source_lock(spec, project_root)
    controls = synthetic_decision_estimator_controls(
        tolerance=spec.calibration.synthetic_tolerance
    )
    return {
        "schema_version": 3,
        "suite_id": spec.suite_id,
        "suite_hash": _suite_hash(spec, suite_path),
        "v2_preserved": True,
        "v2_verdict": source["manifest"]["verdict"],
        "v2_scientific_use": "exploratory_only",
        "v2_confirmatory_used_for_v3_tuning": False,
        "fresh_confirmation_salt": spec.fresh_confirmation_salt,
        "fresh_episode_count": _fresh_episode_count(spec),
        "policy_training_allowed": False,
        "synthetic_controls": controls,
        "source_hashes": source["hashes"],
        "valid": bool(controls["passed"]),
    }


def fit_measurement_representations(
    suite: OfficialMeasurementAuditSuiteV3 | str | Path,
    output_dir: str | Path,
    *,
    progress: Any | None = None,
) -> MeasurementRepresentationManifest:
    """Fit identity encoders on v2 calibration only and freeze calibration/validation features."""

    spec, suite_path, project_root = _suite_context(suite)
    source = _validate_v2_source_lock(spec, project_root)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_index = OfficialTraceIndex.model_validate(source["trace_index"])
    tuning_trace_index = _tuning_trace_index(trace_index)
    trace_store = OfficialCompactTraceStore.prepare(
        tuning_trace_index,
        output / "trace-cache",
        progress=progress,
    )
    libraries = _load_response_libraries(source["response_matrices"])
    work_index_path = output / "representation-work-index.json"
    completed: dict[str, MeasurementRepresentationArtifact] = {}
    if work_index_path.is_file():
        work_index = _read_json(work_index_path)
        if work_index.get("suite_id") != spec.suite_id:
            raise ValueError("representation work index belongs to another v3 suite")
        completed = {
            str(key): MeasurementRepresentationArtifact.model_validate(value)
            for key, value in work_index.get("work_units", {}).items()
        }
    artifacts: list[MeasurementRepresentationArtifact] = []
    calibration_keys: set[tuple[str, int]] = set()
    validation_keys: set[tuple[str, int]] = set()
    for library in libraries:
        layout = next(item for item in spec.layouts if item.layout_id == library.layout_id)
        partner_labels = {partner: index for index, partner in enumerate(library.partner_ids)}
        for evidence_policy in layout.evidence_policies:
            for prefix in spec.representations.prefixes:
                calibration_source = trace_store.decision_sequence_source(
                    library.layout_id,
                    evidence_policy,
                    "calibration",
                    partner_labels,
                    prefix,
                )
                validation_source = trace_store.decision_sequence_source(
                    library.layout_id,
                    evidence_policy,
                    "validation",
                    partner_labels,
                    prefix,
                )
                _require_disjoint_sources(calibration_source, validation_source)
                calibration_keys.update(
                    (library.layout_id, episode.environment_key)
                    for episode in calibration_source.episodes
                )
                validation_keys.update(
                    (library.layout_id, episode.environment_key)
                    for episode in validation_source.episodes
                )
                event_key = (
                    f"{library.layout_id}/{evidence_policy}/{prefix}/event"
                )
                event_artifact = completed.get(event_key)
                if event_artifact is None:
                    event_artifact = _fit_event_artifact(
                        spec,
                        library,
                        evidence_policy,
                        prefix,
                        calibration_source,
                        validation_source,
                        output,
                    )
                    completed[event_key] = event_artifact
                    _write_representation_work_index(work_index_path, spec.suite_id, completed)
                else:
                    _verify_artifact(event_artifact)
                artifacts.append(event_artifact)
                split_training, split_early = deterministic_stratified_calibration_split(
                    calibration_source,
                    fraction=spec.representations.encoder_training_fraction,
                    salt=(
                        f"{spec.suite_id}:{library.layout_id}:{evidence_policy}:"
                        f"{prefix}:encoder-split"
                    ),
                )
                for seed in spec.representations.gru_seeds:
                    gru_key = f"{library.layout_id}/{evidence_policy}/{prefix}/gru/{seed}"
                    gru_artifact = completed.get(gru_key)
                    if gru_artifact is None:
                        gru_artifact = _fit_gru_artifact(
                            spec,
                            library,
                            evidence_policy,
                            prefix,
                            seed,
                            calibration_source,
                            validation_source,
                            split_training,
                            split_early,
                            output,
                        )
                        completed[gru_key] = gru_artifact
                        _write_representation_work_index(
                            work_index_path, spec.suite_id, completed
                        )
                    else:
                        _verify_artifact(gru_artifact)
                    artifacts.append(gru_artifact)
                    if progress is not None:
                        progress(
                            f"v3 representation: {library.layout_id}/{evidence_policy}/"
                            f"{prefix}/gru-{seed}"
                        )
    manifest = MeasurementRepresentationManifest(
        suite_id=spec.suite_id,
        suite_hash=_suite_hash(spec, suite_path),
        source_v2_plan_hash=spec.v2.v2_rollout_plan_hash,
        source_v2_trace_index_hash=spec.v2.v2_trace_index_hash,
        calibration_data_hash=_hash_json(sorted(calibration_keys)),
        validation_data_hash=_hash_json(sorted(validation_keys)),
        artifacts=tuple(artifacts),
        complete=bool(artifacts),
    )
    _atomic_json(output / "measurement-representations.json", manifest.to_dict())
    return manifest


def fit_pairwise_decoders(
    suite: OfficialMeasurementAuditSuiteV3 | str | Path,
    representations: MeasurementRepresentationManifest | str | Path,
    output_dir: str | Path,
    *,
    progress: Any | None = None,
) -> MeasurementFitManifest:
    """Fit and freeze direct pairwise decision heads using calibration/validation only."""

    spec, suite_path, project_root = _suite_context(suite)
    source = _validate_v2_source_lock(spec, project_root)
    representation_manifest, representation_path = _resolve_representation_manifest(
        representations
    )
    if not representation_manifest.complete or representation_manifest.suite_id != spec.suite_id:
        raise ValueError("v3 representation manifest is incomplete or belongs to another suite")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    libraries = {
        item.layout_id: item
        for item in _load_response_libraries(source["response_matrices"])
    }
    groups: dict[tuple[str, str, str, str], list[MeasurementRepresentationArtifact]] = {}
    for artifact in representation_manifest.artifacts:
        key = (
            artifact.layout_id,
            artifact.evidence_policy,
            str(artifact.prefix),
            artifact.representation,
        )
        groups.setdefault(key, []).append(artifact)
    all_decoders: list[PairwiseDecisionDecoder] = []
    selections: list[dict[str, Any]] = []
    for (layout_id, evidence_policy, prefix_text, representation), artifacts in sorted(
        groups.items()
    ):
        library = libraries[cast(Any, layout_id)]
        work_signature = _hash_json(
            {
                "suite": spec.suite_id,
                "layout": layout_id,
                "evidence_policy": evidence_policy,
                "prefix": prefix_text,
                "representation": representation,
                "artifacts": [artifact.artifact_id for artifact in artifacts],
                "decoder": spec.decoder.to_dict(),
                "conflicts": library.conflicting_pairs_by_margin["0.02"],
            }
        )
        work_path = output / "decoder-work-units" / f"{work_signature}.json"
        if work_path.is_file():
            work = _read_json(work_path)
            if work.get("signature") != work_signature:
                raise ValueError("decoder work-unit signature changed")
            resumed_decoders = tuple(
                PairwiseDecisionDecoder.model_validate(item)
                for item in work.get("decoders", ())
            )
            if not resumed_decoders:
                raise ValueError("completed decoder work unit contains no pairwise heads")
            all_decoders.extend(resumed_decoders)
            selections.append(dict(work["selection"]))
            if progress is not None:
                progress(
                    f"v3 decoder resume: {layout_id}/{evidence_policy}/"
                    f"{prefix_text}/{representation}"
                )
            continue
        datasets: list[PairwiseDecoderDataset] = []
        for artifact in sorted(artifacts, key=lambda item: -1 if item.seed is None else item.seed):
            _verify_artifact(artifact)
            calibration, _calibration_keys, calibration_partners, _calibration_commitment = (
                load_encoded_histories(artifact.calibration_embeddings_path)
            )
            validation, _validation_keys, validation_partners, validation_commitment = (
                load_encoded_histories(artifact.validation_embeddings_path)
            )
            datasets.extend(
                _pairwise_datasets(
                    library,
                    artifact.seed,
                    calibration.embeddings,
                    calibration_partners,
                    validation.embeddings,
                    validation_partners,
                    validation_commitment,
                    prefix_text,
                )
            )
        selection = select_global_decoder_configuration(
            datasets,
            layout_id=layout_id,
            evidence_policy=evidence_policy,
            prefix=_parse_prefix(prefix_text),
            representation=representation,
            seed=None,
            ridge_strengths=spec.decoder.ridge_strengths,
            temperatures=spec.decoder.temperatures,
            prior_shrinkages=spec.decoder.prior_shrinkages,
            maximum_iterations=spec.decoder.maximum_iterations,
            convergence_tolerance=spec.decoder.convergence_tolerance,
        )
        all_decoders.extend(selection.decoders)
        selection_payload = {
            "layout_id": layout_id,
            "evidence_policy": evidence_policy,
            "prefix": _parse_prefix(prefix_text),
            "representation": representation,
            "configuration_id": selection.configuration_id,
            "ridge_strength": selection.ridge_strength,
            "temperature": selection.temperature,
            "prior_shrinkage": selection.prior_shrinkage,
            "mean_validation_decision_loss": selection.mean_validation_loss,
            "worst_pair_validation_decision_loss": selection.worst_pair_validation_loss,
            "decoder_count": len(selection.decoders),
        }
        selections.append(selection_payload)
        _atomic_json(
            work_path,
            {
                "schema_version": 1,
                "signature": work_signature,
                "selection": selection_payload,
                "decoders": [item.to_dict() for item in selection.decoders],
            },
        )
        if progress is not None:
            progress(
                f"v3 decoder: {layout_id}/{evidence_policy}/{prefix_text}/"
                f"{representation} -> {selection.configuration_id}"
            )
    decoder_payload = {
        "schema_version": 1,
        "suite_id": spec.suite_id,
        "selection_scope": "one_global_configuration_per_layout_policy_prefix_representation",
        "selection_order": [
            "mean_validation_decision_loss",
            "worst_pair_validation_decision_loss",
            "stronger_prior_shrinkage",
            "larger_ridge_penalty",
            "lexicographic_configuration_id",
        ],
        "selections": selections,
        "decoders": [item.to_dict() for item in all_decoders],
        "v2_confirmatory_used_for_tuning": False,
        "policy_training_performed": False,
    }
    decoder_path = output / "pairwise-decoders.json"
    _atomic_json(decoder_path, decoder_payload)
    decoder_hash = _sha256(decoder_path)
    representation_hash = (
        _sha256(representation_path)
        if representation_path is not None
        else _hash_json(representation_manifest.to_dict())
    )
    frozen_configuration_hash = _hash_json(
        {
            "suite_hash": _suite_hash(spec, suite_path),
            "representation_manifest_hash": representation_hash,
            "decoder_manifest_hash": decoder_hash,
            "algorithm": "stage6-v3-direct-pairwise-decision-v1",
        }
    )
    fit_manifest = MeasurementFitManifest(
        suite_id=spec.suite_id,
        suite_hash=_suite_hash(spec, suite_path),
        source_v2_plan_hash=spec.v2.v2_rollout_plan_hash,
        source_v2_trace_index_hash=spec.v2.v2_trace_index_hash,
        calibration_data_hash=representation_manifest.calibration_data_hash,
        validation_data_hash=representation_manifest.validation_data_hash,
        artifacts=representation_manifest.artifacts,
        decoder_manifest_path=str(decoder_path),
        decoder_manifest_hash=decoder_hash,
        frozen_configuration_hash=frozen_configuration_hash,
        complete=bool(all_decoders),
    )
    _atomic_json(output / "measurement-fit-manifest.json", fit_manifest.to_dict())
    return fit_manifest


def prepare_fresh_confirmation(
    suite: OfficialMeasurementAuditSuiteV3 | str | Path,
    fit_manifest: MeasurementFitManifest | str | Path,
    workspace: str | Path,
) -> OfficialConfirmationPlan:
    """Prepare exactly 9,600 trace-only episodes after the measurement hash is frozen."""

    spec, suite_path, project_root = _suite_context(suite)
    source = _validate_v2_source_lock(spec, project_root)
    fit, fit_path = _resolve_fit_manifest(fit_manifest)
    _validate_fit_manifest(spec, fit, fit_path, suite_path)
    if fit_path is None:
        raise ValueError("fresh confirmation requires a materialized immutable fit manifest")
    v2_plan = OfficialRolloutPlan.model_validate(source["rollout_plan"])
    inventory = load_official_asset_inventory(
        _resolve_source_path(project_root, spec.v2.v2_asset_inventory_path)
    )
    target = Path(workspace).resolve()
    target.mkdir(parents=True, exist_ok=True)
    _link_runtime_assets(v2_plan, target, spec)
    v2_keys = {key for shard in v2_plan.shards for key in shard.episode_keys}
    shards: list[OfficialRolloutShard] = []
    for layout in spec.layouts:
        partners = sorted(
            partner.partner_id
            for partner in inventory.partners
            if partner.layout_id == layout.layout_id
        )
        if len(partners) != layout.expected_partner_count:
            raise ValueError(f"v2 inventory partner count changed for {layout.layout_id}")
        for partner_id in partners:
            for evidence_policy in layout.evidence_policies:
                template = _trace_template(v2_plan, layout.layout_id, partner_id, evidence_policy)
                template_request = _read_json(Path(template.request_path))
                keys = tuple(
                    _fresh_environment_key(
                        spec.fresh_confirmation_salt,
                        layout.layout_id,
                        partner_id,
                        index,
                    )
                    for index in range(layout.fresh_episodes_per_partner_policy)
                )
                if any(key in v2_keys for key in keys):
                    raise ValueError("fresh confirmation key overlaps the complete v2 audit")
                shard_id = (
                    f"v3--{layout.layout_id}--trace--{_slug(partner_id)}--"
                    f"{_slug(evidence_policy)}--confirmatory"
                )
                request_path = target / "requests" / f"{shard_id}.json"
                result_path = target / "results" / "trace" / f"{shard_id}.json.gz"
                request_payload = dict(template_request)
                payload = dict(request_payload["payload"])
                payload.update(
                    {
                        "split": "confirmatory",
                        "episode_keys": list(keys),
                        "balanced_seats": True,
                        "deterministic": True,
                    }
                )
                request_payload.update(
                    {
                        "suite_id": spec.suite_id,
                        "policy_training_allowed": False,
                        "payload": payload,
                    }
                )
                request_payload.pop("request_hash", None)
                request_hash = _hash_json(request_payload)
                request_payload["request_hash"] = request_hash
                _atomic_json(request_path, request_payload)
                shards.append(
                    OfficialRolloutShard(
                        shard_id=shard_id,
                        kind="trace",
                        layout_id=layout.layout_id,
                        request_path=str(request_path),
                        result_path=str(result_path),
                        request_hash=request_hash,
                        partner_id=partner_id,
                        evidence_policy=evidence_policy,
                        split="confirmatory",
                        episode_keys=keys,
                        deterministic=True,
                    )
                )
    if sum(len(shard.episode_keys) for shard in shards) != 9600:
        raise ValueError("fresh confirmation plan does not contain exactly 9,600 episodes")
    suite_file = _resolve_source_path(project_root, spec.v2.v2_suite_path)
    internal_payload = {
        "schema_version": 1,
        "suite_id": spec.suite_id,
        "suite_path": str(suite_file),
        "suite_hash": _suite_hash(spec, suite_path),
        "inventory_path": str(
            _resolve_source_path(project_root, spec.v2.v2_asset_inventory_path)
        ),
        "inventory_hash": spec.v2.v2_inventory_hash,
        "workspace": str(target),
        "shards": [shard.to_dict() for shard in shards],
    }
    rollout_plan = OfficialRolloutPlan.model_validate(
        {**internal_payload, "plan_hash": _hash_json(internal_payload)}
    )
    _atomic_json(target / "official-rollout-plan.json", rollout_plan.to_dict())
    fresh_keys = {key for shard in shards for key in shard.episode_keys}
    public_payload = {
        "schema_version": 1,
        "suite_id": spec.suite_id,
        "suite_path": str(suite_path) if suite_path is not None else "<in-memory>",
        "suite_hash": _suite_hash(spec, suite_path),
        "frozen_configuration_hash": fit.frozen_configuration_hash,
        "fit_manifest_path": str(fit_path),
        "fit_manifest_hash": _sha256(fit_path),
        "source_v2_plan_hash": spec.v2.v2_rollout_plan_hash,
        "source_v2_environment_key_hash": _hash_json(sorted(v2_keys)),
        "fresh_environment_key_hash": _hash_json(sorted(fresh_keys)),
        "workspace": str(target),
        "rollout_plan": rollout_plan.to_dict(),
    }
    plan = OfficialConfirmationPlan.model_validate(
        {**public_payload, "plan_hash": _hash_json(public_payload)}
    )
    _atomic_json(target / "official-confirmation-plan.json", plan.to_dict())
    ledger = OfficialConfirmationLedger(
        suite_id=spec.suite_id,
        plan_hash=plan.plan_hash,
        frozen_configuration_hash=fit.frozen_configuration_hash,
        entries=tuple(
            OfficialRolloutLedgerEntry(shard_id=shard.shard_id, status="pending")
            for shard in shards
        ),
        complete=False,
    )
    _atomic_json(target / "official-confirmation-ledger.json", ledger.to_dict())
    return plan


def run_fresh_confirmation(
    plan: OfficialConfirmationPlan | str | Path,
    *,
    workers: int = 2,
    resume: bool = True,
    executor: Any | None = None,
) -> OfficialConfirmationLedger:
    """Run or resume trace-only confirmation shards through the established CPU runner."""

    confirmation = _resolve_confirmation_plan(plan)
    _validate_confirmation_plan_integrity(confirmation)
    if any(shard.kind != "trace" for shard in confirmation.rollout_plan.shards):
        raise ValueError("Stage 6 v3 confirmation structurally permits trace shards only")
    for shard in confirmation.rollout_plan.shards:
        request = _read_json(Path(shard.request_path))
        if request.get("policy_training_allowed") is not False:
            raise ValueError("fresh confirmation request does not forbid policy training")
    rollout_ledger = run_official_rollouts(
        confirmation.rollout_plan,
        workers=workers,
        resume=resume,
        kinds=("trace",),
        executor=executor,
    )
    ledger = _confirmation_ledger(confirmation, rollout_ledger)
    _atomic_json(
        Path(confirmation.workspace) / "official-confirmation-ledger.json",
        ledger.to_dict(),
    )
    return ledger


def get_fresh_confirmation_status(
    plan: OfficialConfirmationPlan | str | Path,
) -> OfficialConfirmationLedger:
    confirmation = _resolve_confirmation_plan(plan)
    _validate_confirmation_plan_integrity(confirmation)
    runtime_ledger_path = Path(confirmation.workspace) / "official-rollout-ledger.json"
    if runtime_ledger_path.is_file():
        rollout = OfficialRolloutLedger.model_validate(_read_json(runtime_ledger_path))
        return _confirmation_ledger(confirmation, rollout)
    path = Path(confirmation.workspace) / "official-confirmation-ledger.json"
    if path.is_file():
        ledger = OfficialConfirmationLedger.model_validate(_read_json(path))
        if ledger.plan_hash != confirmation.plan_hash:
            raise ValueError("confirmation ledger belongs to a different plan")
        return ledger
    rollout = get_official_rollout_status(confirmation.rollout_plan)
    return _confirmation_ledger(confirmation, rollout)


def _fit_event_artifact(
    spec: OfficialMeasurementAuditSuiteV3,
    library: OfficialResponseValueMatrix,
    evidence_policy: str,
    prefix: int | str,
    calibration_source: Any,
    validation_source: Any,
    output: Path,
) -> MeasurementRepresentationArtifact:
    unit = f"{library.layout_id}--{_slug(evidence_policy)}--{_slug(str(prefix))}--event"
    calibration_matrix = event_representation_matrix(
        calibration_source.episodes,
        prefix,
        width=spec.representations.event_feature_width,
        salt=spec.representations.event_hash_salt,
    )
    validation_matrix = event_representation_matrix(
        validation_source.episodes,
        prefix,
        width=spec.representations.event_feature_width,
        salt=spec.representations.event_hash_salt,
    )
    calibration_encoded = EncodedHistories(
        embeddings=calibration_matrix,
        identity_logits=np.zeros((len(calibration_matrix), len(library.partner_ids))),
        labels=np.asarray(calibration_source.labels, dtype=np.int64),
        row_indices=np.arange(len(calibration_matrix), dtype=np.int64),
    )
    validation_encoded = EncodedHistories(
        embeddings=validation_matrix,
        identity_logits=np.zeros((len(validation_matrix), len(library.partner_ids))),
        labels=np.asarray(validation_source.labels, dtype=np.int64),
        row_indices=np.arange(len(validation_matrix), dtype=np.int64),
    )
    calibration_path = output / "representations" / f"{unit}--calibration.npz"
    validation_path = output / "representations" / f"{unit}--validation.npz"
    calibration_hash = _save_source_encoding(
        calibration_path, calibration_encoded, calibration_source
    )
    validation_hash = _save_source_encoding(
        validation_path, validation_encoded, validation_source
    )
    return MeasurementRepresentationArtifact(
        artifact_id=_hash_json({"unit": unit, "calibration": calibration_hash}),
        layout_id=library.layout_id,
        evidence_policy=evidence_policy,
        prefix=cast(MeasurementPrefix, prefix),
        representation="event",
        calibration_embeddings_path=str(calibration_path),
        calibration_embeddings_hash=calibration_hash,
        validation_embeddings_path=str(validation_path),
        validation_embeddings_hash=validation_hash,
        calibration_key_hash=_source_key_hash(calibration_source),
        validation_key_hash=_source_key_hash(validation_source),
        feature_width=spec.representations.event_feature_width,
    )


def _fit_gru_artifact(
    spec: OfficialMeasurementAuditSuiteV3,
    library: OfficialResponseValueMatrix,
    evidence_policy: str,
    prefix: int | str,
    seed: int,
    calibration_source: Any,
    validation_source: Any,
    split_training: Any,
    split_early: Any,
    output: Path,
) -> MeasurementRepresentationArtifact:
    unit = (
        f"{library.layout_id}--{_slug(evidence_policy)}--{_slug(str(prefix))}--gru-{seed}"
    )
    signature = _hash_json(
        {
            "suite": spec.suite_id,
            "unit": unit,
            "calibration_keys": _source_key_hash(calibration_source),
            "training_fraction": spec.representations.encoder_training_fraction,
            "feature_contract": (
                "ego_observation+previous_ego_action+visible_partner_action+reward+step"
            ),
        }
    )
    checkpoint = fit_gru_identity_representation(
        split_training,
        split_early,
        mode_count=len(library.partner_ids),
        hidden_size=spec.representations.gru_hidden_size,
        seed=seed,
        signature=signature,
        checkpoint_path=output / "encoders" / f"{unit}.pt",
    )
    calibration_encoded = encode_with_frozen_identity_representation(
        checkpoint, calibration_source, signature=signature
    )
    validation_encoded = encode_with_frozen_identity_representation(
        checkpoint, validation_source, signature=signature
    )
    identity_temperature = _identity_temperature(
        validation_encoded.identity_logits,
        validation_encoded.labels,
        spec.decoder.temperatures,
    )
    calibration_path = output / "representations" / f"{unit}--calibration.npz"
    validation_path = output / "representations" / f"{unit}--validation.npz"
    calibration_hash = _save_source_encoding(
        calibration_path, calibration_encoded, calibration_source
    )
    validation_hash = _save_source_encoding(
        validation_path, validation_encoded, validation_source
    )
    return MeasurementRepresentationArtifact(
        artifact_id=_hash_json(
            {"unit": unit, "encoder": checkpoint.checkpoint_hash, "calibration": calibration_hash}
        ),
        layout_id=library.layout_id,
        evidence_policy=evidence_policy,
        prefix=cast(MeasurementPrefix, prefix),
        representation="gru",
        seed=seed,
        identity_temperature=identity_temperature,
        encoder_checkpoint_path=str(checkpoint.checkpoint_path),
        encoder_checkpoint_hash=checkpoint.checkpoint_hash,
        calibration_embeddings_path=str(calibration_path),
        calibration_embeddings_hash=calibration_hash,
        validation_embeddings_path=str(validation_path),
        validation_embeddings_hash=validation_hash,
        calibration_key_hash=_source_key_hash(calibration_source),
        validation_key_hash=_source_key_hash(validation_source),
        feature_width=checkpoint.hidden_size,
    )


def _pairwise_datasets(
    library: OfficialResponseValueMatrix,
    seed: int | None,
    calibration_features: np.ndarray,
    calibration_partner_ids: np.ndarray,
    validation_features: np.ndarray,
    validation_partner_ids: np.ndarray,
    validation_commitment: np.ndarray,
    prefix: str,
) -> list[PairwiseDecoderDataset]:
    losses = np.asarray(library.normalized_losses, dtype=np.float64)
    partner_index = {partner: index for index, partner in enumerate(library.partner_ids)}
    output: list[PairwiseDecoderDataset] = []
    for left, right in library.conflicting_pairs_by_margin["0.02"]:
        calibration_mask = np.isin(calibration_partner_ids, (left, right))
        validation_mask = np.isin(validation_partner_ids, (left, right))
        output.append(
            PairwiseDecoderDataset(
                seed=seed,
                left_partner_id=left,
                right_partner_id=right,
                calibration_features=calibration_features[calibration_mask],
                calibration_labels=(calibration_partner_ids[calibration_mask] == right).astype(
                    np.int64
                ),
                validation_features=validation_features[validation_mask],
                validation_labels=(validation_partner_ids[validation_mask] == right).astype(
                    np.int64
                ),
                loss_matrix=losses[[partner_index[left], partner_index[right]]],
                validation_censored=(
                    ~validation_commitment[validation_mask]
                    if prefix == "pre_commitment"
                    else None
                ),
            )
        )
    return output


def _save_source_encoding(path: Path, encoded: EncodedHistories, source: Any) -> str:
    return save_encoded_histories(
        path,
        encoded,
        environment_keys=[episode.environment_key for episode in source.episodes],
        partner_ids=[episode.partner_id for episode in source.episodes],
        commitment_reached=[episode.commitment_reached for episode in source.episodes],
    )


def _validate_v2_source_lock(
    spec: OfficialMeasurementAuditSuiteV3,
    project_root: Path,
) -> dict[str, Any]:
    paths_and_hashes = {
        "v2_suite": (spec.v2.v2_suite_path, spec.v2.v2_suite_hash),
        "v2_rollout_plan": (
            spec.v2.v2_rollout_plan_path,
            spec.v2.v2_rollout_plan_file_hash,
        ),
        "v2_rollout_ledger": (
            spec.v2.v2_rollout_ledger_path,
            spec.v2.v2_rollout_ledger_hash,
        ),
        "v2_manifest": (spec.v2.v2_manifest_path, spec.v2.v2_manifest_hash),
        "v2_trace_index": (spec.v2.v2_trace_index_path, spec.v2.v2_trace_index_hash),
        "response_matrices": (
            spec.v2.response_value_matrices_path,
            spec.v2.response_value_matrices_hash,
        ),
        "method_evaluation": (
            spec.v2.official_method_evaluation_path,
            spec.v2.official_method_evaluation_hash,
        ),
        "exclusions": (spec.v2.exclusions_path, spec.v2.exclusions_hash),
    }
    loaded: dict[str, Any] = {}
    observed_hashes: dict[str, str] = {}
    for name, (raw_path, expected_hash) in paths_and_hashes.items():
        path = _resolve_source_path(project_root, raw_path)
        observed = _sha256(path)
        if observed != expected_hash:
            raise ValueError(f"immutable v2 source hash changed: {name}")
        loaded[name] = _read_json(path)
        observed_hashes[name] = observed
    plan = OfficialRolloutPlan.model_validate(loaded["v2_rollout_plan"])
    ledger = OfficialRolloutLedger.model_validate(loaded["v2_rollout_ledger"])
    manifest = loaded["v2_manifest"]
    if plan.plan_hash != spec.v2.v2_rollout_plan_hash or ledger.plan_hash != plan.plan_hash:
        raise ValueError("v2 plan/ledger provenance no longer matches the frozen protocol")
    if not ledger.complete or manifest.get("status") != "complete":
        raise ValueError("v3 requires the completed immutable v2 audit")
    if manifest.get("verdict") != "redesign" or manifest.get("source_hash") != (
        spec.v2.v2_source_hash
    ):
        raise ValueError("v2 failed-audit verdict/source hash changed")
    inventory = load_official_asset_inventory(
        _resolve_source_path(project_root, spec.v2.v2_asset_inventory_path)
    )
    if inventory.inventory_hash != spec.v2.v2_inventory_hash or not inventory.complete:
        raise ValueError("v2 official asset inventory changed or is incomplete")
    return {
        "suite": loaded["v2_suite"],
        "rollout_plan": loaded["v2_rollout_plan"],
        "rollout_ledger": loaded["v2_rollout_ledger"],
        "manifest": manifest,
        "trace_index": loaded["v2_trace_index"],
        "response_matrices": loaded["response_matrices"],
        "method_evaluation": loaded["method_evaluation"],
        "exclusions": loaded["exclusions"],
        "hashes": observed_hashes,
    }


def _validate_fit_manifest(
    spec: OfficialMeasurementAuditSuiteV3,
    fit: MeasurementFitManifest,
    fit_path: Path | None,
    suite_path: Path | None,
) -> None:
    if not fit.complete or fit.suite_id != spec.suite_id:
        raise ValueError("fresh confirmation requires a complete v3 fit")
    if fit.suite_hash != _suite_hash(spec, suite_path):
        raise ValueError("fit manifest was produced from a different v3 suite")
    if fit.source_v2_plan_hash != spec.v2.v2_rollout_plan_hash:
        raise ValueError("fit manifest references a different v2 rollout plan")
    decoder_path = Path(fit.decoder_manifest_path)
    if not decoder_path.is_file() or _sha256(decoder_path) != fit.decoder_manifest_hash:
        raise ValueError("frozen pairwise decoder manifest integrity check failed")
    if fit_path is not None and not fit_path.is_file():
        raise ValueError("materialized fit manifest is missing")


def _validate_confirmation_plan_integrity(plan: OfficialConfirmationPlan) -> None:
    fit_path = Path(plan.fit_manifest_path)
    if not fit_path.is_file() or _sha256(fit_path) != plan.fit_manifest_hash:
        raise ValueError("frozen measurement configuration changed after confirmation planning")
    fit = MeasurementFitManifest.model_validate(_read_json(fit_path))
    if fit.frozen_configuration_hash != plan.frozen_configuration_hash:
        raise ValueError("confirmation plan and frozen measurement configuration differ")
    rollout_payload = plan.rollout_plan.to_dict()
    observed_rollout_hash = str(rollout_payload.pop("plan_hash"))
    if observed_rollout_hash != _hash_json(rollout_payload):
        raise ValueError("internal confirmation rollout plan is invalid")
    public_payload = plan.to_dict()
    observed_public_hash = str(public_payload.pop("plan_hash"))
    if observed_public_hash != _hash_json(public_payload):
        raise ValueError("public confirmation plan integrity check failed")


def _confirmation_ledger(
    plan: OfficialConfirmationPlan,
    ledger: OfficialRolloutLedger,
) -> OfficialConfirmationLedger:
    if ledger.plan_hash != plan.rollout_plan.plan_hash:
        raise ValueError("runtime ledger belongs to another confirmation rollout plan")
    return OfficialConfirmationLedger(
        suite_id=plan.suite_id,
        plan_hash=plan.plan_hash,
        frozen_configuration_hash=plan.frozen_configuration_hash,
        entries=ledger.entries,
        complete=ledger.complete,
        failed_shards=ledger.failed_shards,
    )


def _load_response_libraries(payload: Mapping[str, Any]) -> tuple[OfficialResponseValueMatrix, ...]:
    matrices = payload.get("matrices")
    if not isinstance(matrices, list) or not matrices:
        raise ValueError("v2 response-matrix artifact is invalid")
    return tuple(OfficialResponseValueMatrix.model_validate(item) for item in matrices)


def _tuning_trace_index(trace_index: OfficialTraceIndex) -> OfficialTraceIndex:
    """Return the immutable v2 calibration/validation boundary used for all fitting."""

    allowed = {"calibration", "validation"}
    entries = tuple(entry for entry in trace_index.entries if entry.split in allowed)
    if not entries:
        raise ValueError("v3 fitting requires v2 calibration and validation traces")
    if {entry.split for entry in entries} != allowed:
        raise ValueError("v3 fitting requires both calibration and validation traces")
    return trace_index.model_copy(update={"entries": entries})


def _require_disjoint_sources(left: Any, right: Any) -> None:
    left_keys = {(episode.partner_id, episode.environment_key) for episode in left.episodes}
    right_keys = {(episode.partner_id, episode.environment_key) for episode in right.episodes}
    if left_keys & right_keys:
        raise ValueError("v3 calibration and validation histories overlap")


def _verify_artifact(artifact: MeasurementRepresentationArtifact) -> None:
    for path_value, expected in (
        (artifact.calibration_embeddings_path, artifact.calibration_embeddings_hash),
        (artifact.validation_embeddings_path, artifact.validation_embeddings_hash),
    ):
        path = Path(path_value)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen representation artifact changed: {artifact.artifact_id}")
    if artifact.encoder_checkpoint_path is not None:
        checkpoint = Path(artifact.encoder_checkpoint_path)
        if (
            artifact.encoder_checkpoint_hash is None
            or not checkpoint.is_file()
            or _sha256(checkpoint) != artifact.encoder_checkpoint_hash
        ):
            raise ValueError(f"frozen encoder artifact changed: {artifact.artifact_id}")


def _write_representation_work_index(
    path: Path,
    suite_id: str,
    completed: Mapping[str, MeasurementRepresentationArtifact],
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "suite_id": suite_id,
            "work_units": {
                key: artifact.to_dict() for key, artifact in sorted(completed.items())
            },
        },
    )


def _trace_template(
    plan: OfficialRolloutPlan,
    layout_id: str,
    partner_id: str,
    evidence_policy: str,
) -> OfficialRolloutShard:
    matches = [
        shard
        for shard in plan.shards
        if shard.kind == "trace"
        and shard.layout_id == layout_id
        and shard.partner_id == partner_id
        and shard.evidence_policy == evidence_policy
        and shard.split == "calibration"
    ]
    if len(matches) != 1:
        raise ValueError("v2 trace template is missing or ambiguous")
    return matches[0]


def _link_runtime_assets(
    source_plan: OfficialRolloutPlan,
    target: Path,
    suite: OfficialMeasurementAuditSuiteV3,
) -> None:
    source_workspace = Path(source_plan.workspace)
    for relative in (suite.runtime.upstream_directory, suite.runtime.asset_directory):
        source = (source_workspace / relative).resolve()
        destination = target / relative
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError("confirmation workspace contains a different runtime asset link")
            continue
        if not source.exists():
            raise ValueError(f"v2 runtime asset is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=True)


def _fresh_environment_key(
    salt: str,
    layout_id: str,
    partner_id: str,
    index: int,
) -> int:
    digest = hashlib.sha256(
        f"{salt}:{layout_id}:{partner_id}:{index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _source_key_hash(source: Any) -> str:
    return _hash_json(
        sorted((episode.partner_id, episode.environment_key) for episode in source.episodes)
    )


def _fresh_episode_count(spec: OfficialMeasurementAuditSuiteV3) -> int:
    return sum(
        layout.expected_partner_count
        * len(layout.evidence_policies)
        * layout.fresh_episodes_per_partner_policy
        for layout in spec.layouts
    )


def _resolve_representation_manifest(
    value: MeasurementRepresentationManifest | str | Path,
) -> tuple[MeasurementRepresentationManifest, Path | None]:
    if isinstance(value, MeasurementRepresentationManifest):
        return value, None
    path = Path(value).resolve()
    return MeasurementRepresentationManifest.model_validate(_read_json(path)), path


def _resolve_fit_manifest(
    value: MeasurementFitManifest | str | Path,
) -> tuple[MeasurementFitManifest, Path | None]:
    if isinstance(value, MeasurementFitManifest):
        return value, None
    path = Path(value).resolve()
    return MeasurementFitManifest.model_validate(_read_json(path)), path


def _resolve_confirmation_plan(
    value: OfficialConfirmationPlan | str | Path,
) -> OfficialConfirmationPlan:
    if isinstance(value, OfficialConfirmationPlan):
        return value
    return OfficialConfirmationPlan.model_validate(_read_json(Path(value)))


def _suite_context(
    value: OfficialMeasurementAuditSuiteV3 | str | Path,
) -> tuple[OfficialMeasurementAuditSuiteV3, Path | None, Path]:
    if isinstance(value, OfficialMeasurementAuditSuiteV3):
        return value, None, Path(__file__).resolve().parents[2]
    path = Path(value).resolve()
    return load_official_measurement_suite(path), path, path.parents[2]


def _suite_hash(spec: OfficialMeasurementAuditSuiteV3, path: Path | None) -> str:
    return _sha256(path) if path is not None else _hash_json(spec.to_dict())


def _resolve_source_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _parse_prefix(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _identity_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    candidates: Sequence[float],
) -> float:
    selected = (float("inf"), 1.0)
    for temperature in candidates:
        scaled = logits / float(temperature)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss = -float(np.mean(log_probabilities[np.arange(len(labels)), labels]))
        key = (loss, float(temperature))
        if key < selected:
            selected = key
    return selected[1]


def _slug(value: str) -> str:
    return value.replace(":", "-").replace("_", "-")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
