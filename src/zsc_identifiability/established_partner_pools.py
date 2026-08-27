"""Deterministic, resumable Stage 6 partner-pool construction."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from zsc_identifiability.established_models import (
    EstablishedValidationSuite,
    FrozenPartnerPoolBundle,
    FrozenPartnerPoolManifest,
    PartnerCandidateLedgerEntry,
    PartnerCheckpoint,
    PartnerPoolBuildLedger,
    PartnerPoolBuildPlan,
    PartnerPoolBuildStatus,
    PartnerPoolCandidatePlan,
    PartnerPoolSplitStatus,
    PartnerPoolStage,
    SplitName,
    load_established_suite_file,
)
from zsc_identifiability.established_partners import (
    reward_vector_hash,
    split_for_reward_vector,
    vectors_for_split,
)
from zsc_identifiability.established_runtime import (
    dispatch_runtime_request,
    forward_runtime_process_signals,
    validate_upstreams,
    write_runtime_request,
)

SPLITS: tuple[SplitName, ...] = ("train", "validation", "evaluation")
SEED_BASES: dict[SplitName, int] = {
    "train": 41_001,
    "validation": 141_001,
    "evaluation": 241_001,
}
COMPETENCE_KEY_START = 1_729_000

CandidateExecutor = Callable[
    [PartnerPoolBuildPlan, PartnerPoolCandidatePlan, PartnerPoolStage], PartnerCheckpoint
]


def prepare_partner_pool_build(
    suite: EstablishedValidationSuite,
    *,
    suite_path: str | Path,
    layout: str,
    workspace: str | Path,
    project_root: str | Path | None = None,
) -> PartnerPoolBuildPlan:
    """Materialize a deterministic plan and empty ledger without running training."""

    root = _project_root(project_root)
    source = Path(suite_path).resolve()
    destination = Path(workspace).resolve()
    if layout not in {item.layout_id for item in suite.layouts}:
        raise ValueError(f"unknown Stage 6 layout: {layout!r}")
    upstream = validate_upstreams(suite, root)
    if not upstream.passed:
        raise RuntimeError("partner-pool planning requires validated pinned upstreams and runtimes")
    suite_hash = _path_hash(source)
    quotas = _quotas(suite)
    caps = _caps(suite)
    candidates: list[PartnerPoolCandidatePlan] = []
    for split in SPLITS:
        vectors = vectors_for_split(suite, split)
        required_vectors = caps[split] // suite.partner_generation.seeds_per_reward_vector
        if len(vectors) < required_vectors:
            raise ValueError(f"insufficient hash-assigned reward vectors for {split}")
        candidate_index = 0
        for vector_index, vector in enumerate(vectors[:required_vectors]):
            vector_hash = reward_vector_hash(vector)
            for replicate in range(suite.partner_generation.seeds_per_reward_vector):
                seed = partner_seed(split, vector_index, replicate)
                candidates.append(
                    PartnerPoolCandidatePlan(
                        candidate_id=f"{split}-{vector_hash[:12]}-seed{seed}",
                        split=split,
                        candidate_index=candidate_index,
                        reward_vector_index=vector_index,
                        replicate=replicate,
                        seed=seed,
                        reward_vector={key: float(value) for key, value in vector.items()},
                        reward_vector_hash=vector_hash,
                        initially_active=candidate_index < quotas[split],
                    )
                )
                candidate_index += 1
    runtime_source_hash = _path_hash(
        root
        / "phase-6-established-validation"
        / "runtime-overcookedv2"
        / "src"
        / "stage6_overcooked_runtime"
    )
    orchestrator_source_hash = _path_hash(root / "src" / "zsc_identifiability")
    plan_payload: dict[str, Any] = {
        "schema_version": 1,
        "suite_path": str(source),
        "suite_id": suite.suite_id,
        "suite_hash": suite_hash,
        "layout_id": layout,
        "workspace": str(destination),
        "project_root": str(root),
        "upstream_commits": {item.repository_id: item.commit for item in suite.upstreams},
        "orchestrator_source_hash": orchestrator_source_hash,
        "runtime_source_hash": runtime_source_hash,
        "quotas": quotas,
        "caps": caps,
        "expansion_block_size": suite.partner_generation.expansion_block_size,
        "screen_transitions": suite.partner_generation.screen_transitions,
        "finalist_transitions": suite.partner_generation.finalist_transitions,
        "validation_rollouts": suite.partner_generation.validation_rollouts,
        "minimum_correct_delivery_rate": (
            suite.partner_generation.minimum_correct_delivery_rate
        ),
        "competence_environment_keys": tuple(
            range(
                COMPETENCE_KEY_START,
                COMPETENCE_KEY_START + suite.partner_generation.validation_rollouts,
            )
        ),
        "candidates": tuple(item.to_dict() for item in candidates),
    }
    plan_payload["plan_hash"] = _canonical_hash(plan_payload)
    plan = PartnerPoolBuildPlan.model_validate(plan_payload)
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / "build-plan.json"
    if plan_path.exists():
        existing = load_partner_pool_build_plan(plan_path)
        if existing != plan:
            raise ValueError("partner-pool workspace already contains a different immutable plan")
    else:
        _atomic_write_json(plan_path, plan.to_dict())
    ledger_path = destination / "ledger.json"
    if not ledger_path.exists():
        ledger = PartnerPoolBuildLedger(
            plan_hash=plan.plan_hash,
            entries=tuple(
                PartnerCandidateLedgerEntry(
                    candidate_id=item.candidate_id,
                    status="pending_screen" if item.initially_active else "inactive",
                    active=item.initially_active,
                )
                for item in plan.candidates
            ),
            updated_at_utc=_now(),
        )
        _atomic_write_json(ledger_path, ledger.to_dict())
    else:
        _load_ledger(plan)
    return plan


def partner_seed(split: SplitName, vector_index: int, replicate: int) -> int:
    if vector_index < 0 or replicate not in {0, 1}:
        raise ValueError("partner seed requires a non-negative vector index and replicate 0 or 1")
    return SEED_BASES[split] + 2 * vector_index + replicate


def load_partner_pool_build_plan(path: str | Path) -> PartnerPoolBuildPlan:
    payload = _read_object(Path(path))
    plan = PartnerPoolBuildPlan.model_validate(payload)
    unhashed = plan.to_dict()
    observed = unhashed.pop("plan_hash")
    if _canonical_hash(unhashed) != observed:
        raise ValueError("partner-pool build plan hash mismatch")
    return plan


def run_partner_pool_build(
    plan: PartnerPoolBuildPlan | str | Path,
    *,
    splits: Sequence[SplitName] = SPLITS,
    workers: int = 1,
    freeze_on_success: bool = False,
    executor: CandidateExecutor | None = None,
) -> PartnerPoolBuildLedger:
    """Run or resume selected split queues; callers decide whether to detach."""

    resolved = _coerce_plan(plan)
    selected_splits = tuple(dict.fromkeys(splits))
    if not selected_splits or any(item not in SPLITS for item in selected_splits):
        raise ValueError("partner-pool run requires valid non-empty splits")
    if workers < 1 or workers > 4:
        raise ValueError("partner-pool workers must be between one and four")
    _validate_plan_environment(resolved)
    execute = executor or _execute_candidate
    with _workspace_lock(resolved), forward_runtime_process_signals():
        ledger = _recover_interrupted_entries(resolved, _load_ledger(resolved))
        attempted: set[tuple[str, PartnerPoolStage]] = set()
        for split in selected_splits:
            while True:
                runnable: list[
                    tuple[
                        PartnerPoolCandidatePlan,
                        PartnerCandidateLedgerEntry,
                        PartnerPoolStage,
                    ]
                ] = []
                for candidate in resolved.candidates:
                    if candidate.split != split:
                        continue
                    entry = _entry_by_id(ledger, candidate.candidate_id)
                    stage = _pending_stage(entry)
                    if (
                        entry.active
                        and stage is not None
                        and (candidate.candidate_id, stage) not in attempted
                    ):
                        runnable.append((candidate, entry, stage))
                if runnable:
                    for _candidate, entry, stage in runnable:
                        ledger = _replace_entry(
                            ledger,
                            entry.model_copy(
                                update={
                                    "status": (
                                        "screen_running"
                                        if stage == "screen"
                                        else "finalist_running"
                                    )
                                }
                            ),
                        )
                    _write_ledger(resolved, ledger)
                    if workers == 1:
                        for candidate, entry, stage in runnable:
                            updated, completed_stage = _execute_stage(
                                resolved, candidate, entry, stage, execute
                            )
                            attempted.add((updated.candidate_id, completed_stage))
                            ledger = _replace_entry(ledger, updated)
                            _write_ledger(resolved, ledger)
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            futures = {
                                pool.submit(
                                    _execute_stage,
                                    resolved,
                                    candidate,
                                    entry,
                                    stage,
                                    execute,
                                ): (candidate.candidate_id, stage)
                                for candidate, entry, stage in runnable
                            }
                            for future in as_completed(futures):
                                updated, completed_stage = future.result()
                                attempted.add((updated.candidate_id, completed_stage))
                                ledger = _replace_entry(ledger, updated)
                                _write_ledger(resolved, ledger)
                    continue
                if _eligible_count(ledger, resolved, split) >= resolved.quotas[split]:
                    break
                inactive = [
                    candidate
                    for candidate in resolved.candidates
                    if candidate.split == split
                    and not _entry_by_id(ledger, candidate.candidate_id).active
                ]
                if not inactive:
                    break
                activation = inactive[: resolved.expansion_block_size]
                for candidate in activation:
                    entry = _entry_by_id(ledger, candidate.candidate_id).model_copy(
                        update={"active": True, "status": "pending_screen"}
                    )
                    ledger = _replace_entry(ledger, entry)
                _write_ledger(resolved, ledger)
        status = get_partner_pool_status(resolved, ledger=ledger)
        if freeze_on_success and status.complete:
            freeze_partner_pools(resolved, ledger=ledger, lock_held=True)
            ledger = _load_ledger(resolved)
        return ledger


def get_partner_pool_status(
    plan: PartnerPoolBuildPlan | str | Path,
    *,
    ledger: PartnerPoolBuildLedger | None = None,
) -> PartnerPoolBuildStatus:
    resolved = _coerce_plan(plan)
    current = _load_ledger(resolved) if ledger is None else ledger
    split_statuses: list[PartnerPoolSplitStatus] = []
    for split in SPLITS:
        entries = [
            _entry_by_id(current, candidate.candidate_id)
            for candidate in resolved.candidates
            if candidate.split == split
        ]
        active = [item for item in entries if item.active]
        eligible = sum(item.status == "eligible" for item in active)
        rejected = sum(item.status in {"screen_rejected", "finalist_rejected"} for item in active)
        failed = sum(item.status == "failed" for item in active)
        pending = sum(
            item.status
            in {"pending_screen", "screen_running", "pending_finalist", "finalist_running"}
            for item in active
        )
        split_statuses.append(
            PartnerPoolSplitStatus(
                split=split,
                quota=resolved.quotas[split],
                cap=resolved.caps[split],
                active=len(active),
                eligible=eligible,
                rejected=rejected,
                failed=failed,
                pending=pending,
                quota_met=eligible >= resolved.quotas[split],
                cap_exhausted=len(active) == resolved.caps[split],
            )
        )
    unresolved = sum(item.failed for item in split_statuses)
    return PartnerPoolBuildStatus(
        plan_hash=resolved.plan_hash,
        splits=tuple(split_statuses),
        complete=(
            all(item.quota_met and item.pending == 0 for item in split_statuses)
            and unresolved == 0
        ),
        frozen=current.frozen_bundle_path is not None,
        unresolved_failures=unresolved,
    )


def freeze_partner_pools(
    plan: PartnerPoolBuildPlan | str | Path,
    *,
    ledger: PartnerPoolBuildLedger | None = None,
    lock_held: bool = False,
) -> FrozenPartnerPoolBundle:
    resolved = _coerce_plan(plan)
    context = _null_context() if lock_held else _workspace_lock(resolved)
    with context:
        current = _load_ledger(resolved) if ledger is None else ledger
        existing = _existing_frozen_bundle(resolved, current)
        if existing is not None:
            return existing
        _validate_plan_environment(resolved)
        status = get_partner_pool_status(resolved, ledger=current)
        if not status.complete:
            raise RuntimeError(
                "partner pools cannot freeze before all quotas pass without failures"
            )
        timestamp = _freeze_timestamp(resolved)
        frozen_root = Path(resolved.workspace) / "frozen"
        manifests: dict[SplitName, FrozenPartnerPoolManifest] = {}
        for split in SPLITS:
            candidates = [item for item in resolved.candidates if item.split == split]
            eligible = [
                (candidate, _entry_by_id(current, candidate.candidate_id).finalist_checkpoint)
                for candidate in candidates
                if _entry_by_id(current, candidate.candidate_id).active
                and _entry_by_id(current, candidate.candidate_id).status == "eligible"
            ]
            checkpoints = tuple(
                checkpoint
                for candidate, checkpoint in eligible
                if checkpoint is not None and _verify_finalist(resolved, candidate, checkpoint)
            )
            policy: Literal["exact_quota", "all_processed_eligible"] = (
                "all_processed_eligible" if split == "evaluation" else "exact_quota"
            )
            if split != "evaluation":
                checkpoints = checkpoints[: resolved.quotas[split]]
            manifests[split] = FrozenPartnerPoolManifest(
                suite_id=resolved.suite_id,
                suite_hash=resolved.suite_hash,
                plan_hash=resolved.plan_hash,
                layout_id=resolved.layout_id,
                split=split,
                selection_policy=policy,
                quota=resolved.quotas[split],
                checkpoints=checkpoints,
                checkpoint_hashes=tuple(
                    str(item.checkpoint_content_hash) for item in checkpoints
                ),
                frozen_at_utc=timestamp,
            )
        leakage = _audit_frozen_leakage(manifests.values())
        if not leakage["passed"]:
            raise ValueError("frozen partner pools fail the cross-split leakage audit")
        pool_names: dict[SplitName, str] = {
            "train": "train-pool.json",
            "validation": "validation-pool.json",
            "evaluation": "evaluation-candidates.json",
        }
        pool_paths: dict[SplitName, str] = {}
        pool_hashes: dict[SplitName, str] = {}
        for split in SPLITS:
            path = frozen_root / pool_names[split]
            _write_immutable_json(path, manifests[split].to_dict())
            pool_paths[split] = str(path)
            pool_hashes[split] = _path_hash(path)
        leakage_path = frozen_root / "leakage-audit.json"
        _write_immutable_json(leakage_path, leakage)
        summary = _publication_summary(resolved, manifests, timestamp)
        summary_path = frozen_root / "publication-summary.json"
        _write_immutable_json(summary_path, summary)
        bundle = FrozenPartnerPoolBundle(
            suite_id=resolved.suite_id,
            suite_hash=resolved.suite_hash,
            plan_hash=resolved.plan_hash,
            layout_id=resolved.layout_id,
            pool_paths=pool_paths,
            pool_hashes=pool_hashes,
            leakage_audit_path=str(leakage_path),
            leakage_audit_hash=_path_hash(leakage_path),
            publication_summary_path=str(summary_path),
            publication_summary_hash=_path_hash(summary_path),
            frozen_at_utc=timestamp,
        )
        bundle_path = frozen_root / "frozen-pool-bundle.json"
        _write_immutable_json(bundle_path, bundle.to_dict())
        current = current.model_copy(
            update={
                "frozen_bundle_path": str(bundle_path),
                "frozen_bundle_hash": _path_hash(bundle_path),
                "updated_at_utc": _now(),
            }
        )
        _write_ledger(resolved, current)
        return bundle


def _execute_stage(
    plan: PartnerPoolBuildPlan,
    candidate: PartnerPoolCandidatePlan,
    entry: PartnerCandidateLedgerEntry,
    stage: PartnerPoolStage,
    executor: CandidateExecutor,
) -> tuple[PartnerCandidateLedgerEntry, PartnerPoolStage]:
    attempts = entry.attempts + 1
    try:
        checkpoint = executor(plan, candidate, stage)
        if stage == "screen":
            updated = entry.model_copy(
                update={
                    "status": "pending_finalist" if checkpoint.competent else "screen_rejected",
                    "attempts": attempts,
                    "last_error": None,
                    "screen_checkpoint": checkpoint,
                }
            )
        else:
            updated = entry.model_copy(
                update={
                    "status": "eligible" if checkpoint.competent else "finalist_rejected",
                    "attempts": attempts,
                    "last_error": None,
                    "finalist_checkpoint": checkpoint,
                }
            )
        return updated, stage
    except Exception as error:  # noqa: BLE001 - failure is persisted for detached recovery
        return (
            entry.model_copy(
                update={
                    "status": "failed",
                    "attempts": attempts,
                    "last_error": f"{type(error).__name__}: {error}",
                }
            ),
            stage,
        )


def _pending_stage(entry: PartnerCandidateLedgerEntry) -> PartnerPoolStage | None:
    if entry.status in {"pending_screen", "screen_running"}:
        return "screen"
    if entry.status in {"pending_finalist", "finalist_running"}:
        return "finalist"
    if entry.status == "failed":
        return "finalist" if entry.screen_checkpoint is not None else "screen"
    return None


def _execute_candidate(
    plan: PartnerPoolBuildPlan,
    candidate: PartnerPoolCandidatePlan,
    stage: PartnerPoolStage,
) -> PartnerCheckpoint:
    suite = load_established_suite_file(plan.suite_path)
    job_dir = Path(plan.workspace) / "jobs" / candidate.candidate_id / stage
    job_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_stage(plan, candidate, stage, job_dir)
    if completed is not None:
        return completed
    transitions = plan.screen_transitions if stage == "screen" else plan.finalist_transitions
    payload: dict[str, Any] = {
        "method_id": "partner_ippo",
        "layout_id": plan.layout_id,
        "seed": candidate.seed,
        "transitions": transitions,
        "schedule_target_transitions": plan.finalist_transitions,
        "learning_rate": 0.00025,
        "entropy_coefficient": 0.01,
        "behavior_preferences": candidate.reward_vector,
        "reward_vector_id": candidate.reward_vector_hash,
        "split": candidate.split,
        "output_dir": str(job_dir / "checkpoints"),
    }
    resume = _resume_checkpoint(plan, candidate, stage)
    if resume is not None:
        payload["resume_checkpoint"] = str(resume)
    operation = "train_partner"
    latest = _latest_index(job_dir / "checkpoints" / "training-state")
    if (
        latest is not None
        and isinstance(latest.get("completed_transitions"), int)
        and latest.get("completed_transitions") == latest.get("target_transitions")
    ):
        operation = "recover_training"
        payload["resume_checkpoint"] = latest["checkpoint_path"]
    request_path = write_runtime_request(
        suite,
        "overcookedv2_py310",
        operation,
        payload,
        job_dir / "request.json",
    )
    training_result_path = job_dir / "runtime-result.json"
    training_result = dispatch_runtime_request(
        suite,
        "overcookedv2_py310",
        request_path,
        training_result_path,
        plan.project_root,
        log_path=job_dir / "runtime.log",
    )
    checkpoint_path = Path(training_result["payload"]["checkpoint_paths"][-1]).resolve()
    competence_payload = {
        "ego_checkpoint": str(checkpoint_path),
        "partner_checkpoint": str(checkpoint_path),
        "layout_id": plan.layout_id,
        "environment_keys": list(plan.competence_environment_keys),
    }
    competence_request_path = write_runtime_request(
        suite,
        "overcookedv2_py310",
        "evaluate_pair",
        competence_payload,
        job_dir / "competence-request.json",
    )
    competence_result_path = job_dir / "competence-result.json"
    competence_result = dispatch_runtime_request(
        suite,
        "overcookedv2_py310",
        competence_request_path,
        competence_result_path,
        plan.project_root,
        log_path=job_dir / "competence.log",
    )
    checkpoint = _checkpoint_from_results(
        plan,
        candidate,
        stage,
        training_result,
        request_path,
        training_result_path,
        competence_request_path,
        competence_result_path,
        competence_result,
    )
    _atomic_write_json(job_dir / "checkpoint.json", checkpoint.to_dict())
    return checkpoint


def _checkpoint_from_results(
    plan: PartnerPoolBuildPlan,
    candidate: PartnerPoolCandidatePlan,
    stage: PartnerPoolStage,
    training_result: dict[str, Any],
    training_request_path: Path,
    training_result_path: Path,
    competence_request_path: Path,
    competence_result_path: Path,
    competence_result: dict[str, Any],
) -> PartnerCheckpoint:
    training_payload = training_result["payload"]
    checkpoint_path = Path(training_payload["checkpoint_paths"][-1]).resolve()
    training_states = training_payload.get("training_state_paths", [])
    if not training_states:
        raise RuntimeError("partner training did not emit a full resumable state")
    training_state = Path(training_states[-1]).resolve()
    delivery_rate = float(competence_result["payload"]["correct_delivery_episode_rate"])
    request_payload = _read_object(competence_request_path)
    requested = plan.screen_transitions if stage == "screen" else plan.finalist_transitions
    return PartnerCheckpoint(
        partner_id=candidate.candidate_id,
        reward_vector_id=candidate.reward_vector_hash,
        reward_vector_hash=candidate.reward_vector_hash,
        split=candidate.split,
        seed=candidate.seed,
        layout_id=plan.layout_id,
        checkpoint_path=str(checkpoint_path),
        normalized_checkpoint_hash=str(
            training_payload["checkpoint_parameter_hashes"][str(checkpoint_path)]
        ),
        checkpoint_content_hash=_path_hash(checkpoint_path),
        training_state_checkpoint_path=str(training_state),
        training_state_checkpoint_hash=str(
            training_payload["training_state_hashes"][str(training_state)]
        ),
        stage=stage,
        requested_transitions=requested,
        training_request_path=str(training_request_path.resolve()),
        training_request_hash=str(_read_object(training_request_path)["request_hash"]),
        training_result_path=str(training_result_path.resolve()),
        training_result_hash=_path_hash(training_result_path),
        competence_request_path=str(competence_request_path.resolve()),
        competence_request_hash=str(request_payload["request_hash"]),
        competence_result_path=str(competence_result_path.resolve()),
        competence_result_hash=_path_hash(competence_result_path),
        source_plan_hash=plan.plan_hash,
        transitions=int(training_payload["completed_transitions"]),
        validation_correct_delivery_rate=delivery_rate,
        competent=delivery_rate >= plan.minimum_correct_delivery_rate,
    )


def _load_completed_stage(
    plan: PartnerPoolBuildPlan,
    candidate: PartnerPoolCandidatePlan,
    stage: PartnerPoolStage,
    job_dir: Path,
) -> PartnerCheckpoint | None:
    checkpoint_path = job_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        return None
    checkpoint = PartnerCheckpoint.model_validate(_read_object(checkpoint_path))
    _verify_stage_checkpoint(plan, candidate, stage, checkpoint)
    return checkpoint


def _verify_stage_checkpoint(
    plan: PartnerPoolBuildPlan,
    candidate: PartnerPoolCandidatePlan,
    stage: PartnerPoolStage,
    checkpoint: PartnerCheckpoint,
) -> bool:
    expected_transitions = (
        plan.screen_transitions if stage == "screen" else plan.finalist_transitions
    )
    if (
        checkpoint.partner_id != candidate.candidate_id
        or checkpoint.split != candidate.split
        or checkpoint.seed != candidate.seed
        or checkpoint.reward_vector_hash != candidate.reward_vector_hash
        or checkpoint.layout_id != plan.layout_id
        or checkpoint.stage != stage
        or checkpoint.requested_transitions != expected_transitions
        or checkpoint.source_plan_hash != plan.plan_hash
    ):
        raise ValueError(f"stale or mismatched {stage} checkpoint for {candidate.candidate_id}")
    if checkpoint.checkpoint_content_hash is None or _path_hash(
        Path(checkpoint.checkpoint_path)
    ) != checkpoint.checkpoint_content_hash:
        raise ValueError("compact partner checkpoint content hash mismatch")
    if checkpoint.training_state_checkpoint_path is None or _path_hash(
        Path(checkpoint.training_state_checkpoint_path)
    ) != checkpoint.training_state_checkpoint_hash:
        raise ValueError("full partner training-state hash mismatch")
    if checkpoint.transitions <= 0 or checkpoint.transitions > expected_transitions:
        raise ValueError("partner checkpoint has an invalid attainable transition target")
    if checkpoint.training_request_path is None or checkpoint.training_result_path is None:
        raise ValueError("partner checkpoint lacks training request/result provenance")
    training_request = _read_object(Path(checkpoint.training_request_path))
    training_result = _read_object(Path(checkpoint.training_result_path))
    if (
        _path_hash(Path(checkpoint.training_result_path)) != checkpoint.training_result_hash
        or _runtime_request_hash(training_request) != checkpoint.training_request_hash
        or training_request.get("request_hash") != checkpoint.training_request_hash
        or training_result.get("request_hash") != checkpoint.training_request_hash
        or training_result.get("operation") not in {"train_partner", "recover_training"}
        or training_result.get("status") != "complete"
        or int(training_request["payload"]["transitions"]) != expected_transitions
        or int(training_result["payload"]["requested_transitions"]) != expected_transitions
        or int(training_result["payload"]["completed_transitions"]) != checkpoint.transitions
    ):
        raise ValueError("partner training request/result provenance mismatch")
    latest = _latest_index(Path(checkpoint.training_state_checkpoint_path).parent)
    if (
        latest is None
        or int(latest.get("completed_transitions", -1)) != checkpoint.transitions
        or int(latest.get("target_transitions", -1)) != checkpoint.transitions
    ):
        raise ValueError("partner full state does not represent its attainable completed target")
    if checkpoint.competence_request_path is None or checkpoint.competence_result_path is None:
        raise ValueError("partner checkpoint lacks competence provenance")
    request = _read_object(Path(checkpoint.competence_request_path))
    result = _read_object(Path(checkpoint.competence_result_path))
    if (
        _path_hash(Path(checkpoint.competence_result_path)) != checkpoint.competence_result_hash
        or _runtime_request_hash(request) != checkpoint.competence_request_hash
        or request.get("request_hash") != checkpoint.competence_request_hash
        or result.get("request_hash") != checkpoint.competence_request_hash
        or result.get("operation") != "evaluate_pair"
        or result.get("status") != "complete"
        or tuple(request["payload"]["environment_keys"]) != plan.competence_environment_keys
        or int(result["payload"]["episode_count"]) != plan.validation_rollouts
    ):
        raise ValueError("partner competence evaluation provenance mismatch")
    observed_rate = float(result["payload"]["correct_delivery_episode_rate"])
    if abs(observed_rate - checkpoint.validation_correct_delivery_rate) > 1e-12:
        raise ValueError("partner competence rate does not match its evaluation result")
    return True


def _verify_finalist(
    plan: PartnerPoolBuildPlan,
    candidate: PartnerPoolCandidatePlan,
    checkpoint: PartnerCheckpoint,
) -> bool:
    _verify_stage_checkpoint(plan, candidate, "finalist", checkpoint)
    if not checkpoint.competent or (
        checkpoint.validation_correct_delivery_rate + 1e-12
        < plan.minimum_correct_delivery_rate
    ):
        raise ValueError("incompetent finalist cannot enter a frozen partner pool")
    return True


def _resume_checkpoint(
    plan: PartnerPoolBuildPlan,
    candidate: PartnerPoolCandidatePlan,
    stage: PartnerPoolStage,
) -> Path | None:
    job_root = Path(plan.workspace) / "jobs" / candidate.candidate_id
    same_stage = _latest_index(job_root / stage / "checkpoints" / "training-state")
    if same_stage is not None:
        return Path(str(same_stage["checkpoint_path"])).resolve()
    if stage == "finalist":
        screen = job_root / "screen" / "checkpoint.json"
        if screen.is_file():
            checkpoint = PartnerCheckpoint.model_validate(_read_object(screen))
            _verify_stage_checkpoint(plan, candidate, "screen", checkpoint)
            if not checkpoint.competent or checkpoint.training_state_checkpoint_path is None:
                raise ValueError("finalist continuation requires a competent full screen state")
            return Path(checkpoint.training_state_checkpoint_path).resolve()
    return None


def _recover_interrupted_entries(
    plan: PartnerPoolBuildPlan, ledger: PartnerPoolBuildLedger
) -> PartnerPoolBuildLedger:
    entries = []
    for item in ledger.entries:
        if item.status in {"screen_running", "finalist_running", "failed"}:
            status = (
                "pending_finalist"
                if item.screen_checkpoint is not None and item.screen_checkpoint.competent
                else "pending_screen"
            )
            entries.append(item.model_copy(update={"status": status}))
        else:
            entries.append(item)
    recovered = ledger.model_copy(update={"entries": tuple(entries), "updated_at_utc": _now()})
    _write_ledger(plan, recovered)
    return recovered


def _existing_frozen_bundle(
    plan: PartnerPoolBuildPlan, ledger: PartnerPoolBuildLedger
) -> FrozenPartnerPoolBundle | None:
    if ledger.frozen_bundle_path is None:
        return None
    path = Path(ledger.frozen_bundle_path)
    if _path_hash(path) != ledger.frozen_bundle_hash:
        raise ValueError("published frozen bundle hash mismatch")
    bundle = FrozenPartnerPoolBundle.model_validate(_read_object(path))
    if (
        bundle.plan_hash != plan.plan_hash
        or bundle.suite_hash != plan.suite_hash
        or bundle.layout_id != plan.layout_id
    ):
        raise ValueError("published frozen bundle belongs to another plan")
    for split in SPLITS:
        if _path_hash(Path(bundle.pool_paths[split])) != bundle.pool_hashes[split]:
            raise ValueError("published frozen pool hash mismatch")
    if _path_hash(Path(bundle.leakage_audit_path)) != bundle.leakage_audit_hash:
        raise ValueError("published frozen leakage-audit hash mismatch")
    if _path_hash(Path(bundle.publication_summary_path)) != bundle.publication_summary_hash:
        raise ValueError("published frozen summary hash mismatch")
    return bundle


def _audit_frozen_leakage(
    manifests: Iterable[FrozenPartnerPoolManifest],
) -> dict[str, Any]:
    seen: dict[str, dict[str, str]] = {
        "partner_id": {},
        "reward_vector": {},
        "seed": {},
        "normalized_checkpoint": {},
        "checkpoint_content": {},
        "training_state": {},
    }
    collisions: list[str] = []
    for manifest in manifests:
        for checkpoint in manifest.checkpoints:
            values = {
                "partner_id": checkpoint.partner_id,
                "reward_vector": checkpoint.reward_vector_hash,
                "seed": str(checkpoint.seed),
                "normalized_checkpoint": checkpoint.normalized_checkpoint_hash,
                "checkpoint_content": str(checkpoint.checkpoint_content_hash),
                "training_state": str(checkpoint.training_state_checkpoint_hash),
            }
            for category, value in values.items():
                previous = seen[category].setdefault(value, manifest.split)
                if previous != manifest.split:
                    collisions.append(
                        f"{category} {value} crosses {previous}/{manifest.split}"
                    )
    return {"schema_version": 1, "passed": not collisions, "collisions": sorted(collisions)}


def _publication_summary(
    plan: PartnerPoolBuildPlan,
    manifests: dict[SplitName, FrozenPartnerPoolManifest],
    timestamp: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": plan.suite_id,
        "suite_hash": plan.suite_hash,
        "plan_hash": plan.plan_hash,
        "layout_id": plan.layout_id,
        "frozen_at_utc": timestamp,
        "splits": {
            split: {
                "selection_policy": manifests[split].selection_policy,
                "quota": manifests[split].quota,
                "count": len(manifests[split].checkpoints),
                "partner_ids": [item.partner_id for item in manifests[split].checkpoints],
                "reward_vector_hashes": [
                    item.reward_vector_hash for item in manifests[split].checkpoints
                ],
                "normalized_checkpoint_hashes": [
                    item.normalized_checkpoint_hash for item in manifests[split].checkpoints
                ],
                "competence_rates": [
                    item.validation_correct_delivery_rate
                    for item in manifests[split].checkpoints
                ],
            }
            for split in SPLITS
        },
    }


def _validate_plan_environment(plan: PartnerPoolBuildPlan) -> None:
    suite_path = Path(plan.suite_path)
    if _path_hash(suite_path) != plan.suite_hash:
        raise ValueError("partner-pool suite changed after planning")
    suite = load_established_suite_file(suite_path)
    if suite.suite_id != plan.suite_id:
        raise ValueError("partner-pool suite identifier changed after planning")
    audit = validate_upstreams(suite, plan.project_root)
    if not audit.passed:
        raise RuntimeError("partner-pool run requires its pinned upstreams and runtimes")
    observed_commits = {item.repository_id: item.observed_commit for item in audit.repositories}
    if observed_commits != plan.upstream_commits:
        raise ValueError("partner-pool upstream pins changed after planning")
    runtime = (
        Path(plan.project_root)
        / "phase-6-established-validation"
        / "runtime-overcookedv2"
        / "src"
        / "stage6_overcooked_runtime"
    )
    if _path_hash(runtime) != plan.runtime_source_hash:
        raise ValueError("partner-pool runtime source changed after planning")
    orchestrator = Path(plan.project_root) / "src" / "zsc_identifiability"
    if _path_hash(orchestrator) != plan.orchestrator_source_hash:
        raise ValueError("partner-pool orchestrator source changed after planning")
    for candidate in plan.candidates:
        if split_for_reward_vector(candidate.reward_vector, suite) != candidate.split:
            raise ValueError("partner candidate reward-vector split changed after planning")
        if reward_vector_hash(candidate.reward_vector) != candidate.reward_vector_hash:
            raise ValueError("partner candidate reward-vector hash mismatch")
        if candidate.seed != partner_seed(
            candidate.split, candidate.reward_vector_index, candidate.replicate
        ):
            raise ValueError("partner candidate seed mapping changed after planning")
        if candidate.candidate_id != (
            f"{candidate.split}-{candidate.reward_vector_hash[:12]}-seed{candidate.seed}"
        ):
            raise ValueError("partner candidate identifier changed after planning")


def _quotas(suite: EstablishedValidationSuite) -> dict[SplitName, int]:
    spec = suite.partner_generation
    return {
        "train": spec.training_partner_quota,
        "validation": spec.validation_partner_quota,
        "evaluation": spec.evaluation_candidate_quota,
    }


def _caps(suite: EstablishedValidationSuite) -> dict[SplitName, int]:
    spec = suite.partner_generation
    return {
        "train": spec.training_candidate_cap,
        "validation": spec.validation_candidate_cap,
        "evaluation": spec.evaluation_candidate_cap,
    }


def _eligible_count(
    ledger: PartnerPoolBuildLedger, plan: PartnerPoolBuildPlan, split: SplitName
) -> int:
    identifiers = {item.candidate_id for item in plan.candidates if item.split == split}
    return sum(
        item.status == "eligible"
        for item in ledger.entries
        if item.candidate_id in identifiers
    )


def _entry_by_id(
    ledger: PartnerPoolBuildLedger, candidate_id: str
) -> PartnerCandidateLedgerEntry:
    return next(item for item in ledger.entries if item.candidate_id == candidate_id)


def _replace_entry(
    ledger: PartnerPoolBuildLedger, replacement: PartnerCandidateLedgerEntry
) -> PartnerPoolBuildLedger:
    return ledger.model_copy(
        update={
            "entries": tuple(
                replacement if item.candidate_id == replacement.candidate_id else item
                for item in ledger.entries
            ),
            "updated_at_utc": _now(),
        }
    )


def _load_ledger(plan: PartnerPoolBuildPlan) -> PartnerPoolBuildLedger:
    ledger = PartnerPoolBuildLedger.model_validate(
        _read_object(Path(plan.workspace) / "ledger.json")
    )
    if ledger.plan_hash != plan.plan_hash:
        raise ValueError("partner-pool ledger belongs to another plan")
    planned = tuple(item.candidate_id for item in plan.candidates)
    recorded = tuple(item.candidate_id for item in ledger.entries)
    if planned != recorded:
        raise ValueError("partner-pool ledger candidate order changed")
    return ledger


def _write_ledger(plan: PartnerPoolBuildPlan, ledger: PartnerPoolBuildLedger) -> None:
    _atomic_write_json(Path(plan.workspace) / "ledger.json", ledger.to_dict())


def _coerce_plan(plan: PartnerPoolBuildPlan | str | Path) -> PartnerPoolBuildPlan:
    return plan if isinstance(plan, PartnerPoolBuildPlan) else load_partner_pool_build_plan(plan)


def _latest_index(path: Path) -> dict[str, Any] | None:
    index = path / "latest.json"
    return _read_object(index) if index.is_file() else None


def _freeze_timestamp(plan: PartnerPoolBuildPlan) -> str:
    path = Path(plan.workspace) / "frozen" / "freeze-intent.json"
    if path.is_file():
        payload = _read_object(path)
        if payload.get("plan_hash") != plan.plan_hash:
            raise ValueError("freeze intent belongs to another partner-pool plan")
        return str(payload["frozen_at_utc"])
    timestamp = _now()
    _write_immutable_json(
        path,
        {"schema_version": 1, "plan_hash": plan.plan_hash, "frozen_at_utc": timestamp},
    )
    return timestamp


@contextmanager
def _workspace_lock(plan: PartnerPoolBuildPlan) -> Iterator[None]:
    path = Path(plan.workspace) / ".runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another partner-pool runner owns this workspace") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _null_context() -> Iterator[None]:
    yield


def _read_object(path: Path) -> dict[str, Any]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_immutable_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"refusing to overwrite immutable frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def _path_hash(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ValueError(f"cannot hash missing partner-pool asset: {path}")
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
            digest.update(str(item.relative_to(path)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _runtime_request_hash(payload: dict[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("request_hash", None)
    return _canonical_hash(unhashed)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _project_root(value: str | Path | None) -> Path:
    return Path(value).resolve() if value is not None else Path(__file__).resolve().parents[2]
