from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from zsc_identifiability.established_models import (
    EstablishedValidationSuite,
    PartnerCheckpoint,
    PartnerPoolBuildLedger,
    PartnerPoolBuildPlan,
    PartnerPoolCandidatePlan,
    PartnerPoolStage,
    load_established_suite_file,
)
from zsc_identifiability.established_partner_pools import (
    _execute_candidate,
    _resume_checkpoint,
    freeze_partner_pools,
    get_partner_pool_status,
    partner_seed,
    prepare_partner_pool_build,
    run_partner_pool_build,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SUITE = ROOT / "phase-6-established-validation/suites/full-scale-overcookedv2.json"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tree_hash(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _request(path: Path, payload: dict[str, Any]) -> str:
    request = {"schema_version": 1, "payload": payload}
    request_hash = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request["request_hash"] = request_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    return request_hash


def _write_suite(
    tmp_path: Path,
    *,
    train_quota: int = 2,
    train_cap: int = 4,
    validation_quota: int = 2,
    validation_cap: int = 4,
    evaluation_quota: int = 32,
    evaluation_cap: int = 64,
    expansion: int = 2,
) -> tuple[EstablishedValidationSuite, Path]:
    original = load_established_suite_file(CANONICAL_SUITE)
    generation = type(original.partner_generation).model_validate(
        {
            **original.partner_generation.to_dict(),
            "training_partner_quota": train_quota,
            "training_candidate_cap": train_cap,
            "validation_partner_quota": validation_quota,
            "validation_candidate_cap": validation_cap,
            "evaluation_candidate_quota": evaluation_quota,
            "evaluation_candidate_cap": evaluation_cap,
            "expansion_block_size": expansion,
            "screen_transitions": 20,
            "finalist_transitions": 40,
            "validation_rollouts": 4,
        }
    )
    suite = original.model_copy(update={"partner_generation": generation})
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite.to_dict(), indent=2, sort_keys=True) + "\n")
    return suite, path


def _fake_upstream_audit(suite: EstablishedValidationSuite) -> SimpleNamespace:
    return SimpleNamespace(
        passed=True,
        repositories=tuple(
            SimpleNamespace(repository_id=item.repository_id, observed_commit=item.commit)
            for item in suite.upstreams
        ),
    )


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **suite_options: int,
) -> PartnerPoolBuildPlan:
    suite, suite_path = _write_suite(tmp_path, **suite_options)
    monkeypatch.setattr(
        "zsc_identifiability.established_partner_pools.validate_upstreams",
        lambda _suite, _root: _fake_upstream_audit(suite),
    )
    return prepare_partner_pool_build(
        suite,
        suite_path=suite_path,
        layout="demo_cook_simple",
        workspace=tmp_path / "workspace",
        project_root=ROOT,
    )


def _fake_executor(
    competence: Any,
    calls: list[tuple[str, PartnerPoolStage]],
    *,
    duplicate_normalized_hashes: bool = False,
):
    def execute(
        plan: PartnerPoolBuildPlan,
        candidate: PartnerPoolCandidatePlan,
        stage: PartnerPoolStage,
    ) -> PartnerCheckpoint:
        calls.append((candidate.candidate_id, stage))
        competent = bool(competence(candidate, stage))
        requested = plan.screen_transitions if stage == "screen" else plan.finalist_transitions
        job = Path(plan.workspace) / "fake-jobs" / candidate.candidate_id / stage
        compact = job / "compact-policy"
        compact.mkdir(parents=True, exist_ok=True)
        (compact / "params.bin").write_bytes(f"{candidate.candidate_id}:{stage}".encode())
        state_root = job / "training-state"
        state = state_root / f"step-{requested}"
        state.mkdir(parents=True, exist_ok=True)
        (state / "state.bin").write_bytes(f"state:{candidate.candidate_id}:{stage}".encode())
        latest = {
            "checkpoint_path": str(state),
            "completed_transitions": requested,
            "target_transitions": requested,
        }
        (state_root / "latest.json").write_text(json.dumps(latest) + "\n")

        training_request = job / "training-request.json"
        training_hash = _request(training_request, {"transitions": requested})
        training_result = job / "training-result.json"
        training_result.write_text(
            json.dumps(
                {
                    "request_hash": training_hash,
                    "operation": "train_partner",
                    "status": "complete",
                    "payload": {
                        "requested_transitions": requested,
                        "completed_transitions": requested,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        competence_request = job / "competence-request.json"
        competence_hash = _request(
            competence_request,
            {"environment_keys": list(plan.competence_environment_keys)},
        )
        competence_result = job / "competence-result.json"
        rate = 1.0 if competent else 0.0
        competence_result.write_text(
            json.dumps(
                {
                    "request_hash": competence_hash,
                    "operation": "evaluate_pair",
                    "status": "complete",
                    "payload": {
                        "episode_count": plan.validation_rollouts,
                        "correct_delivery_episode_rate": rate,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        normalized = (
            "a" * 64
            if duplicate_normalized_hashes
            else _sha(f"normalized:{candidate.candidate_id}:{stage}")
        )
        return PartnerCheckpoint(
            partner_id=candidate.candidate_id,
            reward_vector_id=candidate.reward_vector_hash,
            reward_vector_hash=candidate.reward_vector_hash,
            split=candidate.split,
            seed=candidate.seed,
            layout_id=plan.layout_id,
            checkpoint_path=str(compact),
            normalized_checkpoint_hash=normalized,
            checkpoint_content_hash=_tree_hash(compact),
            training_state_checkpoint_path=str(state),
            training_state_checkpoint_hash=_tree_hash(state),
            stage=stage,
            requested_transitions=requested,
            training_request_path=str(training_request),
            training_request_hash=training_hash,
            training_result_path=str(training_result),
            training_result_hash=_tree_hash(training_result),
            competence_request_path=str(competence_request),
            competence_request_hash=competence_hash,
            competence_result_path=str(competence_result),
            competence_result_hash=_tree_hash(competence_result),
            source_plan_hash=plan.plan_hash,
            transitions=requested,
            validation_correct_delivery_rate=rate,
            competent=competent,
        )

    return execute


def test_canonical_plan_has_registered_caps_and_disjoint_seed_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = load_established_suite_file(CANONICAL_SUITE)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite.to_dict(), indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(
        "zsc_identifiability.established_partner_pools.validate_upstreams",
        lambda _suite, _root: _fake_upstream_audit(suite),
    )
    plan = prepare_partner_pool_build(
        suite,
        suite_path=suite_path,
        layout="demo_cook_simple",
        workspace=tmp_path / "workspace",
        project_root=ROOT,
    )
    assert plan.quotas == {"train": 24, "validation": 8, "evaluation": 32}
    assert plan.caps == {"train": 48, "validation": 16, "evaluation": 64}
    assert len(plan.candidates) == 128
    assert len({item.seed for item in plan.candidates}) == 128
    assert partner_seed("train", 0, 0) == 41_001
    assert partner_seed("validation", 0, 0) == 141_001
    assert partner_seed("evaluation", 0, 0) == 241_001
    assert prepare_partner_pool_build(
        suite,
        suite_path=suite_path,
        layout="demo_cook_simple",
        workspace=tmp_path / "workspace",
        project_root=ROOT,
    ) == plan


def test_runner_expands_promotes_and_freezes_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)
    calls: list[tuple[str, PartnerPoolStage]] = []

    def competence(candidate: PartnerPoolCandidatePlan, stage: PartnerPoolStage) -> bool:
        if candidate.split == "train" and candidate.candidate_index == 0 and stage == "screen":
            return False
        if (
            candidate.split == "validation"
            and candidate.candidate_index == 0
            and stage == "finalist"
        ):
            return False
        return not (
            candidate.split == "evaluation"
            and candidate.candidate_index == 0
            and stage == "screen"
        )

    ledger = run_partner_pool_build(
        plan,
        workers=2,
        executor=_fake_executor(competence, calls),
    )
    status = get_partner_pool_status(plan, ledger=ledger)
    assert status.complete
    assert all(item.pending == 0 and item.failed == 0 for item in status.splits)
    assert next(item for item in status.splits if item.split == "train").active == 4
    assert next(item for item in status.splits if item.split == "validation").active == 4
    assert next(item for item in status.splits if item.split == "evaluation").active == 34
    train_zero = next(item for item in plan.candidates if item.split == "train")
    assert (train_zero.candidate_id, "finalist") not in calls
    split_by_id = {item.candidate_id: item.split for item in plan.candidates}
    train_positions = [
        index for index, (item, _) in enumerate(calls) if split_by_id[item] == "train"
    ]
    validation_positions = [
        index for index, (item, _) in enumerate(calls) if split_by_id[item] == "validation"
    ]
    evaluation_positions = [
        index for index, (item, _) in enumerate(calls) if split_by_id[item] == "evaluation"
    ]
    assert max(train_positions) < min(validation_positions)
    assert max(validation_positions) < min(evaluation_positions)

    bundle = freeze_partner_pools(plan, ledger=ledger)
    train = json.loads(Path(bundle.pool_paths["train"]).read_text())
    validation = json.loads(Path(bundle.pool_paths["validation"]).read_text())
    evaluation = json.loads(Path(bundle.pool_paths["evaluation"]).read_text())
    assert len(train["checkpoints"]) == 2
    assert len(validation["checkpoints"]) == 2
    assert len(evaluation["checkpoints"]) == 33
    assert json.loads(Path(bundle.leakage_audit_path).read_text())["passed"]
    assert freeze_partner_pools(plan) == bundle
    prior_call_count = len(calls)
    run_partner_pool_build(plan, executor=_fake_executor(competence, calls))
    assert len(calls) == prior_call_count
    Path(bundle.pool_paths["train"]).write_text("{}\n")
    with pytest.raises(ValueError, match="pool hash mismatch"):
        freeze_partner_pools(plan)


def test_runner_reports_cap_exhaustion_without_relaxing_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)
    calls: list[tuple[str, PartnerPoolStage]] = []
    run_partner_pool_build(
        plan,
        splits=("train",),
        executor=_fake_executor(lambda _candidate, _stage: False, calls),
    )
    status = get_partner_pool_status(plan)
    train = next(item for item in status.splits if item.split == "train")
    assert train.cap_exhausted
    assert train.eligible == 0
    assert not train.quota_met
    assert all(stage == "screen" for _, stage in calls)
    with pytest.raises(RuntimeError, match="cannot freeze"):
        freeze_partner_pools(plan)


def test_runner_recovers_failed_and_running_jobs_on_next_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(
        tmp_path,
        monkeypatch,
        train_quota=2,
        train_cap=2,
        validation_quota=2,
        validation_cap=2,
    )
    attempts: dict[str, int] = {}
    calls: list[tuple[str, PartnerPoolStage]] = []
    successful = _fake_executor(lambda _candidate, _stage: True, calls)

    def fails_once(plan_arg, candidate, stage):
        attempts[candidate.candidate_id] = attempts.get(candidate.candidate_id, 0) + 1
        if candidate.candidate_index == 0 and attempts[candidate.candidate_id] == 1:
            raise RuntimeError("simulated interruption")
        return successful(plan_arg, candidate, stage)

    first = run_partner_pool_build(plan, splits=("train",), executor=fails_once)
    assert get_partner_pool_status(plan, ledger=first).unresolved_failures == 1
    second = run_partner_pool_build(plan, splits=("train",), executor=fails_once)
    train = next(
        item
        for item in get_partner_pool_status(plan, ledger=second).splits
        if item.split == "train"
    )
    assert train.quota_met and train.failed == 0

    raw = json.loads((Path(plan.workspace) / "ledger.json").read_text())
    raw["entries"][0]["status"] = "finalist_running"
    raw["entries"][0]["finalist_checkpoint"] = None
    (Path(plan.workspace) / "ledger.json").write_text(json.dumps(raw) + "\n")
    recovered = run_partner_pool_build(plan, splits=("train",), executor=successful)
    first_entry = recovered.entries[0]
    assert first_entry.status == "eligible"
    assert first_entry.finalist_checkpoint is not None


def test_stale_source_and_cross_split_checkpoint_leakage_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(
        tmp_path,
        monkeypatch,
        train_quota=2,
        train_cap=2,
        validation_quota=2,
        validation_cap=2,
    )
    stale = plan.model_copy(update={"orchestrator_source_hash": "0" * 64})
    with pytest.raises(ValueError, match="orchestrator source changed"):
        run_partner_pool_build(stale, executor=lambda *_args: None)  # type: ignore[arg-type]

    calls: list[tuple[str, PartnerPoolStage]] = []
    ledger = run_partner_pool_build(
        plan,
        executor=_fake_executor(
            lambda _candidate, _stage: True,
            calls,
            duplicate_normalized_hashes=True,
        ),
    )
    assert get_partner_pool_status(plan, ledger=ledger).complete
    with pytest.raises(ValueError, match="leakage audit"):
        freeze_partner_pools(plan, ledger=ledger)


def test_status_and_freeze_do_not_dispatch_runtime_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)
    calls: list[tuple[str, PartnerPoolStage]] = []
    ledger = run_partner_pool_build(
        plan,
        executor=_fake_executor(lambda _candidate, _stage: True, calls),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("status/freeze dispatched runtime training")

    monkeypatch.setattr(
        "zsc_identifiability.established_partner_pools.dispatch_runtime_request",
        forbidden,
    )
    assert get_partner_pool_status(plan, ledger=ledger).complete
    assert freeze_partner_pools(plan, ledger=ledger).plan_hash == plan.plan_hash


def test_completed_state_uses_recovery_only_export_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)
    candidate = next(item for item in plan.candidates if item.split == "train")
    job = Path(plan.workspace) / "jobs" / candidate.candidate_id / "screen"
    state_root = job / "checkpoints" / "training-state"
    state = state_root / f"step-{plan.screen_transitions}"
    state.mkdir(parents=True)
    (state / "state.bin").write_bytes(b"completed-state")
    (state_root / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(state),
                "completed_transitions": plan.screen_transitions,
                "target_transitions": plan.screen_transitions,
            }
        )
        + "\n"
    )
    operations: list[str] = []

    def dispatch(_suite, _runtime, request_path, result_path, _root, **_kwargs):
        request = json.loads(Path(request_path).read_text())
        operation = str(request["operation"])
        operations.append(operation)
        if operation == "recover_training":
            compact = job / "checkpoints" / "run_0" / "ckpt_final"
            compact.mkdir(parents=True)
            (compact / "params.bin").write_bytes(b"recovered-policy")
            payload = {
                "requested_transitions": plan.screen_transitions,
                "completed_transitions": plan.screen_transitions,
                "checkpoint_paths": [str(compact)],
                "checkpoint_parameter_hashes": {str(compact): _sha("parameters")},
                "training_state_paths": [str(state)],
                "training_state_hashes": {str(state): _tree_hash(state)},
            }
        else:
            payload = {
                "episode_count": plan.validation_rollouts,
                "correct_delivery_episode_rate": 1.0,
            }
        result = {
            "schema_version": 1,
            "request_hash": request["request_hash"],
            "operation": operation,
            "status": "complete",
            "payload": payload,
        }
        Path(result_path).write_text(json.dumps(result, sort_keys=True) + "\n")
        return result

    monkeypatch.setattr(
        "zsc_identifiability.established_partner_pools.dispatch_runtime_request",
        dispatch,
    )
    checkpoint = _execute_candidate(plan, candidate, "screen")
    assert checkpoint.competent
    assert operations == ["recover_training", "evaluate_pair"]
    assert "train_partner" not in operations
    assert _resume_checkpoint(plan, candidate, "finalist") == Path(
        str(checkpoint.training_state_checkpoint_path)
    ).resolve()


def test_ledger_model_rejects_duplicate_candidate_identifiers() -> None:
    with pytest.raises(ValueError, match="identifiers must be unique"):
        PartnerPoolBuildLedger(
            plan_hash="0" * 64,
            entries=(
                {"candidate_id": "duplicate", "status": "inactive", "active": False},
                {"candidate_id": "duplicate", "status": "inactive", "active": False},
            ),
            updated_at_utc="now",
        )
