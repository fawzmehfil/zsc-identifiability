"""Resumable CPU rollout orchestration for the official-checkpoint audit."""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from zsc_identifiability.established_official_assets import load_official_asset_inventory
from zsc_identifiability.established_official_models import (
    OfficialAssetInventory,
    OfficialCheckpointAuditSuiteV2,
    OfficialRolloutLedger,
    OfficialRolloutLedgerEntry,
    OfficialRolloutPlan,
    OfficialRolloutShard,
    load_official_checkpoint_suite,
)

RuntimeExecutor = Callable[[OfficialRolloutShard, Path, Path], str]


def prepare_official_rollouts(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    assets: OfficialAssetInventory | str | Path,
    workspace: str | Path,
) -> OfficialRolloutPlan:
    """Materialize every preregistered atomic shard without running inference."""

    spec, suite_path = _resolve_suite(suite)
    inventory, inventory_path = _resolve_inventory(assets)
    if not inventory.complete:
        raise ValueError("official rollout preparation requires a complete asset inventory")
    if inventory.suite_id != spec.suite_id:
        raise ValueError("official asset inventory belongs to a different suite")
    target = Path(workspace).resolve()
    target.mkdir(parents=True, exist_ok=True)
    shards: list[OfficialRolloutShard] = []
    partners_by_layout = {
        layout.layout_id: tuple(
            partner for partner in inventory.partners if partner.layout_id == layout.layout_id
        )
        for layout in spec.layouts
    }
    methods_by_layout = {
        layout.layout_id: tuple(
            method for method in inventory.methods if method.layout_id == layout.layout_id
        )
        for layout in spec.layouts
    }
    for layout in spec.layouts:
        partners = partners_by_layout[layout.layout_id]
        methods = methods_by_layout[layout.layout_id]
        if len(partners) != layout.expected_partner_count:
            raise ValueError(f"asset inventory is incomplete for {layout.layout_id}")
        reference = next(
            method
            for method in methods
            if method.method_id == spec.evidence.passive_reference_method
            and method.seed == spec.evidence.passive_reference_seed
        )
        shards.append(
            _make_shard(
                target,
                spec,
                kind="parity",
                layout_id=layout.layout_id,
                suffix="runtime-parity",
                payload={
                    "checkpoint_path": reference.checkpoint_path,
                    "checkpoint_architecture": reference.policy_architecture,
                    "partner_checkpoint_path": partners[0].partner_checkpoint_path,
                    "deterministic": True,
                    "episode_keys": [_episode_key(spec, layout.layout_id, "parity", 0)],
                    "seat_assignments": [0, 1],
                },
                deterministic=True,
            )
        )
        for partner in partners:
            for response in partners:
                keys = _episode_keys(
                    spec,
                    layout.layout_id,
                    f"response:{partner.partner_id}",
                    layout.response_episodes_per_pair,
                )
                shards.append(
                    _make_shard(
                        target,
                        spec,
                        kind="response",
                        layout_id=layout.layout_id,
                        suffix=f"{_slug(partner.partner_id)}--{_slug(response.partner_id)}",
                        partner_id=partner.partner_id,
                        response_id=response.partner_id,
                        episode_keys=keys,
                        payload={
                            "partner_checkpoint_path": partner.partner_checkpoint_path,
                            "response_checkpoint_path": response.response_checkpoint_path,
                            "episode_keys": list(keys),
                            "balanced_seats": True,
                            "deterministic": False,
                        },
                    )
                )
            for evidence_policy in layout.diagnostic_options:
                for split, count in layout.trace_episodes.items():
                    keys = _episode_keys(
                        spec,
                        layout.layout_id,
                        f"trace:{partner.partner_id}:{split}",
                        count,
                        split=split,
                    )
                    shards.append(
                        _make_shard(
                            target,
                            spec,
                            kind="trace",
                            layout_id=layout.layout_id,
                            suffix=(
                                f"{_slug(partner.partner_id)}--{_slug(evidence_policy)}--{split}"
                            ),
                            partner_id=partner.partner_id,
                            evidence_policy=evidence_policy,
                            split=split,
                            episode_keys=keys,
                            deterministic=True,
                            payload={
                                "partner_checkpoint_path": partner.partner_checkpoint_path,
                                "reference_checkpoint_path": reference.checkpoint_path,
                                "evidence_policy": evidence_policy,
                                "split": split,
                                "episode_keys": list(keys),
                                "balanced_seats": True,
                                "deterministic": True,
                                "maximum_option_steps": spec.evidence.maximum_option_steps,
                                "prefix_steps": list(spec.evidence.prefix_steps),
                            },
                        )
                    )
            for method in methods:
                keys = _episode_keys(
                    spec,
                    layout.layout_id,
                    f"method:{partner.partner_id}",
                    layout.method_episodes,
                )
                for deployment in ("stochastic", "greedy"):
                    shards.append(
                        _make_shard(
                            target,
                            spec,
                            kind="method",
                            layout_id=layout.layout_id,
                            suffix=(
                                f"{method.method_id}-s{method.seed}-{deployment}--"
                                f"{_slug(partner.partner_id)}"
                            ),
                            partner_id=partner.partner_id,
                            method_id=method.method_id,
                            method_seed=method.seed,
                            deployment=deployment,
                            episode_keys=keys,
                            deterministic=deployment == "greedy",
                            payload={
                                "partner_checkpoint_path": partner.partner_checkpoint_path,
                                "method_checkpoint_path": method.checkpoint_path,
                                "method_id": method.method_id,
                                "method_seed": method.seed,
                                "policy_architecture": method.policy_architecture,
                                "deployment": deployment,
                                "episode_keys": list(keys),
                                "balanced_seats": True,
                                "deterministic": deployment == "greedy",
                                "retain_full_observations": False,
                            },
                        )
                    )
    suite_hash = _sha256_path(suite_path) if suite_path is not None else _hash_json(spec.to_dict())
    payload = {
        "schema_version": 1,
        "suite_id": spec.suite_id,
        "suite_path": str(suite_path or "<in-memory>"),
        "suite_hash": suite_hash,
        "inventory_path": str(inventory_path or "<in-memory>"),
        "inventory_hash": inventory.inventory_hash,
        "workspace": str(target),
        "shards": [shard.to_dict() for shard in shards],
    }
    plan = OfficialRolloutPlan.model_validate({**payload, "plan_hash": _hash_json(payload)})
    _atomic_json(target / "official-rollout-plan.json", plan.to_dict())
    initial = OfficialRolloutLedger(
        suite_id=spec.suite_id,
        plan_hash=plan.plan_hash,
        entries=tuple(
            OfficialRolloutLedgerEntry(shard_id=shard.shard_id, status="pending")
            for shard in shards
        ),
        complete=False,
    )
    _atomic_json(target / "official-rollout-ledger.json", initial.to_dict())
    return plan


def run_official_rollouts(
    plan: OfficialRolloutPlan | str | Path,
    *,
    workers: int = 2,
    resume: bool = True,
    kinds: tuple[str, ...] | None = None,
    executor: RuntimeExecutor | None = None,
) -> OfficialRolloutLedger:
    """Run atomic shards, preserving completed results and every failure."""

    rollout_plan = _resolve_plan(plan)
    if workers < 1 or workers > 4:
        raise ValueError("official audit workers must be between one and four")
    workspace = Path(rollout_plan.workspace).resolve()
    ledger_path = workspace / "official-rollout-ledger.json"
    runtime = executor or _default_executor(rollout_plan)
    with _exclusive_lock(workspace / ".official-runner.lock"):
        ledger = _load_or_initialize_ledger(rollout_plan, ledger_path, resume)
        current = {entry.shard_id: entry for entry in ledger.entries}
        selected = [
            shard
            for shard in rollout_plan.shards
            if (kinds is None or shard.kind in kinds)
            and not _valid_complete_result(shard, current.get(shard.shard_id))
        ]

        def execute(shard: OfficialRolloutShard) -> tuple[str, str | None, str | None]:
            try:
                result_hash = runtime(shard, workspace, workspace / "logs")
                return shard.shard_id, result_hash, None
            except Exception as error:  # noqa: BLE001 - every shard failure belongs in ledger
                return shard.shard_id, None, f"{type(error).__name__}: {error}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[concurrent.futures.Future[tuple[str, str | None, str | None]], str] = {}
            for shard in selected:
                previous = current.get(shard.shard_id)
                current[shard.shard_id] = OfficialRolloutLedgerEntry(
                    shard_id=shard.shard_id,
                    status="running",
                    attempts=0 if previous is None else previous.attempts,
                )
                _write_ledger(rollout_plan, ledger_path, current, kinds)
                futures[pool.submit(execute, shard)] = shard.shard_id
            for future in concurrent.futures.as_completed(futures):
                shard_id, result_hash, error = future.result()
                previous = current[shard_id]
                current[shard_id] = OfficialRolloutLedgerEntry(
                    shard_id=shard_id,
                    status="complete" if error is None else "failed",
                    attempts=previous.attempts + 1,
                    result_hash=result_hash,
                    error=error,
                )
                _write_ledger(rollout_plan, ledger_path, current, kinds)
        return _write_ledger(rollout_plan, ledger_path, current, kinds)


def get_official_rollout_status(
    plan: OfficialRolloutPlan | str | Path,
) -> OfficialRolloutLedger:
    rollout_plan = _resolve_plan(plan)
    ledger_path = Path(rollout_plan.workspace) / "official-rollout-ledger.json"
    return _load_or_initialize_ledger(rollout_plan, ledger_path, resume=True)


def _make_shard(
    workspace: Path,
    suite: OfficialCheckpointAuditSuiteV2,
    *,
    kind: Literal["parity", "response", "trace", "method"],
    layout_id: str,
    suffix: str,
    payload: dict[str, Any],
    partner_id: str | None = None,
    response_id: str | None = None,
    method_id: str | None = None,
    method_seed: int | None = None,
    deployment: str | None = None,
    evidence_policy: str | None = None,
    split: str | None = None,
    episode_keys: tuple[int, ...] = (),
    deterministic: bool = False,
) -> OfficialRolloutShard:
    shard_id = f"{layout_id}--{kind}--{suffix}"
    request_path = workspace / "requests" / f"{shard_id}.json"
    result_path = workspace / "results" / kind / f"{shard_id}.json.gz"
    operation = {
        "parity": "official_parity",
        "response": "official_response_rollout",
        "trace": "official_trace_rollout",
        "method": "official_method_rollout",
    }[kind]
    request_payload = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "runtime": "zsceval_py39",
        "operation": operation,
        "policy_training_allowed": False,
        "layout_id": layout_id,
        "max_episode_steps": 400,
        "repository_commit": suite.upstream.repository_commit,
        "policy_pool_revision": suite.upstream.policy_pool_revision,
        "payload": payload,
    }
    request_hash = _hash_json(request_payload)
    request_payload["request_hash"] = request_hash
    _atomic_json(request_path, request_payload)
    return OfficialRolloutShard(
        shard_id=shard_id,
        kind=kind,
        layout_id=layout_id,  # type: ignore[arg-type]
        request_path=str(request_path),
        result_path=str(result_path),
        request_hash=request_hash,
        partner_id=partner_id,
        response_id=response_id,
        method_id=method_id,  # type: ignore[arg-type]
        method_seed=method_seed,
        deployment=deployment,  # type: ignore[arg-type]
        evidence_policy=evidence_policy,
        split=split,  # type: ignore[arg-type]
        episode_keys=episode_keys,
        deterministic=deterministic,
    )


def _default_executor(plan: OfficialRolloutPlan) -> RuntimeExecutor:
    suite = load_official_checkpoint_suite(plan.suite_path)
    project_root = Path(plan.suite_path).resolve().parents[2]
    runtime_project = (project_root / suite.runtime.runtime_project).resolve()
    source_root = (Path(plan.workspace) / suite.runtime.upstream_directory).resolve()
    pool_root = (Path(plan.workspace) / suite.runtime.asset_directory).resolve()

    def execute(shard: OfficialRolloutShard, _workspace: Path, log_dir: Path) -> str:
        result = Path(shard.result_path)
        result.parent.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{shard.shard_id}.log"
        command = (
            "uv",
            "run",
            "--project",
            str(runtime_project),
            "python",
            "-m",
            "stage6_legacy_runtime",
            "--request",
            shard.request_path,
            "--result",
            shard.result_path,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "ZSC_EVAL_SOURCE": str(source_root),
                "ZSC_EVAL_POLICY_POOL": str(pool_root),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if process.returncode:
            raise RuntimeError(f"runtime exited {process.returncode}; inspect {log_path}")
        if not result.is_file():
            raise RuntimeError("runtime completed without writing its result shard")
        return _sha256_path(result)

    return execute


def _episode_keys(
    suite: OfficialCheckpointAuditSuiteV2,
    layout: str,
    namespace: str,
    count: int,
    *,
    split: str = "confirmatory",
) -> tuple[int, ...]:
    if count % 2:
        raise ValueError("official paired-seat episode counts must be even")
    return tuple(
        _episode_key(suite, layout, f"{split}:{namespace}", index) for index in range(count)
    )


def _episode_key(
    suite: OfficialCheckpointAuditSuiteV2,
    layout: str,
    namespace: str,
    index: int,
) -> int:
    split = next((name for name in suite.split_key_salts if namespace.startswith(name)), None)
    salt = suite.split_key_salts[split or "confirmatory"]
    digest = hashlib.sha256(f"{salt}:{layout}:{namespace}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _valid_complete_result(
    shard: OfficialRolloutShard,
    entry: OfficialRolloutLedgerEntry | None,
) -> bool:
    if entry is None or entry.status != "complete" or entry.result_hash is None:
        return False
    path = Path(shard.result_path)
    return path.is_file() and _sha256_path(path) == entry.result_hash


def _load_or_initialize_ledger(
    plan: OfficialRolloutPlan,
    path: Path,
    resume: bool,
) -> OfficialRolloutLedger:
    if path.exists() and resume:
        ledger = OfficialRolloutLedger.model_validate(_read_json(path))
        if ledger.plan_hash != plan.plan_hash:
            raise ValueError("rollout ledger belongs to a different plan")
        reset = tuple(
            entry.model_copy(update={"status": "pending", "error": "interrupted"})
            if entry.status == "running"
            else entry
            for entry in ledger.entries
        )
        return ledger.model_copy(update={"entries": reset})
    return OfficialRolloutLedger(
        suite_id=plan.suite_id,
        plan_hash=plan.plan_hash,
        entries=tuple(
            OfficialRolloutLedgerEntry(shard_id=shard.shard_id, status="pending")
            for shard in plan.shards
        ),
        complete=False,
    )


def _write_ledger(
    plan: OfficialRolloutPlan,
    path: Path,
    entries: dict[str, OfficialRolloutLedgerEntry],
    kinds: tuple[str, ...] | None,
) -> OfficialRolloutLedger:
    ordered = tuple(entries[shard.shard_id] for shard in plan.shards)
    del kinds  # Ledger completeness always describes the complete immutable plan.
    failed = tuple(entry.shard_id for entry in ordered if entry.status == "failed")
    ledger = OfficialRolloutLedger(
        suite_id=plan.suite_id,
        plan_hash=plan.plan_hash,
        entries=ordered,
        complete=bool(ordered) and all(entry.status == "complete" for entry in ordered),
        failed_shards=failed,
    )
    _atomic_json(path, ledger.to_dict())
    return ledger


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another official rollout runner owns this workspace") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _resolve_suite(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> tuple[OfficialCheckpointAuditSuiteV2, Path | None]:
    if isinstance(suite, OfficialCheckpointAuditSuiteV2):
        return suite, None
    path = Path(suite).resolve()
    return load_official_checkpoint_suite(path), path


def _resolve_inventory(
    assets: OfficialAssetInventory | str | Path,
) -> tuple[OfficialAssetInventory, Path | None]:
    if isinstance(assets, OfficialAssetInventory):
        return assets, None
    path = Path(assets).resolve()
    return load_official_asset_inventory(path), path


def _resolve_plan(plan: OfficialRolloutPlan | str | Path) -> OfficialRolloutPlan:
    if isinstance(plan, OfficialRolloutPlan):
        return plan
    return OfficialRolloutPlan.model_validate(_read_json(Path(plan)))


def _slug(value: str) -> str:
    return value.replace(":", "-").replace("_", "-")


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
