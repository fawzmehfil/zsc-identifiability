from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from zsc_identifiability.established_official_analysis import (
    build_official_response_library,
)
from zsc_identifiability.established_official_assets import (
    load_official_asset_inventory,
    prepare_official_asset_lock,
)
from zsc_identifiability.established_official_models import (
    OfficialAssetInventory,
    OfficialAssetRecord,
    OfficialCheckpointAuditSuiteV2,
    OfficialMethodAsset,
    OfficialPartnerAsset,
    load_official_checkpoint_suite,
)
from zsc_identifiability.established_official_reporting import _load_plan_inventory
from zsc_identifiability.established_official_rollouts import (
    get_official_rollout_status,
    prepare_official_rollouts,
    run_official_rollouts,
)
from zsc_identifiability.established_official_statistics import (
    clustered_dri_coefficient_interval,
    nested_leave_one_scheme_out_feature_regression,
    nested_leave_one_scheme_out_regression,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "phase-6-established-validation/suites/canonical.json"


def test_canonical_official_suite_is_inference_only_and_complete() -> None:
    suite = load_official_checkpoint_suite(SUITE_PATH)
    assert suite.policy_training_allowed is False
    assert [layout.layout_id for layout in suite.layouts] == ["random3_m", "small_corridor"]
    assert {method.method_id for method in suite.methods} == {
        "fcp",
        "mep",
        "trajedi",
        "hsp",
        "cole",
        "e3t",
    }
    architectures = {method.method_id: method.architecture_by_layout for method in suite.methods}
    assert architectures["e3t"] == {"random3_m": "mlp", "small_corridor": "mlp"}
    assert architectures["cole"] == {"random3_m": "mlp", "small_corridor": "rnn"}


def test_official_suite_rejects_training_and_split_leakage() -> None:
    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="policy training is forbidden"):
        OfficialCheckpointAuditSuiteV2.model_validate({**raw, "training_budget": 1})
    leaked = dict(raw)
    leaked["split_key_salts"] = {
        "calibration": "same",
        "validation": "same",
        "confirmatory": "different",
    }
    with pytest.raises(ValidationError, match="disjoint"):
        OfficialCheckpointAuditSuiteV2.model_validate(leaked)


def test_asset_lock_uses_only_official_yaml_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _minimal_suite()

    def fake_download(url: str) -> bytes:
        layout = "random3_m" if "random3_m" in url else "small_corridor"
        return _benchmark_yaml(layout)

    monkeypatch.setattr(
        "zsc_identifiability.established_official_assets._download_bytes", fake_download
    )
    lock = prepare_official_asset_lock(suite, tmp_path)
    roles = [entry.role for entry in lock.entries]
    assert roles.count("partner") == 4
    assert roles.count("response") == 4
    assert roles.count("method") == 8
    assert not any("w2" in entry.relative_path for entry in lock.entries)
    assert (tmp_path / "official-asset-lock.json").is_file()


def test_rollout_plan_is_disjoint_resumable_and_training_free(tmp_path: Path) -> None:
    suite = _minimal_suite()
    inventory = _minimal_inventory(suite, tmp_path)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite.to_dict()), encoding="utf-8")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory.to_dict()), encoding="utf-8")
    plan = prepare_official_rollouts(suite_path, inventory_path, tmp_path / "run")
    assert plan.inventory_path == str(inventory_path.resolve())
    all_keys = {
        split: {
            key
            for shard in plan.shards
            if shard.kind == "trace" and shard.split == split
            for key in shard.episode_keys
        }
        for split in ("calibration", "validation", "confirmatory")
    }
    assert not all_keys["calibration"] & all_keys["validation"]
    assert not all_keys["calibration"] & all_keys["confirmatory"]
    assert not all_keys["validation"] & all_keys["confirmatory"]
    for shard in plan.shards:
        request = json.loads(Path(shard.request_path).read_text(encoding="utf-8"))
        assert request["policy_training_allowed"] is False
        if shard.kind == "method":
            assert request["payload"]["policy_architecture"] in {"mlp", "rnn"}
        if shard.kind == "parity":
            assert request["payload"]["checkpoint_architecture"] == "rnn"

    for layout in ("random3_m", "small_corridor"):
        partners = sorted(
            {
                shard.partner_id
                for shard in plan.shards
                if shard.layout_id == layout and shard.partner_id is not None
            }
        )
        for partner in partners:
            response_keys = {
                shard.episode_keys
                for shard in plan.shards
                if shard.kind == "response" and shard.partner_id == partner
            }
            assert len(response_keys) == 1
            method_keys = {
                shard.episode_keys
                for shard in plan.shards
                if shard.kind == "method" and shard.partner_id == partner
            }
            assert len(method_keys) == 1
            assert {
                shard.deployment
                for shard in plan.shards
                if shard.kind == "method" and shard.partner_id == partner
            } == {"greedy", "stochastic"}
            for split in ("calibration", "validation", "confirmatory"):
                trace_keys = {
                    shard.episode_keys
                    for shard in plan.shards
                    if shard.kind == "trace"
                    and shard.partner_id == partner
                    and shard.split == split
                }
                assert len(trace_keys) == 1

    calls: list[str] = []

    def executor(shard: Any, _workspace: Path, _logs: Path) -> str:
        calls.append(shard.shard_id)
        result = Path(shard.result_path)
        result.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(result, "wt", encoding="utf-8") as handle:
            json.dump({"shard_id": shard.shard_id}, handle)
        return hashlib.sha256(result.read_bytes()).hexdigest()

    first = run_official_rollouts(plan, workers=2, kinds=("parity",), executor=executor)
    assert not first.complete
    parity_ids = {shard.shard_id for shard in plan.shards if shard.kind == "parity"}
    assert all(
        entry.status == "complete" for entry in first.entries if entry.shard_id in parity_ids
    )
    first_call_count = len(calls)
    second = run_official_rollouts(plan, workers=2, kinds=("parity",), executor=executor)
    assert not second.complete
    assert len(calls) == first_call_count
    assert get_official_rollout_status(plan).plan_hash == plan.plan_hash


def test_rollout_plan_materializes_in_memory_provenance(tmp_path: Path) -> None:
    suite = _minimal_suite()
    inventory = _minimal_inventory(suite, tmp_path)
    workspace = tmp_path / "run"

    plan = prepare_official_rollouts(suite, inventory, workspace)

    assert plan.suite_path == str((workspace / "official-audit-suite.json").resolve())
    assert plan.inventory_path == str((workspace / "official-asset-inventory.json").resolve())
    assert load_official_checkpoint_suite(plan.suite_path) == suite
    assert load_official_asset_inventory(plan.inventory_path) == inventory


def test_analysis_recovers_hash_validated_legacy_inventory_path(tmp_path: Path) -> None:
    suite = _minimal_suite()
    inventory = _minimal_inventory(suite, tmp_path)
    workspace = tmp_path / "run"
    plan = prepare_official_rollouts(suite, inventory, workspace)
    legacy_plan = plan.model_copy(update={"inventory_path": "<in-memory>"})

    assert _load_plan_inventory(legacy_plan) == inventory

    mismatched = legacy_plan.model_copy(update={"inventory_hash": "f" * 64})
    with pytest.raises(ValueError, match="inventory hash does not match"):
        _load_plan_inventory(mismatched)


def test_response_library_uses_ratio_loss_and_margin_sensitivity(tmp_path: Path) -> None:
    suite = _minimal_suite()
    paths: list[Path] = []
    for layout in suite.layouts:
        partners = [f"{layout.layout_id}:hsp1:mid", f"{layout.layout_id}:hsp1:final"]
        matrix = ((100.0, 50.0), (50.0, 100.0))
        for left, partner in enumerate(partners):
            for right, response in enumerate(partners):
                path = tmp_path / f"{layout.layout_id}-{left}-{right}.json.gz"
                payload = {
                    "operation": "official_response_rollout",
                    "policy_training_performed": False,
                    "partner_deployment": "stochastic",
                    "layout_id": layout.layout_id,
                    "partner_id": partner,
                    "response_id": response,
                    "episode_returns": [matrix[left][right]] * layout.response_episodes_per_pair,
                    "mean_ego_event_features": [float(right == 0), float(right == 1)],
                }
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                paths.append(path)
    libraries = build_official_response_library(paths, suite)
    assert len(libraries) == 2
    for library in libraries:
        assert library.normalized_losses == ((0.0, 0.5), (0.5, 0.0))
        assert len(library.conflicting_pairs_by_margin["0.01"]) == 1
        assert len(library.conflicting_pairs_by_margin["0.05"]) == 1


def test_nested_regression_never_uses_br_prox() -> None:
    rows: list[dict[str, Any]] = []
    schemes = ("a", "b", "c", "d", "e", "f", "g", "h")
    for left_index, left in enumerate(schemes):
        for right in schemes[left_index + 1 :]:
            for method_number, method in enumerate(("fcp", "mep")):
                dri = 0.1 * (left_index + 1)
                rows.append(
                    {
                        "normalized_response_library_regret": 0.5 - dri + 0.01 * method_number,
                        "left_scheme_id": left,
                        "right_scheme_id": right,
                        "method_id": method,
                        "layout_id": "random3_m",
                        "precommitment_dri": dri,
                        "identity_mi_nats": 0.25 * dri,
                        "partner_competence": 1.0,
                        "prior_confusion_risk": 0.5,
                        "conflict_coefficient": 1.0,
                        "rahman_brdiv_return": 0.5,
                        "zsceval_br_div_raw": 1.0,
                        "visible_action_predictability": 0.2,
                        "prefix_tv": dri / 2,
                        "br_prox": 0.99,
                    }
                )
    report = nested_leave_one_scheme_out_regression(rows)
    assert report["br_prox_used_as_predictor"] is False
    assert "br_prox" not in report["baseline_features"]
    assert report["folds"]
    identity_report = nested_leave_one_scheme_out_feature_regression(
        rows, incremental_feature="identity_mi_nats"
    )
    prefix_report = nested_leave_one_scheme_out_feature_regression(
        rows, incremental_feature="prefix_tv"
    )
    assert identity_report["incremental_feature"] == "identity_mi_nats"
    assert "prefix_tv" not in prefix_report["baseline_features"]
    assert prefix_report["incremental_feature"] == "prefix_tv"


def test_adjusted_clustered_interval_uses_all_three_registered_clusters() -> None:
    rows: list[dict[str, Any]] = []
    schemes = ("a", "b", "c", "d", "e", "f", "g", "h")
    for left_index, left in enumerate(schemes):
        for right_index, right in enumerate(schemes[left_index + 1 :], left_index + 1):
            dri = 0.07 + 0.005 * (left_index + 1) * (right_index + 2)
            for method_number, method in enumerate(("fcp", "mep")):
                for method_seed in (1, 2):
                    for episode_key in range(4):
                        rows.append(
                            {
                                "normalized_response_library_regret": (
                                    0.6 - 0.4 * dri + 0.01 * method_number + 0.002 * episode_key
                                ),
                                "left_scheme_id": left,
                                "right_scheme_id": right,
                                "method_id": method,
                                "method_seed": method_seed,
                                "layout_id": "random3_m",
                                "environment_key": f"random3_m:{episode_key}",
                                "precommitment_dri": dri,
                                "partner_competence": 0.8 + 0.01 * right_index,
                                "prior_confusion_risk": 0.2 + 0.01 * left_index,
                                "conflict_coefficient": 0.4 + 0.02 * right_index,
                                "rahman_brdiv_return": 0.3 + 0.01 * left_index,
                                "zsceval_br_div_raw": 0.5 + 0.02 * right_index,
                                "visible_action_predictability": 0.7 + 0.01 * episode_key,
                                "prefix_tv": 0.1 + 0.007 * ((left_index + 2 * right_index) % 5),
                            }
                        )
    interval = clustered_dri_coefficient_interval(rows, resamples=128, seed=7)
    assert interval["adjusted_for_registered_controls"] is True
    assert interval["cluster_dimensions"] == [
        "method_seed",
        "hsp_scheme",
        "episode_key",
    ]
    assert interval["resamples_completed"] > 100


def test_official_runtime_source_contains_no_training_path() -> None:
    source = (
        ROOT
        / "phase-6-established-validation/runtime-legacy/src/stage6_legacy_runtime/official_eval.py"
    ).read_text(encoding="utf-8")
    assert "policy_training_allowed" in source
    assert "import trainer" not in source.lower()
    assert "torch.optim" not in source.lower()
    assert "optimizer =" not in source.lower()
    assert '"partner_deployment": "stochastic"' in source
    assert "partner_deterministic" in source


def test_script_controller_bridges_new_recipe_capacity_name() -> None:
    runtime = _load_official_runtime_module()
    mdp = SimpleNamespace(max_num_items_for_soup=3)
    state = object()

    class CapacityReadingPeriod:
        def reset(self, candidate: Any, received_state: Any, player: int) -> None:
            assert candidate is mdp
            assert candidate.num_items_for_soup == 3
            assert received_state is state
            assert player == 1

        def done(self, candidate: Any, received_state: Any, player: int) -> bool:
            assert candidate.num_items_for_soup == 3
            assert received_state is state
            assert player == 1
            return False

    controller = runtime._Controller(CapacityReadingPeriod(), physical_ego=1)
    env = SimpleNamespace(base_env=SimpleNamespace(mdp=mdp, state=state))

    assert controller.done(env) is False
    assert mdp.num_items_for_soup == mdp.max_num_items_for_soup == 3


def test_script_controller_preserves_old_capacity_and_rejects_ambiguity() -> None:
    runtime = _load_official_runtime_module()
    old_mdp = SimpleNamespace(num_items_for_soup=3)
    assert runtime._script_controller_mdp(old_mdp) is old_mdp
    assert not hasattr(old_mdp, "max_num_items_for_soup")

    conflicting = SimpleNamespace(num_items_for_soup=2, max_num_items_for_soup=3)
    with pytest.raises(ValueError, match="soup-capacity fields disagree"):
        runtime._script_controller_mdp(conflicting)

    with pytest.raises(AttributeError, match="requires a positive"):
        runtime._script_controller_mdp(SimpleNamespace())
    with pytest.raises(AttributeError, match="requires a positive"):
        runtime._script_controller_mdp(SimpleNamespace(num_items_for_soup=0))


def _load_official_runtime_module() -> Any:
    path = (
        ROOT
        / "phase-6-established-validation/runtime-legacy/src/stage6_legacy_runtime/official_eval.py"
    )
    spec = importlib.util.spec_from_file_location("stage6_test_official_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_suite() -> OfficialCheckpointAuditSuiteV2:
    raw = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    for layout in raw["layouts"]:
        layout["expected_partner_count"] = 2
        layout["expected_scheme_count"] = 1
        layout["response_episodes_per_pair"] = 2
        layout["trace_episodes"] = {
            "calibration": 2,
            "validation": 2,
            "confirmatory": 2,
        }
        layout["method_episodes"] = 2
        layout["diagnostic_options"] = ["ordinary_progress"]
    raw["methods"] = [{**method, "seeds": [1]} for method in raw["methods"][:4]]
    return OfficialCheckpointAuditSuiteV2.model_validate(raw)


def _benchmark_yaml(layout: str) -> bytes:
    return f"""
bias1_mid:
  policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
  featurize_type: ppo
  train: false
  model_path:
    actor: {layout}/hsp/s1/hsp/hsp1_mid_w0_actor.pt
bias1_final:
  policy_config_path: {layout}/policy_config/mlp_policy_config.pkl
  featurize_type: ppo
  train: false
  model_path:
    actor: {layout}/hsp/s1/hsp/hsp1_final_w0_actor.pt
agent_name:
  policy_config_path: {layout}/policy_config/rnn_policy_config.pkl
  featurize_type: ppo
  train: false
  model_path:
    actor: {layout}/algorithm/s2/population/seed.pt
""".encode()


def _minimal_inventory(
    suite: OfficialCheckpointAuditSuiteV2, tmp_path: Path
) -> OfficialAssetInventory:
    partners = tuple(
        OfficialPartnerAsset(
            partner_id=f"{layout.layout_id}:hsp1:{stage}",
            layout_id=layout.layout_id,
            scheme_id="hsp1",
            training_stage=stage,
            partner_checkpoint_path=str(tmp_path / f"{layout.layout_id}-{stage}-w0.pt"),
            response_checkpoint_path=str(tmp_path / f"{layout.layout_id}-{stage}-w1.pt"),
            partner_asset_id=f"{layout.layout_id}-{stage}-w0",
            response_asset_id=f"{layout.layout_id}-{stage}-w1",
        )
        for layout in suite.layouts
        for stage in ("mid", "final")
    )
    methods = tuple(
        OfficialMethodAsset(
            method_id=method.method_id,
            layout_id=layout.layout_id,
            seed=1,
            checkpoint_path=str(tmp_path / f"{layout.layout_id}-{method.method_id}.pt"),
            asset_id=f"{layout.layout_id}-{method.method_id}",
            policy_architecture=method.architecture_by_layout[layout.layout_id],
            recurrent=method.architecture_by_layout[layout.layout_id] == "rnn",
        )
        for layout in suite.layouts
        for method in suite.methods
    )
    records = tuple(
        OfficialAssetRecord(
            asset_id=asset_id,
            local_path=str(tmp_path / asset_id),
            size=1,
            file_hash="a" * 64,
        )
        for asset_id in {
            *(item.partner_asset_id for item in partners),
            *(item.response_asset_id for item in partners),
            *(item.asset_id for item in methods),
        }
    )
    payload = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "lock_hash": "b" * 64,
        "partners": [item.to_dict() for item in partners],
        "methods": [item.to_dict() for item in methods],
        "assets": [item.to_dict() for item in records],
        "duplicate_tensor_groups": [],
        "complete": True,
        "missing_asset_ids": [],
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return OfficialAssetInventory.model_validate({**payload, "inventory_hash": content_hash})
