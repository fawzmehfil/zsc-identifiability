from pathlib import Path

import pytest
import torch

from zsc_identifiability.learning_models import load_learning_suite_file
from zsc_identifiability.learning_pools import generate_learning_pools
from zsc_identifiability.learning_runner import (
    SMOKE_ACTIVE_DIAGNOSTICS,
    SMOKE_ACTIVE_METHODS,
    SMOKE_NO_IDENTIFICATION_METHODS,
    SMOKE_RECURRENT_METHODS,
    SMOKE_TOM_PASSIVE_CONTROL,
    _smoke_expectation,
    assess_smoke_matrix,
)
from zsc_identifiability.learning_trainer import (
    _tom_oracle_routing,
    load_checkpoint,
    train_method,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "phase-4-learned-audit" / "suites" / "canonical.json"


def _smoke_evaluation(method_id: str, cell_id: str, team_return: float) -> dict[str, object]:
    return {
        "method_id": method_id,
        "cell_id": cell_id,
        "evaluation": {"team_return": team_return},
    }


def _passing_smoke_evaluations() -> dict[str, dict[str, object]]:
    evaluations: dict[str, dict[str, object]] = {}

    def add(method_id: str, cell_id: str, team_return: float) -> None:
        evaluations[f"{method_id}/{cell_id}"] = _smoke_evaluation(method_id, cell_id, team_return)

    for method_id in SMOKE_NO_IDENTIFICATION_METHODS:
        add(method_id, "no_identification_needed", 100.0)
    for method_id in (*SMOKE_ACTIVE_METHODS, *SMOKE_ACTIVE_DIAGNOSTICS):
        add(method_id, "active_only", 100.0)
    add("tom_selector_style", "active_only", 80.0)
    add(*SMOKE_TOM_PASSIVE_CONTROL, 100.0)
    add("mlp_ppo", "remember_response", 80.0)
    for method_id in SMOKE_RECURRENT_METHODS:
        add(method_id, "remember_response", 100.0)
    return evaluations


def test_smoke_expectations_are_method_appropriate() -> None:
    assert _smoke_expectation("no_identification_needed", "pace_style") == (
        "universal_response_per_method",
        99.0,
    )
    assert _smoke_expectation("active_only", "tom_selector_style") == (
        "active_identifiability_diagnostic",
        None,
    )
    assert _smoke_expectation("active_only", "gru_ppo_active") == (
        "aggregate_active_capability",
        None,
    )
    assert _smoke_expectation("passive_early", "tom_selector_style") == (
        "tom_passive_evidence_sanity",
        98.0,
    )


def test_tom_hidden_routing_is_limited_to_first_quarter() -> None:
    assert _tom_oracle_routing(0, 20_000)
    assert _tom_oracle_routing(4_999, 20_000)
    assert not _tom_oracle_routing(5_000, 20_000)
    assert not _tom_oracle_routing(20_000, 20_000)


def test_aggregate_smoke_gate_preserves_tom_active_failure() -> None:
    report = assess_smoke_matrix(_passing_smoke_evaluations())
    assert report["status"] == "complete"
    assert report["passed"] is True
    assert report["checks"]["tom_active_diagnostic_preserved"] is True


def test_aggregate_smoke_gate_requires_tom_passive_sanity() -> None:
    evaluations = _passing_smoke_evaluations()
    evaluations["tom_selector_style/passive_early"]["evaluation"] = {"team_return": 80.0}
    report = assess_smoke_matrix(evaluations)
    assert report["status"] == "complete"
    assert report["passed"] is False
    assert report["checks"]["tom_passive_evidence_sanity"] is False


def test_aggregate_smoke_gate_reports_missing_runs() -> None:
    evaluations = _passing_smoke_evaluations()
    del evaluations["pace_style/remember_response"]
    report = assess_smoke_matrix(evaluations)
    assert report["status"] == "incomplete"
    assert report["passed"] is False
    assert report["missing_runs"] == ["pace_style/remember_response"]


def test_tiny_training_checkpoint_round_trip(tmp_path) -> None:
    suite = load_learning_suite_file(SUITE_PATH)
    suite = suite.model_copy(
        update={
            "budget": suite.budget.model_copy(update={"num_envs": 8, "checkpoint_interval": 32})
        }
    )
    pools = generate_learning_pools(suite, suite_path=SUITE_PATH)
    method = next(item for item in suite.methods if item.method_id == "gru_ppo_active")
    method = method.model_copy(
        update={
            "config": method.config.model_copy(
                update={
                    "transitions_per_update": 32,
                    "minibatch_size": 16,
                    "optimization_epochs": 1,
                }
            )
        }
    )
    manifest = train_method(
        suite,
        method,
        pools.by_cell()["active_only"],
        tmp_path,
        seed=29,
        transitions=32,
    )
    assert manifest.completed_transitions >= 32
    assert Path(manifest.checkpoint_path).exists()
    model, payload = load_checkpoint(manifest.checkpoint_path)
    assert model.method_id == "gru_ppo_active"
    assert payload["configuration_hash"] == manifest.configuration_hash


def test_cpu_resume_matches_uninterrupted_training(tmp_path) -> None:
    suite = load_learning_suite_file(SUITE_PATH)
    suite = suite.model_copy(
        update={
            "budget": suite.budget.model_copy(update={"num_envs": 8, "checkpoint_interval": 32})
        }
    )
    pools = generate_learning_pools(suite, suite_path=SUITE_PATH)
    method = next(item for item in suite.methods if item.method_id == "gru_ppo_active")
    method = method.model_copy(
        update={
            "config": method.config.model_copy(
                update={
                    "transitions_per_update": 32,
                    "minibatch_size": 16,
                    "optimization_epochs": 1,
                }
            )
        }
    )
    cell = pools.by_cell()["active_only"]
    first = train_method(suite, method, cell, tmp_path / "resumed", seed=41, transitions=32)
    numbered = next((Path(first.checkpoint_path).parent / "checkpoints").glob("step-*.pt"))
    resumed = train_method(
        suite,
        method,
        cell,
        tmp_path / "resumed",
        seed=41,
        transitions=64,
        resume_checkpoint=numbered,
    )
    uninterrupted = train_method(
        suite,
        method,
        cell,
        tmp_path / "uninterrupted",
        seed=41,
        transitions=64,
    )
    _, resumed_payload = load_checkpoint(resumed.checkpoint_path)
    _, uninterrupted_payload = load_checkpoint(uninterrupted.checkpoint_path)
    assert resumed.completed_transitions == uninterrupted.completed_transitions
    for name, tensor in resumed_payload["model_state"].items():
        assert torch.equal(tensor, uninterrupted_payload["model_state"][name]), name


@pytest.mark.parametrize(
    "method_id",
    [
        "mlp_ppo",
        "gru_ppo_passive",
        "odits_style",
        "pace_aux",
        "pace_style",
        "talents_style",
        "tom_selector_style",
        "csp_style_reconnaissance",
    ],
)
def test_each_learning_baseline_completes_one_update(tmp_path, method_id: str) -> None:
    suite = load_learning_suite_file(SUITE_PATH)
    suite = suite.model_copy(
        update={
            "budget": suite.budget.model_copy(update={"num_envs": 4, "checkpoint_interval": 16})
        }
    )
    pools = generate_learning_pools(suite, suite_path=SUITE_PATH)
    method = next(item for item in suite.methods if item.method_id == method_id)
    method = method.model_copy(
        update={
            "config": method.config.model_copy(
                update={
                    "transitions_per_update": 32,
                    "minibatch_size": 16,
                    "optimization_epochs": 1,
                }
            )
        }
    )
    cell_id = "active_only"
    manifest = train_method(
        suite,
        method,
        pools.by_cell()[cell_id],
        tmp_path / method_id,
        seed=53,
        transitions=32,
    )
    assert manifest.completed_transitions >= 32
    _, payload = load_checkpoint(manifest.checkpoint_path)
    assert payload["method"]["method_id"] == method_id
    if method_id == "talents_style":
        metadata = payload["pretraining_metadata"]
        assert metadata["trajectory_count"] == 96
        assert metadata["hidden_mode_labels_used"] is False
        assert sum(metadata["cluster_sizes"]) == 96
