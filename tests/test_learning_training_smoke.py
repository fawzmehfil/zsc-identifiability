from pathlib import Path

import pytest
import torch

from zsc_identifiability.learning_models import load_learning_suite_file
from zsc_identifiability.learning_pools import generate_learning_pools
from zsc_identifiability.learning_trainer import load_checkpoint, train_method

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "phase-4-learned-audit" / "suites" / "canonical.json"


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
