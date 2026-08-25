from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from zsc_identifiability.learning_models import LearningAuditSuite, load_learning_suite_file
from zsc_identifiability.learning_pools import (
    audit_learning_pool_leakage,
    audit_learning_pool_matching,
    generate_learning_pools,
    generate_symmetry_pool,
    make_smoke_pool,
)
from zsc_identifiability.learning_tuning import candidate_method_configs

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "phase-4-learned-audit" / "suites" / "canonical.json"


@pytest.fixture(scope="module")
def learning_suite():
    return load_learning_suite_file(SUITE_PATH)


@pytest.fixture(scope="module")
def learning_pools(learning_suite):
    return generate_learning_pools(learning_suite, suite_path=SUITE_PATH)


def test_learning_suite_generates_disjoint_valid_v1_games(learning_pools) -> None:
    assert len(learning_pools.cells) == 13
    assert all(len(item.train) == 3 for item in learning_pools.cells)
    assert all(len(item.validation) == len(item.test) == 1 for item in learning_pools.cells)
    assert all(
        game.game.schema_version == 1
        for cell in learning_pools.cells
        for split in (cell.train, cell.validation, cell.test)
        for game in split
    )
    report = audit_learning_pool_leakage(learning_pools)
    assert report["passed"] is True
    assert audit_learning_pool_matching(learning_pools)["passed"] is True


def test_test_games_are_canonical_but_never_training_dynamics(learning_pools) -> None:
    for cell in learning_pools.cells:
        assert cell.test[0].game == cell.source_population.game
        assert cell.test[0].dynamics_hash not in {item.dynamics_hash for item in cell.train}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda data: data["profiles"]["validation"][0].update(
                profile_id=data["profiles"]["train"][0]["profile_id"]
            ),
            "identifiers",
        ),
        (
            lambda data: data["profiles"]["train"][0].update(reliability="2/5"),
            "reliability",
        ),
        (lambda data: data.update(cells=[]), "at least one cell"),
        (
            lambda data: data["methods"].append(deepcopy(data["methods"][0])),
            "duplicate learning method",
        ),
    ],
)
def test_invalid_learning_suite_is_rejected(learning_suite, mutation, message: str) -> None:
    data = learning_suite.model_dump(mode="json")
    mutation(data)
    with pytest.raises(ValidationError, match=message):
        LearningAuditSuite.model_validate(data)


def test_preregistered_search_grid_is_method_specific(learning_suite) -> None:
    methods = {method.method_id: method for method in learning_suite.methods}
    assert len(candidate_method_configs(methods["gru_ppo_active"])) == 4
    assert len(candidate_method_configs(methods["pace_aux"])) == 8
    assert len(candidate_method_configs(methods["pace_style"])) == 16
    assert len(candidate_method_configs(methods["odits_style"])) == 16


def test_selected_symmetry_population_is_independently_generated(learning_suite) -> None:
    pool = generate_symmetry_pool(
        learning_suite,
        "active_only",
        "role_signal_swap",
        suite_path=SUITE_PATH,
    )
    assert pool.cell_id == "active_only--symmetry-role_signal_swap"
    assert pool.source_population.descriptor.symmetry_id == "role_signal_swap"
    assert {item.dynamics_hash for item in pool.train}.isdisjoint(
        {item.dynamics_hash for item in pool.test}
    )


def test_smoke_pool_is_perfectly_diagnostic_and_zero_cost(learning_pools) -> None:
    smoke = make_smoke_pool(learning_pools.by_cell()["active_only"])
    game = smoke.train[0].game
    assert all(outcome.cost == "0" for row in game.kernels for outcome in row.outcomes)
    staged = [row for row in game.kernels if row.action == "stage_shared_item" and row.time == 0]
    assert all(
        sorted(outcome.probability for outcome in row.outcomes) == ["0", "1"] for row in staged
    )
