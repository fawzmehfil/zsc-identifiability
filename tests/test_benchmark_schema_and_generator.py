from copy import deepcopy
from fractions import Fraction

import pytest
from pydantic import ValidationError

from zsc_identifiability.benchmark_generator import generate
from zsc_identifiability.benchmark_models import MatchedBenchmarkSuite


def test_canonical_suite_generates_every_declared_population(benchmark_set) -> None:
    assert len(benchmark_set.populations) == 94
    assert len(benchmark_set.by_id()) == 94
    assert {item.descriptor.family_kind for item in benchmark_set.populations} == {
        "binary_role_allocation",
        "factorized_identity_memory",
    }
    assert all(item.game.schema_version == 1 for item in benchmark_set.populations)
    assert all(item.descriptor.game_hash for item in benchmark_set.populations)
    assert all(
        item.descriptor.suite_hash == benchmark_set.suite_hash
        for item in benchmark_set.populations
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["families"][0].update(reliability="2/5"), "reliability"),
        (lambda data: data["families"][0].update(intervention_cost="-1"), "nonnegative"),
        (
            lambda data: data["families"][0]["sweeps"][0].update(values=["2/5"]),
            "reliability",
        ),
        (
            lambda data: data["families"][1]["sweeps"][1].update(values=["3"]),
            r"\[0, 2\]",
        ),
        (lambda data: data.update(base_team_return="39"), "cost-plus-loss"),
        (
            lambda data: data["families"][0]["cells"].append(
                deepcopy(data["families"][0]["cells"][0])
            ),
            "duplicate binary cell",
        ),
        (
            lambda data: data["families"][0]["sweeps"][0].update(cell_id="missing"),
            "unknown cell",
        ),
    ],
)
def test_invalid_suite_parameters_are_rejected(
    benchmark_suite, mutation, message: str
) -> None:
    data = benchmark_suite.model_dump(mode="json")
    mutation(data)
    with pytest.raises(ValidationError, match=message):
        MatchedBenchmarkSuite.model_validate(data)


def test_matching_contract_must_reference_generated_population(benchmark_suite) -> None:
    data = benchmark_suite.model_dump(mode="json")
    data["matching_contracts"][0]["right_population_id"] = "missing-population"
    parsed = MatchedBenchmarkSuite.model_validate(data)
    with pytest.raises(ValueError, match="missing populations"):
        generate(parsed)


def test_symmetries_preserve_prior_and_loss_geometry(benchmark_set) -> None:
    populations = benchmark_set.by_id()
    reference = populations["binary-role-allocation--active_only--identity"]
    variants = [
        item
        for item in benchmark_set.populations
        if item.descriptor.family_id == "binary-role-allocation"
        and item.descriptor.cell_id == "active_only"
        and not item.descriptor.symmetry_id.startswith("sweep_")
    ]
    assert len(variants) == 4
    assert all(item.game.modes == reference.game.modes for item in variants)
    assert all(
        sorted(loss.loss for loss in item.game.decision_losses)
        == sorted(loss.loss for loss in reference.game.decision_losses)
        for item in variants
    )
    assert Fraction(reference.game.loss_exact("partner_alpha", "take_role_b")) == 40
