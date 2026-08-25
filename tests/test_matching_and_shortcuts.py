from fractions import Fraction

from hypothesis import given
from hypothesis import strategies as st

from zsc_identifiability.benchmark_audit import audit_pair, audit_shortcuts
from zsc_identifiability.benchmark_generator import generate
from zsc_identifiability.benchmark_models import MatchedBenchmarkSuite
from zsc_identifiability.population_metrics import compute


def test_every_canonical_matching_contract_passes(benchmark_set) -> None:
    populations = benchmark_set.by_id()
    for contract in benchmark_set.suite.matching_contracts:
        result = audit_pair(
            populations[contract.left_population_id],
            populations[contract.right_population_id],
            contract,
        )
        assert result.passed, result.to_dict()
        assert all(item.status == "pass" for item in result.items)


def test_memory_and_evidence_blind_shortcuts_fail_as_intended(benchmark_set) -> None:
    item = benchmark_set.by_id()[
        "factorized-identity-memory--remember_response--identity"
    ]
    audit = audit_shortcuts(item)
    assert audit.passed
    assert audit.best_fixed_risk == 20
    assert audit.evidence_blind_risk == 20
    assert audit.memoryless_risk == 20
    assert audit.history_aware_risk == 8


def test_late_evidence_and_boundary_do_not_leak_into_policy_choice(benchmark_set) -> None:
    populations = benchmark_set.by_id()
    late = audit_shortcuts(
        populations["binary-role-allocation--precommit_inseparable--identity"]
    )
    boundary = audit_shortcuts(
        populations["binary-role-allocation--active_boundary--identity"]
    )
    assert late.postcommit_leak_free
    assert late.history_aware_risk == 20
    assert boundary.valueless_probe_tie_break_ok


@given(
    numerator=st.integers(min_value=1, max_value=10),
    denominator=st.integers(min_value=2, max_value=10),
)
def test_response_dri_is_monotone_in_reliability(
    benchmark_suite: MatchedBenchmarkSuite,
    numerator: int,
    denominator: int,
) -> None:
    q = max(Fraction(1, 2), min(Fraction(1), Fraction(numerator, denominator)))
    data = benchmark_suite.model_dump(mode="json")
    binary = data["families"][0]
    binary["reliability"] = str(q)
    binary["generate_symmetries"] = False
    binary["cells"] = [
        cell for cell in binary["cells"] if cell["cell_id"] == "active_only"
    ]
    binary["sweeps"] = []
    data["families"] = [binary]
    data["matching_contracts"] = []
    suite = MatchedBenchmarkSuite.model_validate(data)
    item = compute(generate(suite).populations[0])
    assert item.values["active_dri"] == 2 * q - 1


def test_intervention_cost_changes_net_value_not_information_value(benchmark_set) -> None:
    populations = benchmark_set.by_id()
    cheap = compute(populations["binary-role-allocation--active_only--sweep_cost_5"])
    costly = compute(populations["binary-role-allocation--active_only--sweep_cost_15"])
    assert cheap.values["active_dri"] == costly.values["active_dri"] == Fraction(3, 5)
    assert cheap.values["active_net_regret"] == 13
    assert costly.values["active_net_regret"] == 20
