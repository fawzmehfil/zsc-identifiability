from fractions import Fraction

import pytest

from zsc_identifiability.population_metrics import compute


def population(benchmark_set, identifier: str):
    return benchmark_set.by_id()[identifier]


def test_binary_matched_cells_isolate_precommitment_dri(benchmark_set) -> None:
    early = compute(population(benchmark_set, "binary-role-allocation--passive_early--identity"))
    active = compute(population(benchmark_set, "binary-role-allocation--active_only--identity"))
    inseparable = compute(
        population(
            benchmark_set,
            "binary-role-allocation--precommit_inseparable--identity",
        )
    )

    assert early.values["rahman_brdiv_return"] == active.values["rahman_brdiv_return"] == 40
    assert early.values["zsceval_br_div_raw"] == active.values["zsceval_br_div_raw"] == 1
    assert early.values["lobp_action_oracle_score_nats"] == pytest.approx(
        active.values["lobp_action_oracle_score_nats"], abs=1e-12
    )
    assert early.values["passive_dri"] == Fraction(3, 5)
    assert active.values["passive_dri"] == 0
    assert active.values["active_dri"] == Fraction(3, 5)
    assert inseparable.values["active_dri"] == 0
    assert active.values["active_net_regret"] == 13
    assert active.values["best_fixed_response_value"] == 80
    assert active.values["task_active_oracle_return"] == 87
    assert active.values["information_only_return"] == 87
    assert active.applicability_flags["active_dri"]


def test_factorized_signals_match_identity_information_but_not_decision_value(
    benchmark_set,
) -> None:
    response = compute(
        population(
            benchmark_set,
            "factorized-identity-memory--remember_response--identity",
        )
    )
    subtype = compute(
        population(
            benchmark_set,
            "factorized-identity-memory--remember_subtype--identity",
        )
    )

    assert response.values["identity_mutual_information_bits"] == pytest.approx(
        subtype.values["identity_mutual_information_bits"], abs=1e-12
    )
    assert response.values["decision_signature_mutual_information_bits"] > 0
    assert subtype.values["decision_signature_mutual_information_bits"] == pytest.approx(0.0)
    assert response.values["passive_dri"] == Fraction(3, 5)
    assert subtype.values["passive_dri"] == 0
    assert sorted(response.prefix_tv_curves.values()) == sorted(subtype.prefix_tv_curves.values())
    assert response.values["zsceval_br_div_raw"] == subtype.values["zsceval_br_div_raw"] == 0


def test_raw_and_code_normalized_zsceval_determinants_are_distinct(benchmark_set) -> None:
    item = compute(population(benchmark_set, "binary-role-allocation--active_only--identity"))
    assert item.values["zsceval_br_div_raw"] == 1
    assert item.values["zsceval_br_div_code"] == Fraction(10**12, 1004006004001)
    assert item.brdiv_matrices["zsceval_float_slogdet_raw"]["sign"] == 1.0


def test_no_identification_needed_reports_unavailable_dri(benchmark_set) -> None:
    item = compute(
        population(
            benchmark_set,
            "binary-role-allocation--no_identification_needed--identity",
        )
    )
    assert item.values["prior_risk"] == 0
    assert item.values["passive_dri"] is None
    assert item.values["active_dri"] is None
    assert not item.applicability_flags["passive_dri"]


def test_fraction_and_float_backends_agree(benchmark_set) -> None:
    item = population(benchmark_set, "factorized-identity-memory--active_response--identity")
    exact = compute(item, "fraction")
    approximate = compute(item, "float")
    for key in (
        "prior_risk",
        "passive_residual_risk",
        "active_residual_risk",
        "rahman_brdiv_return",
        "zsceval_br_div_raw",
    ):
        assert float(exact.values[key]) == pytest.approx(approximate.values[key], abs=1e-10)
