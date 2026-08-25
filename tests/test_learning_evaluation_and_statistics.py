from pathlib import Path

import numpy as np
import pytest
import torch

from zsc_identifiability.learning_env import build_observation_layout
from zsc_identifiability.learning_evaluation import (
    _distance_to_frontier,
    evaluate_neural_policy_exact,
    evaluate_neural_policy_sampled,
)
from zsc_identifiability.learning_methods import LearnedPolicy, NetworkOutput
from zsc_identifiability.learning_models import load_learning_suite_file
from zsc_identifiability.learning_pools import generate_learning_pools
from zsc_identifiability.learning_runner import _oracle_controls, _plot_results
from zsc_identifiability.learning_statistics import (
    kendall_rank_correlation,
    paired_bootstrap_interval,
    strict_ranking_reversals,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "phase-4-learned-audit" / "suites" / "canonical.json"


class ScriptedActivePolicy(LearnedPolicy):
    def __init__(self, cell) -> None:
        super().__init__()
        self.method_id = "scripted"
        self.hidden_size = 1
        self.recurrent = True
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        game = cell.test[0].game
        self.layout = build_observation_layout(game)
        self.stage = self.layout.action_ids.index("stage_shared_item")
        self.commit_a = self.layout.action_ids.index("commit:take_role_a")
        self.commit_b = self.layout.action_ids.index("commit:take_role_b")
        self.signal_zero = self.layout.feature_names.index("observation:signal_zero")
        self.start = self.layout.feature_names.index("observation:<start>")

    def forward_step(self, observations, hidden, action_masks):
        logits = (
            torch.full(
                (observations.shape[0], len(self.layout.action_ids)),
                -1000.0,
                device=observations.device,
            )
            + self.anchor
        )
        for index in range(observations.shape[0]):
            if observations[index, self.start] > 0.5:
                action = self.stage
            elif observations[index, self.signal_zero] > 0.5:
                action = self.commit_a
            else:
                action = self.commit_b
            logits[index, action] = 0
        return NetworkOutput(
            self.mask_logits(logits, action_masks),
            torch.zeros(observations.shape[0]),
            hidden,
        )


def active_cell():
    suite = load_learning_suite_file(SUITE_PATH)
    return generate_learning_pools(suite, suite_path=SUITE_PATH).by_cell()["active_only"]


def test_exact_learned_evaluator_recovers_active_game_decomposition() -> None:
    cell = active_cell()
    result = evaluate_neural_policy_exact(
        ScriptedActivePolicy(cell), cell.test[0], method_id="scripted", mode="greedy"
    )
    assert result.team_return == pytest.approx(87)
    assert result.expected_intervention_cost == pytest.approx(5)
    assert result.actual_confusion_loss == pytest.approx(8)
    assert result.residual_bayes_risk == pytest.approx(8)
    assert result.decision_utilization_gap == pytest.approx(0)
    assert result.policy_dri == pytest.approx(0.6)
    assert result.probe_probability == pytest.approx(1)


def test_sampled_policy_evaluation_calibrates_to_exact_value() -> None:
    cell = active_cell()
    policy = ScriptedActivePolicy(cell)
    exact = evaluate_neural_policy_exact(policy, cell.test[0], method_id="scripted", mode="greedy")
    sampled = evaluate_neural_policy_sampled(
        policy,
        cell.test[0],
        method_id="scripted",
        mode="greedy",
        episodes=20_000,
        batch_size=1_000,
        seed=1729,
    )
    assert sampled.evaluator == "sampled_neural_policy_20000_episodes"
    assert sampled.team_return == pytest.approx(exact.team_return, abs=0.5)
    assert sampled.residual_bayes_risk == pytest.approx(exact.residual_bayes_risk, abs=0.5)


def test_paired_statistics_detect_a_strict_reversal() -> None:
    rows = []
    for seed in range(10):
        rows.extend(
            [
                {"cell_id": "left", "method_id": "a", "seed": seed, "team_return": 10.0},
                {"cell_id": "left", "method_id": "b", "seed": seed, "team_return": 0.0},
                {"cell_id": "right", "method_id": "a", "seed": seed, "team_return": 0.0},
                {"cell_id": "right", "method_id": "b", "seed": seed, "team_return": 10.0},
            ]
        )
    report = strict_ranking_reversals(rows, "left", "right", resamples=1000)
    assert report["strict_reversal_count"] == 1
    interval = paired_bootstrap_interval(np.ones(10), np.zeros(10), resamples=1000)
    assert interval.lower > 0
    assert kendall_rank_correlation({"a": 1, "b": 0}, {"a": 0, "b": 1}) == -1


def test_frontier_distance_includes_randomized_mixture_segments() -> None:
    assert _distance_to_frontier(2.5, 14, ((0, 20), (5, 8)), 40) == pytest.approx(0)


def test_complete_report_figures_accept_exact_rational_controls(tmp_path) -> None:
    suite = load_learning_suite_file(SUITE_PATH)
    pools = generate_learning_pools(suite, suite_path=SUITE_PATH)
    controls = _oracle_controls(pools, suite)
    rows = []
    for cell in pools.cells:
        for method in ("mlp_ppo", "gru_ppo_active", "gru_ppo_passive"):
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "method_id": method,
                    "seed": 2001,
                    "mode": "greedy",
                    "team_return": 80.0,
                    "policy_dri": 0.0,
                    "expected_intervention_cost": 0.0,
                    "residual_bayes_risk": 20.0,
                    "decision_utilization_gap": 0.0,
                    "identity_mutual_information_bits": 0.0,
                    "probe_probability": 0.0,
                }
            )
    _plot_results(tmp_path / "figures", rows, controls, tmp_path / "runs", [])
    assert (tmp_path / "figures" / "learned-policies-active-frontier.pdf").exists()
    assert (tmp_path / "figures" / "precommitment-versus-eventual-evidence.png").exists()
