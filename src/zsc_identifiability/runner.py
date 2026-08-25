"""Reproducible canonical-suite execution and artifact generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from zsc_identifiability.frontier import compute as compute_frontier_impl
from zsc_identifiability.metrics import compute_distributions, evaluate
from zsc_identifiability.models import FiniteConventionGame, load_game_file
from zsc_identifiability.numeric import Backend, close, serialize_number
from zsc_identifiability.oracles import action_then_commit_policy, immediate_commitment_policy
from zsc_identifiability.results import FrontierResult, RunManifest
from zsc_identifiability.solver import solve
from zsc_identifiability.theory import (
    TheoryCheck,
    multitype_pairwise_lower_bound,
    one_intervention_is_strictly_optimal,
    verify_binary_identities,
)


class SuiteSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    suite_id: str
    games: tuple[str, ...]
    action_classes: tuple[str, ...]
    backends: tuple[Backend, ...]
    parameter_sweeps: dict[str, tuple[str, ...]]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _history_dict(game: FiniteConventionGame, policy: Any) -> dict[str, Any]:
    histories = compute_distributions(game, policy, "fraction")
    return {
        "game_id": histories.game_id,
        "by_mode": {
            mode: {history: serialize_number(value) for history, value in values.items()}
            for mode, values in histories.by_mode.items()
        },
        "expected_cost_by_mode": {
            mode: serialize_number(value) for mode, value in histories.expected_cost_by_mode.items()
        },
        "decisions_by_history": histories.decisions_by_history,
        "commitment_time_by_history": histories.commitment_time_by_history,
        "posterior_by_history": {
            history: {mode: serialize_number(value) for mode, value in values.items()}
            for history, values in histories.posterior_by_history.items()
        },
    }


def _oracle_policies(game: FiniteConventionGame, backend: Backend) -> dict[str, Any]:
    result: dict[str, Any] = {
        "fixed_response": immediate_commitment_policy(game, backend),
        "passive_net": solve(game, "passive", "net_regret", backend).policy,
        "task_active_net": solve(game, "task", "net_regret", backend).policy,
        "task_information": solve(game, "task", "information", backend).policy,
        "reconnaissance": solve(game, "reconnaissance", "information", backend).policy,
    }
    for action in game.available_actions(game.initial_state, 0):
        result[f"fixed_action__{action}"] = action_then_commit_policy(game, action, backend)
    return result


def _summary_row(
    game: FiniteConventionGame,
    oracle: str,
    backend: Backend,
    policy: Any,
    evaluation: Any,
) -> dict[str, Any]:
    return {
        "game_id": game.game_id,
        "oracle": oracle,
        "backend": backend,
        "prior_risk": serialize_number(evaluation.prior_risk),
        "residual_risk": serialize_number(evaluation.residual_risk_precommitment),
        "eventual_risk": serialize_number(evaluation.residual_risk_eventual),
        "intervention_cost": serialize_number(evaluation.expected_intervention_cost),
        "net_regret": serialize_number(evaluation.net_oracle_regret),
        "dri": serialize_number(evaluation.dri_precommitment),
        "eventual_dri": serialize_number(evaluation.dri_eventual),
        "identity_mi_bits": evaluation.identity_mutual_information_bits,
        "decision_signature_mi_bits": evaluation.decision_signature_mutual_information_bits,
        "map_type_accuracy": serialize_number(evaluation.map_type_accuracy),
        "decision_accuracy": serialize_number(evaluation.decision_accuracy),
        "expected_commitment_time": serialize_number(evaluation.expected_commitment_time),
        "root_choice": (
            f"commit:{policy.decision}" if policy.kind == "commit" else f"act:{policy.action}"
        ),
    }


def _scalar_signature(evaluation: Any) -> tuple[Any, ...]:
    return (
        evaluation.prior_risk,
        evaluation.residual_risk_precommitment,
        evaluation.residual_risk_eventual,
        evaluation.expected_intervention_cost,
        evaluation.net_oracle_regret,
        evaluation.dri_precommitment,
        evaluation.dri_eventual,
        evaluation.map_type_accuracy,
        evaluation.decision_accuracy,
        evaluation.expected_commitment_time,
    )


def _assert_backend_agreement(exact: Any, approximate: Any, label: str) -> None:
    for left, right in zip(_scalar_signature(exact), _scalar_signature(approximate), strict=True):
        if left is None or right is None:
            if left is not None or right is not None:
                raise AssertionError(f"backend null mismatch for {label}")
            continue
        if not close(left, right, 1e-10):
            raise AssertionError(f"fraction/float mismatch for {label}: {left} versus {right}")
    if (
        abs(exact.identity_mutual_information_bits - approximate.identity_mutual_information_bits)
        > 1e-10
    ):
        raise AssertionError(f"identity MI backend mismatch for {label}")


def _parameter_sweep(spec: SuiteSpec) -> list[dict[str, Any]]:
    parameters = spec.parameter_sweeps
    rows = []
    for q_text in parameters["q"]:
        for cost_text in parameters["cost"]:
            for remaining_text in parameters["remaining_decisions"]:
                for mismatch_text in parameters["mismatch_loss"]:
                    for horizon_text in parameters["horizon"]:
                        for prior_text in parameters["priors"]:
                            q = Fraction(q_text)
                            cost = Fraction(cost_text)
                            remaining = int(remaining_text)
                            mismatch = Fraction(mismatch_text)
                            prior = Fraction(prior_text)
                            scale = remaining * mismatch
                            immediate = min(prior, 1 - prior) * scale
                            error = min(prior * q, (1 - prior) * (1 - q)) + min(
                                prior * (1 - q), (1 - prior) * q
                            )
                            probe_risk = error * scale
                            value = immediate - probe_risk
                            rows.append(
                                {
                                    "q": str(q),
                                    "cost": str(cost),
                                    "remaining_decisions": remaining,
                                    "mismatch_loss": str(mismatch),
                                    "horizon": int(horizon_text),
                                    "prior_mode_zero": str(prior),
                                    "immediate_risk": str(immediate),
                                    "post_probe_risk": str(probe_risk),
                                    "risk_reduction_threshold": str(value),
                                    "probe_strictly_optimal": cost < value,
                                }
                            )
    return rows


def _theory_checks(games: dict[str, FiniteConventionGame]) -> list[TheoryCheck]:
    checks: list[TheoryCheck] = []
    for game_id in ("active-separable", "fundamentally-inseparable", "passive-easy"):
        game = games[game_id]
        action = "stage_shared_item" if game_id != "passive-easy" else "advance_task"
        policy = action_then_commit_policy(game, action, "fraction")
        checks.extend(verify_binary_identities(game, policy))

    multitype = games["multitype-asymmetric-loss"]
    multi_policy = action_then_commit_policy(multitype, "share_resource", "fraction")
    multi_evaluation = evaluate(multitype, multi_policy, "fraction")
    bound = multitype_pairwise_lower_bound(multitype, multi_policy, "mode_zero", "mode_two")
    checks.append(
        TheoryCheck(
            "multitype_pairwise_lower_bound",
            multi_evaluation.residual_risk_precommitment >= bound,
            str(multi_evaluation.residual_risk_precommitment),
            str(bound),
            "The full Bayes risk is no smaller than the conflicting-pair overlap bound.",
        )
    )
    cases = ((Fraction(5), True), (Fraction(12), False), (Fraction(15), False))
    for cost, expected in cases:
        actual = one_intervention_is_strictly_optimal(cost, 4, Fraction(10), Fraction(4, 5))
        checks.append(
            TheoryCheck(
                f"one_intervention_threshold_cost_{cost}",
                actual is expected,
                str(actual).lower(),
                str(expected).lower(),
                "Strict intervention choice follows c < NM(q - 1/2).",
            )
        )

    inseparable = games["fundamentally-inseparable"]
    inseparable_policy = action_then_commit_policy(inseparable, "stage_shared_item")
    inseparable_evaluation = evaluate(inseparable, inseparable_policy, "fraction")
    inseparable_tv = next(iter(inseparable_evaluation.pairwise_total_variation.values()))
    checks.append(
        TheoryCheck(
            "observational_equivalence_impossibility",
            inseparable_tv == 0 and inseparable_evaluation.residual_risk_precommitment > 0,
            f"tv={inseparable_tv},risk={inseparable_evaluation.residual_risk_precommitment}",
            "tv=0,risk>0",
            "Conflicting modes with identical pre-commitment histories retain positive risk.",
        )
    )

    active = games["active-separable"]
    passive_information = solve(active, "passive", "information", "fraction")
    task_information = solve(active, "task", "information", "fraction")
    checks.append(
        TheoryCheck(
            "policy_class_enlargement",
            task_information.residual_decision_risk <= passive_information.residual_decision_risk,
            str(task_information.residual_decision_risk),
            str(passive_information.residual_decision_risk),
            "Expanding the action class cannot worsen optimal information risk.",
        )
    )
    fixed_evaluation = evaluate(active, immediate_commitment_policy(active), "fraction")
    informed_evaluation = evaluate(active, task_information.policy, "fraction")
    checks.append(
        TheoryCheck(
            "usable_evidence_nonincrease",
            informed_evaluation.residual_risk_precommitment
            <= fixed_evaluation.residual_risk_precommitment,
            str(informed_evaluation.residual_risk_precommitment),
            str(fixed_evaluation.residual_risk_precommitment),
            "The Bayes decision maker can ignore evidence, so usable evidence cannot add risk.",
        )
    )

    active_frontier = compute_frontier_impl(active, "task", "fraction")
    ordered = sorted(active_frontier.deterministic_points, key=lambda point: point.expected_cost)
    dri_values = [point.dri for point in ordered]
    monotone = all(
        left is not None and right is not None and left <= right
        for left, right in zip(dri_values, dri_values[1:], strict=False)
    )
    checks.append(
        TheoryCheck(
            "frontier_budget_monotonicity",
            monotone,
            ",".join(str(value) for value in dri_values),
            "non-decreasing",
            "The best attainable DRI cannot fall as the permitted cost budget grows.",
        )
    )
    dri_bounded = all(
        point.dri is None or (0 <= point.dri <= 1)
        for game in games.values()
        for point in compute_frontier_impl(game, "task", "fraction").deterministic_points
    )
    checks.append(
        TheoryCheck(
            "dri_bounds",
            dri_bounded,
            str(dri_bounded).lower(),
            "true",
            "When prior risk is positive, normalized Bayes-risk reduction lies in [0,1].",
        )
    )

    left, right = active_frontier.deterministic_points
    mixture_weight = Fraction(2, 5)
    mixture_cost = mixture_weight * left.expected_cost + (1 - mixture_weight) * right.expected_cost
    mixture_risk = mixture_weight * left.residual_risk + (1 - mixture_weight) * right.residual_risk
    checks.append(
        TheoryCheck(
            "episode_randomization_linearity",
            mixture_cost == 3 and mixture_risk == Fraction(64, 5),
            f"cost={mixture_cost},risk={mixture_risk}",
            "cost=3,risk=64/5",
            "Episode-start randomization linearly interpolates deterministic cost and risk.",
        )
    )

    boundary_solution = solve(games["threshold-boundary"], "task", "net_regret", "fraction")
    checks.append(
        TheoryCheck(
            "threshold_equality_tie_break",
            boundary_solution.policy.kind == "commit"
            and boundary_solution.total_cost_plus_risk == 20,
            f"{boundary_solution.policy.kind},{boundary_solution.total_cost_plus_risk}",
            "commit,20",
            "At exact value equality, lower cost and earlier commitment select stopping.",
        )
    )

    late = games["late-reveal"]
    late_evaluation = evaluate(late, action_then_commit_policy(late, "advance_task"), "fraction")
    checks.append(
        TheoryCheck(
            "late_evidence_exclusion",
            late_evaluation.dri_precommitment == 0 and late_evaluation.dri_eventual == 1,
            f"pre={late_evaluation.dri_precommitment},eventual={late_evaluation.dri_eventual}",
            "pre=0,eventual=1",
            "Post-commitment revelation is not credited to the earlier decision.",
        )
    )
    return checks


def _plot_threshold(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row
        for row in rows
        if row["remaining_decisions"] == 4
        and row["mismatch_loss"] == "10"
        and row["horizon"] == 1
        and row["prior_mode_zero"] == "1/2"
    ]
    qs = sorted({float(Fraction(row["q"])) for row in selected})
    costs = sorted({float(Fraction(row["cost"])) for row in selected})
    matrix = [
        [
            int(
                next(
                    row["probe_strictly_optimal"]
                    for row in selected
                    if float(Fraction(row["q"])) == q and float(Fraction(row["cost"])) == cost
                )
            )
            for q in qs
        ]
        for cost in costs
    ]

    def boundaries(values: list[float]) -> list[float]:
        midpoints = [(left + right) / 2 for left, right in zip(values, values[1:], strict=False)]
        return [
            values[0] - (midpoints[0] - values[0]),
            *midpoints,
            values[-1] + (values[-1] - midpoints[-1]),
        ]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    image = ax.pcolormesh(
        boundaries(qs),
        boundaries(costs),
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        shading="flat",
    )
    ax.set_xticks(qs, [f"{q:g}" for q in qs])
    ax.set_yticks(costs, [f"{cost:g}" for cost in costs])
    ax.set_xlabel("Diagnostic correctness q")
    ax.set_ylabel("Intervention cost c")
    ax.set_title("One-intervention optimality (N=4, M=10)")
    threshold = [40 * (q - 0.5) for q in qs]
    ax.plot(qs, threshold, color="darkorange", marker="o", label="c = NM(q-1/2)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(image, ax=ax, ticks=[0, 1], label="Probe strictly optimal")
    fig.tight_layout()
    _save_figure(fig, output, "intervention-optimality")


def _save_figure(fig: Any, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_frontier(frontiers: dict[tuple[str, str], FrontierResult], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for action_class, marker, zorder in (("task", "s", 2), ("passive", "o", 3)):
        frontier = frontiers[("active-separable", action_class)]
        points = frontier.deterministic_points
        ax.plot(
            [float(point.expected_cost) for point in points],
            [float(point.dri or 0) for point in points],
            marker=marker,
            markersize=8,
            zorder=zorder,
            label=f"{action_class} deterministic",
        )
    ax.set_xlabel("Expected intervention cost")
    ax.set_ylabel("Pre-commitment DRI")
    ax.set_title("Passive and task-active frontiers")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output, "passive-vs-active-frontier")


def _plot_identity_vs_dri(game: FiniteConventionGame, output: Path) -> None:
    policies = {
        "immediate": immediate_commitment_policy(game),
        "parity action": action_then_commit_policy(game, "yield_role"),
        "response action": action_then_commit_policy(game, "stage_shared_item"),
    }
    values = [(label, evaluate(game, policy, "fraction")) for label, policy in policies.items()]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for label, item in values:
        ax.scatter(item.identity_mutual_information_bits, float(item.dri_precommitment or 0), s=70)
        ax.annotate(
            label,
            (item.identity_mutual_information_bits, float(item.dri_precommitment or 0)),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Identity mutual information (bits)")
    ax.set_ylabel("Pre-commitment DRI")
    ax.set_title("Identity information need not be decision-useful")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 0.7)
    fig.tight_layout()
    _save_figure(fig, output, "dri-vs-identity-information")


def _plot_late_reveal(game: FiniteConventionGame, output: Path) -> None:
    item = evaluate(game, action_then_commit_policy(game, "advance_task"), "fraction")
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.bar(
        ["Before commitment", "Eventually"],
        [float(item.dri_precommitment or 0), float(item.dri_eventual or 0)],
        color=["#4c78a8", "#f58518"],
    )
    ax.set_ylabel("DRI")
    ax.set_ylim(0, 1.05)
    ax.set_title("Late evidence cannot repair an earlier decision")
    fig.tight_layout()
    _save_figure(fig, output, "precommitment-vs-eventual-dri")


def _plot_risk_tv(games: dict[str, FiniteConventionGame], output: Path) -> None:
    points: list[tuple[float, float]] = []
    for game_id in ("active-separable", "passive-easy", "fundamentally-inseparable"):
        game = games[game_id]
        for action in game.available_actions(game.initial_state, 0):
            item = evaluate(game, action_then_commit_policy(game, action), "fraction")
            tv = next(iter(item.pairwise_total_variation.values()))
            points.append((float(tv), float(item.residual_risk_precommitment)))
    grouped: dict[tuple[float, float], int] = {}
    for point in points:
        grouped[point] = grouped.get(point, 0) + 1
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs = [index / 100 for index in range(101)]
    ax.plot(xs, [20 * (1 - value) for value in xs], color="black", label="R = 20(1-TV)")
    for (tv, risk), count in grouped.items():
        ax.scatter(tv, risk, s=55 + 12 * count, zorder=3)
        label = "uninformative policies" if tv == 0 else "q=0.8 diagnostic policies"
        ax.annotate(
            f"{label} (n={count})",
            (tv, risk),
            xytext=(8, -16 if tv == 0 else 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Total variation between history distributions")
    ax.set_ylabel("Residual Bayes risk")
    ax.set_title("Binary risk–distinguishability identity")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output, "binary-risk-vs-tv")


def _plot_multitype(game: FiniteConventionGame, output: Path) -> None:
    policy = action_then_commit_policy(game, "share_resource")
    item = evaluate(game, policy, "fraction")
    bound = multitype_pairwise_lower_bound(game, policy, "mode_zero", "mode_two")
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.bar(
        ["Actual Bayes risk", "Pairwise lower bound"],
        [float(item.residual_risk_precommitment), float(bound)],
        color=["#4c78a8", "#e45756"],
    )
    ax.set_ylabel("Expected confusion loss")
    ax.set_title("Multi-type risk and valid pairwise bound")
    fig.tight_layout()
    _save_figure(fig, output, "multitype-risk-bound")


def _source_hash(project_root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [
        project_root / "pyproject.toml",
        project_root / ".python-version",
        project_root / "uv.lock",
        project_root / "README.md",
    ]
    for directory in (project_root / "src", project_root / "tests"):
        candidates.extend(path for path in directory.rglob("*") if path.is_file())
    phase_directory = project_root / "phase-2-exact-model"
    candidates.extend(
        path
        for path in phase_directory.rglob("*")
        if path.is_file() and "artifacts" not in path.relative_to(phase_directory).parts
    )
    for path in sorted(set(candidates)):
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def execute_suite(suite_path: str | Path, output_dir: str | Path) -> RunManifest:
    suite_path = Path(suite_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = SuiteSpec.model_validate_json(suite_path.read_text(encoding="utf-8"))
    games = {
        game.game_id: game
        for game in (load_game_file((suite_path.parent / item).resolve()) for item in spec.games)
    }
    exact_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    frontiers: dict[tuple[str, str], FrontierResult] = {}
    for game in games.values():
        evaluations_by_backend: dict[Backend, dict[str, Any]] = {}
        for backend in spec.backends:
            policies = _oracle_policies(game, backend)
            evaluations_by_backend[backend] = {}
            for oracle, policy in policies.items():
                item = evaluate(game, policy, backend)
                evaluations_by_backend[backend][oracle] = item
                row = _summary_row(game, oracle, backend, policy, item)
                csv_rows.append(row)
                if backend == "fraction":
                    exact_rows.append(row)
                    _write_json(
                        output / "policies" / game.game_id / f"{oracle}.json", policy.to_dict()
                    )
                    _write_json(
                        output / "histories" / game.game_id / f"{oracle}.json",
                        _history_dict(game, policy),
                    )
        for oracle, exact in evaluations_by_backend["fraction"].items():
            _assert_backend_agreement(
                exact, evaluations_by_backend["float"][oracle], f"{game.game_id}/{oracle}"
            )
        for action_class in ("passive", "task"):
            frontier = compute_frontier_impl(game, action_class, "fraction")
            frontiers[(game.game_id, action_class)] = frontier
            _write_json(
                output / "frontiers" / game.game_id / f"{action_class}.json", frontier.to_dict()
            )

    sweep_rows = _parameter_sweep(spec)
    pd.DataFrame(csv_rows).to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "summary.json",
        {
            "suite_id": spec.suite_id,
            "exact_results": exact_rows,
            "reported_scalar_oracles": {
                "known_mode_risk": "0",
                "known_response_signature_risk": "0",
                "note": (
                    "Both are zero because every validated mode has at least one "
                    "zero-loss response."
                ),
            },
            "float_agreement_tolerance": 1e-10,
        },
    )
    pd.DataFrame(sweep_rows).to_csv(output / "parameter-sweep.csv", index=False)
    _write_json(output / "parameter-sweep.json", sweep_rows)
    checks = _theory_checks(games)
    if not all(check.passed for check in checks):
        failures = [check.name for check in checks if not check.passed]
        raise AssertionError(f"theory checks failed: {failures}")
    _write_json(output / "theorem-checks.json", [check.__dict__ for check in checks])

    figures = output / "figures"
    _plot_threshold(sweep_rows, figures)
    _plot_frontier(frontiers, figures)
    _plot_identity_vs_dri(games["decision-irrelevant-identity"], figures)
    _plot_late_reveal(games["late-reveal"], figures)
    _plot_risk_tv(games, figures)
    _plot_multitype(games["multitype-asymmetric-loss"], figures)

    project_root = Path(__file__).resolve().parents[2]
    dependencies = {}
    for package in (
        "pydantic",
        "numpy",
        "pandas",
        "matplotlib",
        "pytest",
        "hypothesis",
        "ruff",
        "mypy",
    ):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    generated = tuple(
        str(path.relative_to(output))
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = RunManifest(
        schema_version=1,
        suite_id=spec.suite_id,
        configuration_hash=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        source_tree_hash=_source_hash(project_root),
        python_version=platform.python_version(),
        dependency_versions=dependencies,
        invoked_command=(
            f"python -m zsc_identifiability run-suite --suite {suite_path} --output {output}"
        ),
        generated_files=generated,
    )
    _write_json(output / "manifest.json", manifest.to_dict())
    return manifest
