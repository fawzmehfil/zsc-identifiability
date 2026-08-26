"""Materialization, training-matrix execution, and Stage 4 reporting."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from zsc_identifiability.benchmark_audit import audit_shortcuts
from zsc_identifiability.learning_env import build_observation_layout
from zsc_identifiability.learning_evaluation import (
    evaluate_neural_policy_exact,
    evaluate_neural_policy_sampled,
    evaluate_reconnaissance_policy,
)
from zsc_identifiability.learning_methods import UniformMaskedPolicy
from zsc_identifiability.learning_models import (
    GeneratedLearningPools,
    LearningAuditManifest,
    LearningAuditSuite,
    LearningCellPools,
    LearningMethodSpec,
    load_learning_suite_file,
    serialize_learning,
)
from zsc_identifiability.learning_pools import (
    audit_learning_pool_leakage,
    audit_learning_pool_matching,
    generate_evaluation_variant,
    generate_learning_pools,
    generate_symmetry_pool,
    make_smoke_pool,
)
from zsc_identifiability.learning_statistics import (
    kendall_rank_correlation,
    paired_bootstrap_interval,
    strict_ranking_reversals,
)
from zsc_identifiability.learning_trainer import load_checkpoint, train_method
from zsc_identifiability.learning_tuning import selected_method_from_report
from zsc_identifiability.numeric import parse_rational
from zsc_identifiability.population_metrics import compute as compute_population_metrics

Gate = Literal["smoke", "development", "confirmatory", "rescue"]

SMOKE_NO_IDENTIFICATION_METHODS = (
    "mlp_ppo",
    "gru_ppo_passive",
    "gru_ppo_active",
    "odits_style",
    "pace_aux",
    "pace_style",
)
SMOKE_ACTIVE_METHODS = (
    "mlp_ppo",
    "gru_ppo_active",
    "odits_style",
    "pace_aux",
    "pace_style",
    "talents_style",
)
SMOKE_ACTIVE_DIAGNOSTICS = (
    "tom_selector_style",
    "csp_style_reconnaissance",
)
SMOKE_MEMORYLESS_METHOD = "mlp_ppo"
SMOKE_RECURRENT_METHODS = (
    "gru_ppo_passive",
    "gru_ppo_active",
    "odits_style",
    "pace_aux",
    "pace_style",
    "talents_style",
    "tom_selector_style",
)
SMOKE_TOM_PASSIVE_CONTROL = ("tom_selector_style", "passive_early")


def _planned_smoke_pairs() -> tuple[tuple[str, str], ...]:
    pairs = [
        *((method, "no_identification_needed") for method in SMOKE_NO_IDENTIFICATION_METHODS),
        *((method, "active_only") for method in SMOKE_ACTIVE_METHODS),
        *((method, "active_only") for method in SMOKE_ACTIVE_DIAGNOSTICS),
        SMOKE_TOM_PASSIVE_CONTROL,
        (SMOKE_MEMORYLESS_METHOD, "remember_response"),
        *((method, "remember_response") for method in SMOKE_RECURRENT_METHODS),
    ]
    return tuple(pairs)


def materialize_learning_pools(
    pools: GeneratedLearningPools, output_dir: str | Path
) -> tuple[str, ...]:
    output = Path(output_dir).resolve()
    written: list[str] = []
    index: list[dict[str, Any]] = []
    for cell in pools.cells:
        for split_name in ("train", "validation", "test"):
            for item in getattr(cell, split_name):
                path = output / "games" / cell.cell_id / split_name / f"{item.profile_id}.json"
                _write_json(path, item.game.model_dump(mode="json"))
                relative = str(path.relative_to(output))
                written.append(relative)
                index.append(
                    {
                        "cell_id": cell.cell_id,
                        "split": split_name,
                        "profile_id": item.profile_id,
                        "source_population_id": item.source_population_id,
                        "dynamics_hash": item.dynamics_hash,
                        "game_file": relative,
                    }
                )
    _write_json(
        output / "index.json",
        {
            "schema_version": 1,
            "suite_id": pools.suite.suite_id,
            "source_suite_hash": pools.source_suite_hash,
            "learning_suite_hash": hashlib.sha256(
                json.dumps(
                    pools.suite.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "games": index,
        },
    )
    written.append("index.json")
    return tuple(sorted(written))


def execute_training_matrix(
    suite_path: str | Path,
    runs_dir: str | Path,
    *,
    method_id: str,
    gate: Gate,
    cell_id: str | None = None,
    seed_override: int | None = None,
    resume_checkpoint: str | Path | None = None,
    selection_report: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    suite = load_learning_suite_file(suite_path)
    pools = generate_learning_pools(suite, suite_path=suite_path)
    method = next((item for item in suite.methods if item.method_id == method_id), None)
    if method is None or not method.enabled:
        raise ValueError(f"unknown or disabled learning method: {method_id!r}")
    if selection_report is not None:
        method = selected_method_from_report(method, selection_report)
    cells = [item for item in pools.cells if cell_id is None or item.cell_id == cell_id]
    cells = [item for item in cells if _method_applies(suite, method, item.cell_id)]
    if not cells:
        raise ValueError("no applicable learning cells selected")
    if gate == "smoke":
        cells = [make_smoke_pool(cell) for cell in cells]
        method = method.model_copy(
            update={
                "config": method.config.model_copy(
                    update={
                        "transitions_per_update": 1_024,
                        "minibatch_size": 256,
                    }
                )
            }
        )
    if resume_checkpoint is not None and (len(cells) != 1 or seed_override is None):
        raise ValueError("resume requires exactly one --cell and an explicit --seed")
    if seed_override is not None:
        seeds: tuple[int, ...] = (seed_override,)
    elif gate == "smoke":
        seeds = (101,)
    elif gate == "development":
        seeds = suite.budget.development_seeds
    else:
        seeds = suite.budget.confirmatory_seeds
    transitions = {
        "smoke": suite.budget.smoke_transitions,
        "development": suite.budget.development_transitions,
        "confirmatory": suite.budget.confirmatory_transitions,
        "rescue": suite.budget.rescue_transitions,
    }[gate]
    if gate == "rescue":
        method = method.model_copy(
            update={
                "config": method.config.model_copy(
                    update={"hidden_size": 128, "learning_rate": 1e-4}
                )
            }
        )
    manifests = []
    for cell in cells:
        for seed in seeds:
            manifest = train_method(
                suite,
                method,
                cell,
                runs_dir,
                seed=seed,
                transitions=transitions,
                resume_checkpoint=resume_checkpoint,
            )
            payload = manifest.to_dict()
            if gate == "smoke":
                model, checkpoint = load_checkpoint(manifest.checkpoint_path)
                action_class = "passive" if method.method_id == "gru_ppo_passive" else "task"
                evaluation = evaluate_neural_policy_exact(
                    model,
                    cell.test[0],
                    method_id=method.method_id,
                    mode="greedy",
                    action_class=action_class,
                    base_team_return=float(parse_rational(suite.base_team_return)),
                    loss_scale=float(parse_rational(suite.loss_scale)),
                    identity_label_response_classes=tuple(
                        int(item) for item in checkpoint["partner_response_classes"]
                    ),
                )
                criterion, required = _smoke_expectation(cell.cell_id, method.method_id)
                passed = required is None or evaluation.team_return >= required
                payload["smoke_gate"] = {
                    "criterion": criterion,
                    "enforced_per_run": required is not None,
                    "team_return": evaluation.team_return,
                    "required_return": required,
                    "passed": passed,
                    "probe_probability": evaluation.probe_probability,
                    "policy_dri": evaluation.policy_dri,
                }
                if not passed:
                    raise RuntimeError(
                        f"smoke gate failed for {method.method_id}/{cell.cell_id}: "
                        f"return {evaluation.team_return}, required {required}"
                    )
            manifests.append(payload)
    return tuple(manifests)


def _smoke_expectation(cell_id: str, method_id: str) -> tuple[str, float | None]:
    if cell_id == "no_identification_needed":
        return "universal_response_per_method", 99.0
    if (method_id, cell_id) == SMOKE_TOM_PASSIVE_CONTROL:
        return "tom_passive_evidence_sanity", 98.0
    if cell_id == "active_only":
        if method_id in SMOKE_ACTIVE_METHODS:
            return "aggregate_active_capability", None
        return "active_identifiability_diagnostic", None
    if cell_id == "remember_response":
        if method_id == SMOKE_MEMORYLESS_METHOD:
            return "memoryless_negative_control", None
        if method_id in SMOKE_RECURRENT_METHODS:
            return "aggregate_recurrent_memory", None
    return "diagnostic_only", None


def audit_smoke_matrix(
    suite_path: str | Path,
    runs_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate completed smoke checkpoints and apply capability-level gates."""
    suite = load_learning_suite_file(suite_path)
    smoke_cells = {
        cell.cell_id: make_smoke_pool(cell)
        for cell in generate_learning_pools(suite, suite_path=suite_path).cells
    }
    evaluations: dict[str, dict[str, Any]] = {}
    run_root = Path(runs_dir).resolve()
    for manifest_path in sorted(run_root.glob("*--seed-101/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        method_id = str(manifest["method_id"])
        cell_id = str(manifest["cell_id"])
        if cell_id not in smoke_cells:
            continue
        checkpoint_path = Path(str(manifest["checkpoint_path"]))
        if not checkpoint_path.exists():
            checkpoint_path = manifest_path.parent / "best.pt"
        if not checkpoint_path.exists():
            continue
        model, checkpoint = load_checkpoint(checkpoint_path)
        action_class: Literal["passive", "task"] = (
            "passive" if method_id == "gru_ppo_passive" else "task"
        )
        evaluation = evaluate_neural_policy_exact(
            model,
            smoke_cells[cell_id].test[0],
            method_id=method_id,
            mode="greedy",
            action_class=action_class,
            base_team_return=float(parse_rational(suite.base_team_return)),
            loss_scale=float(parse_rational(suite.loss_scale)),
            identity_label_response_classes=tuple(
                int(item) for item in checkpoint["partner_response_classes"]
            ),
        )
        criterion, required = _smoke_expectation(cell_id, method_id)
        key = _smoke_key(method_id, cell_id)
        evaluations[key] = {
            "method_id": method_id,
            "cell_id": cell_id,
            "criterion": criterion,
            "required_return": required,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_hash": manifest["checkpoint_hash"],
            "source_tree_hash": manifest["source_tree_hash"],
            "evaluation": evaluation.to_dict(),
        }
    report = assess_smoke_matrix(evaluations)
    report.update(
        {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "runs_directory": str(run_root),
            "evaluations": evaluations,
        }
    )
    if output_path is not None:
        _write_json(Path(output_path).resolve(), report)
    return report


def assess_smoke_matrix(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered smoke criteria to exact checkpoint evaluations."""
    planned = tuple(_smoke_key(method, cell) for method, cell in _planned_smoke_pairs())
    missing = tuple(key for key in planned if key not in evaluations)

    def team_return(method: str, cell: str) -> float | None:
        payload = evaluations.get(_smoke_key(method, cell))
        if payload is None:
            return None
        return float(payload["evaluation"]["team_return"])

    universal_returns = {
        method: team_return(method, "no_identification_needed")
        for method in SMOKE_NO_IDENTIFICATION_METHODS
    }
    active_returns = {method: team_return(method, "active_only") for method in SMOKE_ACTIVE_METHODS}
    memoryless_return = team_return(SMOKE_MEMORYLESS_METHOD, "remember_response")
    recurrent_returns = {
        method: team_return(method, "remember_response") for method in SMOKE_RECURRENT_METHODS
    }
    tom_passive_return = team_return(*SMOKE_TOM_PASSIVE_CONTROL)
    available_active = tuple(value for value in active_returns.values() if value is not None)
    available_recurrent = tuple(value for value in recurrent_returns.values() if value is not None)
    checks = {
        "matrix_complete": not missing,
        "universal_response_control": (
            all(value is not None for value in universal_returns.values())
            and all(value >= 99.0 for value in universal_returns.values() if value is not None)
        ),
        "active_capability_anchor": bool(available_active) and max(available_active) >= 98.0,
        "tom_passive_evidence_sanity": tom_passive_return is not None
        and tom_passive_return >= 98.0,
        "recurrent_memory_capability": (
            memoryless_return is not None
            and bool(available_recurrent)
            and max(available_recurrent) >= 98.0
            and max(available_recurrent) > memoryless_return
        ),
        "tom_active_diagnostic_preserved": (
            _smoke_key("tom_selector_style", "active_only") in evaluations
        ),
    }
    required_checks = tuple(key for key in checks if key != "tom_active_diagnostic_preserved")
    return {
        "status": "incomplete" if missing else "complete",
        "passed": not missing and all(checks[key] for key in required_checks),
        "checks": checks,
        "missing_runs": list(missing),
        "planned_runs": list(planned),
        "summary": {
            "universal_response_returns": universal_returns,
            "active_capability_returns": active_returns,
            "tom_passive_return": tom_passive_return,
            "memoryless_return": memoryless_return,
            "recurrent_memory_returns": recurrent_returns,
        },
    }


def _smoke_key(method_id: str, cell_id: str) -> str:
    return f"{method_id}/{cell_id}"


def execute_rescue_matrix(
    suite_path: str | Path,
    runs_dir: str | Path,
    *,
    cell_id: str,
) -> tuple[dict[str, Any], ...]:
    """Run the prespecified larger-budget rescue for every central comparator."""
    suite = load_learning_suite_file(suite_path)
    selection_directory = Path(suite_path).resolve().parents[1] / "development"
    manifests: list[dict[str, Any]] = []
    for method in suite.methods:
        if not method.enabled or method.method_id == "csp_style_reconnaissance":
            continue
        if not _method_applies(suite, method, cell_id):
            continue
        manifests.extend(
            execute_training_matrix(
                suite_path,
                runs_dir,
                method_id=method.method_id,
                gate="rescue",
                cell_id=cell_id,
                selection_report=(
                    selection_directory / f"{method.method_id}-selection.json"
                    if (selection_directory / f"{method.method_id}-selection.json").exists()
                    else None
                ),
            )
        )
    return tuple(manifests)


def execute_symmetry_matrix(
    suite_path: str | Path,
    runs_dir: str | Path,
    *,
    method_id: str,
    selection_report: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Independently retrain one method on every selected relabeling."""
    suite = load_learning_suite_file(suite_path)
    method = next((item for item in suite.methods if item.method_id == method_id), None)
    if method is None or not method.enabled:
        raise ValueError(f"unknown or disabled learning method: {method_id!r}")
    if method.method_id == "csp_style_reconnaissance":
        return ()
    if selection_report is not None:
        method = selected_method_from_report(method, selection_report)
    manifests: list[dict[str, Any]] = []
    for symmetry in suite.symmetry_audits:
        if not _method_applies(suite, method, symmetry.cell_id):
            continue
        pool = generate_symmetry_pool(
            suite,
            symmetry.cell_id,
            symmetry.symmetry_id,
            suite_path=suite_path,
        )
        for seed in suite.budget.confirmatory_seeds:
            manifests.append(
                train_method(
                    suite,
                    method,
                    pool,
                    runs_dir,
                    seed=seed,
                    transitions=suite.budget.confirmatory_transitions,
                ).to_dict()
            )
    return tuple(manifests)


def execute_learning_audit(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    runs_dir: str | Path | None = None,
    rescue_runs_dir: str | Path | None = None,
) -> LearningAuditManifest:
    suite_path = Path(suite_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = Path(runs_dir).resolve() if runs_dir is not None else suite_path.parents[1] / "runs"
    rescue_runs = (
        Path(rescue_runs_dir).resolve()
        if rescue_runs_dir is not None
        else suite_path.parents[1] / "rescue-runs"
    )
    suite = load_learning_suite_file(suite_path)
    pools = generate_learning_pools(suite, suite_path=suite_path)
    materialize_learning_pools(pools, output / "generated")
    _write_json(output / "partner-pool-manifest.json", _pool_manifest(pools))
    leakage = audit_learning_pool_leakage(pools)
    _write_json(output / "kernel-leakage-audit.json", leakage)
    _write_json(output / "learning-matching-audit.json", audit_learning_pool_matching(pools))
    controls = _oracle_controls(pools, suite)
    _write_json(output / "oracle-controls.json", controls)
    evaluations: list[dict[str, Any]] = []
    reconnaissance_evaluations: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    symmetry_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in pools.cells:
        for method in suite.methods:
            if not method.enabled or not _method_applies(suite, method, cell.cell_id):
                continue
            for seed in suite.budget.confirmatory_seeds:
                run_id = f"{cell.cell_id}--{method.method_id}--seed-{seed}"
                checkpoint = runs / run_id / "best.pt"
                if not checkpoint.exists():
                    missing.append(run_id)
                    continue
                model, payload = load_checkpoint(checkpoint)
                action_class = "passive" if method.method_id == "gru_ppo_passive" else "task"
                if method.method_id == "csp_style_reconnaissance":
                    reconnaissance_result = evaluate_reconnaissance_policy(
                        model,
                        cell.test[0],
                        episodes=10_000,
                        seed=seed,
                        base_team_return=float(parse_rational(suite.base_team_return)),
                        loss_scale=float(parse_rational(suite.loss_scale)),
                    )
                    reconnaissance_evaluations.append(
                        {
                            "cell_id": cell.cell_id,
                            "method_id": method.method_id,
                            "seed": seed,
                            "protocol": "one_unscored_reconnaissance_episode",
                            **reconnaissance_result.to_dict(),
                        }
                    )
                    continue
                for evaluation_mode in ("greedy", "stochastic"):
                    learned_result = evaluate_neural_policy_exact(
                        model,
                        cell.test[0],
                        method_id=method.method_id,
                        mode=evaluation_mode,
                        action_class=action_class,
                        base_team_return=float(parse_rational(suite.base_team_return)),
                        loss_scale=float(parse_rational(suite.loss_scale)),
                        identity_label_response_classes=tuple(
                            int(item) for item in payload["partner_response_classes"]
                        ),
                    )
                    evaluations.append(
                        {
                            "cell_id": cell.cell_id,
                            "method_id": method.method_id,
                            "seed": seed,
                            **learned_result.to_dict(),
                        }
                    )
                    if (
                        evaluation_mode == "stochastic"
                        and method.method_id == "gru_ppo_active"
                        and seed == suite.budget.confirmatory_seeds[0]
                        and cell.cell_id in {"passive_early", "active_only"}
                    ):
                        sampled = evaluate_neural_policy_sampled(
                            model,
                            cell.test[0],
                            method_id=method.method_id,
                            mode="stochastic",
                            action_class=action_class,
                            episodes=100_000,
                            seed=seed,
                            base_team_return=float(parse_rational(suite.base_team_return)),
                            loss_scale=float(parse_rational(suite.loss_scale)),
                            identity_label_response_classes=tuple(
                                int(item) for item in payload["partner_response_classes"]
                            ),
                        )
                        calibration_rows.append(
                            _calibration_record(
                                cell.cell_id,
                                method.method_id,
                                seed,
                                learned_result.to_dict(),
                                sampled.to_dict(),
                                episodes=100_000,
                                loss_scale=float(parse_rational(suite.loss_scale)),
                            )
                        )
    for symmetry in suite.symmetry_audits:
        symmetry_pool = generate_symmetry_pool(
            suite,
            symmetry.cell_id,
            symmetry.symmetry_id,
            suite_path=suite_path,
        )
        for method in suite.methods:
            if (
                not method.enabled
                or method.method_id == "csp_style_reconnaissance"
                or not _method_applies(suite, method, symmetry.cell_id)
            ):
                continue
            for seed in suite.budget.confirmatory_seeds:
                run_id = f"{symmetry_pool.cell_id}--{method.method_id}--seed-{seed}"
                checkpoint = runs / run_id / "best.pt"
                if not checkpoint.exists():
                    missing.append(f"symmetry:{run_id}")
                    continue
                model, payload = load_checkpoint(checkpoint)
                action_class = "passive" if method.method_id == "gru_ppo_passive" else "task"
                learned_result = evaluate_neural_policy_exact(
                    model,
                    symmetry_pool.test[0],
                    method_id=method.method_id,
                    mode="greedy",
                    action_class=action_class,
                    base_team_return=float(parse_rational(suite.base_team_return)),
                    loss_scale=float(parse_rational(suite.loss_scale)),
                    identity_label_response_classes=tuple(
                        int(item) for item in payload["partner_response_classes"]
                    ),
                )
                symmetry_rows.append(
                    {
                        "cell_id": symmetry.cell_id,
                        "symmetry_id": symmetry.symmetry_id,
                        "method_id": method.method_id,
                        "seed": seed,
                        **learned_result.to_dict(),
                    }
                )
    sweep_variants = {
        "reliability": (
            "sweep_reliability_1_over_2",
            "sweep_reliability_3_over_5",
            "sweep_reliability_4_over_5",
            "sweep_reliability_1",
        ),
        "cost": (
            "sweep_cost_0",
            "sweep_cost_5",
            "sweep_cost_12",
            "sweep_cost_15",
        ),
    }
    sweep_games = {
        variant_id: generate_evaluation_variant(
            suite,
            "active_only",
            variant_id,
            suite_path=suite_path,
        )
        for variants in sweep_variants.values()
        for variant_id in variants
    }
    for method in suite.methods:
        if not method.enabled or method.method_id == "csp_style_reconnaissance":
            continue
        for seed in suite.budget.confirmatory_seeds:
            run_id = f"active_only--{method.method_id}--seed-{seed}"
            checkpoint = runs / run_id / "best.pt"
            if not checkpoint.exists():
                continue
            model, payload = load_checkpoint(checkpoint)
            action_class = "passive" if method.method_id == "gru_ppo_passive" else "task"
            for sweep_name, variants in sweep_variants.items():
                for variant_id in variants:
                    variant = sweep_games[variant_id]
                    result = evaluate_neural_policy_exact(
                        model,
                        variant,
                        method_id=method.method_id,
                        mode="greedy",
                        action_class=action_class,
                        base_team_return=float(parse_rational(suite.base_team_return)),
                        loss_scale=float(parse_rational(suite.loss_scale)),
                        identity_label_response_classes=tuple(
                            int(item) for item in payload["partner_response_classes"]
                        ),
                    )
                    sweep_rows.append(
                        {
                            "method_id": method.method_id,
                            "seed": seed,
                            "sweep": sweep_name,
                            "variant_id": variant_id,
                            **result.to_dict(),
                        }
                    )
    _write_json(output / "learned-policy-evaluations.json", evaluations)
    _write_json(
        output / "reconnaissance-protocol-evaluations.json",
        reconnaissance_evaluations,
    )
    _write_json(output / "frozen-policy-sweep-evaluations.json", sweep_rows)
    selection_files = sorted((output.parent / "development").glob("*-selection.json"))
    selections = [json.loads(path.read_text(encoding="utf-8")) for path in selection_files]
    _write_json(output / "hyperparameter-selection-table.json", selections)
    if selections:
        pd.DataFrame(
            [
                {
                    "method_id": row["method_id"],
                    **row["selected"],
                }
                for row in selections
            ]
        ).to_csv(output / "hyperparameter-selection-table.csv", index=False)
    _write_json(output / "exact-versus-monte-carlo-calibration.json", calibration_rows)
    _write_json(
        output / "symmetry-audit.json",
        _symmetry_report(evaluations, symmetry_rows, suite),
    )
    if evaluations:
        pd.DataFrame(evaluations).to_csv(output / "learned-policy-evaluations.csv", index=False)
    status: Literal["complete", "incomplete", "invalid"] = "incomplete" if missing else "complete"
    verdict: Literal[
        "continue_to_repair", "continue_without_repair", "redesign", "stop", "pending"
    ] = "pending"
    if not missing:
        confirmatory = [item for item in evaluations if item["mode"] == "greedy"]
        statistics = _statistical_reports(confirmatory, suite)
        _write_json(output / "ranking-reversal-report.json", statistics)
        checks, verdict = _scientific_verdict(confirmatory, controls)
        if verdict == "continue_to_repair":
            rescue_evaluations, rescue_missing = _evaluate_rescue_runs(
                suite,
                pools.by_cell()["active_only"],
                rescue_runs,
            )
            _write_json(
                output / "optimization-rescue-evaluations.json",
                {
                    "cell_id": "active_only",
                    "prespecified_hidden_size": 128,
                    "prespecified_learning_rate": 1e-4,
                    "prespecified_transitions": suite.budget.rescue_transitions,
                    "evaluations": rescue_evaluations,
                    "missing_runs": rescue_missing,
                },
            )
            if rescue_missing:
                missing.extend(f"rescue:{item}" for item in rescue_missing)
                status = "incomplete"
                verdict = "pending"
                checks["active_gap_survives_rescue"] = False
            else:
                rescue_best = max(float(item["team_return"]) for item in rescue_evaluations)
                active_oracle = _as_float(controls["active_only"]["task_active_oracle_return"])
                checks["active_gap_survives_rescue"] = active_oracle - rescue_best > 1
                verdict = (
                    "continue_to_repair"
                    if checks["active_gap_survives_rescue"]
                    else "continue_without_repair"
                )
        _write_json(output / "scientific-checks.json", checks)
        _plot_results(output / "figures", confirmatory, controls, runs, sweep_rows)
    project_root = Path(__file__).resolve().parents[2]
    generated_files = tuple(
        str(path.relative_to(output))
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = LearningAuditManifest(
        schema_version=1,
        suite_id=suite.suite_id,
        status=status,
        scientific_verdict=verdict,
        implementation_passed=True,
        configuration_hash=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        source_tree_hash=_source_hash(project_root),
        generated_files=generated_files,
        missing_runs=tuple(sorted(missing)),
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(),
        invoked_command=tuple(sys.argv),
        rng_configuration={
            "environment_bit_generator": "NumPy PCG64",
            "bootstrap_seed": suite.statistics.bootstrap_seed,
            "development_seeds": suite.budget.development_seeds,
            "confirmatory_seeds": suite.budget.confirmatory_seeds,
            "torch_deterministic_algorithms": True,
        },
    )
    _write_json(output / "manifest.json", manifest.to_dict())
    return manifest


def _evaluate_rescue_runs(
    suite: LearningAuditSuite,
    cell: LearningCellPools,
    runs_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    evaluations: list[dict[str, Any]] = []
    missing: list[str] = []
    for method in suite.methods:
        if (
            not method.enabled
            or method.method_id == "csp_style_reconnaissance"
            or not _method_applies(suite, method, cell.cell_id)
        ):
            continue
        for seed in suite.budget.confirmatory_seeds:
            run_id = f"{cell.cell_id}--{method.method_id}--seed-{seed}"
            checkpoint = runs_dir / run_id / "best.pt"
            if not checkpoint.exists():
                missing.append(run_id)
                continue
            model, payload = load_checkpoint(checkpoint)
            action_class = "passive" if method.method_id == "gru_ppo_passive" else "task"
            result = evaluate_neural_policy_exact(
                model,
                cell.test[0],
                method_id=method.method_id,
                mode="greedy",
                action_class=action_class,
                base_team_return=float(parse_rational(suite.base_team_return)),
                loss_scale=float(parse_rational(suite.loss_scale)),
                identity_label_response_classes=tuple(
                    int(item) for item in payload["partner_response_classes"]
                ),
            )
            evaluations.append(
                {
                    "cell_id": cell.cell_id,
                    "method_id": method.method_id,
                    "seed": seed,
                    **result.to_dict(),
                }
            )
    return evaluations, missing


def _method_applies(suite: LearningAuditSuite, method: LearningMethodSpec, cell_id: str) -> bool:
    if method.method_id in {"talents_style", "tom_selector_style"}:
        return cell_id in suite.specialist_cells
    if method.method_id == "csp_style_reconnaissance":
        return cell_id in suite.reconnaissance_cells
    return cell_id in suite.cells


def _pool_manifest(pools: GeneratedLearningPools) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": pools.suite.suite_id,
        "source_suite_hash": pools.source_suite_hash,
        "cells": {
            cell.cell_id: {
                split: [
                    {
                        "profile_id": item.profile_id,
                        "dynamics_hash": item.dynamics_hash,
                        "source_population_id": item.source_population_id,
                    }
                    for item in getattr(cell, split)
                ]
                for split in ("train", "validation", "test")
            }
            for cell in pools.cells
        },
    }


def _oracle_controls(pools: GeneratedLearningPools, suite: LearningAuditSuite) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cell in pools.cells:
        population = cell.source_population
        metrics = compute_population_metrics(population, "fraction")
        shortcuts = audit_shortcuts(population, "fraction")
        layout = build_observation_layout(cell.test[0].game)
        random_policy = UniformMaskedPolicy(layout.action_size)
        random_result = evaluate_neural_policy_exact(
            random_policy,
            cell.test[0],
            method_id="random_intervention",
            mode="stochastic",
            action_class="task",
            base_team_return=float(parse_rational(suite.base_team_return)),
            loss_scale=float(parse_rational(suite.loss_scale)),
            device=torch.device("cpu"),
        )
        result[cell.cell_id] = {
            "population_id": population.descriptor.population_id,
            "known_mode_return": metrics.values["known_mode_return_mean"],
            "task_active_oracle_return": metrics.values["task_active_oracle_return"],
            "passive_oracle_return": metrics.values["passive_oracle_return"],
            "information_only_return": metrics.values["information_only_return"],
            "best_fixed_response_return": metrics.values["best_fixed_response_value"],
            "passive_dri": metrics.values["passive_dri"],
            "active_dri": metrics.values["active_dri"],
            "eventual_dri": metrics.values["eventual_dri"],
            "evidence_blind_risk": shortcuts.evidence_blind_risk,
            "memoryless_risk": shortcuts.memoryless_risk,
            "history_aware_risk": shortcuts.history_aware_risk,
            "random_intervention": random_result.to_dict(),
            "active_frontier": metrics.active_frontier,
        }
    return cast(dict[str, Any], serialize_learning(result))


def _statistical_reports(rows: list[dict[str, Any]], suite: LearningAuditSuite) -> dict[str, Any]:
    pairs = (
        ("passive_early", "active_only"),
        ("active_only", "precommit_inseparable"),
        ("remember_response", "remember_subtype"),
        ("active_response", "active_identity_only"),
    )
    reversals = {
        f"{left}__vs__{right}": strict_ranking_reversals(
            rows,
            left,
            right,
            resamples=suite.statistics.bootstrap_resamples,
            confidence_level=suite.statistics.confidence_level,
            seed=suite.statistics.bootstrap_seed,
        )
        for left, right in pairs
    }
    cell_method_values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        cell_method_values.setdefault(str(row["cell_id"]), {}).setdefault(
            str(row["method_id"]), []
        ).append(float(row["team_return"]))
    means = {
        cell: {method: float(np.mean(values)) for method, values in methods.items()}
        for cell, methods in cell_method_values.items()
    }
    rank_matrix = {
        cell: {
            str(method): float(rank)
            for method, rank in pd.Series(methods, dtype="float64")
            .rank(method="average", ascending=False)
            .items()
        }
        for cell, methods in means.items()
    }
    kendall = {
        f"{left}__vs__{right}": kendall_rank_correlation(means[left], means[right])
        for left, right in pairs
        if left in means and right in means and len(set(means[left]) & set(means[right])) >= 2
    }
    return {
        "ranking_reversals": reversals,
        "strict_ranking_reversal_count": sum(
            int(report["strict_reversal_count"]) for report in reversals.values()
        ),
        "mean_return_by_cell_and_method": means,
        "rank_matrix": rank_matrix,
        "rank_tie_policy": "average rank for exactly equal mean returns",
        "kendall_rank_correlations": kendall,
    }


def _calibration_record(
    cell_id: str,
    method_id: str,
    seed: int,
    exact: dict[str, Any],
    sampled: dict[str, Any],
    *,
    episodes: int,
    loss_scale: float,
) -> dict[str, Any]:
    """Distribution-free 95% calibration interval for bounded episode returns."""
    half_width = loss_scale * np.sqrt(np.log(40.0) / (2 * episodes))
    sampled_return = float(sampled["team_return"])
    exact_return = float(exact["team_return"])
    return {
        "cell_id": cell_id,
        "method_id": method_id,
        "seed": seed,
        "episodes": episodes,
        "interval": "two-sided Hoeffding 95% interval",
        "exact_team_return": exact_return,
        "sampled_team_return": sampled_return,
        "lower": sampled_return - half_width,
        "upper": sampled_return + half_width,
        "absolute_error": abs(exact_return - sampled_return),
        "passed": abs(exact_return - sampled_return) <= half_width,
    }


def _symmetry_report(
    canonical_rows: list[dict[str, Any]],
    symmetry_rows: list[dict[str, Any]],
    suite: LearningAuditSuite,
) -> dict[str, Any]:
    canonical = {
        (str(row["cell_id"]), str(row["method_id"]), int(row["seed"])): float(row["team_return"])
        for row in canonical_rows
        if row["mode"] == "greedy"
    }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in symmetry_rows:
        groups.setdefault(
            (str(row["cell_id"]), str(row["symmetry_id"]), str(row["method_id"])), []
        ).append(row)
    comparisons: list[dict[str, Any]] = []
    for (cell_id, symmetry_id, method_id), rows in sorted(groups.items()):
        seeds = sorted(
            int(row["seed"]) for row in rows if (cell_id, method_id, int(row["seed"])) in canonical
        )
        if len(seeds) < 2:
            continue
        variant_by_seed = {int(row["seed"]): float(row["team_return"]) for row in rows}
        interval = paired_bootstrap_interval(
            np.asarray([variant_by_seed[seed] for seed in seeds]),
            np.asarray([canonical[(cell_id, method_id, seed)] for seed in seeds]),
            resamples=suite.statistics.bootstrap_resamples,
            confidence_level=suite.statistics.confidence_level,
            seed=suite.statistics.bootstrap_seed,
        )
        comparisons.append(
            {
                "cell_id": cell_id,
                "symmetry_id": symmetry_id,
                "method_id": method_id,
                "paired_seed_count": len(seeds),
                "difference": interval.to_dict(),
                "equivalence_margin": 1.0,
                "passed": interval.lower >= -1 and interval.upper <= 1,
            }
        )
    return {
        "protocol": "independent label-mapped retraining; no mixed codebooks",
        "raw_evaluations": symmetry_rows,
        "comparisons": comparisons,
        "passed": bool(comparisons) and all(row["passed"] for row in comparisons),
    }


def _scientific_verdict(
    rows: list[dict[str, Any]], controls: dict[str, Any]
) -> tuple[dict[str, bool], Literal["continue_to_repair", "continue_without_repair", "redesign"]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["cell_id"]), str(row["method_id"])), []).append(
            float(row["team_return"])
        )
    no_id_best = max(
        (
            np.mean(values)
            for (cell, _), values in grouped.items()
            if cell == "no_identification_needed"
        ),
        default=-np.inf,
    )
    passive_best = max(
        (np.mean(values) for (cell, _), values in grouped.items() if cell == "passive_early"),
        default=-np.inf,
    )
    memoryless = np.mean(grouped.get(("remember_response", "mlp_ppo"), [-np.inf]))
    recurrent_best = max(
        (
            np.mean(values)
            for (cell, method), values in grouped.items()
            if cell == "remember_response" and method != "mlp_ppo"
        ),
        default=-np.inf,
    )
    active_best = max(
        (np.mean(values) for (cell, _), values in grouped.items() if cell == "active_only"),
        default=-np.inf,
    )
    active_oracle = _as_float(controls["active_only"]["task_active_oracle_return"])
    inseparable_best = max(
        (
            np.mean(values)
            for (cell, _), values in grouped.items()
            if cell == "precommit_inseparable"
        ),
        default=np.inf,
    )
    fundamental_best = max(
        (
            np.mean(values)
            for (cell, _), values in grouped.items()
            if cell == "fundamentally_inseparable"
        ),
        default=np.inf,
    )
    subtype_dri = [
        abs(float(row["policy_dri"]))
        for row in rows
        if row["cell_id"] in {"remember_subtype", "active_identity_only"}
        and row["policy_dri"] is not None
    ]
    fixed_return = _as_float(controls["active_only"]["best_fixed_response_return"])
    checks = {
        "no_identification_sanity": bool(no_id_best >= 99),
        "passive_sanity": bool(passive_best >= 90.8),
        "memory_capability": bool(recurrent_best > memoryless and recurrent_best >= 90.8),
        "active_gap_exceeds_one_point": bool(active_oracle - active_best > 1),
        "precommit_inseparable_has_no_leakage": bool(inseparable_best <= fixed_return + 1),
        "fundamentally_inseparable_has_no_leakage": bool(fundamental_best <= fixed_return + 1),
        "identity_only_evidence_receives_zero_dri": bool(subtype_dri and max(subtype_dri) <= 1e-10),
    }
    if not all(
        (
            checks["no_identification_sanity"],
            checks["passive_sanity"],
            checks["memory_capability"],
            checks["precommit_inseparable_has_no_leakage"],
            checks["fundamentally_inseparable_has_no_leakage"],
            checks["identity_only_evidence_receives_zero_dri"],
        )
    ):
        verdict: Literal["continue_to_repair", "continue_without_repair", "redesign"] = "redesign"
    elif checks["active_gap_exceeds_one_point"]:
        verdict = "continue_to_repair"
    else:
        verdict = "continue_without_repair"
    return checks, verdict


def _plot_results(
    directory: Path,
    rows: list[dict[str, Any]],
    controls: dict[str, Any],
    runs_dir: Path,
    sweep_rows: list[dict[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    means = frame.groupby(["cell_id", "method_id"], as_index=False).mean(numeric_only=True)
    pivot = means.pivot(index="method_id", columns="cell_id", values="team_return")
    figure, axis = plt.subplots(figsize=(12, 5))
    image = axis.imshow(pivot.rank(axis=0, ascending=False), aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Learned-method rank by identifiability cell")
    figure.colorbar(image, ax=axis, label="Rank (1 is best)")
    figure.tight_layout()
    _save_figure(figure, directory / "algorithm-rank-heatmap")

    figure, axis = plt.subplots(figsize=(9, 5))
    for method, group in means.groupby("method_id"):
        axis.scatter(group["policy_dri"], group["team_return"], label=method, s=28)
    axis.set_xlabel("Learned policy DRI")
    axis.set_ylabel("Team return")
    axis.set_title("Return versus decision-relevant identifiability")
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    _save_figure(figure, directory / "return-versus-policy-dri")

    active = means[
        means["cell_id"].isin(["active_only", "active_boundary", "active_too_expensive"])
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    for method, group in active.groupby("method_id"):
        axis.plot(
            group["expected_intervention_cost"], group["probe_probability"], "o-", label=method
        )
    axis.set_xlabel("Expected intervention cost")
    axis.set_ylabel("Probe probability")
    axis.set_title("Cost sensitivity of learned intervention")
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    _save_figure(figure, directory / "probe-cost-sensitivity")

    if sweep_rows:
        sweep_frame = pd.DataFrame(sweep_rows)
        sweep_means = sweep_frame.groupby(["sweep", "variant_id", "method_id"], as_index=False)[
            "probe_probability"
        ].mean()
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        for axis, sweep_name in zip(axes, ("reliability", "cost"), strict=True):
            subset = sweep_means[sweep_means["sweep"] == sweep_name]
            for method, group in subset.groupby("method_id"):
                group = group.sort_values("variant_id")
                axis.plot(group["variant_id"], group["probe_probability"], "o-", label=method)
            axis.set_title(f"Frozen-policy probe rate across {sweep_name}")
            axis.set_ylabel("Probe probability")
            axis.tick_params(axis="x", rotation=35)
        axes[1].legend(fontsize=6, ncol=2)
        figure.tight_layout()
        _save_figure(figure, directory / "probe-probability-reliability-and-cost")

    diagnostic_cells = [
        "passive_early",
        "active_only",
        "precommit_inseparable",
        "remember_response",
        "remember_subtype",
        "active_response",
        "active_identity_only",
    ]
    selected = means[means["cell_id"].isin(diagnostic_cells)]
    figure, axis = plt.subplots(figsize=(12, 5))
    for method, group in selected.groupby("method_id"):
        ordered = group.set_index("cell_id").reindex(diagnostic_cells)
        axis.plot(diagnostic_cells, ordered["team_return"], "o-", label=method)
    axis.set_ylabel("Exact greedy team return")
    axis.set_title("Learned return across matched identifiability strata")
    axis.tick_params(axis="x", rotation=40)
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    _save_figure(figure, directory / "learned-return-matched-dri")

    figure, axis = plt.subplots(figsize=(10, 5))
    cell_means = selected.groupby("cell_id", as_index=False)["team_return"].mean()
    x = np.arange(len(cell_means))
    learned = cell_means["team_return"].to_numpy()
    passive = np.asarray(
        [_as_float(controls[cell]["passive_oracle_return"]) for cell in cell_means["cell_id"]]
    )
    active_oracle = np.asarray(
        [_as_float(controls[cell]["task_active_oracle_return"]) for cell in cell_means["cell_id"]]
    )
    axis.bar(x - 0.25, learned, width=0.25, label="learned mean")
    axis.bar(x, passive, width=0.25, label="passive oracle")
    axis.bar(x + 0.25, active_oracle, width=0.25, label="active oracle")
    axis.set_xticks(x, cell_means["cell_id"], rotation=40, ha="right")
    axis.set_ylabel("Team return")
    axis.set_title("Learned, passive-oracle, and active-oracle performance")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, directory / "passive-active-oracle-gaps")

    decomposition = means.groupby("method_id", as_index=False)[
        ["expected_intervention_cost", "residual_bayes_risk", "decision_utilization_gap"]
    ].mean()
    figure, axis = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(decomposition))
    for field, label in (
        ("expected_intervention_cost", "intervention cost"),
        ("residual_bayes_risk", "residual risk"),
        ("decision_utilization_gap", "utilization gap"),
    ):
        values = decomposition[field].to_numpy()
        axis.bar(decomposition["method_id"], values, bottom=bottom, label=label)
        bottom += values
    axis.tick_params(axis="x", rotation=40)
    axis.set_ylabel("Regret contribution")
    axis.set_title("Learned regret decomposition")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, directory / "regret-decomposition")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(means["identity_mutual_information_bits"], means["policy_dri"], s=24)
    axis.set_xlabel("Identity mutual information (bits)")
    axis.set_ylabel("Policy DRI")
    axis.set_title("Identity information is not necessarily decision-relevant")
    figure.tight_layout()
    _save_figure(figure, directory / "identity-information-versus-dri")

    response_subtype = means[means["cell_id"].isin(["remember_response", "remember_subtype"])]
    figure, axis = plt.subplots(figsize=(9, 5))
    for method, group in response_subtype.groupby("method_id"):
        axis.plot(group["cell_id"], group["team_return"], "o-", label=method)
    axis.set_ylabel("Team return")
    axis.set_title("Response-relevant versus subtype-only evidence")
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    _save_figure(figure, directory / "response-versus-subtype-evidence")

    memory_methods = means[
        means["cell_id"].isin(["remember_response", "remember_subtype"])
        & means["method_id"].isin(["mlp_ppo", "gru_ppo_active", "gru_ppo_passive"])
    ]
    memory_pivot = memory_methods.pivot(index="cell_id", columns="method_id", values="team_return")
    figure, axis = plt.subplots(figsize=(8, 5))
    memory_pivot.plot.bar(ax=axis)
    axis.set_ylabel("Team return")
    axis.set_title("Memoryless versus recurrent policies")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    _save_figure(figure, directory / "memoryless-versus-recurrent")

    timing_cells = ["passive_early", "active_only", "precommit_inseparable"]
    figure, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(len(timing_cells))
    pre = np.asarray([_as_float(controls[cell]["active_dri"] or 0) for cell in timing_cells])
    eventual = np.asarray([_as_float(controls[cell]["eventual_dri"] or 0) for cell in timing_cells])
    axis.bar(x - 0.18, pre, 0.36, label="pre-commitment active DRI")
    axis.bar(x + 0.18, eventual, 0.36, label="eventual DRI")
    axis.set_xticks(x, timing_cells, rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_title("Pre-commitment versus eventual evidence")
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, directory / "precommitment-versus-eventual-evidence")

    frontier_points = controls["active_only"]["active_frontier"]["deterministic_points"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [_as_float(point["expected_cost"]) for point in frontier_points],
        [_as_float(point["residual_risk"]) for point in frontier_points],
        "k-o",
        label="exact active frontier",
    )
    active_learned = means[means["cell_id"] == "active_only"]
    for _, row in active_learned.iterrows():
        axis.scatter(
            row["expected_intervention_cost"],
            row["residual_bayes_risk"],
            label=row["method_id"],
            s=35,
        )
    axis.set_xlabel("Expected intervention cost")
    axis.set_ylabel("Residual Bayes risk")
    axis.set_title("Learned policies against the active-identifiability frontier")
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    _save_figure(figure, directory / "learned-policies-active-frontier")

    _plot_learning_curves(directory, runs_dir)

    _write_json(directory.parent / "control-summary.json", controls)


def _plot_learning_curves(directory: Path, runs_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    curve_count = 0
    for path in sorted(runs_dir.glob("*/training-metrics.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        points = [row for row in rows if "validation_team_return" in row]
        if not points:
            continue
        label = path.parent.name.rsplit("--seed-", 1)[0]
        axis.plot(
            [row["transitions"] for row in points],
            [row["validation_team_return"] for row in points],
            alpha=0.25,
            label=label if curve_count < 12 else None,
        )
        curve_count += 1
    axis.set_xlabel("Training transitions")
    axis.set_ylabel("Validation team return")
    axis.set_title("Validation learning curves across seeds")
    if curve_count:
        axis.legend(fontsize=5, ncol=2)
    else:
        axis.text(0.5, 0.5, "No training curves found", ha="center", va="center")
    figure.tight_layout()
    _save_figure(figure, directory / "learning-curves")


def _save_figure(figure: Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _as_float(value: Any) -> float:
    return float(parse_rational(value)) if isinstance(value, str) else float(value)


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "numpy", "pydantic", "pandas", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_learning(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
