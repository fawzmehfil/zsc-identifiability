"""Validation-only global hyperparameter selection for Stage 4 methods."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from zsc_identifiability.learning_evaluation import evaluate_neural_policy_exact
from zsc_identifiability.learning_models import (
    LearningAuditSuite,
    LearningMethodSpec,
    load_learning_suite_file,
    serialize_learning,
)
from zsc_identifiability.learning_pools import generate_learning_pools
from zsc_identifiability.learning_trainer import load_checkpoint, train_method
from zsc_identifiability.numeric import parse_rational
from zsc_identifiability.solver import solve

SELECTION_CELLS = (
    "passive_early",
    "active_only",
    "remember_response",
    "remember_subtype",
)


def candidate_method_configs(
    method: LearningMethodSpec,
) -> tuple[tuple[str, LearningMethodSpec], ...]:
    """Return the preregistered grid, changing only parameters used by a method."""
    common: dict[str, tuple[float | int, ...]] = {
        "learning_rate": (1e-4, 3e-4),
        "entropy_coefficient": (0.001, 0.01),
    }
    if method.method_id in {"pace_aux", "pace_style", "csp_style_reconnaissance"}:
        common["auxiliary_coefficient"] = (0.1, 1.0)
    if method.method_id in {"pace_style", "csp_style_reconnaissance"}:
        common["pace_bonus_initial"] = (0.1, 0.5)
    if method.method_id in {"odits_style", "talents_style"}:
        common["latent_dimension"] = (4, 8)
    if method.method_id == "odits_style":
        common["kl_coefficient"] = (0.1, 1.0)
    keys = tuple(common)
    candidates: list[tuple[str, LearningMethodSpec]] = []
    for values in itertools.product(*(common[key] for key in keys)):
        updates = dict(zip(keys, values, strict=True))
        config = method.config.model_copy(update=updates)
        candidate = method.model_copy(update={"config": config})
        encoded = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
        identifier = hashlib.sha256(encoded).hexdigest()[:12]
        candidates.append((identifier, candidate))
    return tuple(candidates)


def run_development_search(
    suite_path: str | Path,
    output_dir: str | Path,
    *,
    method_id: str,
) -> dict[str, Any]:
    """Select one configuration per method across all declared validation tasks."""
    suite = load_learning_suite_file(suite_path)
    pools = generate_learning_pools(suite, suite_path=suite_path)
    base_method = next(
        (method for method in suite.methods if method.method_id == method_id and method.enabled),
        None,
    )
    if base_method is None:
        raise ValueError(f"unknown or disabled learning method: {method_id!r}")
    applicable = [
        pools.by_cell()[cell_id]
        for cell_id in SELECTION_CELLS
        if _applies(suite, base_method, cell_id)
    ]
    if not applicable:
        raise ValueError(f"method {method_id!r} has no development-selection cells")
    root = Path(output_dir).resolve()
    rows: list[dict[str, Any]] = []
    for config_id, candidate in candidate_method_configs(base_method):
        for cell in applicable:
            for seed in suite.budget.development_seeds:
                manifest = train_method(
                    suite,
                    candidate,
                    cell,
                    root / "runs" / config_id,
                    seed=seed,
                    transitions=suite.budget.development_transitions,
                )
                model, payload = load_checkpoint(manifest.checkpoint_path)
                action_class = "passive" if method_id == "gru_ppo_passive" else "task"
                evaluation = evaluate_neural_policy_exact(
                    model,
                    cell.validation[0],
                    method_id=method_id,
                    mode="greedy",
                    action_class=action_class,
                    base_team_return=float(parse_rational(suite.base_team_return)),
                    loss_scale=float(parse_rational(suite.loss_scale)),
                    identity_label_response_classes=tuple(
                        int(item) for item in payload["partner_response_classes"]
                    ),
                )
                oracle = solve(
                    cell.validation[0].game,
                    action_class,  # type: ignore[arg-type]
                    "net_regret",
                    "float",
                    commitment_states=cell.validation[0].commitment_states,
                )
                oracle_return = float(parse_rational(suite.base_team_return)) - float(
                    oracle.total_cost_plus_risk
                )
                rows.append(
                    {
                        "config_id": config_id,
                        "configuration": candidate.config.model_dump(mode="json"),
                        "cell_id": cell.cell_id,
                        "seed": seed,
                        "oracle_normalized_return": evaluation.oracle_normalized_return,
                        "validation_oracle_ratio": evaluation.team_return / oracle_return,
                        "validation_oracle_return": oracle_return,
                        "team_return": evaluation.team_return,
                        "intervention_cost": evaluation.expected_intervention_cost,
                        "parameter_count": int(payload["model_metadata"]["parameter_count"]),
                        "checkpoint": manifest.checkpoint_path,
                    }
                )
    aggregate: list[dict[str, Any]] = []
    for config_id, candidate in candidate_method_configs(base_method):
        selected = [row for row in rows if row["config_id"] == config_id]
        normalized = [float(row["validation_oracle_ratio"]) for row in selected]
        aggregate.append(
            {
                "config_id": config_id,
                "configuration": candidate.config.model_dump(mode="json"),
                "mean_oracle_normalized_return": float(np.mean(normalized)),
                "mean_intervention_cost": float(
                    np.mean([float(row["intervention_cost"]) for row in selected])
                ),
                "parameter_count": int(selected[0]["parameter_count"]),
                "evaluation_count": len(selected),
            }
        )
    winner = sorted(
        aggregate,
        key=lambda row: (
            -float(row["mean_oracle_normalized_return"]),
            float(row["mean_intervention_cost"]),
            int(row["parameter_count"]),
            str(row["config_id"]),
        ),
    )[0]
    report = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "method_id": method_id,
        "selection_cells": [cell.cell_id for cell in applicable],
        "development_seeds": list(suite.budget.development_seeds),
        "test_data_used": False,
        "selection_rule": (
            "mean oracle-normalized validation return; then lower intervention cost; "
            "then fewer parameters; then lexicographic config ID"
        ),
        "selected": winner,
        "candidates": aggregate,
        "per_seed": rows,
    }
    _write_json(root / f"{method_id}-selection.json", report)
    return report


def selected_method_from_report(
    method: LearningMethodSpec, report_path: str | Path
) -> LearningMethodSpec:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("method_id") != method.method_id:
        raise ValueError("hyperparameter report method does not match requested method")
    if report.get("test_data_used") is not False:
        raise ValueError("hyperparameter report must certify that test data was unused")
    return method.model_copy(
        update={"config": method.config.model_validate(report["selected"]["configuration"])}
    )


def _applies(suite: LearningAuditSuite, method: LearningMethodSpec, cell_id: str) -> bool:
    if method.method_id in {"talents_style", "tom_selector_style"}:
        return cell_id in suite.specialist_cells
    if method.method_id == "csp_style_reconnaissance":
        return cell_id in suite.reconnaissance_cells
    return cell_id in suite.cells


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serialize_learning(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
