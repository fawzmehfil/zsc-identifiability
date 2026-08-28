"""End-to-end reporting for the inference-only official-checkpoint audit."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from zsc_identifiability.established_dri import (
    fit_event_posterior,
    predict_event_posteriors,
    summarize_posteriors,
)
from zsc_identifiability.established_official_analysis import (
    _pairwise_rows,
    analyze_official_checkpoint_audit,
    audit_official_estimator_calibration,
    build_official_response_library,
    build_official_trace_index,
    estimate_official_pairwise_dri,
)
from zsc_identifiability.established_official_assets import load_official_asset_inventory
from zsc_identifiability.established_official_models import (
    OfficialAssetInventory,
    OfficialCheckpointAuditManifest,
    OfficialCheckpointAuditSuiteV2,
    OfficialResponseValueMatrix,
    OfficialRolloutLedger,
    OfficialRolloutPlan,
    OfficialTraceIndex,
    PairwiseIdentifiabilityRow,
    load_official_checkpoint_suite,
)
from zsc_identifiability.established_official_statistics import (
    clustered_dri_coefficient_interval,
    nested_leave_one_scheme_out_feature_regression,
    nested_leave_one_scheme_out_regression,
)
from zsc_identifiability.learning_statistics import holm_adjust


def run_complete_official_checkpoint_analysis(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    plan: OfficialRolloutPlan | str | Path,
    ledger: OfficialRolloutLedger | str | Path,
    output_dir: str | Path,
) -> OfficialCheckpointAuditManifest:
    """Reproduce every compact Stage 6 v2 analysis from completed shards."""

    spec, suite_path = _suite_and_path(suite)
    rollout_plan = _plan(plan)
    rollout_ledger = _ledger(ledger)
    _require_complete_plan(rollout_plan, rollout_ledger)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory = load_official_asset_inventory(rollout_plan.inventory_path)

    libraries = build_official_response_library(
        tuple(shard.result_path for shard in rollout_plan.shards if shard.kind == "response"),
        spec,
    )
    trace_index = build_official_trace_index(rollout_plan, rollout_ledger)
    full_population: list[dict[str, Any]] = []
    pairwise = tuple(
        row
        for library in libraries
        for row in estimate_official_pairwise_dri(
            trace_index,
            library,
            spec,
            full_population_rows=full_population,
        )
    )
    calibration = audit_official_estimator_calibration(trace_index, libraries, pairwise, spec)
    exclusions = _build_exclusion_report(inventory, libraries)
    method_report = _build_method_report(rollout_plan, libraries, exclusions, trace_index)
    validation_pairwise = _estimate_validation_event_rows(trace_index, libraries, spec)
    intervention = _build_intervention_audit(
        trace_index,
        libraries,
        pairwise,
        validation_pairwise,
        spec,
    )
    regression_rows = _build_regression_rows(
        method_report["partner_method_rows"],
        libraries,
        pairwise,
        trace_index,
        exclusions,
    )
    episode_regression_rows = _build_episode_regression_rows(
        regression_rows, method_report["episode_rows"]
    )
    regression = _run_regressions(regression_rows, episode_regression_rows, spec)
    ranking_reversals = _ranking_reversal_report(regression_rows, spec)
    sensitivity = _build_sensitivity_report(
        libraries, pairwise, regression_rows, regression, intervention, spec
    )

    artifacts: dict[str, Any] = {
        "response-value-matrices.json": {"matrices": [item.to_dict() for item in libraries]},
        "response-conflict-audit.json": _response_conflict_report(libraries, exclusions),
        "official-trace-index.json": trace_index.to_dict(),
        "pairwise-identifiability.json": [item.to_dict() for item in pairwise],
        "full-population-identifiability.json": full_population,
        "estimator-calibration-report.json": calibration,
        "official-method-evaluation.json": method_report,
        "natural-intervention-audit.json": intervention,
        "held-out-regression-report.json": regression,
        "ranking-reversal-report.json": ranking_reversals,
        "sensitivity-report.json": sensitivity,
        "partner-inventory-and-exclusions.json": exclusions,
        "reproducibility-manifest.json": _reproducibility_manifest(
            spec,
            rollout_plan,
            rollout_ledger,
            inventory,
            trace_index,
            suite_path,
            output,
        ),
    }
    generated: list[str] = []
    for name, payload in artifacts.items():
        path = output / name
        _atomic_json(path, payload)
        generated.append(str(path))
    figures = _generate_figures(
        output,
        libraries,
        pairwise,
        method_report,
        intervention,
        regression,
        regression_rows,
    )
    generated.extend(figures)

    ordinary_pre = [
        row
        for row in pairwise
        if row.estimator == "gru"
        and row.evidence_policy == "ordinary_progress"
        and row.prefix == "pre_commitment"
    ]
    commitment_rate = float(np.mean([row.commitment_rate for row in ordinary_pre]))
    top_condition = bool(
        intervention["natural_intervention_qualifies"] or intervention["systematic_method_gap"]
    )
    analysis_inputs = {
        "asset_integrity_passed": inventory.complete,
        "runtime_parity_passed": _parity_passed(rollout_plan, rollout_ledger),
        "competent_primary_partners": exclusions["central_partner_counts"]["random3_m"],
        "robust_conflict_present": all(
            bool(library.conflicting_pairs_by_margin["0.01"])
            and bool(library.conflicting_pairs_by_margin["0.05"])
            for library in libraries
        ),
        "commitment_reliable": commitment_rate >= 0.8,
        "estimator_calibration_passed": bool(calibration["passed"]),
        "no_dri_asset_selection": spec.partner_selection_criterion
        == "official_benchmark_yaml_only",
        "dri_incremental_value": bool(regression["overall"]["incremental_value"]),
        "robustness_direction_reproduced": bool(regression["directionally_reproduced"]),
        "qualifying_intervention_or_systematic_gap": top_condition,
        "natural_intervention_qualifies": bool(intervention["natural_intervention_qualifies"]),
        "generated_files": generated,
        "missing_artifacts": [],
        "total_episodes": sum(len(shard.episode_keys) for shard in rollout_plan.shards),
        "total_environment_steps": _observed_environment_steps(rollout_plan),
        "peak_workers": spec.runtime.default_workers,
        "peak_memory_bytes": _peak_memory_bytes(rollout_plan),
        "inventory_hash": rollout_plan.inventory_hash,
        "source_hash": _analysis_source_hash(suite_path),
        "invoked_command": (
            "zsc-identifiability",
            "established",
            "official",
            "analyze",
        ),
    }
    manifest = analyze_official_checkpoint_audit(
        suite_path if suite_path is not None else spec,
        analysis_inputs,
    )
    manifest_path = output / "official-checkpoint-audit-manifest.json"
    _atomic_json(manifest_path, manifest.to_dict())
    return manifest


def _build_method_report(
    plan: OfficialRolloutPlan,
    libraries: Sequence[OfficialResponseValueMatrix],
    exclusions: Mapping[str, Any],
    trace_index: OfficialTraceIndex,
) -> dict[str, Any]:
    library_by_layout = {item.layout_id: item for item in libraries}
    duplicate_seed_ids = set(str(item) for item in exclusions["duplicate_method_asset_ids"])
    inventory = load_official_asset_inventory(plan.inventory_path)
    asset_for_method = {
        (item.layout_id, item.method_id, item.seed): item.asset_id for item in inventory.methods
    }
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    total_steps = 0
    for shard in plan.shards:
        if shard.kind != "method":
            continue
        result = _read_result(Path(shard.result_path))
        _require_result(result, "official_method_rollout")
        if shard.partner_id is None or shard.method_id is None or shard.method_seed is None:
            raise ValueError("method shard is missing its registered identifiers")
        if shard.deployment is None:
            raise ValueError("method shard is missing its deployment semantics")
        library = library_by_layout[shard.layout_id]
        partner_index = library.partner_ids.index(shard.partner_id)
        best = max(library.raw_values[partner_index])
        values = [float(value) for value in result["episode_returns"]]
        if len(values) != len(shard.episode_keys):
            raise ValueError("method result episode count differs from the locked shard")
        asset_id = asset_for_method[(shard.layout_id, shard.method_id, shard.method_seed)]
        excluded_duplicate = asset_id in duplicate_seed_ids
        mean_return = float(np.mean(values))
        row = {
            "layout_id": shard.layout_id,
            "partner_id": shard.partner_id,
            "scheme_id": _scheme(shard.partner_id),
            "training_stage": _stage(shard.partner_id),
            "method_id": shard.method_id,
            "method_seed": shard.method_seed,
            "deployment": shard.deployment,
            "method_asset_id": asset_id,
            "duplicate_seed_excluded_from_inference": excluded_duplicate,
            "episode_count": len(values),
            "mean_sparse_return": mean_return,
            "best_response_library_return": best,
            "normalized_response_library_regret": 1.0 - mean_return / best,
            "br_prox": mean_return / best,
            "commitment_rate": float(
                np.mean(
                    [bool(item.get("commitment_reached", False)) for item in result["episodes"]]
                )
            ),
            "mean_commitment_step": _mean_optional(
                [item.get("commitment_step") for item in result["episodes"]]
            ),
            "diagnostic_pattern_frequency": _diagnostic_pattern_frequency(result["episodes"]),
        }
        rows.append(row)
        for ordinal, (key, value, episode) in enumerate(
            zip(shard.episode_keys, values, result["episodes"], strict=True)
        ):
            steps = episode.get("steps", ())
            total_steps += len(steps) if steps else 400
            episode_rows.append(
                {
                    **{
                        key_name: row[key_name]
                        for key_name in (
                            "layout_id",
                            "partner_id",
                            "scheme_id",
                            "method_id",
                            "method_seed",
                            "deployment",
                            "duplicate_seed_excluded_from_inference",
                        )
                    },
                    "environment_key": int(key),
                    "episode_ordinal": ordinal,
                    "ego_seat": int(episode["ego_seat"]),
                    "sparse_return": value,
                    "normalized_response_library_regret": 1.0 - value / best,
                    "commitment_reached": bool(episode.get("commitment_reached", False)),
                }
            )
    induced_dri = _method_induced_event_dri(rows, plan, trace_index, libraries)
    deployment_sensitivity = _deployment_sensitivity(rows)
    return {
        "schema_version": 1,
        "partner_method_rows": rows,
        "episode_rows": episode_rows,
        "method_induced_event_dri": induced_dri,
        "greedy_vs_stochastic_sensitivity": deployment_sensitivity,
        "seat_sensitivity": _seat_sensitivity(episode_rows),
        "duplicate_weights_count_as_independent_seeds": False,
        "observed_environment_steps": total_steps,
        "peak_memory_bytes": None,
    }


def _seat_sensitivity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (str(row["layout_id"]), str(row["method_id"]), int(row["ego_seat"]))
            for row in rows
            if row["deployment"] == "stochastic"
            and not bool(row["duplicate_seed_excluded_from_inference"])
        }
    )
    for layout_id, method_id, seat in groups:
        selected = [
            row
            for row in rows
            if row["layout_id"] == layout_id
            and row["method_id"] == method_id
            and int(row["ego_seat"]) == seat
            and row["deployment"] == "stochastic"
            and not bool(row["duplicate_seed_excluded_from_inference"])
        ]
        output.append(
            {
                "layout_id": layout_id,
                "method_id": method_id,
                "ego_seat": seat,
                "episode_count": len(selected),
                "mean_regret": float(
                    np.mean([float(row["normalized_response_library_regret"]) for row in selected])
                ),
            }
        )
    return output


def _deployment_sensitivity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["layout_id"]),
            str(row["partner_id"]),
            str(row["method_id"]),
            int(row["method_seed"]),
            str(row["deployment"]),
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (str(row["layout_id"]), str(row["method_id"]), int(row["method_seed"]))
            for row in rows
            if not bool(row["duplicate_seed_excluded_from_inference"])
        }
    )
    for layout_id, method_id, seed in groups:
        partner_ids = sorted(
            {
                str(row["partner_id"])
                for row in rows
                if row["layout_id"] == layout_id
                and row["method_id"] == method_id
                and int(row["method_seed"]) == seed
            }
        )
        differences = [
            float(
                lookup[(layout_id, partner, method_id, seed, "greedy")][
                    "normalized_response_library_regret"
                ]
            )
            - float(
                lookup[(layout_id, partner, method_id, seed, "stochastic")][
                    "normalized_response_library_regret"
                ]
            )
            for partner in partner_ids
        ]
        output.append(
            {
                "layout_id": layout_id,
                "method_id": method_id,
                "method_seed": seed,
                "greedy_minus_stochastic_regret": float(np.mean(differences)),
                "partner_count": len(differences),
            }
        )
    return output


def _method_induced_event_dri(
    method_rows: Sequence[Mapping[str, Any]],
    plan: OfficialRolloutPlan,
    trace_index: OfficialTraceIndex,
    libraries: Sequence[OfficialResponseValueMatrix],
) -> list[dict[str, Any]]:
    shards = [shard for shard in plan.shards if shard.kind == "method"]
    results = {shard.shard_id: _read_result(Path(shard.result_path)) for shard in shards}
    output: list[dict[str, Any]] = []
    for library in libraries:
        partner_indices = {partner: index for index, partner in enumerate(library.partner_ids)}
        losses = np.asarray(library.normalized_losses, dtype=np.float64)
        prior = np.full(len(library.partner_ids), 1.0 / len(library.partner_ids))
        signatures = tuple(
            library.response_ids[int(np.argmin(losses[index]))]
            for index in range(len(library.partner_ids))
        )
        traces = _trace_groups(trace_index, library.layout_id)
        calibration = traces[("ordinary_progress", "calibration")]
        calibration_labels = [partner_indices[str(item["partner_id"])] for item in calibration]
        model = fit_event_posterior(
            [_event_tokens(item, "pre_commitment") for item in calibration],
            calibration_labels,
            len(library.partner_ids),
        )
        combinations = sorted(
            {
                (str(row["method_id"]), int(row["method_seed"]), str(row["deployment"]))
                for row in method_rows
                if row["layout_id"] == library.layout_id
                and not bool(row["duplicate_seed_excluded_from_inference"])
            }
        )
        for method_id, seed, deployment in combinations:
            selected_shards = [
                shard
                for shard in shards
                if shard.layout_id == library.layout_id
                and shard.method_id == method_id
                and shard.method_seed == seed
                and shard.deployment == deployment
            ]
            episodes: list[Mapping[str, Any]] = []
            labels: list[int] = []
            for shard in selected_shards:
                if shard.partner_id is None:
                    continue
                shard_episodes = results[shard.shard_id]["episodes"]
                episodes.extend(shard_episodes)
                labels.extend([partner_indices[shard.partner_id]] * len(shard_episodes))
            histories = [_event_tokens(item, "pre_commitment") for item in episodes]
            posteriors = predict_event_posteriors(model, histories, prior.tolist())
            for index, episode in enumerate(episodes):
                if not bool(episode.get("commitment_reached", False)):
                    posteriors[index] = prior
            summary = summarize_posteriors(
                prior.tolist(),
                losses.tolist(),
                posteriors.tolist(),
                response_signatures=signatures,
                true_modes=labels,
            )
            output.append(
                {
                    "layout_id": library.layout_id,
                    "method_id": method_id,
                    "method_seed": seed,
                    "deployment": deployment,
                    "episode_count": len(episodes),
                    "estimator_training_policy": "fcp_seed1_greedy_ordinary_progress",
                    "dri": summary.dri,
                    "residual_risk": summary.residual_risk,
                    "identity_mi_nats": summary.identity_mutual_information_nats,
                    "decision_signature_mi_nats": (
                        summary.response_signature_mutual_information_nats
                    ),
                }
            )
    return output


def _build_regression_rows(
    method_rows: Sequence[Mapping[str, Any]],
    libraries: Sequence[OfficialResponseValueMatrix],
    pairwise: Sequence[PairwiseIdentifiabilityRow],
    trace_index: OfficialTraceIndex,
    exclusions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dri_lookup = {
        (row.layout_id, row.left_partner_id, row.right_partner_id): row
        for row in pairwise
        if row.estimator == "gru"
        and row.evidence_policy == "ordinary_progress"
        and row.prefix == "pre_commitment"
    }
    predictability = _pairwise_visible_action_predictability(trace_index, libraries)
    method_lookup = {
        (
            str(row["layout_id"]),
            str(row["partner_id"]),
            str(row["method_id"]),
            int(row["method_seed"]),
        ): row
        for row in method_rows
        if not bool(row["duplicate_seed_excluded_from_inference"])
        and row["deployment"] == "stochastic"
    }
    excluded_partners = set(str(item) for item in exclusions["central_excluded_partner_ids"])
    output: list[dict[str, Any]] = []
    for library in libraries:
        partner_index = {partner: index for index, partner in enumerate(library.partner_ids)}
        raw = np.asarray(library.raw_values, dtype=np.float64)
        losses = np.asarray(library.normalized_losses, dtype=np.float64)
        competence_scale = float(np.max(raw.max(axis=1)))
        for left, right in library.conflicting_pairs_by_margin["0.02"]:
            if left in excluded_partners or right in excluded_partners:
                continue
            pair_key = (library.layout_id, left, right)
            dri_row = dri_lookup[pair_key]
            left_index, right_index = partner_index[left], partner_index[right]
            pair_losses = losses[[left_index, right_index]]
            prior_risk = float(np.min(np.mean(pair_losses, axis=0)))
            features = np.asarray(
                [
                    library.best_response_event_features[left],
                    library.best_response_event_features[right],
                ],
                dtype=np.float64,
            )
            pair_brdiv = _gram_determinant(features)
            pair_rahman = _pair_rahman(raw, left_index, right_index)
            competence = float(
                np.mean([raw[left_index].max(), raw[right_index].max()]) / competence_scale
            )
            for method_id in sorted({str(row["method_id"]) for row in method_rows}):
                seeds = sorted(
                    {
                        int(row["method_seed"])
                        for row in method_rows
                        if row["layout_id"] == library.layout_id
                        and row["method_id"] == method_id
                        and row["deployment"] == "stochastic"
                        and not bool(row["duplicate_seed_excluded_from_inference"])
                    }
                )
                for seed in seeds:
                    left_method = method_lookup.get((library.layout_id, left, method_id, seed))
                    right_method = method_lookup.get((library.layout_id, right, method_id, seed))
                    if left_method is None or right_method is None:
                        continue
                    output.append(
                        {
                            "layout_id": library.layout_id,
                            "left_partner_id": left,
                            "right_partner_id": right,
                            "left_scheme_id": _scheme(left),
                            "right_scheme_id": _scheme(right),
                            "method_id": method_id,
                            "method_seed": seed,
                            "normalized_response_library_regret": float(
                                np.mean(
                                    [
                                        left_method["normalized_response_library_regret"],
                                        right_method["normalized_response_library_regret"],
                                    ]
                                )
                            ),
                            "br_prox": float(
                                np.mean([left_method["br_prox"], right_method["br_prox"]])
                            ),
                            "precommitment_dri": float(dri_row.dri or 0.0),
                            "identity_mi_nats": float(dri_row.identity_mi_nats),
                            "partner_competence": competence,
                            "prior_confusion_risk": prior_risk,
                            "conflict_coefficient": library.conflict_coefficients[
                                f"{left}|{right}"
                            ],
                            "rahman_brdiv_return": pair_rahman,
                            "zsceval_br_div_raw": pair_brdiv,
                            "visible_action_predictability": predictability[pair_key],
                            "prefix_tv": dri_row.prefix_tv,
                        }
                    )
    return output


def _run_regressions(
    rows: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    suite: OfficialCheckpointAuditSuiteV2,
) -> dict[str, Any]:
    overall = nested_leave_one_scheme_out_regression(
        rows, ridge_strengths=suite.statistics.ridge_strengths
    )
    by_layout: dict[str, Any] = {}
    for layout in suite.layouts:
        selected = [row for row in rows if row["layout_id"] == layout.layout_id]
        by_layout[layout.layout_id] = nested_leave_one_scheme_out_regression(
            selected, ridge_strengths=suite.statistics.ridge_strengths
        )
    reproduced = all(
        report["delta_mae"] < 0 and report["delta_mse"] < 0 for report in by_layout.values()
    )
    return {
        "schema_version": 1,
        "outcome": "normalized_response_library_regret",
        "br_prox_used_as_predictor": False,
        "overall": overall,
        "by_layout": by_layout,
        "directionally_reproduced": reproduced,
        "row_count": len(rows),
        "episode_row_count": len(episode_rows),
        "dri_coefficient_interval": clustered_dri_coefficient_interval(
            episode_rows,
            resamples=suite.statistics.bootstrap_resamples,
        ),
    }


def _ranking_reversal_report(
    rows: Sequence[Mapping[str, Any]], suite: OfficialCheckpointAuditSuiteV2
) -> dict[str, Any]:
    method_ids = sorted({str(row["method_id"]) for row in rows})
    comparisons: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for layout in suite.layouts:
        layout_rows = [row for row in rows if row["layout_id"] == layout.layout_id]
        pair_dri = {
            (str(row["left_partner_id"]), str(row["right_partner_id"])): float(
                row["precommitment_dri"]
            )
            for row in layout_rows
        }
        threshold = float(np.median(list(pair_dri.values())))
        lookup = {
            (
                str(row["left_partner_id"]),
                str(row["right_partner_id"]),
                str(row["method_id"]),
                int(row["method_seed"]),
            ): float(row["normalized_response_library_regret"])
            for row in layout_rows
        }
        for left_index, left_method in enumerate(method_ids):
            for right_method in method_ids[left_index + 1 :]:
                effects: dict[str, list[float]] = {"low": [], "high": []}
                for partner_pair, dri in pair_dri.items():
                    stratum = "low" if dri <= threshold else "high"
                    seeds = sorted(
                        {
                            int(row["method_seed"])
                            for row in layout_rows
                            if row["method_id"] in {left_method, right_method}
                            and (
                                str(row["left_partner_id"]),
                                str(row["right_partner_id"]),
                            )
                            == partner_pair
                        }
                    )
                    for seed in seeds:
                        left_key = (*partner_pair, left_method, seed)
                        right_key = (*partner_pair, right_method, seed)
                        if left_key in lookup and right_key in lookup:
                            effects[stratum].append(lookup[left_key] - lookup[right_key])
                intervals = {
                    stratum: _paired_bootstrap_interval(
                        values,
                        suite.statistics.bootstrap_resamples,
                        seed=9101
                        + sum(
                            f"{layout.layout_id}:{left_method}:{right_method}:{stratum}".encode()
                        ),
                    )
                    for stratum, values in effects.items()
                }
                comparison_id = f"{layout.layout_id}:{left_method}:{right_method}"
                for stratum in ("low", "high"):
                    raw_p[f"{comparison_id}:{stratum}"] = float(intervals[stratum]["two_sided_p"])
                comparisons.append(
                    {
                        "comparison_id": comparison_id,
                        "layout_id": layout.layout_id,
                        "left_method": left_method,
                        "right_method": right_method,
                        "dri_median_threshold": threshold,
                        "low_dri": intervals["low"],
                        "high_dri": intervals["high"],
                    }
                )
    adjusted = holm_adjust(raw_p)
    strict_count = 0
    for item in comparisons:
        comparison_id = str(item["comparison_id"])
        low = item["low_dri"]
        high = item["high_dri"]
        item["holm_p_low"] = adjusted[f"{comparison_id}:low"]
        item["holm_p_high"] = adjusted[f"{comparison_id}:high"]
        item["strict_reversal"] = bool(
            float(low["mean"]) * float(high["mean"]) < 0
            and float(low["ci_low"]) * float(low["ci_high"]) > 0
            and float(high["ci_low"]) * float(high["ci_high"]) > 0
            and item["holm_p_low"] < 0.05
            and item["holm_p_high"] < 0.05
        )
        strict_count += int(item["strict_reversal"])
    return {
        "schema_version": 1,
        "secondary_analysis": True,
        "strict_reversal_count": strict_count,
        "comparisons": comparisons,
    }


def _build_episode_regression_rows(
    aggregate_rows: Sequence[Mapping[str, Any]],
    method_episode_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    episode_lookup = {
        (
            str(row["layout_id"]),
            str(row["partner_id"]),
            str(row["method_id"]),
            int(row["method_seed"]),
            int(row["episode_ordinal"]),
        ): row
        for row in method_episode_rows
        if row["deployment"] == "stochastic"
        and not bool(row["duplicate_seed_excluded_from_inference"])
    }
    output: list[dict[str, Any]] = []
    for aggregate in aggregate_rows:
        layout = str(aggregate["layout_id"])
        left = str(aggregate["left_partner_id"])
        right = str(aggregate["right_partner_id"])
        method = str(aggregate["method_id"])
        seed = int(aggregate["method_seed"])
        ordinals = sorted(
            {
                int(row["episode_ordinal"])
                for row in method_episode_rows
                if row["layout_id"] == layout
                and row["method_id"] == method
                and int(row["method_seed"]) == seed
                and row["deployment"] == "stochastic"
            }
        )
        for ordinal in ordinals:
            left_row = episode_lookup.get((layout, left, method, seed, ordinal))
            right_row = episode_lookup.get((layout, right, method, seed, ordinal))
            if left_row is None or right_row is None:
                continue
            output.append(
                {
                    **aggregate,
                    "normalized_response_library_regret": float(
                        np.mean(
                            [
                                left_row["normalized_response_library_regret"],
                                right_row["normalized_response_library_regret"],
                            ]
                        )
                    ),
                    "environment_key": f"{layout}:{ordinal}",
                }
            )
    return output


def _estimate_validation_event_rows(
    trace_index: OfficialTraceIndex,
    libraries: Sequence[OfficialResponseValueMatrix],
    suite: OfficialCheckpointAuditSuiteV2,
) -> tuple[PairwiseIdentifiabilityRow, ...]:
    output: list[PairwiseIdentifiabilityRow] = []
    for library in libraries:
        grouped = _trace_groups(trace_index, library.layout_id)
        partner_indices = {partner: index for index, partner in enumerate(library.partner_ids)}
        schemes = {partner: _scheme(partner) for partner in library.partner_ids}
        losses = np.asarray(library.normalized_losses, dtype=np.float64)
        prior = np.full(len(library.partner_ids), 1.0 / len(library.partner_ids))
        for evidence_policy in sorted({key[0] for key in grouped if key[1] == "calibration"}):
            calibration = grouped[(evidence_policy, "calibration")]
            validation = grouped[(evidence_policy, "validation")]
            calibration_labels = [partner_indices[str(item["partner_id"])] for item in calibration]
            validation_labels = [partner_indices[str(item["partner_id"])] for item in validation]
            calibration_histories = [_event_tokens(item, "pre_commitment") for item in calibration]
            validation_histories = [_event_tokens(item, "pre_commitment") for item in validation]
            model = fit_event_posterior(
                calibration_histories,
                calibration_labels,
                len(library.partner_ids),
                smoothing=suite.estimator.event_laplace_alpha,
            )
            posteriors = predict_event_posteriors(model, validation_histories, prior.tolist())
            output.extend(
                _pairwise_rows(
                    library,
                    library.conflicting_pairs_by_margin["0.02"],
                    schemes,
                    evidence_policy,
                    "event",
                    "pre_commitment",
                    posteriors,
                    validation,
                    validation_labels,
                    losses,
                    validation_histories,
                )
            )
    return tuple(output)


def _build_intervention_audit(
    trace_index: OfficialTraceIndex,
    libraries: Sequence[OfficialResponseValueMatrix],
    confirmatory_rows: Sequence[PairwiseIdentifiabilityRow],
    validation_rows: Sequence[PairwiseIdentifiabilityRow],
    suite: OfficialCheckpointAuditSuiteV2,
) -> dict[str, Any]:
    library_by_layout = {item.layout_id: item for item in libraries}
    layout_reports: list[dict[str, Any]] = []
    any_qualifies = False
    for layout in suite.layouts:
        library = library_by_layout[layout.layout_id]
        groups = _trace_groups(trace_index, layout.layout_id)
        validation_risk = _option_effects(validation_rows, layout.layout_id, "event", metric="risk")
        confirm_event_dri = _option_effects(
            confirmatory_rows, layout.layout_id, "event", metric="dri"
        )
        confirm_gru_dri = _option_effects(confirmatory_rows, layout.layout_id, "gru", metric="dri")
        confirm_gru_risk = _option_effects(
            confirmatory_rows, layout.layout_id, "gru", metric="risk"
        )
        validation_metrics = _option_trace_metrics(
            groups, library, layout.diagnostic_options, "validation"
        )
        confirmatory_metrics = _option_trace_metrics(
            groups, library, layout.diagnostic_options, "confirmatory"
        )
        options = [item for item in layout.diagnostic_options if item != "ordinary_progress"]
        selection_rows: list[dict[str, Any]] = []
        for option in options:
            effect = float(np.mean(validation_risk.get(option, (0.0,))))
            cost = float(validation_metrics[option]["normalized_task_cost"])
            selection_rows.append(
                {
                    "option": option,
                    "validation_decision_risk_reduction": effect,
                    "validation_normalized_task_cost": cost,
                    "validation_net_information_value": effect - cost,
                }
            )
        selected = max(
            selection_rows,
            key=lambda item: (
                float(item["validation_net_information_value"]),
                -options.index(str(item["option"])),
            ),
        )
        raw_p: dict[str, float] = {}
        option_rows: list[dict[str, Any]] = []
        for option in options:
            event_dri_values = confirm_event_dri.get(option, ())
            gru_dri_values = confirm_gru_dri.get(option, ())
            gru_risk_values = confirm_gru_risk.get(option, ())
            interval = _paired_bootstrap_interval(
                gru_risk_values,
                suite.statistics.bootstrap_resamples,
                seed=8201 + sum(option.encode()),
            )
            raw_p[option] = float(interval["two_sided_p"])
            metrics = confirmatory_metrics[option]
            row = {
                "option": option,
                "selected_on_validation": option == selected["option"],
                "confirmatory_event_dri_effect": (
                    float(np.mean(event_dri_values)) if event_dri_values else None
                ),
                "confirmatory_gru_dri_effect": (
                    float(np.mean(gru_dri_values)) if gru_dri_values else None
                ),
                "confirmatory_decision_risk_reduction": (
                    float(np.mean(gru_risk_values)) if gru_risk_values else None
                ),
                "confirmatory_risk_reduction_interval": interval,
                **metrics,
            }
            option_rows.append(row)
        adjusted = holm_adjust(raw_p)
        for row in option_rows:
            option = str(row["option"])
            row["holm_adjusted_p"] = adjusted[option]
            dri_effect = row["confirmatory_gru_dri_effect"]
            risk_effect = row["confirmatory_decision_risk_reduction"]
            interval = row["confirmatory_risk_reduction_interval"]
            measurable_cost = (
                float(row["normalized_task_cost"]) > 1e-12
                or float(row["mean_commitment_delay_steps"]) > 1e-12
            )
            row["qualifies_as_diagnostic"] = bool(
                row["selected_on_validation"]
                and dri_effect is not None
                and float(dri_effect) > 0
                and risk_effect is not None
                and float(risk_effect) > 0
                and float(interval["ci_low"]) > 0
                and adjusted[option] < 0.05
                and float(row["completion_before_commitment_rate"]) >= 0.8
                and float(row["partner_response_tv"]) > 0
                and measurable_cost
                and float(risk_effect) - float(row["normalized_task_cost"]) > 0
            )
            any_qualifies = any_qualifies or bool(row["qualifies_as_diagnostic"])
        layout_reports.append(
            {
                "layout_id": layout.layout_id,
                "selection_uses_validation_only": True,
                "selection_table": selection_rows,
                "selected_option": selected["option"],
                "confirmatory_options": option_rows,
                "restricted_empirical_frontier": _restricted_empirical_frontier(
                    confirmatory_rows,
                    layout.layout_id,
                    confirmatory_metrics,
                ),
            }
        )
    return {
        "schema_version": 1,
        "comparator": "ordinary_progress",
        "paired_environment_keys": True,
        "holm_corrected": True,
        "layouts": layout_reports,
        "natural_intervention_qualifies": any_qualifies,
        "systematic_method_gap": False,
        "broad_causal_claim_authorized": False,
    }


def _option_effects(
    rows: Sequence[PairwiseIdentifiabilityRow],
    layout_id: str,
    estimator: str,
    *,
    metric: str,
) -> dict[str, tuple[float, ...]]:
    if metric not in {"dri", "risk"}:
        raise ValueError(f"unsupported intervention metric: {metric!r}")
    ordinary = {
        (row.left_partner_id, row.right_partner_id): (
            float(row.dri or 0.0) if metric == "dri" else row.residual_risk
        )
        for row in rows
        if row.layout_id == layout_id
        and row.estimator == estimator
        and row.prefix == "pre_commitment"
        and row.evidence_policy == "ordinary_progress"
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.layout_id != layout_id
            or row.estimator != estimator
            or row.prefix != "pre_commitment"
            or row.evidence_policy == "ordinary_progress"
        ):
            continue
        key = (row.left_partner_id, row.right_partner_id)
        if key in ordinary and row.dri is not None:
            option_value = float(row.dri) if metric == "dri" else row.residual_risk
            effect = (
                option_value - ordinary[key] if metric == "dri" else ordinary[key] - option_value
            )
            grouped[row.evidence_policy].append(effect)
    return {key: tuple(value) for key, value in grouped.items()}


def _restricted_empirical_frontier(
    rows: Sequence[PairwiseIdentifiabilityRow],
    layout_id: str,
    option_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    policy_ids = sorted(
        {
            row.evidence_policy
            for row in rows
            if row.layout_id == layout_id
            and row.estimator == "gru"
            and row.prefix == "pre_commitment"
        }
    )
    points: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        values = [
            float(row.dri)
            for row in rows
            if row.layout_id == layout_id
            and row.estimator == "gru"
            and row.prefix == "pre_commitment"
            and row.evidence_policy == policy_id
            and row.dri is not None
        ]
        points.append(
            {
                "policy_id": policy_id,
                "normalized_task_cost": (
                    0.0
                    if policy_id == "ordinary_progress"
                    else float(option_metrics[policy_id]["normalized_task_cost"])
                ),
                "mean_pairwise_dri": float(np.mean(values)),
            }
        )
    nondominated = [
        point
        for point in points
        if not any(
            other is not point
            and float(other["normalized_task_cost"]) <= float(point["normalized_task_cost"]) + 1e-12
            and float(other["mean_pairwise_dri"]) >= float(point["mean_pairwise_dri"]) - 1e-12
            and (
                float(other["normalized_task_cost"]) < float(point["normalized_task_cost"]) - 1e-12
                or float(other["mean_pairwise_dri"]) > float(point["mean_pairwise_dri"]) + 1e-12
            )
            for other in points
        )
    ]
    ordered = sorted(
        nondominated,
        key=lambda item: (item["normalized_task_cost"], -item["mean_pairwise_dri"]),
    )
    unique_cost: list[dict[str, Any]] = []
    for point in ordered:
        if (
            unique_cost
            and abs(
                float(point["normalized_task_cost"])
                - float(unique_cost[-1]["normalized_task_cost"])
            )
            <= 1e-12
        ):
            if float(point["mean_pairwise_dri"]) > float(unique_cost[-1]["mean_pairwise_dri"]):
                unique_cost[-1] = point
        else:
            unique_cost.append(point)
    hull: list[dict[str, Any]] = []
    for point in unique_cost:
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            first_slope = (
                float(second["mean_pairwise_dri"]) - float(first["mean_pairwise_dri"])
            ) / (float(second["normalized_task_cost"]) - float(first["normalized_task_cost"]))
            next_slope = (
                float(point["mean_pairwise_dri"]) - float(second["mean_pairwise_dri"])
            ) / (float(point["normalized_task_cost"]) - float(second["normalized_task_cost"]))
            if next_slope < first_slope - 1e-12:
                break
            hull.pop()
        hull.append(point)
    mixtures = [
        {
            "left_policy_id": left["policy_id"],
            "right_policy_id": right["policy_id"],
            "mixture_probability_range": [0.0, 1.0],
            "cost_endpoints": [
                left["normalized_task_cost"],
                right["normalized_task_cost"],
            ],
            "dri_endpoints": [left["mean_pairwise_dri"], right["mean_pairwise_dri"]],
        }
        for left, right in zip(hull, hull[1:], strict=False)
    ]
    return {
        "exact_bayes_frontier": False,
        "deterministic_points": points,
        "nondominated_points": ordered,
        "convexified_envelope_points": hull,
        "episode_level_randomized_mixture_segments": mixtures,
    }


def _option_trace_metrics(
    groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    library: OfficialResponseValueMatrix,
    options: Sequence[str],
    split: str,
) -> dict[str, dict[str, float]]:
    ordinary = groups[("ordinary_progress", split)]
    ordinary_by_key = {
        (str(item["partner_id"]), int(item["environment_key"])): item for item in ordinary
    }
    maxima = {
        partner: max(library.raw_values[index]) for index, partner in enumerate(library.partner_ids)
    }
    output: dict[str, dict[str, float]] = {}
    for option in options:
        if option == "ordinary_progress":
            continue
        selected = groups[(option, split)]
        costs: list[float] = []
        delays: list[float] = []
        completed: list[float] = []
        response_histories: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        for item in selected:
            partner = str(item["partner_id"])
            key = (partner, int(item["environment_key"]))
            baseline = ordinary_by_key.get(key)
            if baseline is None:
                raise ValueError("intervention and ordinary traces do not share environment keys")
            costs.append(
                (float(baseline["sparse_return"]) - float(item["sparse_return"])) / maxima[partner]
            )
            baseline_commit = baseline.get("commitment_step")
            option_commit = item.get("commitment_step")
            delays.append(
                float(
                    (400 if option_commit is None else option_commit)
                    - (400 if baseline_commit is None else baseline_commit)
                )
            )
            completion = item.get("intervention_completed_step")
            completed.append(
                float(
                    completion is not None
                    and (option_commit is None or int(completion) < int(option_commit))
                )
            )
            response_histories[partner].append(
                tuple(
                    int(step["visible_partner_action"])
                    for step in item["steps"]
                    if step.get("visible_partner_action") is not None
                    and (
                        item.get("commitment_step") is None
                        or int(step["step"]) < int(item["commitment_step"])
                    )
                )
            )
        pair_tvs = [
            _history_tv(response_histories[left], response_histories[right])
            for left, right in library.conflicting_pairs_by_margin["0.02"]
        ]
        output[option] = {
            "normalized_task_cost": float(np.mean(costs)),
            "mean_commitment_delay_steps": float(np.mean(delays)),
            "completion_before_commitment_rate": float(np.mean(completed)),
            "partner_response_tv": float(np.mean(pair_tvs)) if pair_tvs else 0.0,
        }
    return output


def _pairwise_visible_action_predictability(
    trace_index: OfficialTraceIndex,
    libraries: Sequence[OfficialResponseValueMatrix],
) -> dict[tuple[str, str, str], float]:
    output: dict[tuple[str, str, str], float] = {}
    for library in libraries:
        groups = _trace_groups(trace_index, library.layout_id)
        calibration = groups[("ordinary_progress", "calibration")]
        confirmatory = groups[("ordinary_progress", "confirmatory")]
        for left, right in library.conflicting_pairs_by_margin["0.02"]:
            selected_train = [item for item in calibration if item["partner_id"] in {left, right}]
            selected_test = [item for item in confirmatory if item["partner_id"] in {left, right}]
            counts: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
            global_counts: Counter[int] = Counter()
            for episode in selected_train:
                for context, action in _visible_action_targets(episode):
                    counts[context][action] += 1
                    global_counts[action] += 1
            losses: list[float] = []
            for episode in selected_test:
                for context, action in _visible_action_targets(episode):
                    row = counts.get(context, global_counts)
                    total = sum(row.values()) + 7.0
                    losses.append(-math.log((row.get(action, 0) + 1.0) / total))
            output[(library.layout_id, left, right)] = (
                float(np.mean(losses)) if losses else math.log(7.0)
            )
    return output


def _build_exclusion_report(
    inventory: OfficialAssetInventory,
    libraries: Sequence[OfficialResponseValueMatrix],
) -> dict[str, Any]:
    record_by_id = {item.asset_id: item for item in inventory.assets}
    partner_by_asset = {item.partner_asset_id: item for item in inventory.partners}
    method_by_asset = {item.asset_id: item for item in inventory.methods}
    excluded_partners: set[str] = set()
    duplicate_method_assets: set[str] = set()
    duplicate_rows: list[dict[str, Any]] = []
    for group in inventory.duplicate_tensor_groups:
        partners = sorted(
            (partner_by_asset[item] for item in group if item in partner_by_asset),
            key=lambda item: (item.layout_id, item.scheme_id, item.training_stage),
        )
        methods = sorted(
            (method_by_asset[item] for item in group if item in method_by_asset),
            key=lambda item: (item.layout_id, item.method_id, item.seed),
        )
        for layout_id in sorted({item.layout_id for item in partners}):
            same_layout = [item for item in partners if item.layout_id == layout_id]
            excluded_partners.update(item.partner_id for item in same_layout[1:])
        for layout_id, method_id in sorted({(item.layout_id, item.method_id) for item in methods}):
            same_method = [
                item
                for item in methods
                if item.layout_id == layout_id and item.method_id == method_id
            ]
            duplicate_method_assets.update(item.asset_id for item in same_method[1:])
        duplicate_rows.append(
            {
                "normalized_tensor_hash": (
                    record_by_id[group[0]].normalized_tensor_hash if group else None
                ),
                "asset_ids": list(group),
                "partner_ids": [item.partner_id for item in partners],
                "methods": [
                    {
                        "layout_id": item.layout_id,
                        "method_id": item.method_id,
                        "seed": item.seed,
                    }
                    for item in methods
                ],
            }
        )
    central_counts = {
        library.layout_id: sum(partner not in excluded_partners for partner in library.partner_ids)
        for library in libraries
    }
    return {
        "schema_version": 1,
        "selection_source": "official_benchmark_yaml_only",
        "all_official_partners_reported": True,
        "competence_rule": "positive best available response-library return",
        "central_excluded_partner_ids": sorted(excluded_partners),
        "duplicate_method_asset_ids": sorted(duplicate_method_assets),
        "duplicate_groups": duplicate_rows,
        "central_partner_counts": central_counts,
    }


def _response_conflict_report(
    libraries: Sequence[OfficialResponseValueMatrix], exclusions: Mapping[str, Any]
) -> dict[str, Any]:
    excluded = set(str(item) for item in exclusions["central_excluded_partner_ids"])
    layouts: list[dict[str, Any]] = []
    for library in libraries:
        layouts.append(
            {
                "layout_id": library.layout_id,
                "official_partner_count": len(library.partner_ids),
                "central_partner_count": sum(
                    partner not in excluded for partner in library.partner_ids
                ),
                "conflicting_pair_counts": {
                    margin: sum(
                        left not in excluded and right not in excluded for left, right in pairs
                    )
                    for margin, pairs in library.conflicting_pairs_by_margin.items()
                },
                "adequacy_margin_sensitivity_passed": all(
                    any(left not in excluded and right not in excluded for left, right in pairs)
                    for pairs in library.conflicting_pairs_by_margin.values()
                ),
                "rahman_brdiv_return": library.rahman_brdiv_return,
                "zsceval_br_div_raw": library.zsceval_br_div_raw,
                "zsceval_br_div_code": library.zsceval_br_div_code,
            }
        )
    return {"schema_version": 1, "layouts": layouts}


def _build_sensitivity_report(
    libraries: Sequence[OfficialResponseValueMatrix],
    pairwise: Sequence[PairwiseIdentifiabilityRow],
    regression_rows: Sequence[Mapping[str, Any]],
    regression: Mapping[str, Any],
    intervention: Mapping[str, Any],
    suite: OfficialCheckpointAuditSuiteV2,
) -> dict[str, Any]:
    conflict_counts = {
        library.layout_id: {
            margin: len(pairs) for margin, pairs in library.conflicting_pairs_by_margin.items()
        }
        for library in libraries
    }
    estimator_directions: dict[str, dict[str, float]] = {}
    for layout in suite.layouts:
        estimator_directions[layout.layout_id] = {}
        for estimator in ("event", "gru"):
            values = [
                float(row.dri)
                for row in pairwise
                if row.layout_id == layout.layout_id
                and row.estimator == estimator
                and row.evidence_policy == "ordinary_progress"
                and row.prefix == "pre_commitment"
                and row.dri is not None
            ]
            estimator_directions[layout.layout_id][estimator] = float(np.mean(values))
    stage_reports: dict[str, Any] = {}
    for stage in ("mid", "final"):
        selected = [
            row
            for row in regression_rows
            if _stage(str(row["left_partner_id"])) == stage
            and _stage(str(row["right_partner_id"])) == stage
        ]
        stage_reports[stage] = _safe_regression(selected, suite)
    deployment_note = {
        "implemented": True,
        "primary": "stochastic",
        "sensitivity": "greedy",
        "reported_in": "official-method-evaluation.json",
    }
    return {
        "schema_version": 1,
        "adequacy_margin_conflict_counts": conflict_counts,
        "event_vs_gru_mean_precommitment_dri": estimator_directions,
        "fixed_step_vs_commitment": _prefix_sensitivity(pairwise),
        "mid_only_and_final_only_regression": stage_reports,
        "seat_specific": {
            "implemented_in_rollouts": True,
            "central_results_average_balanced_seats": True,
        },
        "full_pool_vs_competent_only": {
            "full_pool_reported": True,
            "central_rule": "positive best response-library return and unique tensor weights",
        },
        "greedy_vs_stochastic": deployment_note,
        "identity_mi_substitution": _safe_feature_regression(
            regression_rows, suite, "identity_mi_nats"
        ),
        "prefix_tv_substitution": _safe_feature_regression(regression_rows, suite, "prefix_tv"),
        "primary_regression": regression["overall"],
        "intervention_direction_by_layout": [
            {
                "layout_id": item["layout_id"],
                "selected_option": item["selected_option"],
                "qualifies": any(
                    bool(row["qualifies_as_diagnostic"]) for row in item["confirmatory_options"]
                ),
            }
            for item in intervention["layouts"]
        ],
        "claim_direction_stable": bool(regression["directionally_reproduced"]),
    }


def _safe_regression(
    rows: Sequence[Mapping[str, Any]], suite: OfficialCheckpointAuditSuiteV2
) -> dict[str, Any]:
    try:
        return nested_leave_one_scheme_out_regression(
            rows, ridge_strengths=suite.statistics.ridge_strengths
        )
    except ValueError as error:
        return {"status": "not_estimable", "reason": str(error), "row_count": len(rows)}


def _safe_feature_regression(
    rows: Sequence[Mapping[str, Any]],
    suite: OfficialCheckpointAuditSuiteV2,
    feature: str,
) -> dict[str, Any]:
    try:
        return nested_leave_one_scheme_out_feature_regression(
            rows,
            incremental_feature=feature,
            ridge_strengths=suite.statistics.ridge_strengths,
        )
    except ValueError as error:
        return {
            "status": "not_estimable",
            "reason": str(error),
            "row_count": len(rows),
            "incremental_feature": feature,
        }


def _prefix_sensitivity(
    rows: Sequence[PairwiseIdentifiabilityRow],
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = defaultdict(dict)
    for layout_id in sorted({row.layout_id for row in rows}):
        for prefix in ("32", "pre_commitment"):
            values = [
                float(row.dri)
                for row in rows
                if row.layout_id == layout_id
                and row.estimator == "gru"
                and row.evidence_policy == "ordinary_progress"
                and row.prefix == prefix
                and row.dri is not None
            ]
            output[layout_id][prefix] = float(np.mean(values)) if values else None
    return dict(output)


def _generate_figures(
    output: Path,
    libraries: Sequence[OfficialResponseValueMatrix],
    pairwise: Sequence[PairwiseIdentifiabilityRow],
    method_report: Mapping[str, Any],
    intervention: Mapping[str, Any],
    regression: Mapping[str, Any],
    regression_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    def save(name: str) -> None:
        for suffix in ("pdf", "png"):
            path = figure_dir / f"{name}.{suffix}"
            plt.savefig(path, bbox_inches="tight", dpi=180)
            generated.append(str(path))
        plt.close()

    primary = next(item for item in libraries if item.layout_id == "random3_m")
    plt.figure(figsize=(7, 6))
    plt.imshow(np.asarray(primary.normalized_losses), aspect="auto", cmap="magma")
    plt.colorbar(label="normalized response loss")
    plt.xlabel("response checkpoint")
    plt.ylabel("official partner checkpoint")
    save("response-library-cross-play-heatmap")

    _plot_dri_curves(pairwise, plt)
    save("precommitment-and-eventual-dri-curves")
    _plot_scatter(pairwise, "prefix_tv", "dri", plt)
    save("prefix-tv-versus-dri")
    _plot_scatter(pairwise, "identity_mi_nats", "dri", plt)
    save("identity-information-versus-decision-relevant-information")
    _plot_dri_controls(regression_rows, plt)
    save("dri-versus-brdiv-and-predictability")

    predictions = regression["overall"]["predictions"]
    truth = [item["truth"] for item in predictions]
    plt.figure(figsize=(6, 5))
    plt.scatter(truth, [item["baseline_prediction"] for item in predictions], label="controls")
    plt.scatter(truth, [item["full_prediction"] for item in predictions], label="controls + DRI")
    plt.xlabel("held-out regret")
    plt.ylabel("prediction")
    plt.legend()
    save("held-out-regret-predictions")

    _plot_method_dri(method_report, pairwise, plt)
    save("official-method-performance-across-dri")
    _plot_partial_effect(regression_rows, plt)
    save("partial-effect-of-dri-by-method")
    _plot_intervention(intervention, plt)
    save("passive-versus-intervention-dri-and-cost")
    _plot_layout_comparison(pairwise, plt)
    save("primary-versus-robustness-layout-effects")
    return generated


def _plot_dri_curves(rows: Sequence[PairwiseIdentifiabilityRow], plt: Any) -> None:
    order = ("0", "8", "16", "32", "pre_commitment", "eventual")
    plt.figure(figsize=(7, 4))
    for layout in sorted({row.layout_id for row in rows}):
        values = []
        for prefix in order:
            selected = [
                float(row.dri)
                for row in rows
                if row.layout_id == layout
                and row.estimator == "gru"
                and row.evidence_policy == "ordinary_progress"
                and row.prefix == prefix
                and row.dri is not None
            ]
            values.append(float(np.mean(selected)) if selected else np.nan)
        plt.plot(order, values, marker="o", label=layout)
    plt.ylabel("mean pairwise DRI")
    plt.xlabel("evidence endpoint")
    plt.legend()


def _plot_scatter(
    rows: Sequence[PairwiseIdentifiabilityRow], x_field: str, y_field: str, plt: Any
) -> None:
    selected = [
        row
        for row in rows
        if row.estimator == "gru"
        and row.evidence_policy == "ordinary_progress"
        and row.prefix == "pre_commitment"
        and row.dri is not None
    ]
    plt.figure(figsize=(6, 5))
    for layout in sorted({row.layout_id for row in selected}):
        values = [row for row in selected if row.layout_id == layout]
        plt.scatter(
            [float(getattr(row, x_field)) for row in values],
            [float(getattr(row, y_field)) for row in values],
            label=layout,
            alpha=0.7,
        )
    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.legend()


def _plot_dri_controls(rows: Sequence[Mapping[str, Any]], plt: Any) -> None:
    unique: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["layout_id"]),
            str(row["left_partner_id"]),
            str(row["right_partner_id"]),
        )
        unique.setdefault(key, row)
    values = list(unique.values())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for layout in sorted({str(row["layout_id"]) for row in values}):
        selected = [row for row in values if row["layout_id"] == layout]
        axes[0].scatter(
            [float(row["zsceval_br_div_raw"]) for row in selected],
            [float(row["precommitment_dri"]) for row in selected],
            label=layout,
            alpha=0.65,
        )
        axes[1].scatter(
            [float(row["visible_action_predictability"]) for row in selected],
            [float(row["precommitment_dri"]) for row in selected],
            label=layout,
            alpha=0.65,
        )
    axes[0].set_xlabel("pairwise ZSC-Eval BR-Div")
    axes[1].set_xlabel("visible-action cross-entropy")
    for axis in axes:
        axis.set_ylabel("pre-commitment DRI")
        axis.legend()
    figure.tight_layout()


def _plot_partial_effect(rows: Sequence[Mapping[str, Any]], plt: Any) -> None:
    control_names = (
        "partner_competence",
        "prior_confusion_risk",
        "conflict_coefficient",
        "rahman_brdiv_return",
        "zsceval_br_div_raw",
        "visible_action_predictability",
        "prefix_tv",
    )
    x = np.asarray(
        [[1.0, *(float(row[name]) for name in control_names)] for row in rows],
        dtype=np.float64,
    )
    y = np.asarray(
        [float(row["normalized_response_library_regret"]) for row in rows],
        dtype=np.float64,
    )
    residual = y - x @ (np.linalg.pinv(x) @ y)
    plt.figure(figsize=(7, 5))
    for method in sorted({str(row["method_id"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["method_id"] == method]
        plt.scatter(
            [float(rows[index]["precommitment_dri"]) for index in indices],
            residual[indices],
            label=method,
            alpha=0.5,
        )
    plt.xlabel("pre-commitment DRI")
    plt.ylabel("regret residual after registered continuous controls")
    plt.legend(ncol=2)


def _plot_method_dri(
    method_report: Mapping[str, Any],
    pairwise: Sequence[PairwiseIdentifiabilityRow],
    plt: Any,
    *,
    partial: bool = False,
) -> None:
    partner_dri: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in pairwise:
        if (
            row.estimator == "gru"
            and row.evidence_policy == "ordinary_progress"
            and row.prefix == "pre_commitment"
            and row.dri is not None
        ):
            partner_dri[(row.layout_id, row.left_partner_id)].append(float(row.dri))
            partner_dri[(row.layout_id, row.right_partner_id)].append(float(row.dri))
    plt.figure(figsize=(7, 5))
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row in method_report["partner_method_rows"]:
        if row["deployment"] != "stochastic" or row["duplicate_seed_excluded_from_inference"]:
            continue
        key = (str(row["layout_id"]), str(row["partner_id"]))
        if key not in partner_dri:
            continue
        x, y = grouped.setdefault(str(row["method_id"]), ([], []))
        x.append(float(np.mean(partner_dri[key])))
        y.append(float(row["normalized_response_library_regret"]))
    for method, (x, y) in grouped.items():
        if partial and x:
            centered = np.asarray(y) - np.mean(y)
            plt.scatter(x, centered, label=method, alpha=0.5)
        else:
            plt.scatter(x, y, label=method, alpha=0.5)
    plt.xlabel("partner mean pre-commitment DRI")
    plt.ylabel("centered regret" if partial else "normalized response-library regret")
    plt.legend(ncol=2)


def _plot_intervention(report: Mapping[str, Any], plt: Any) -> None:
    labels: list[str] = []
    dri_effects: list[float] = []
    risk_effects: list[float] = []
    costs: list[float] = []
    for layout in report["layouts"]:
        for row in layout["confirmatory_options"]:
            labels.append(f"{layout['layout_id']}\n{row['option']}")
            dri_effects.append(float(row["confirmatory_gru_dri_effect"] or 0.0))
            risk_effects.append(float(row["confirmatory_decision_risk_reduction"] or 0.0))
            costs.append(float(row["normalized_task_cost"]))
    positions = np.arange(len(labels))
    plt.figure(figsize=(max(7, len(labels) * 1.3), 5))
    plt.bar(positions - 0.25, dri_effects, width=0.25, label="DRI effect")
    plt.bar(positions, risk_effects, width=0.25, label="decision-risk reduction")
    plt.bar(positions + 0.25, costs, width=0.25, label="task cost")
    plt.xticks(positions, labels, rotation=30, ha="right")
    plt.legend()


def _plot_layout_comparison(rows: Sequence[PairwiseIdentifiabilityRow], plt: Any) -> None:
    layouts = sorted({row.layout_id for row in rows})
    passive: list[float] = []
    active: list[float] = []
    for layout in layouts:
        selected = [
            row
            for row in rows
            if row.layout_id == layout
            and row.estimator == "gru"
            and row.prefix == "pre_commitment"
            and row.dri is not None
        ]
        passive.append(
            float(
                np.mean(
                    [
                        float(row.dri or 0.0)
                        for row in selected
                        if row.evidence_policy == "ordinary_progress"
                    ]
                )
            )
        )
        active.append(
            max(
                float(
                    np.mean(
                        [float(row.dri or 0.0) for row in selected if row.evidence_policy == option]
                    )
                )
                for option in {row.evidence_policy for row in selected}
            )
        )
    positions = np.arange(len(layouts))
    plt.figure(figsize=(6, 5))
    plt.bar(positions - 0.2, passive, width=0.4, label="passive")
    plt.bar(positions + 0.2, active, width=0.4, label="best audited option")
    plt.xticks(positions, layouts)
    plt.ylabel("mean pairwise DRI")
    plt.legend()


def _trace_groups(
    index: OfficialTraceIndex, layout_id: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in index.entries:
        if entry.layout_id != layout_id:
            continue
        result = _read_result(Path(entry.path))
        _require_result(result, "official_trace_rollout")
        for episode in result["episodes"]:
            record = dict(episode)
            record["partner_id"] = entry.partner_id
            grouped[(entry.evidence_policy, entry.split)].append(record)
    return grouped


def _event_tokens(episode: Mapping[str, Any], prefix: str) -> tuple[str, ...]:
    steps = list(episode["steps"])
    if prefix == "pre_commitment":
        commitment = episode.get("commitment_step")
        steps = [] if commitment is None else [item for item in steps if item["step"] < commitment]
    tokens: list[str] = []
    for step in steps:
        tokens.append(f"ego_action:{step['ego_action']}")
        if step.get("visible_partner_action") is not None:
            tokens.append(f"partner_action:{step['visible_partner_action']}")
        tokens.extend(str(item) for item in step.get("events", ()))
        reward = float(step.get("reward", 0.0))
        tokens.append(
            "reward:positive" if reward > 0 else "reward:negative" if reward < 0 else "reward:zero"
        )
    return tuple(tokens) or ("zero_step",)


def _visible_action_targets(
    episode: Mapping[str, Any],
) -> list[tuple[tuple[int, int, int], int]]:
    output: list[tuple[tuple[int, int, int], int]] = []
    previous_ego = -1
    previous_reward = 0
    commitment = episode.get("commitment_step")
    for step in episode["steps"]:
        if commitment is not None and int(step["step"]) >= int(commitment):
            break
        action = step.get("visible_partner_action")
        if action is not None:
            output.append(
                (
                    (min(int(step["step"]) // 8, 7), previous_ego, previous_reward),
                    int(action),
                )
            )
        previous_ego = int(step["ego_action"])
        reward = float(step.get("reward", 0.0))
        previous_reward = 1 if reward > 0 else -1 if reward < 0 else 0
    return output


def _paired_bootstrap_interval(
    values: Sequence[float], resamples: int, *, seed: int
) -> dict[str, float | int]:
    if not values:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "two_sided_p": 1.0,
            "resamples": 0,
        }
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(resamples, len(array)))].mean(axis=1)
    p_value = min(
        1.0,
        2.0 * min(float(np.mean(sampled <= 0)), float(np.mean(sampled >= 0))),
    )
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "two_sided_p": p_value,
        "resamples": resamples,
    }


def _history_tv(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> float:
    if not left or not right:
        return 0.0
    left_counts = Counter(tuple(item) for item in left)
    right_counts = Counter(tuple(item) for item in right)
    support = left_counts.keys() | right_counts.keys()
    return float(
        0.5
        * sum(
            abs(left_counts[item] / len(left) - right_counts[item] / len(right)) for item in support
        )
    )


def _pair_rahman(matrix: np.ndarray, left: int, right: int) -> float:
    return float(
        0.25
        * (
            (matrix[left, left] - matrix[left, right])
            + (matrix[left, left] - matrix[right, left])
            + (matrix[right, right] - matrix[right, left])
            + (matrix[right, right] - matrix[left, right])
        )
    )


def _gram_determinant(features: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(features @ features.T)
    return 0.0 if sign <= 0 or not np.isfinite(logdet) else float(np.exp(logdet))


def _diagnostic_pattern_frequency(episodes: Sequence[Mapping[str, Any]]) -> float:
    if not episodes:
        return 0.0
    matched = 0
    markers = ("PLACEMENT_ON_COUNTER", "POT", "PICKUP")
    for episode in episodes:
        commitment = episode.get("commitment_step")
        events = [
            str(event)
            for step in episode.get("steps", ())
            if commitment is None or int(step["step"]) < int(commitment)
            for event in step.get("events", ())
        ]
        matched += int(any(marker in event for marker in markers for event in events))
    return matched / len(episodes)


def _mean_optional(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _scheme(partner_id: str) -> str:
    parts = partner_id.split(":")
    if len(parts) < 3:
        raise ValueError(f"invalid official partner id: {partner_id!r}")
    return parts[-2]


def _stage(partner_id: str) -> str:
    parts = partner_id.split(":")
    if len(parts) < 3 or parts[-1] not in {"mid", "final"}:
        raise ValueError(f"invalid official partner stage: {partner_id!r}")
    return parts[-1]


def _parity_passed(plan: OfficialRolloutPlan, ledger: OfficialRolloutLedger) -> bool:
    status = {item.shard_id: item for item in ledger.entries}
    return all(
        status[shard.shard_id].status == "complete"
        for shard in plan.shards
        if shard.kind == "parity"
    )


def _observed_environment_steps(plan: OfficialRolloutPlan) -> int:
    total = 0
    for shard in plan.shards:
        result = _read_result(Path(shard.result_path))
        lengths = result.get("episode_lengths")
        if lengths is None:
            total += 400 * len(shard.episode_keys)
        else:
            total += sum(int(value) for value in lengths)
    return total


def _peak_memory_bytes(plan: OfficialRolloutPlan) -> int | None:
    values = [
        int(value)
        for shard in plan.shards
        if (value := _read_result(Path(shard.result_path)).get("peak_memory_bytes")) is not None
    ]
    return max(values) if values else None


def _reproducibility_manifest(
    suite: OfficialCheckpointAuditSuiteV2,
    plan: OfficialRolloutPlan,
    ledger: OfficialRolloutLedger,
    inventory: OfficialAssetInventory,
    trace_index: OfficialTraceIndex,
    suite_path: Path | None,
    output: Path,
) -> dict[str, Any]:
    parity_results = [
        _read_result(Path(shard.result_path)) for shard in plan.shards if shard.kind == "parity"
    ]
    versions = {
        result["layout_id"]: result.get("dependency_versions", {}) for result in parity_results
    }
    return {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "suite_hash": plan.suite_hash,
        "plan_hash": plan.plan_hash,
        "inventory_hash": inventory.inventory_hash,
        "repository_commit": suite.upstream.repository_commit,
        "policy_pool_revision": suite.upstream.policy_pool_revision,
        "asset_hashes": {
            item.repository_path or item.asset_id: item.file_hash for item in inventory.assets
        },
        "normalized_tensor_hashes": {
            item.repository_path or item.asset_id: item.normalized_tensor_hash
            for item in inventory.assets
            if item.normalized_tensor_hash is not None
        },
        "request_hashes": {item.shard_id: item.request_hash for item in plan.shards},
        "result_hashes": {
            item.shard_id: item.result_hash
            for item in ledger.entries
            if item.result_hash is not None
        },
        "trace_hashes": {item.trace_id: item.content_hash for item in trace_index.entries},
        "analysis_source_hash": _analysis_source_hash(suite_path),
        "runtime_versions_by_layout": versions,
        "device": "cpu",
        "default_workers": suite.runtime.default_workers,
        "maximum_workers": suite.runtime.maximum_workers,
        "total_episodes": sum(len(shard.episode_keys) for shard in plan.shards),
        "total_environment_steps": _observed_environment_steps(plan),
        "peak_memory_bytes": _peak_memory_bytes(plan),
        "invoked_command": [
            "zsc-identifiability",
            "established",
            "official",
            "analyze",
            "--suite",
            str(suite_path or "<in-memory>"),
            "--plan",
            str(Path(plan.workspace) / "official-rollout-plan.json"),
            "--ledger",
            str(Path(plan.workspace) / "official-rollout-ledger.json"),
            "--output",
            str(output),
        ],
    }


def _analysis_source_hash(suite_path: Path | None) -> str:
    if suite_path is None:
        return hashlib.sha256(b"in-memory-official-analysis").hexdigest()
    root = suite_path.resolve().parents[2]
    runtime_source = (
        root
        / "phase-6-established-validation/runtime-legacy/src/stage6_legacy_runtime"
        / "official_eval.py"
    )
    paths = sorted(
        (
            *root.glob("src/zsc_identifiability/established_official*.py"),
            runtime_source,
        ),
        key=lambda item: str(item.relative_to(root)),
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_complete_plan(plan: OfficialRolloutPlan, ledger: OfficialRolloutLedger) -> None:
    if plan.plan_hash != ledger.plan_hash:
        raise ValueError("official analysis plan and ledger do not match")
    by_id = {item.shard_id: item for item in ledger.entries}
    incomplete = [
        shard.shard_id
        for shard in plan.shards
        if by_id.get(shard.shard_id) is None or by_id[shard.shard_id].status != "complete"
    ]
    if incomplete:
        raise ValueError(
            f"official analysis requires every rollout shard; {len(incomplete)} remain incomplete"
        )


def _require_result(result: Mapping[str, Any], operation: str) -> None:
    if result.get("operation") != operation:
        raise ValueError(f"unexpected official runtime operation: {result.get('operation')!r}")
    if result.get("policy_training_performed") is not False:
        raise ValueError("official result does not prove inference-only execution")
    if operation != "official_parity" and result.get("partner_deployment") != "stochastic":
        raise ValueError("scientific rollouts must preserve official stochastic partner sampling")


def _read_result(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"official result is not an object: {path}")
    return value


def _suite_and_path(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> tuple[OfficialCheckpointAuditSuiteV2, Path | None]:
    if isinstance(suite, OfficialCheckpointAuditSuiteV2):
        return suite, None
    path = Path(suite).resolve()
    return load_official_checkpoint_suite(path), path


def _plan(plan: OfficialRolloutPlan | str | Path) -> OfficialRolloutPlan:
    if isinstance(plan, OfficialRolloutPlan):
        return plan
    return OfficialRolloutPlan.model_validate(_read_json(Path(plan)))


def _ledger(ledger: OfficialRolloutLedger | str | Path) -> OfficialRolloutLedger:
    if isinstance(ledger, OfficialRolloutLedger):
        return ledger
    return OfficialRolloutLedger.model_validate(_read_json(Path(ledger)))


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
