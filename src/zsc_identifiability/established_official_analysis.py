"""Response-conflict, DRI, and final-gate analysis for official checkpoints."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from zsc_identifiability.established_dri import (
    fit_event_posterior,
    predict_event_posteriors,
    summarize_posteriors,
    synthetic_dri_calibration,
)
from zsc_identifiability.established_gru import fit_cross_fitted_gru_posterior
from zsc_identifiability.established_official_models import (
    OfficialCheckpointAuditManifest,
    OfficialCheckpointAuditSuiteV2,
    OfficialResponseValueMatrix,
    OfficialRolloutLedger,
    OfficialRolloutPlan,
    OfficialTraceIndex,
    OfficialTraceIndexEntry,
    PairwiseIdentifiabilityRow,
    load_official_checkpoint_suite,
)


def build_official_response_library(
    response_results: str | Path | Iterable[str | Path],
    config: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> tuple[OfficialResponseValueMatrix, ...]:
    """Aggregate response shards into fixed empirical loss matrices."""

    suite = _suite(config)
    paths = _result_paths(response_results, kind="response")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        result = _read_result(path)
        _validate_result(result, "official_response_rollout")
        grouped[str(result["layout_id"])].append(result)
    outputs: list[OfficialResponseValueMatrix] = []
    for layout in suite.layouts:
        rows = grouped.get(layout.layout_id, [])
        if not rows:
            raise ValueError(f"no response results found for {layout.layout_id}")
        values: dict[tuple[str, str], float] = {}
        intervals: dict[tuple[str, str], tuple[float, float]] = {}
        features: dict[tuple[str, str], tuple[float, ...]] = {}
        for row in rows:
            partner = str(row["partner_id"])
            response = str(row["response_id"])
            episode_returns = np.asarray(row["episode_returns"], dtype=np.float64)
            if len(episode_returns) != layout.response_episodes_per_pair:
                raise ValueError("response shard has the wrong paired episode count")
            values[(partner, response)] = float(np.mean(episode_returns))
            standard_error = (
                float(np.std(episode_returns, ddof=1) / math.sqrt(len(episode_returns)))
                if len(episode_returns) > 1
                else 0.0
            )
            intervals[(partner, response)] = (
                float(np.mean(episode_returns) - 1.96 * standard_error),
                float(np.mean(episode_returns) + 1.96 * standard_error),
            )
            feature = tuple(float(value) for value in row["mean_ego_event_features"])
            if not feature or not all(math.isfinite(value) for value in feature):
                raise ValueError("response shards require finite best-response event features")
            features[(partner, response)] = feature
        partner_ids = tuple(sorted({key[0] for key in values}))
        response_ids = tuple(sorted({key[1] for key in values}))
        if len(values) != len(partner_ids) * len(response_ids):
            raise ValueError("response cross-play matrix is incomplete")
        raw = np.asarray(
            [[values[(partner, response)] for response in response_ids] for partner in partner_ids],
            dtype=np.float64,
        )
        maxima = raw.max(axis=1)
        if np.any(maxima <= 0):
            raise ValueError("normalized response loss requires positive best-library returns")
        losses = 1.0 - raw / maxima[:, None]
        adequate_sets: dict[str, tuple[str, ...]] = {}
        conflicts_by_margin: dict[str, tuple[tuple[str, str], ...]] = {}
        for margin in suite.statistics.adequacy_margins:
            adequate = {
                partner: tuple(
                    response_ids[index]
                    for index, loss in enumerate(losses[row_index])
                    if loss <= margin + 1e-12
                )
                for row_index, partner in enumerate(partner_ids)
            }
            conflicts_by_margin[f"{margin:.2f}"] = tuple(
                (left, right)
                for left_index, left in enumerate(partner_ids)
                for right in partner_ids[left_index + 1 :]
                if not set(adequate[left]) & set(adequate[right])
            )
            if abs(margin - suite.statistics.primary_adequacy_margin) <= 1e-12:
                adequate_sets = adequate
        conflict_coefficients = {
            f"{partner_ids[left]}|{partner_ids[right]}": float(np.min(losses[left] + losses[right]))
            for left in range(len(partner_ids))
            for right in range(left + 1, len(partner_ids))
        }
        best_indices = np.argmax(raw, axis=1)
        best_features = np.asarray(
            [
                features[(partner, response_ids[int(best_indices[index])])]
                for index, partner in enumerate(partner_ids)
            ],
            dtype=np.float64,
        )
        if len({row.shape for row in best_features}) != 1:
            raise ValueError("best-response event feature widths differ")
        outputs.append(
            OfficialResponseValueMatrix(
                suite_id=suite.suite_id,
                layout_id=layout.layout_id,
                partner_ids=partner_ids,
                response_ids=response_ids,
                raw_values=tuple(tuple(float(value) for value in row) for row in raw),
                raw_value_intervals_95=tuple(
                    tuple(intervals[(partner, response)] for response in response_ids)
                    for partner in partner_ids
                ),
                normalized_losses=tuple(tuple(float(value) for value in row) for row in losses),
                adequate_response_sets=adequate_sets,
                conflicting_pairs_by_margin=conflicts_by_margin,
                conflict_coefficients=conflict_coefficients,
                best_response_event_features={
                    partner: tuple(float(value) for value in best_features[index])
                    for index, partner in enumerate(partner_ids)
                },
                rahman_brdiv_return=_rahman_brdiv(raw),
                zsceval_br_div_raw=_gram_determinant(best_features),
                zsceval_br_div_code=_gram_determinant(_official_normalize(best_features)),
            )
        )
    return tuple(outputs)


def build_official_trace_index(
    plan: OfficialRolloutPlan | str | Path,
    ledger: OfficialRolloutLedger | str | Path,
) -> OfficialTraceIndex:
    rollout_plan = _plan(plan)
    rollout_ledger = _ledger(ledger)
    if rollout_plan.plan_hash != rollout_ledger.plan_hash:
        raise ValueError("trace ledger belongs to a different rollout plan")
    ledger_by_id = {entry.shard_id: entry for entry in rollout_ledger.entries}
    entries: list[OfficialTraceIndexEntry] = []
    for shard in rollout_plan.shards:
        if shard.kind != "trace":
            continue
        status = ledger_by_id[shard.shard_id]
        if status.status != "complete" or status.result_hash is None:
            raise ValueError(f"trace shard is incomplete: {shard.shard_id}")
        if shard.partner_id is None or shard.evidence_policy is None or shard.split is None:
            raise ValueError("trace shard omits required identity fields")
        entries.append(
            OfficialTraceIndexEntry(
                trace_id=shard.shard_id,
                layout_id=shard.layout_id,
                partner_id=shard.partner_id,
                evidence_policy=shard.evidence_policy,
                split=shard.split,
                path=shard.result_path,
                content_hash=status.result_hash,
                episodes=len(shard.episode_keys),
            )
        )
    return OfficialTraceIndex(suite_id=rollout_plan.suite_id, entries=tuple(entries))


def estimate_official_pairwise_dri(
    trace_index: OfficialTraceIndex | str | Path,
    response_library: OfficialResponseValueMatrix,
    config: OfficialCheckpointAuditSuiteV2 | str | Path,
    *,
    full_population_rows: list[dict[str, Any]] | None = None,
) -> tuple[PairwiseIdentifiabilityRow, ...]:
    """Fit full-population estimators, then audit all conflicting pairs."""

    suite = _suite(config)
    index = _trace_index(trace_index)
    partner_ids = response_library.partner_ids
    partner_to_label = {partner: number for number, partner in enumerate(partner_ids)}
    schemes = {partner: partner.split(":")[-2] for partner in partner_ids}
    losses = np.asarray(response_library.normalized_losses, dtype=np.float64)
    prior = np.full(len(partner_ids), 1.0 / len(partner_ids), dtype=np.float64)
    primary_conflicts = response_library.conflicting_pairs_by_margin["0.02"]
    entries = [entry for entry in index.entries if entry.layout_id == response_library.layout_id]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        result = _read_result(Path(entry.path))
        _validate_result(result, "official_trace_rollout")
        for episode in result["episodes"]:
            record = dict(episode)
            record["partner_id"] = entry.partner_id
            grouped[(entry.evidence_policy, entry.split)].append(record)
    prefixes: tuple[int | str, ...] = (*suite.evidence.prefix_steps, "pre_commitment", "eventual")
    rows: list[PairwiseIdentifiabilityRow] = []
    evidence_policies = tuple(sorted({entry.evidence_policy for entry in entries}))
    for evidence_policy in evidence_policies:
        calibration = grouped[(evidence_policy, "calibration")]
        validation = grouped[(evidence_policy, "validation")]
        confirmatory = grouped[(evidence_policy, "confirmatory")]
        _assert_disjoint_episode_keys(calibration, validation, confirmatory)
        for prefix in prefixes:
            event_train = [_event_tokens(item, prefix) for item in calibration]
            event_confirm = [_event_tokens(item, prefix) for item in confirmatory]
            calibration_labels = [partner_to_label[str(item["partner_id"])] for item in calibration]
            confirmatory_labels = [
                partner_to_label[str(item["partner_id"])] for item in confirmatory
            ]
            event_model = fit_event_posterior(
                event_train,
                calibration_labels,
                len(partner_ids),
                smoothing=suite.estimator.event_laplace_alpha,
            )
            event_posteriors = predict_event_posteriors(event_model, event_confirm, prior.tolist())
            event_full_posteriors = event_posteriors.copy()
            if prefix == "pre_commitment":
                _assign_prior_to_censored(event_full_posteriors, confirmatory, prior.tolist())
            if full_population_rows is not None:
                full_population_rows.append(
                    _full_population_row(
                        response_library,
                        evidence_policy,
                        "event",
                        str(prefix),
                        prior,
                        losses,
                        event_full_posteriors,
                        confirmatory_labels,
                        confirmatory,
                    )
                )
            rows.extend(
                _pairwise_rows(
                    response_library,
                    primary_conflicts,
                    schemes,
                    evidence_policy,
                    "event",
                    str(prefix),
                    event_posteriors,
                    confirmatory,
                    confirmatory_labels,
                    losses,
                    event_confirm,
                )
            )
            gru_train = [_gru_features(item, prefix) for item in calibration]
            gru_validation = [_gru_features(item, prefix) for item in validation]
            gru_confirm = [_gru_features(item, prefix) for item in confirmatory]
            validation_labels = [partner_to_label[str(item["partner_id"])] for item in validation]
            ensemble: list[np.ndarray] = []
            for seed in suite.estimator.gru_seeds:
                fit = fit_cross_fitted_gru_posterior(
                    gru_train,
                    calibration_labels,
                    gru_validation,
                    validation_labels,
                    gru_confirm,
                    confirmatory_labels,
                    prior.tolist(),
                    losses.tolist(),
                    hidden_size=suite.estimator.gru_hidden_size,
                    seed=seed,
                )
                ensemble.append(fit.posteriors)
            gru_posteriors = np.mean(np.stack(ensemble), axis=0)
            gru_full_posteriors = gru_posteriors.copy()
            if prefix == "pre_commitment":
                _assign_prior_to_censored(gru_full_posteriors, confirmatory, prior.tolist())
            if full_population_rows is not None:
                seed_dri: list[float | None] = []
                for posterior in ensemble:
                    seed_posterior = posterior.copy()
                    if prefix == "pre_commitment":
                        _assign_prior_to_censored(seed_posterior, confirmatory, prior.tolist())
                    seed_dri.append(
                        _full_population_row(
                            response_library,
                            evidence_policy,
                            "gru",
                            str(prefix),
                            prior,
                            losses,
                            seed_posterior,
                            confirmatory_labels,
                            confirmatory,
                        )["dri"]
                    )
                full = _full_population_row(
                    response_library,
                    evidence_policy,
                    "gru",
                    str(prefix),
                    prior,
                    losses,
                    gru_full_posteriors,
                    confirmatory_labels,
                    confirmatory,
                )
                finite_seed_dri = [float(value) for value in seed_dri if value is not None]
                full["gru_seed_dri_standard_deviation"] = (
                    float(np.std(finite_seed_dri, ddof=1)) if len(finite_seed_dri) > 1 else 0.0
                )
                full_population_rows.append(full)
            rows.extend(
                _pairwise_rows(
                    response_library,
                    primary_conflicts,
                    schemes,
                    evidence_policy,
                    "gru",
                    str(prefix),
                    gru_posteriors,
                    confirmatory,
                    confirmatory_labels,
                    losses,
                    event_confirm,
                )
            )
    return tuple(rows)


def _full_population_row(
    library: OfficialResponseValueMatrix,
    evidence_policy: str,
    estimator: str,
    prefix: str,
    prior: np.ndarray,
    losses: np.ndarray,
    posteriors: np.ndarray,
    labels: Sequence[int],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    signatures = tuple(
        library.response_ids[int(np.argmin(losses[index]))]
        for index in range(len(library.partner_ids))
    )
    summary = summarize_posteriors(
        prior.tolist(),
        losses.tolist(),
        posteriors.tolist(),
        response_signatures=signatures,
        true_modes=labels,
    )
    return {
        "layout_id": library.layout_id,
        "evidence_policy": evidence_policy,
        "estimator": estimator,
        "prefix": prefix,
        "prior_risk": summary.prior_risk,
        "residual_risk": summary.residual_risk,
        "dri": summary.dri,
        "identity_mi_nats": summary.identity_mutual_information_nats,
        "decision_signature_mi_nats": summary.response_signature_mutual_information_nats,
        "commitment_rate": float(
            np.mean([bool(item.get("commitment_reached", False)) for item in episodes])
        ),
    }


def audit_official_estimator_calibration(
    trace_index: OfficialTraceIndex | str | Path,
    response_libraries: Sequence[OfficialResponseValueMatrix],
    pairwise_rows: Sequence[PairwiseIdentifiabilityRow],
    config: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> dict[str, Any]:
    """Run preregistered synthetic, treatment, refit, and label-shuffle controls."""

    suite = _suite(config)
    index = _trace_index(trace_index)
    treatment = _estimator_treatment_agreement(pairwise_rows, suite)
    refits: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    all_candidates: list[tuple[str, str, str]] = []
    for library in response_libraries:
        all_candidates.extend(
            (library.layout_id, left, right)
            for left, right in library.conflicting_pairs_by_margin["0.02"]
        )
    selected = tuple(
        sorted(
            all_candidates,
            key=lambda item: hashlib.sha256("|".join(item).encode()).hexdigest(),
        )[: suite.estimator.direct_pairwise_refit_count]
    )
    row_lookup = {
        (
            row.layout_id,
            row.left_partner_id,
            row.right_partner_id,
            row.evidence_policy,
            row.estimator,
            row.prefix,
        ): row
        for row in pairwise_rows
    }
    for library in response_libraries:
        grouped = _load_trace_groups(index, library.layout_id)
        partners = {partner: number for number, partner in enumerate(library.partner_ids)}
        losses = np.asarray(library.normalized_losses, dtype=np.float64)
        ordinary = {
            split: grouped[("ordinary_progress", split)]
            for split in ("calibration", "validation", "confirmatory")
        }
        shuffle_rows.append(
            _label_shuffle_control(ordinary, partners, losses, suite, library.layout_id)
        )
        for layout_id, left, right in selected:
            if layout_id != library.layout_id:
                continue
            direct = _direct_pairwise_refit(
                ordinary,
                left,
                right,
                partners,
                losses,
                suite,
            )
            for estimator in ("event", "gru"):
                restricted = row_lookup[
                    (layout_id, left, right, "ordinary_progress", estimator, "pre_commitment")
                ]
                direct_value = direct[f"{estimator}_dri"]
                difference = (
                    None
                    if restricted.dri is None or direct_value is None
                    else abs(float(restricted.dri) - float(direct_value))
                )
                refits.append(
                    {
                        "layout_id": layout_id,
                        "left_partner_id": left,
                        "right_partner_id": right,
                        "estimator": estimator,
                        "restricted_dri": restricted.dri,
                        "direct_refit_dri": direct_value,
                        "absolute_difference": difference,
                        "gru_seed_standard_deviation": (
                            direct.get("gru_seed_standard_deviation")
                            if estimator == "gru"
                            else None
                        ),
                    }
                )
    finite_refit_differences = [
        float(row["absolute_difference"])
        for row in refits
        if row["absolute_difference"] is not None
    ]
    maximum_refit_difference = max(finite_refit_differences, default=float("inf"))
    shuffle_maximum = max(
        (abs(float(row["mean_dri"])) for row in shuffle_rows),
        default=float("inf"),
    )
    synthetic = synthetic_dri_calibration()
    tolerance = suite.estimator.treatment_agreement_tolerance
    passed = bool(
        synthetic["passed"]
        and treatment["passed"]
        and finite_refit_differences
        and maximum_refit_difference <= tolerance
        and shuffle_rows
        and shuffle_maximum <= tolerance
    )
    return {
        "schema_version": 1,
        "synthetic_phase3_controls": synthetic,
        "event_gru_treatment_agreement": treatment,
        "direct_pairwise_refits": refits,
        "direct_pair_count": len(selected),
        "maximum_direct_refit_difference": maximum_refit_difference,
        "label_shuffle": shuffle_rows,
        "maximum_absolute_label_shuffle_dri": shuffle_maximum,
        "tolerance": tolerance,
        "passed": passed,
    }


def analyze_official_checkpoint_audit(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    results: Mapping[str, Any],
) -> OfficialCheckpointAuditManifest:
    """Apply the preregistered scientific gates without manufacturing a verdict."""

    spec, suite_path = _suite_with_path(suite)
    gates = {
        "minimum_competent_primary_partners": _optional_bool(
            results.get("competent_primary_partners"), lambda value: int(value) >= 12
        ),
        "response_conflict": _optional_bool(results.get("robust_conflict_present")),
        "commitment_reliable": _optional_bool(results.get("commitment_reliable")),
        "estimator_calibration": _optional_bool(results.get("estimator_calibration_passed")),
        "no_dri_asset_selection": _optional_bool(results.get("no_dri_asset_selection")),
        "dri_incremental_value": _optional_bool(results.get("dri_incremental_value")),
        "robustness_direction_reproduced": _optional_bool(
            results.get("robustness_direction_reproduced")
        ),
        "qualifying_intervention_or_systematic_gap": _optional_bool(
            results.get("qualifying_intervention_or_systematic_gap")
        ),
    }
    required = tuple(gates.values())
    complete = all(value is not None for value in required)
    core_valid = all(
        gates[name] is True
        for name in (
            "minimum_competent_primary_partners",
            "response_conflict",
            "commitment_reliable",
            "estimator_calibration",
            "no_dri_asset_selection",
        )
    )
    if not complete:
        verdict = "pending"
        status = "incomplete"
    elif not core_valid:
        verdict = "redesign"
        status = "complete"
    elif gates["dri_incremental_value"] is not True:
        verdict = "stop"
        status = "complete"
    elif gates["robustness_direction_reproduced"] is not True:
        verdict = "redesign"
        status = "complete"
    elif gates["qualifying_intervention_or_systematic_gap"] is True:
        verdict = "continue_top_paper_package"
        status = "complete"
    elif results.get("natural_intervention_qualifies") is True:
        verdict = "complete_evaluation_only"
        status = "complete"
    else:
        verdict = "complete_measurement_only"
        status = "complete"
    generated = tuple(str(item) for item in results.get("generated_files", ()))
    missing = tuple(str(item) for item in results.get("missing_artifacts", ()))
    return OfficialCheckpointAuditManifest(
        suite_id=spec.suite_id,
        status=status,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        asset_integrity_passed=bool(results.get("asset_integrity_passed", False)),
        runtime_parity_passed=bool(results.get("runtime_parity_passed", False)),
        scientific_gates=gates,
        generated_files=generated,
        missing_artifacts=missing,
        total_episodes=int(results.get("total_episodes", 0)),
        total_environment_steps=int(results.get("total_environment_steps", 0)),
        peak_workers=int(results.get("peak_workers", 0)),
        peak_memory_bytes=(
            None if results.get("peak_memory_bytes") is None else int(results["peak_memory_bytes"])
        ),
        suite_hash=_sha256_path(suite_path) if suite_path else _hash_json(spec.to_dict()),
        inventory_hash=results.get("inventory_hash"),
        source_hash=str(results.get("source_hash", "0" * 64)),
        invoked_command=tuple(str(item) for item in results.get("invoked_command", ())),
    )


def _pairwise_rows(
    library: OfficialResponseValueMatrix,
    conflicts: Sequence[tuple[str, str]],
    schemes: Mapping[str, str],
    evidence_policy: str,
    estimator: str,
    prefix: str,
    posteriors: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    losses: np.ndarray,
    history_keys: Sequence[Sequence[str]],
) -> list[PairwiseIdentifiabilityRow]:
    index = {partner: number for number, partner in enumerate(library.partner_ids)}
    output: list[PairwiseIdentifiabilityRow] = []
    for left, right in conflicts:
        left_index, right_index = index[left], index[right]
        selected = [
            number for number, label in enumerate(labels) if label in {left_index, right_index}
        ]
        pair_posteriors = posteriors[selected][:, [left_index, right_index]]
        denominators = pair_posteriors.sum(axis=1, keepdims=True)
        pair_posteriors = np.divide(
            pair_posteriors,
            denominators,
            out=np.full_like(pair_posteriors, 0.5),
            where=denominators > 1e-15,
        )
        if prefix == "pre_commitment":
            for pair_row, episode_number in enumerate(selected):
                if not bool(episodes[episode_number].get("commitment_reached", False)):
                    pair_posteriors[pair_row] = (0.5, 0.5)
        true_modes = [0 if labels[number] == left_index else 1 for number in selected]
        pair_losses = losses[[left_index, right_index]]
        summary = summarize_posteriors(
            (0.5, 0.5),
            pair_losses.tolist(),
            pair_posteriors,
            response_signatures=(left, right),
            true_modes=true_modes,
        )
        left_histories = [
            history_keys[number] for number in selected if labels[number] == left_index
        ]
        right_histories = [
            history_keys[number] for number in selected if labels[number] == right_index
        ]
        prefix_tv = _empirical_history_tv(left_histories, right_histories)
        commitment_rate = float(
            np.mean(
                [bool(episodes[number].get("commitment_reached", False)) for number in selected]
            )
        )
        output.append(
            PairwiseIdentifiabilityRow(
                layout_id=library.layout_id,
                left_partner_id=left,
                right_partner_id=right,
                left_scheme_id=schemes[left],
                right_scheme_id=schemes[right],
                evidence_policy=evidence_policy,
                estimator=estimator,  # type: ignore[arg-type]
                prefix=prefix,
                prior_risk=summary.prior_risk,
                residual_risk=summary.residual_risk,
                dri=summary.dri,
                identity_mi_nats=summary.identity_mutual_information_nats,
                decision_mi_nats=summary.response_signature_mutual_information_nats or 0.0,
                prefix_tv=prefix_tv,
                commitment_rate=commitment_rate,
            )
        )
    return output


def _estimator_treatment_agreement(
    rows: Sequence[PairwiseIdentifiabilityRow],
    suite: OfficialCheckpointAuditSuiteV2,
) -> dict[str, Any]:
    lookup: dict[tuple[str, str, str, str, str], PairwiseIdentifiabilityRow] = {
        (
            row.layout_id,
            row.left_partner_id,
            row.right_partner_id,
            row.evidence_policy,
            row.estimator,
        ): row
        for row in rows
        if row.prefix == "pre_commitment" and row.dri is not None
    }
    comparisons: list[dict[str, Any]] = []
    for layout in suite.layouts:
        options = tuple(
            option for option in layout.diagnostic_options if option != "ordinary_progress"
        )
        pair_keys = {
            (row.left_partner_id, row.right_partner_id)
            for row in rows
            if row.layout_id == layout.layout_id
            and row.prefix == "pre_commitment"
            and row.evidence_policy == "ordinary_progress"
        }
        for left, right in sorted(pair_keys):
            for option in options:
                keys = {
                    (estimator, evidence): (
                        layout.layout_id,
                        left,
                        right,
                        evidence,
                        estimator,
                    )
                    for estimator in ("event", "gru")
                    for evidence in ("ordinary_progress", option)
                }
                if not all(key in lookup for key in keys.values()):
                    continue
                event_effect = float(lookup[keys[("event", option)]].dri or 0.0) - float(
                    lookup[keys[("event", "ordinary_progress")]].dri or 0.0
                )
                gru_effect = float(lookup[keys[("gru", option)]].dri or 0.0) - float(
                    lookup[keys[("gru", "ordinary_progress")]].dri or 0.0
                )
                comparisons.append(
                    {
                        "layout_id": layout.layout_id,
                        "left_partner_id": left,
                        "right_partner_id": right,
                        "option": option,
                        "event_treatment_effect": event_effect,
                        "gru_treatment_effect": gru_effect,
                        "absolute_difference": abs(event_effect - gru_effect),
                    }
                )
    aggregate: list[dict[str, Any]] = []
    groups = sorted({(item["layout_id"], item["option"]) for item in comparisons})
    for layout_id, option in groups:
        selected = [
            item
            for item in comparisons
            if item["layout_id"] == layout_id and item["option"] == option
        ]
        event_mean = float(np.mean([item["event_treatment_effect"] for item in selected]))
        gru_mean = float(np.mean([item["gru_treatment_effect"] for item in selected]))
        aggregate.append(
            {
                "layout_id": layout_id,
                "option": option,
                "pair_count": len(selected),
                "event_mean_treatment_effect": event_mean,
                "gru_mean_treatment_effect": gru_mean,
                "absolute_difference": abs(event_mean - gru_mean),
            }
        )
    differences = [float(item["absolute_difference"]) for item in aggregate]
    tolerance = suite.estimator.treatment_agreement_tolerance
    return {
        "comparison_count": len(comparisons),
        "maximum_absolute_difference": max(differences, default=float("inf")),
        "mean_absolute_difference": float(np.mean(differences)) if differences else None,
        "tolerance": tolerance,
        "passed": bool(differences) and max(differences) <= tolerance,
        "aggregate_comparisons": aggregate,
        "pairwise_diagnostics": comparisons,
    }


def _load_trace_groups(
    index: OfficialTraceIndex, layout_id: str
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in index.entries:
        if entry.layout_id != layout_id:
            continue
        result = _read_result(Path(entry.path))
        _validate_result(result, "official_trace_rollout")
        for episode in result["episodes"]:
            record = dict(episode)
            record["partner_id"] = entry.partner_id
            grouped[(entry.evidence_policy, entry.split)].append(record)
    return grouped


def _direct_pairwise_refit(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    left: str,
    right: str,
    partner_indices: Mapping[str, int],
    full_losses: np.ndarray,
    suite: OfficialCheckpointAuditSuiteV2,
) -> dict[str, Any]:
    modes = (left, right)
    filtered = {
        split: [row for row in rows if str(row["partner_id"]) in modes]
        for split, rows in split_rows.items()
    }
    labels = {
        split: [0 if str(row["partner_id"]) == left else 1 for row in rows]
        for split, rows in filtered.items()
    }
    pair_losses = full_losses[[partner_indices[left], partner_indices[right]]]
    prior = (0.5, 0.5)
    calibration_histories = [
        _event_tokens(row, "pre_commitment") for row in filtered["calibration"]
    ]
    confirmatory_histories = [
        _event_tokens(row, "pre_commitment") for row in filtered["confirmatory"]
    ]
    event_model = fit_event_posterior(
        calibration_histories,
        labels["calibration"],
        2,
        smoothing=suite.estimator.event_laplace_alpha,
    )
    event_posteriors = predict_event_posteriors(event_model, confirmatory_histories, prior)
    _assign_prior_to_censored(event_posteriors, filtered["confirmatory"], prior)
    event_summary = summarize_posteriors(
        prior,
        pair_losses.tolist(),
        event_posteriors.tolist(),
        response_signatures=modes,
        true_modes=labels["confirmatory"],
    )
    gru_dri: list[float] = []
    for seed in suite.estimator.gru_seeds:
        fit = fit_cross_fitted_gru_posterior(
            [_gru_features(row, "pre_commitment") for row in filtered["calibration"]],
            labels["calibration"],
            [_gru_features(row, "pre_commitment") for row in filtered["validation"]],
            labels["validation"],
            [_gru_features(row, "pre_commitment") for row in filtered["confirmatory"]],
            labels["confirmatory"],
            prior,
            pair_losses.tolist(),
            response_signatures=modes,
            hidden_size=suite.estimator.gru_hidden_size,
            seed=seed,
        )
        posteriors = fit.posteriors.copy()
        _assign_prior_to_censored(posteriors, filtered["confirmatory"], prior)
        summary = summarize_posteriors(
            prior,
            pair_losses.tolist(),
            posteriors.tolist(),
            response_signatures=modes,
            true_modes=labels["confirmatory"],
        )
        if summary.dri is not None:
            gru_dri.append(float(summary.dri))
    return {
        "event_dri": event_summary.dri,
        "gru_dri": float(np.mean(gru_dri)) if gru_dri else None,
        "gru_seed_standard_deviation": (
            float(np.std(gru_dri, ddof=1)) if len(gru_dri) > 1 else 0.0
        ),
    }


def _label_shuffle_control(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    partner_indices: Mapping[str, int],
    losses: np.ndarray,
    suite: OfficialCheckpointAuditSuiteV2,
    layout_id: str,
) -> dict[str, Any]:
    calibration = list(split_rows["calibration"])
    confirmatory = list(split_rows["confirmatory"])
    train_histories = [_event_tokens(row, "pre_commitment") for row in calibration]
    confirm_histories = [_event_tokens(row, "pre_commitment") for row in confirmatory]
    labels = np.asarray(
        [partner_indices[str(row["partner_id"])] for row in calibration], dtype=np.int64
    )
    confirm_labels = [partner_indices[str(row["partner_id"])] for row in confirmatory]
    prior = np.full(len(partner_indices), 1.0 / len(partner_indices), dtype=np.float64)
    rng = np.random.default_rng(7301 + sum(layout_id.encode()))
    values: list[float] = []
    for _ in range(suite.estimator.label_shuffle_repeats):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        model = fit_event_posterior(
            train_histories,
            shuffled.tolist(),
            len(partner_indices),
            smoothing=suite.estimator.event_laplace_alpha,
        )
        posteriors = predict_event_posteriors(model, confirm_histories, prior.tolist())
        _assign_prior_to_censored(posteriors, confirmatory, prior.tolist())
        summary = summarize_posteriors(
            prior.tolist(),
            losses.tolist(),
            posteriors.tolist(),
            true_modes=confirm_labels,
        )
        if summary.dri is not None:
            values.append(float(summary.dri))
    return {
        "layout_id": layout_id,
        "repeats": len(values),
        "mean_dri": float(np.mean(values)) if values else float("inf"),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _assign_prior_to_censored(
    posteriors: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
    prior: Sequence[float],
) -> None:
    prior_row = np.asarray(prior, dtype=np.float64)
    for index, episode in enumerate(episodes):
        if not bool(episode.get("commitment_reached", False)):
            posteriors[index] = prior_row


def _event_tokens(episode: Mapping[str, Any], prefix: int | str) -> tuple[str, ...]:
    steps = _prefix_steps(episode, prefix)
    tokens: list[str] = []
    for step in steps:
        tokens.append(f"ego_action:{step['ego_action']}")
        if step.get("visible_partner_action") is not None:
            tokens.append(f"partner_action:{step['visible_partner_action']}")
        tokens.extend(str(item) for item in step.get("events", ()))
        reward = float(step.get("reward", 0.0))
        reward_token = (
            "reward:positive" if reward > 0 else "reward:negative" if reward < 0 else "reward:zero"
        )
        tokens.append(reward_token)
    return tuple(tokens) or ("zero_step",)


def _gru_features(episode: Mapping[str, Any], prefix: int | str) -> np.ndarray:
    steps = _prefix_steps(episode, prefix)
    width = int(episode["observation_width"])
    rows: list[np.ndarray] = []
    for step in steps:
        observation = np.asarray(step["ego_observation"], dtype=np.float32)
        if observation.shape != (width,):
            raise ValueError("official ego-observation width changed within a trace")
        extras = np.asarray(
            [
                float(step["ego_action"]),
                float(step.get("visible_partner_action", -1)),
                float(step.get("reward", 0.0)),
                float(step["step"]) / 400.0,
                1.0,
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate((observation, extras)))
    return np.stack(rows) if rows else np.zeros((1, width + 5), dtype=np.float32)


def _prefix_steps(episode: Mapping[str, Any], prefix: int | str) -> list[Mapping[str, Any]]:
    steps = list(episode["steps"])
    if isinstance(prefix, int):
        return steps[:prefix]
    if prefix == "pre_commitment":
        commitment = episode.get("commitment_step")
        return [] if commitment is None else [step for step in steps if step["step"] < commitment]
    if prefix == "eventual":
        delivery = episode.get("first_delivery_step")
        return steps if delivery is None else [step for step in steps if step["step"] <= delivery]
    raise ValueError(f"unknown official trace prefix: {prefix!r}")


def _empirical_history_tv(left: Sequence[Sequence[str]], right: Sequence[Sequence[str]]) -> float:
    """Empirical TV between the two mode-conditioned history distributions."""

    if not left or not right:
        return 0.0
    left_counts = Counter(tuple(history) for history in left)
    right_counts = Counter(tuple(history) for history in right)
    support = left_counts.keys() | right_counts.keys()
    return float(
        0.5
        * sum(
            abs(left_counts[item] / len(left) - right_counts[item] / len(right)) for item in support
        )
    )


def _rahman_brdiv(matrix: np.ndarray) -> float:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("co-trained response BRDiv requires a square cross-play matrix")
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for left in range(n):
        for right in range(n):
            if left != right:
                total += (matrix[left, left] - matrix[left, right]) + (
                    matrix[left, left] - matrix[right, left]
                )
    return float(total / (2 * n * (n - 1)))


def _official_normalize(features: np.ndarray) -> np.ndarray:
    maxima = np.max(np.abs(features), axis=0)
    return features / (maxima + 1e-3)


def _gram_determinant(features: np.ndarray) -> float:
    gram = features @ features.T
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0 or not np.isfinite(logdet):
        return 0.0
    return float(np.exp(logdet))


def _assert_disjoint_episode_keys(*groups: Sequence[Mapping[str, Any]]) -> None:
    keys = [{int(item["environment_key"]) for item in group} for group in groups]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise ValueError("official calibration, validation, and confirmatory keys overlap")


def _validate_result(result: Mapping[str, Any], operation: str) -> None:
    if result.get("operation") != operation:
        raise ValueError(f"unexpected official runtime operation: {result.get('operation')!r}")
    if result.get("policy_training_performed") is not False:
        raise ValueError("official runtime result does not prove inference-only execution")
    if operation != "official_parity" and result.get("partner_deployment") != "stochastic":
        raise ValueError("scientific rollouts must preserve official stochastic partner sampling")


def _result_paths(source: str | Path | Iterable[str | Path], *, kind: str) -> tuple[Path, ...]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return tuple(sorted(path.rglob(f"*--{kind}--*.json.gz")))
        if path.suffix == ".json":
            plan = _plan(path)
            return tuple(Path(shard.result_path) for shard in plan.shards if shard.kind == kind)
        return (path,)
    return tuple(Path(item) for item in source)


def _read_result(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"official result is not a JSON object: {path}")
    return value


def _optional_bool(value: Any, transform: Any | None = None) -> bool | None:
    if value is None:
        return None
    return bool(transform(value) if transform is not None else value)


def _suite(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> OfficialCheckpointAuditSuiteV2:
    return _suite_with_path(suite)[0]


def _suite_with_path(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> tuple[OfficialCheckpointAuditSuiteV2, Path | None]:
    if isinstance(suite, OfficialCheckpointAuditSuiteV2):
        return suite, None
    path = Path(suite).resolve()
    return load_official_checkpoint_suite(path), path


def _trace_index(index: OfficialTraceIndex | str | Path) -> OfficialTraceIndex:
    if isinstance(index, OfficialTraceIndex):
        return index
    return OfficialTraceIndex.model_validate(_read_json(Path(index)))


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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
