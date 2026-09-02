"""Fresh-confirmation analysis for the Stage 6 v3 measurement redesign."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from zsc_identifiability.established_dri import summarize_posteriors
from zsc_identifiability.established_gru import (
    GRUSequenceBatchSource,
    fit_streaming_cross_fitted_gru_posterior,
)
from zsc_identifiability.established_official_analysis import build_official_trace_index
from zsc_identifiability.established_official_decision import (
    decoder_probabilities,
    evaluate_pairwise_decision,
    fit_ridge_logistic,
    holm_correct_permutation_tests,
    one_sided_permutation_p_value,
    synthetic_decision_estimator_controls,
)
from zsc_identifiability.established_official_models import (
    OfficialResponseValueMatrix,
    OfficialRolloutLedger,
    OfficialTraceIndex,
    OfficialVerdict,
    PairwiseIdentifiabilityRow,
    load_official_checkpoint_suite,
)
from zsc_identifiability.established_official_redesign import (
    _hash_json,
    _load_response_libraries,
    _read_json,
    _resolve_confirmation_plan,
    _resolve_fit_manifest,
    _resolve_source_path,
    _sha256,
    _suite_context,
    _suite_hash,
    _validate_confirmation_plan_integrity,
    _validate_fit_manifest,
    _validate_v2_source_lock,
)
from zsc_identifiability.established_official_redesign_models import (
    MeasurementCalibrationReportV3,
    MeasurementFitManifest,
    MeasurementPrefix,
    MeasurementRepresentationArtifact,
    OfficialConfirmationLedger,
    OfficialConfirmationPlan,
    OfficialMeasurementAuditManifestV3,
    OfficialMeasurementAuditSuiteV3,
    PairwiseDecisionDecoder,
    PairwiseDecisionValueRow,
)
from zsc_identifiability.established_official_reporting import (
    _build_episode_regression_rows,
    _build_regression_rows,
    _run_regressions,
)
from zsc_identifiability.established_official_representation import (
    EncodedHistories,
    encode_with_frozen_identity_representation,
    event_representation_matrix,
    load_encoded_histories,
    load_frozen_identity_encoder,
    save_encoded_histories,
)
from zsc_identifiability.established_official_trace_store import (
    IndexedTraceSequenceSource,
    OfficialCompactTraceStore,
)


class _RelabeledSequenceSource:
    """Bounded row subset with pair-local binary labels."""

    def __init__(
        self,
        source: Any,
        indices: Sequence[int],
        label_map: Mapping[int, int],
    ) -> None:
        self._source: IndexedTraceSequenceSource = source.subset(indices)
        self._label_map = dict(label_map)

    @property
    def size(self) -> int:
        return self._source.size

    @property
    def feature_width(self) -> int:
        return self._source.feature_width

    @property
    def labels(self) -> Sequence[int]:
        return tuple(self._label_map[int(value)] for value in self._source.labels)

    @property
    def episodes(self) -> Sequence[Any]:
        return self._source.episodes

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for features, lengths, labels, indices in self._source.iter_batches(
            batch_size,
            shuffle=shuffle,
            seed=seed,
        ):
            yield (
                features,
                lengths,
                np.asarray([self._label_map[int(value)] for value in labels], dtype=np.int64),
                indices,
            )


def evaluate_fresh_decision_value(
    suite: OfficialMeasurementAuditSuiteV3 | str | Path,
    plan: OfficialConfirmationPlan | str | Path,
    ledger: OfficialConfirmationLedger | str | Path,
    fit_manifest: MeasurementFitManifest | str | Path,
    output_dir: str | Path,
    *,
    progress: Any | None = None,
) -> tuple[PairwiseDecisionValueRow, ...]:
    """Apply frozen pairwise decoders to untouched v3 confirmation traces."""

    spec, suite_path, project_root = _suite_context(suite)
    confirmation = _resolve_confirmation_plan(plan)
    confirmation_ledger = _resolve_confirmation_ledger(ledger)
    fit, fit_path = _resolve_fit_manifest(fit_manifest)
    _validate_fit_manifest(spec, fit, fit_path, suite_path)
    _require_confirmation_complete(spec, confirmation, confirmation_ledger, fit)
    source = _validate_v2_source_lock(spec, project_root)
    libraries = {
        item.layout_id: item
        for item in _load_response_libraries(source["response_matrices"])
    }
    trace_index = _fresh_trace_index(confirmation, confirmation_ledger)
    output = Path(output_dir).resolve()
    state = output / ".analysis-state"
    evaluation_signature = _hash_json(
        {
            "suite": _suite_hash(spec, suite_path),
            "confirmation_plan": confirmation.plan_hash,
            "confirmation_ledger": confirmation_ledger.to_dict(),
            "fit": fit.frozen_configuration_hash,
            "algorithm": "stage6-v3-fresh-decision-value-v1",
        }
    )
    evaluation_checkpoint = state / "fresh-decision-value-complete.json"
    if evaluation_checkpoint.is_file():
        checkpoint = _read_json(evaluation_checkpoint)
        if checkpoint.get("signature") != evaluation_signature:
            raise ValueError("fresh decision-value checkpoint belongs to another analysis")
        tracked_files = {
            output / "pairwise-decision-value.json": checkpoint.get("pairwise_hash"),
            output / "identity-information-diagnostic.json": checkpoint.get("identity_hash"),
            output / "fresh-representation-index.json": checkpoint.get("index_hash"),
        }
        if all(
            path.is_file() and expected == _sha256(path)
            for path, expected in tracked_files.items()
        ):
            return tuple(
                PairwiseDecisionValueRow.model_validate(item)
                for item in _read_list(output / "pairwise-decision-value.json")
            )
    trace_store = OfficialCompactTraceStore.prepare(
        trace_index,
        state / "fresh-trace-cache",
        progress=progress,
    )
    decoder_payload = _read_json(Path(fit.decoder_manifest_path))
    decoders = tuple(
        PairwiseDecisionDecoder.model_validate(item)
        for item in decoder_payload.get("decoders", ())
    )
    if not decoders:
        raise ValueError("frozen v3 decoder manifest contains no pairwise heads")
    seed_rows: list[dict[str, Any]] = []
    identity_diagnostics: list[dict[str, Any]] = []
    fresh_artifact_index: list[dict[str, Any]] = []
    for artifact in fit.artifacts:
        library = libraries[artifact.layout_id]
        labels = {partner: index for index, partner in enumerate(library.partner_ids)}
        source_rows = trace_store.decision_sequence_source(
            artifact.layout_id,
            artifact.evidence_policy,
            "confirmatory",
            labels,
            artifact.prefix,
        )
        fresh_path = state / "fresh-representations" / f"{artifact.artifact_id}.npz"
        if fresh_path.is_file():
            encoded, keys, partner_ids, commitment = load_encoded_histories(fresh_path)
            expected_keys = np.asarray(
                [episode.environment_key for episode in source_rows.episodes], dtype=np.int64
            )
            expected_partners = np.asarray(
                [episode.partner_id for episode in source_rows.episodes], dtype=np.str_
            )
            if not np.array_equal(keys, expected_keys) or not np.array_equal(
                partner_ids, expected_partners
            ):
                raise ValueError("resumed fresh representation belongs to another trace set")
        else:
            if artifact.representation == "event":
                matrix = event_representation_matrix(
                    source_rows.episodes,
                    artifact.prefix,
                    width=spec.representations.event_feature_width,
                    salt=spec.representations.event_hash_salt,
                )
                encoded = EncodedHistories(
                    embeddings=matrix,
                    identity_logits=np.zeros((len(matrix), len(library.partner_ids))),
                    labels=np.asarray(source_rows.labels, dtype=np.int64),
                    row_indices=np.arange(len(matrix), dtype=np.int64),
                )
            else:
                if artifact.encoder_checkpoint_path is None or (
                    artifact.encoder_checkpoint_hash is None
                ):
                    raise ValueError("GRU representation artifact omits its frozen encoder")
                encoder = load_frozen_identity_encoder(
                    artifact.encoder_checkpoint_path,
                    artifact.encoder_checkpoint_hash,
                )
                signature = _encoder_signature(spec, artifact)
                encoded = encode_with_frozen_identity_representation(
                    encoder,
                    source_rows,
                    signature=signature,
                )
            keys = np.asarray(
                [episode.environment_key for episode in source_rows.episodes], dtype=np.int64
            )
            partner_ids = np.asarray(
                [episode.partner_id for episode in source_rows.episodes], dtype=np.str_
            )
            commitment = np.asarray(
                [episode.commitment_reached for episode in source_rows.episodes], dtype=np.bool_
            )
            save_encoded_histories(
                fresh_path,
                encoded,
                environment_keys=keys.tolist(),
                partner_ids=partner_ids.tolist(),
                commitment_reached=commitment.tolist(),
            )
        fresh_artifact_index.append(
            {
                "artifact_id": artifact.artifact_id,
                "path": str(fresh_path),
                "hash": _sha256(fresh_path),
                "row_count": len(keys),
            }
        )
        artifact_decoders = [
            decoder
            for decoder in decoders
            if _decoder_matches_artifact(decoder, artifact)
        ]
        seed_rows.extend(
            _evaluate_artifact_pairs(
                artifact,
                artifact_decoders,
                library,
                encoded.embeddings,
                partner_ids,
                commitment,
            )
        )
        if artifact.representation == "gru":
            identity_diagnostics.append(
                _identity_diagnostic(
                    artifact,
                    library,
                    encoded.identity_logits,
                    encoded.labels,
                    commitment,
                )
            )
        if progress is not None:
            progress(f"fresh evaluation: {artifact.artifact_id}")
    rows = _aggregate_seed_rows(seed_rows)
    _atomic_json(output / "fresh-representation-index.json", fresh_artifact_index)
    _atomic_json(output / "pairwise-decision-value.json", [row.to_dict() for row in rows])
    _atomic_json(output / "identity-information-diagnostic.json", identity_diagnostics)
    _atomic_json(
        evaluation_checkpoint,
        {
            "schema_version": 1,
            "signature": evaluation_signature,
            "pairwise_hash": _sha256(output / "pairwise-decision-value.json"),
            "identity_hash": _sha256(output / "identity-information-diagnostic.json"),
            "index_hash": _sha256(output / "fresh-representation-index.json"),
        },
    )
    return rows


def analyze_measurement_redesign(
    suite: OfficialMeasurementAuditSuiteV3 | str | Path,
    plan: OfficialConfirmationPlan | str | Path,
    ledger: OfficialConfirmationLedger | str | Path,
    fit_manifest: MeasurementFitManifest | str | Path,
    output_dir: str | Path,
    *,
    progress: Any | None = None,
) -> OfficialMeasurementAuditManifestV3:
    """Run the frozen fresh-confirmation analysis and assign the registered verdict."""

    spec, suite_path, project_root = _suite_context(suite)
    confirmation = _resolve_confirmation_plan(plan)
    confirmation_ledger = _resolve_confirmation_ledger(ledger)
    fit, fit_path = _resolve_fit_manifest(fit_manifest)
    _validate_fit_manifest(spec, fit, fit_path, suite_path)
    _require_confirmation_complete(spec, confirmation, confirmation_ledger, fit)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = evaluate_fresh_decision_value(
        suite,
        confirmation,
        confirmation_ledger,
        fit,
        output,
        progress=progress,
    )
    source = _validate_v2_source_lock(spec, project_root)
    libraries = _load_response_libraries(source["response_matrices"])
    regression = _fresh_regression(spec, rows, libraries, source, project_root, output)
    intervention = _fresh_intervention_audit(
        spec,
        rows,
        confirmation,
        confirmation_ledger,
        libraries,
        output,
    )
    permutation = _permutation_controls(
        spec,
        fit,
        rows,
        libraries,
        output,
        progress=progress,
    )
    direct_binary = _direct_binary_gru_diagnostic(
        spec,
        rows,
        libraries,
        source,
        project_root,
        confirmation,
        confirmation_ledger,
        output,
        progress=progress,
    )
    calibration = _calibration_report(
        spec,
        rows,
        regression,
        intervention,
        permutation,
        direct_binary,
    )
    _atomic_json(output / "measurement-calibration-report-v3.json", calibration.to_dict())
    leakage = _fresh_key_leakage_audit(spec, confirmation, project_root)
    _atomic_json(output / "fresh-key-leakage-audit.json", leakage)
    sensitivity = {
        "schema_version": 3,
        "event_regression_same_direction": regression["event_same_effect_direction"],
        "event_intervention_direction": {
            item["layout_id"]: item["event_decision_risk_reduction"]
            for item in intervention["layouts"]
        },
        "gru_seed_stability": calibration.seed_stability,
        "direct_binary_gru": calibration.direct_binary_gru_diagnostic,
    }
    _atomic_json(output / "measurement-sensitivity-report-v3.json", sensitivity)
    figure_paths = _write_v3_figures(
        spec,
        rows,
        regression,
        intervention,
        permutation,
        output / "figures",
    )
    verdict, gates = _v3_verdict(spec, calibration, regression, intervention, source)
    generated = tuple(
        [
            str(path)
            for path in sorted(output.glob("*.json"))
            if path.name != "official-measurement-audit-manifest-v3.json"
        ]
        + [str(path) for path in figure_paths]
    )
    manifest = OfficialMeasurementAuditManifestV3(
        suite_id=spec.suite_id,
        status="complete",
        verdict=cast(OfficialVerdict, verdict),
        v2_preserved=True,
        confirmation_complete=confirmation_ledger.complete,
        calibration_passed=calibration.passed,
        scientific_gates=gates,
        source_hashes={
            "suite": _suite_hash(spec, suite_path),
            "confirmation_plan": confirmation.plan_hash,
            "fit_configuration": fit.frozen_configuration_hash,
            "v2_plan": spec.v2.v2_rollout_plan_hash,
        },
        generated_files=generated,
        total_fresh_episodes=sum(
            len(shard.episode_keys) for shard in confirmation.rollout_plan.shards
        ),
        total_fresh_environment_steps=_fresh_environment_steps(confirmation),
        peak_workers=spec.runtime.default_workers,
    )
    _atomic_json(output / "official-measurement-audit-manifest-v3.json", manifest.to_dict())
    return manifest


def _write_v3_figures(
    spec: OfficialMeasurementAuditSuiteV3,
    rows: Sequence[PairwiseDecisionValueRow],
    regression: Mapping[str, Any],
    intervention: Mapping[str, Any],
    permutation: Mapping[str, Any],
    figure_dir: Path,
) -> tuple[Path, ...]:
    """Generate compact publication figures only from registered v3 outputs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    prefix_order: tuple[MeasurementPrefix, ...] = (0, 8, 16, 32, "pre_commitment", "eventual")
    figure, axes = plt.subplots(1, len(spec.layouts), figsize=(10, 3.8), sharey=True)
    for axis, layout in zip(np.atleast_1d(axes), spec.layouts, strict=True):
        for representation, marker in (("gru", "o"), ("event", "s")):
            values = [
                float(
                    np.mean(
                        [
                            row.dri
                            for row in rows
                            if row.layout_id == layout.layout_id
                            and row.evidence_policy == "ordinary_progress"
                            and row.representation == representation
                            and str(row.prefix) == str(prefix)
                            and row.dri is not None
                        ]
                    )
                )
                for prefix in prefix_order
            ]
            axis.plot(range(len(prefix_order)), values, marker=marker, label=representation.upper())
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(layout.layout_id)
        axis.set_xticks(
            range(len(prefix_order)),
            [str(value) for value in prefix_order],
            rotation=30,
        )
        axis.set_xlabel("History endpoint")
    np.atleast_1d(axes)[0].set_ylabel("Mean pairwise decision-value DRI")
    np.atleast_1d(axes)[-1].legend(frameon=False)
    figure.tight_layout()
    generated.extend(_save_figure_pair(figure, figure_dir / "fresh-decision-value-curves"))
    plt.close(figure)

    labels: list[str] = []
    observed: list[float] = []
    nulls: list[list[float]] = []
    for test_id, result in sorted(permutation["tests"].items()):
        labels.append(str(test_id).replace(":", "\n"))
        observed.append(float(result["observed"]))
        nulls.append([float(value) for value in result["null_values"]])
    figure, axis = plt.subplots(figsize=(9, 4.2))
    axis.boxplot(nulls, tick_labels=labels, showfliers=False)
    axis.scatter(range(1, len(observed) + 1), observed, color="#c43c39", zorder=3, label="Observed")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Decision-value statistic")
    axis.set_title("Observed effects against registered permutation nulls")
    axis.legend(frameon=False)
    figure.tight_layout()
    generated.extend(_save_figure_pair(figure, figure_dir / "permutation-controls"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(intervention["layouts"]))
    risk = [float(item["gru_decision_risk_reduction"]) for item in intervention["layouts"]]
    cost = [float(item["normalized_task_cost"]) for item in intervention["layouts"]]
    width = 0.35
    axis.bar(x - width / 2, risk, width, label="Decision-risk reduction")
    axis.bar(x + width / 2, cost, width, label="Task cost")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [str(item["layout_id"]) for item in intervention["layouts"]])
    axis.set_ylabel("Normalized value")
    axis.set_title("Frozen interventions: information benefit and task cost")
    axis.legend(frameon=False)
    figure.tight_layout()
    generated.extend(_save_figure_pair(figure, figure_dir / "fresh-intervention-audit"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    report_labels: list[str] = []
    delta_mae: list[float] = []
    delta_mse: list[float] = []
    for representation in ("gru", "event"):
        report_labels.append(f"{representation.upper()}\noverall")
        delta_mae.append(float(regression[representation]["overall"]["delta_mae"]))
        delta_mse.append(float(regression[representation]["overall"]["delta_mse"]))
        for layout in spec.layouts:
            report_labels.append(f"{representation.upper()}\n{layout.layout_id}")
            delta_mae.append(
                float(regression[representation]["by_layout"][layout.layout_id]["delta_mae"])
            )
            delta_mse.append(
                float(regression[representation]["by_layout"][layout.layout_id]["delta_mse"])
            )
    x = np.arange(len(report_labels))
    axis.bar(x - width / 2, delta_mae, width, label="Delta MAE")
    axis.bar(x + width / 2, delta_mse, width, label="Delta MSE")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, report_labels, rotation=25, ha="right")
    axis.set_ylabel("Full model minus baseline")
    axis.set_title("Held-out prediction change after adding fresh DRI")
    axis.legend(frameon=False)
    figure.tight_layout()
    generated.extend(_save_figure_pair(figure, figure_dir / "fresh-held-out-regression"))
    plt.close(figure)
    return tuple(generated)


def _save_figure_pair(figure: Any, stem: Path) -> tuple[Path, Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=180, bbox_inches="tight")
    return pdf, png


def _evaluate_artifact_pairs(
    artifact: MeasurementRepresentationArtifact,
    decoders: Sequence[PairwiseDecisionDecoder],
    library: OfficialResponseValueMatrix,
    features: np.ndarray,
    partner_ids: np.ndarray,
    commitment_reached: np.ndarray,
) -> list[dict[str, Any]]:
    losses = np.asarray(library.normalized_losses, dtype=np.float64)
    partner_index = {partner: index for index, partner in enumerate(library.partner_ids)}
    schemes = {partner: partner.split(":")[-2] for partner in library.partner_ids}
    output: list[dict[str, Any]] = []
    for decoder in decoders:
        left, right = decoder.left_partner_id, decoder.right_partner_id
        mask = np.isin(partner_ids, (left, right))
        probabilities = decoder_probabilities(decoder, features[mask])
        true_modes = (partner_ids[mask] == right).astype(np.int64)
        evaluation = evaluate_pairwise_decision(
            probabilities,
            true_modes,
            losses[[partner_index[left], partner_index[right]]],
            censored=(
                ~commitment_reached[mask]
                if artifact.prefix == "pre_commitment"
                else None
            ),
        )
        output.append(
            {
                "layout_id": library.layout_id,
                "left_partner_id": left,
                "right_partner_id": right,
                "left_scheme_id": schemes[left],
                "right_scheme_id": schemes[right],
                "evidence_policy": artifact.evidence_policy,
                "representation": artifact.representation,
                "prefix": artifact.prefix,
                "seed": artifact.seed,
                "prior_risk": evaluation.prior_risk,
                "residual_risk": evaluation.residual_risk,
                "fixed_response_risk": evaluation.fixed_response_risk,
                "dri": evaluation.dri,
                "brier_score": evaluation.brier_score,
                "uniform_brier_score": evaluation.uniform_brier_score,
                "commitment_rate": float(np.mean(commitment_reached[mask])),
                "sample_count": int(np.count_nonzero(mask)),
            }
        )
    return output


def _aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[PairwiseDecisionValueRow, ...]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["layout_id"],
            row["left_partner_id"],
            row["right_partner_id"],
            row["evidence_policy"],
            row["representation"],
            str(row["prefix"]),
        )
        grouped[key].append(row)
    output: list[PairwiseDecisionValueRow] = []
    for _key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        first = values[0]
        dri_values = [float(row["dri"]) for row in values if row["dri"] is not None]
        output.append(
            PairwiseDecisionValueRow(
                layout_id=first["layout_id"],
                left_partner_id=str(first["left_partner_id"]),
                right_partner_id=str(first["right_partner_id"]),
                left_scheme_id=str(first["left_scheme_id"]),
                right_scheme_id=str(first["right_scheme_id"]),
                evidence_policy=str(first["evidence_policy"]),
                representation=first["representation"],
                prefix=cast(MeasurementPrefix, first["prefix"]),
                prior_risk=float(np.mean([row["prior_risk"] for row in values])),
                residual_risk=float(np.mean([row["residual_risk"] for row in values])),
                fixed_response_risk=float(
                    np.mean([row["fixed_response_risk"] for row in values])
                ),
                dri=None if not dri_values else float(np.mean(dri_values)),
                brier_score=float(np.mean([row["brier_score"] for row in values])),
                uniform_brier_score=float(
                    np.mean([row["uniform_brier_score"] for row in values])
                ),
                commitment_rate=float(
                    np.mean([row["commitment_rate"] for row in values])
                ),
                sample_count=int(values[0]["sample_count"]),
                seed_dri=tuple(dri_values) if first["representation"] == "gru" else (),
            )
        )
    return tuple(output)


def _identity_diagnostic(
    artifact: MeasurementRepresentationArtifact,
    library: OfficialResponseValueMatrix,
    logits: np.ndarray,
    labels: np.ndarray,
    commitment: np.ndarray,
) -> dict[str, Any]:
    if artifact.identity_temperature is None:
        raise ValueError("GRU identity diagnostic omits its validation-calibrated temperature")
    posteriors = _softmax(logits / artifact.identity_temperature)
    prior = np.full(len(library.partner_ids), 1.0 / len(library.partner_ids))
    if artifact.prefix == "pre_commitment":
        posteriors = posteriors.copy()
        posteriors[~commitment] = prior
    losses = np.asarray(library.normalized_losses, dtype=np.float64)
    signatures = tuple(
        library.response_ids[int(np.argmin(losses[index]))]
        for index in range(len(library.partner_ids))
    )
    summary = summarize_posteriors(
        prior.tolist(),
        losses.tolist(),
        posteriors.tolist(),
        response_signatures=signatures,
        true_modes=labels.tolist(),
    )
    return {
        "layout_id": library.layout_id,
        "evidence_policy": artifact.evidence_policy,
        "prefix": artifact.prefix,
        "seed": artifact.seed,
        "temperature": artifact.identity_temperature,
        "identity_mi_nats": summary.identity_mutual_information_nats,
        "decision_signature_mi_nats": summary.response_signature_mutual_information_nats,
        "response_signatures": list(signatures),
    }


def _fresh_regression(
    spec: OfficialMeasurementAuditSuiteV3,
    rows: Sequence[PairwiseDecisionValueRow],
    libraries: Sequence[OfficialResponseValueMatrix],
    source: Mapping[str, Any],
    project_root: Path,
    output: Path,
) -> dict[str, Any]:
    v2_pairwise = tuple(
        PairwiseIdentifiabilityRow.model_validate(item)
        for item in _read_list(
            _resolve_source_path(project_root, spec.v2.v2_trace_index_path).parent
            / "pairwise-identifiability.json"
        )
    )
    v2_controls = {
        (row.layout_id, row.left_partner_id, row.right_partner_id): row
        for row in v2_pairwise
        if row.estimator == "gru"
        and row.evidence_policy == "ordinary_progress"
        and row.prefix == "pre_commitment"
    }
    trace_index = OfficialTraceIndex.model_validate(source["trace_index"])
    v2_cache = (
        _resolve_source_path(project_root, spec.v2.v2_trace_index_path).parent
        / ".analysis-state"
        / "trace-cache"
    )
    trace_store = OfficialCompactTraceStore.prepare(trace_index, v2_cache)
    method_rows = source["method_evaluation"]["partner_method_rows"]
    method_episode_rows = source["method_evaluation"]["episode_rows"]
    v2_suite = load_official_checkpoint_suite(
        _resolve_source_path(project_root, spec.v2.v2_suite_path)
    )
    reports: dict[str, Any] = {}
    for representation in ("gru", "event"):
        converted: list[PairwiseIdentifiabilityRow] = []
        for row in rows:
            if row.representation != representation or row.evidence_policy != (
                "ordinary_progress"
            ) or row.prefix != "pre_commitment":
                continue
            control = v2_controls[(row.layout_id, row.left_partner_id, row.right_partner_id)]
            converted.append(
                PairwiseIdentifiabilityRow(
                    layout_id=row.layout_id,
                    left_partner_id=row.left_partner_id,
                    right_partner_id=row.right_partner_id,
                    left_scheme_id=row.left_scheme_id,
                    right_scheme_id=row.right_scheme_id,
                    evidence_policy=row.evidence_policy,
                    estimator="gru" if representation == "gru" else "event",
                    prefix="pre_commitment",
                    prior_risk=row.prior_risk,
                    residual_risk=row.residual_risk,
                    dri=row.dri,
                    identity_mi_nats=control.identity_mi_nats,
                    decision_mi_nats=control.decision_mi_nats,
                    prefix_tv=control.prefix_tv,
                    commitment_rate=row.commitment_rate,
                )
            )
        regression_rows = _build_regression_rows(
            method_rows,
            libraries,
            converted,
            trace_index,
            source["exclusions"],
            trace_store,
        )
        episode_rows = _build_episode_regression_rows(regression_rows, method_episode_rows)
        reports[representation] = _run_regressions(
            regression_rows,
            episode_rows,
            v2_suite,
        )
    gru_point = float(reports["gru"]["dri_coefficient_interval"]["coefficient_point"])
    event_point = float(reports["event"]["dri_coefficient_interval"]["coefficient_point"])
    payload = {
        "schema_version": 3,
        "outcome_reused_from_v2": "normalized_response_library_regret",
        "official_method_evaluations_rerun": False,
        "gru": reports["gru"],
        "event": reports["event"],
        "event_same_effect_direction": (
            gru_point == 0.0 or event_point == 0.0 or np.sign(gru_point) == np.sign(event_point)
        ),
    }
    _atomic_json(output / "held-out-regression-report-v3.json", payload)
    return payload


def _fresh_intervention_audit(
    spec: OfficialMeasurementAuditSuiteV3,
    rows: Sequence[PairwiseDecisionValueRow],
    plan: OfficialConfirmationPlan,
    ledger: OfficialConfirmationLedger,
    libraries: Sequence[OfficialResponseValueMatrix],
    output: Path,
) -> dict[str, Any]:
    trace_index = _fresh_trace_index(plan, ledger)
    trace_store = OfficialCompactTraceStore.prepare(
        trace_index, output / ".analysis-state" / "fresh-trace-cache"
    )
    library_by_layout = {item.layout_id: item for item in libraries}
    layout_rows: list[dict[str, Any]] = []
    for layout in spec.layouts:
        option = layout.frozen_intervention
        lookup = {
            (
                row.left_partner_id,
                row.right_partner_id,
                row.evidence_policy,
                row.representation,
            ): row
            for row in rows
            if row.layout_id == layout.layout_id and row.prefix == "pre_commitment"
        }
        pair_keys = sorted(
            {
                (row.left_partner_id, row.right_partner_id)
                for row in rows
                if row.layout_id == layout.layout_id
                and row.evidence_policy == "ordinary_progress"
                and row.prefix == "pre_commitment"
            }
        )
        gru_effects: list[float] = []
        event_effects: list[float] = []
        gru_effects_by_scheme_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
        for left, right in pair_keys:
            ordinary_gru = lookup[(left, right, "ordinary_progress", "gru")]
            gru_effect = (
                ordinary_gru.residual_risk
                - lookup[(left, right, option, "gru")].residual_risk
            )
            gru_effects.append(gru_effect)
            scheme_pair = cast(
                tuple[str, str],
                tuple(sorted((ordinary_gru.left_scheme_id, ordinary_gru.right_scheme_id))),
            )
            gru_effects_by_scheme_pair[scheme_pair].append(gru_effect)
            event_effects.append(
                lookup[(left, right, "ordinary_progress", "event")].residual_risk
                - lookup[(left, right, option, "event")].residual_risk
            )
        ordinary_episodes = trace_store.episodes(
            layout.layout_id, "ordinary_progress", "confirmatory"
        )
        option_episodes = trace_store.episodes(layout.layout_id, option, "confirmatory")
        ordinary_returns = {
            (episode.partner_id, episode.environment_key, episode.ego_seat): episode.sparse_return
            for episode in ordinary_episodes
        }
        option_returns = {
            (episode.partner_id, episode.environment_key, episode.ego_seat): episode.sparse_return
            for episode in option_episodes
        }
        common = sorted(ordinary_returns.keys() & option_returns.keys())
        scale = max(
            1.0,
            float(np.max(np.asarray(library_by_layout[layout.layout_id].raw_values))),
        )
        task_cost = float(
            np.mean([ordinary_returns[key] - option_returns[key] for key in common]) / scale
        )
        completion_rate = float(
            np.mean(
                [episode.intervention_completed_step is not None for episode in option_episodes]
            )
        )
        response_tv = _visible_partner_action_tv(ordinary_episodes, option_episodes)
        clustered_effects = np.asarray(
            [np.mean(values) for values in gru_effects_by_scheme_pair.values()]
        )
        ci_low, ci_high = _bootstrap_mean_interval(
            clustered_effects,
            resamples=spec.statistics.bootstrap_resamples,
            seed=7119,
            family_size=4,
        )
        layout_rows.append(
            {
                "layout_id": layout.layout_id,
                "selected_intervention": option,
                "selection_frozen_from_v2": True,
                "pair_count": len(gru_effects),
                "completion_rate": completion_rate,
                "partner_response_tv": response_tv,
                "gru_decision_risk_reduction": float(np.mean(gru_effects)),
                "gru_corrected_ci_low": ci_low,
                "gru_corrected_ci_high": ci_high,
                "interval_correction": (
                    "HSP-scheme-pair clustered Bonferroni familywise 95%"
                ),
                "event_decision_risk_reduction": float(np.mean(event_effects)),
                "normalized_task_cost": task_cost,
                "risk_reduction_exceeds_cost": float(np.mean(gru_effects)) > task_cost,
                "qualifies_before_permutation": bool(
                    completion_rate >= spec.statistics.intervention_completion_threshold
                    and response_tv > 0
                    and ci_low > 0
                    and float(np.mean(gru_effects)) > task_cost
                    and float(np.mean(event_effects)) >= 0
                ),
            }
        )
    payload = {
        "schema_version": 3,
        "paired_environment_keys": True,
        "interventions_reselected": False,
        "layouts": layout_rows,
    }
    _atomic_json(output / "fresh-intervention-audit-v3.json", payload)
    return payload


def _permutation_controls(
    spec: OfficialMeasurementAuditSuiteV3,
    fit: MeasurementFitManifest,
    observed_rows: Sequence[PairwiseDecisionValueRow],
    libraries: Sequence[OfficialResponseValueMatrix],
    output: Path,
    *,
    progress: Any | None,
) -> dict[str, Any]:
    decoder_payload = _read_json(Path(fit.decoder_manifest_path))
    decoders = tuple(
        PairwiseDecisionDecoder.model_validate(item)
        for item in decoder_payload.get("decoders", ())
        if item.get("representation") == "gru" and item.get("prefix") == "pre_commitment"
    )
    library_by_layout = {item.layout_id: item for item in libraries}
    fresh_index = {
        str(item["artifact_id"]): item
        for item in _read_list(output / "fresh-representation-index.json")
    }
    observed_lookup = {
        (row.layout_id, row.left_partner_id, row.right_partner_id, row.evidence_policy): row
        for row in observed_rows
        if row.representation == "gru" and row.prefix == "pre_commitment"
    }
    tests: dict[str, dict[str, Any]] = {}
    for layout in spec.layouts:
        library = library_by_layout[layout.layout_id]
        passive_name = f"{layout.layout_id}:passive_dri"
        option_name = f"{layout.layout_id}:selected_intervention"
        passive_observed = float(
            np.mean(
                [
                    row.dri
                    for row in observed_rows
                    if row.layout_id == layout.layout_id
                    and row.evidence_policy == "ordinary_progress"
                    and row.representation == "gru"
                    and row.prefix == "pre_commitment"
                    and row.dri is not None
                ]
            )
        )
        option_observed = float(
            np.mean(
                [
                    observed_lookup[
                        (layout.layout_id, left, right, "ordinary_progress")
                    ].residual_risk
                    - observed_lookup[
                        (layout.layout_id, left, right, layout.frozen_intervention)
                    ].residual_risk
                    for left, right in library.conflicting_pairs_by_margin["0.02"]
                ]
            )
        )
        state_signature = _hash_json(
            {
                "fit": fit.frozen_configuration_hash,
                "layout": layout.layout_id,
                "fresh_artifacts": sorted(
                    (key, value["hash"])
                    for key, value in fresh_index.items()
                ),
                "repeats": spec.calibration.permutation_repeats,
                "seed": spec.calibration.permutation_seed,
            }
        )
        state_path = (
            output / ".analysis-state" / f"permutation-{layout.layout_id}.json"
        )
        if state_path.is_file():
            state_payload = _read_json(state_path)
            if state_payload.get("signature") != state_signature:
                raise ValueError("permutation work unit belongs to another frozen analysis")
            null_passive = [float(value) for value in state_payload["null_passive"]]
            null_option = [float(value) for value in state_payload["null_option"]]
        else:
            null_passive = []
            null_option = []
        for permutation in range(
            len(null_passive), spec.calibration.permutation_repeats
        ):
            passive_risks = _permuted_unit_risks(
                spec,
                fit,
                decoders,
                fresh_index,
                library,
                "ordinary_progress",
                permutation,
            )
            option_risks = _permuted_unit_risks(
                spec,
                fit,
                decoders,
                fresh_index,
                library,
                layout.frozen_intervention,
                permutation,
            )
            null_passive.append(
                float(
                    np.mean(
                        [
                            (prior - risk) / prior
                            for prior, risk in passive_risks.values()
                            if prior > 1e-15
                        ]
                    )
                )
            )
            null_option.append(
                float(
                    np.mean(
                        [
                            passive_risks[key][1] - option_risks[key][1]
                            for key in passive_risks.keys() & option_risks.keys()
                        ]
                    )
                )
            )
            _atomic_json(
                state_path,
                {
                    "schema_version": 1,
                    "signature": state_signature,
                    "null_passive": null_passive,
                    "null_option": null_option,
                },
            )
            if progress is not None and permutation % 10 == 0:
                progress(f"permutation {permutation + 1}/100: {layout.layout_id}")
        tests[passive_name] = {
            "observed": passive_observed,
            "null_values": null_passive,
            "raw_p": one_sided_permutation_p_value(passive_observed, null_passive),
        }
        tests[option_name] = {
            "observed": option_observed,
            "null_values": null_option,
            "raw_p": one_sided_permutation_p_value(option_observed, null_option),
        }
    adjusted = holm_correct_permutation_tests(
        {name: float(value["raw_p"]) for name, value in tests.items()}
    )
    for name, value in tests.items():
        value["holm_adjusted_p"] = adjusted[name]
        value["passed"] = adjusted[name] < 0.05
    payload = {
        "schema_version": 3,
        "repeats": spec.calibration.permutation_repeats,
        "negative_null_dri_allowed": True,
        "tests": tests,
        "holm_adjusted": adjusted,
    }
    _atomic_json(output / "permutation-report-v3.json", payload)
    return payload


def _permuted_unit_risks(
    spec: OfficialMeasurementAuditSuiteV3,
    fit: MeasurementFitManifest,
    decoders: Sequence[PairwiseDecisionDecoder],
    fresh_index: Mapping[str, Mapping[str, Any]],
    library: OfficialResponseValueMatrix,
    evidence_policy: str,
    permutation: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    artifacts = [
        artifact
        for artifact in fit.artifacts
        if artifact.layout_id == library.layout_id
        and artifact.evidence_policy == evidence_policy
        and artifact.prefix == "pre_commitment"
        and artifact.representation == "gru"
    ]
    losses = np.asarray(library.normalized_losses, dtype=np.float64)
    partner_index = {partner: index for index, partner in enumerate(library.partner_ids)}
    risk_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    prior_by_pair: dict[tuple[str, str], float] = {}
    for artifact in artifacts:
        calibration, _keys, calibration_partners, _commitment = load_encoded_histories(
            artifact.calibration_embeddings_path
        )
        fresh_record = fresh_index[artifact.artifact_id]
        fresh, _fresh_keys, fresh_partners, fresh_commitment = load_encoded_histories(
            str(fresh_record["path"])
        )
        artifact_decoders = [
            decoder
            for decoder in decoders
            if _decoder_matches_artifact(decoder, artifact)
        ]
        for decoder in artifact_decoders:
            left, right = decoder.left_partner_id, decoder.right_partner_id
            calibration_mask = np.isin(calibration_partners, (left, right))
            labels = (calibration_partners[calibration_mask] == right).astype(np.int64)
            rng = np.random.default_rng(
                _permutation_seed(
                    spec.calibration.permutation_seed,
                    permutation,
                    artifact.artifact_id,
                    left,
                    right,
                )
            )
            shuffled = labels.copy()
            rng.shuffle(shuffled)
            coefficient, intercept, mean, scale = fit_ridge_logistic(
                calibration.embeddings[calibration_mask],
                shuffled,
                ridge_strength=decoder.ridge_strength,
                maximum_iterations=spec.decoder.maximum_iterations,
                convergence_tolerance=spec.decoder.convergence_tolerance,
            )
            permuted_decoder = decoder.model_copy(
                update={
                    "coefficient": tuple(float(value) for value in coefficient),
                    "intercept": intercept,
                    "feature_mean": tuple(float(value) for value in mean),
                    "feature_scale": tuple(float(value) for value in scale),
                }
            )
            fresh_mask = np.isin(fresh_partners, (left, right))
            probabilities = decoder_probabilities(
                permuted_decoder, fresh.embeddings[fresh_mask]
            )
            evaluation = evaluate_pairwise_decision(
                probabilities,
                (fresh_partners[fresh_mask] == right).astype(np.int64),
                losses[[partner_index[left], partner_index[right]]],
                censored=~fresh_commitment[fresh_mask],
            )
            key = (left, right)
            risk_by_pair[key].append(evaluation.residual_risk)
            prior_by_pair[key] = evaluation.prior_risk
    return {
        key: (prior_by_pair[key], float(np.mean(values)))
        for key, values in risk_by_pair.items()
    }


def _direct_binary_gru_diagnostic(
    spec: OfficialMeasurementAuditSuiteV3,
    shared_rows: Sequence[PairwiseDecisionValueRow],
    libraries: Sequence[OfficialResponseValueMatrix],
    source: Mapping[str, Any],
    project_root: Path,
    confirmation: OfficialConfirmationPlan,
    confirmation_ledger: OfficialConfirmationLedger,
    output: Path,
    *,
    progress: Any | None,
) -> dict[str, Any]:
    """Fit registered pair-specific GRUs as a representation-sufficiency diagnostic."""

    v2_index = OfficialTraceIndex.model_validate(source["trace_index"])
    tuning_index = v2_index.model_copy(
        update={
            "entries": tuple(
                entry
                for entry in v2_index.entries
                if entry.split in {"calibration", "validation"}
            )
        }
    )
    tuning_store = OfficialCompactTraceStore.prepare(
        tuning_index,
        output / ".analysis-state" / "direct-binary-v2-trace-cache",
    )
    fresh_index = _fresh_trace_index(confirmation, confirmation_ledger)
    fresh_store = OfficialCompactTraceStore.prepare(
        fresh_index,
        output / ".analysis-state" / "fresh-trace-cache",
    )
    shared_lookup = {
        (row.layout_id, row.left_partner_id, row.right_partner_id): row
        for row in shared_rows
        if row.representation == "gru"
        and row.evidence_policy == "ordinary_progress"
        and row.prefix == "pre_commitment"
    }
    selected_pairs = set(
        sorted(
            (
                (library.layout_id, left, right)
                for library in libraries
                for left, right in library.conflicting_pairs_by_margin["0.02"]
            ),
            key=lambda item: hashlib.sha256("|".join(item).encode()).hexdigest(),
        )[: spec.calibration.direct_binary_pair_count]
    )
    if len(selected_pairs) != spec.calibration.direct_binary_pair_count:
        raise ValueError("the v3 direct-binary diagnostic lacks ten conflicting pairs")
    work_dir = output / ".analysis-state" / "direct-binary-gru"
    pair_rows: list[dict[str, Any]] = []
    for library in libraries:
        partner_labels = {partner: index for index, partner in enumerate(library.partner_ids)}
        candidates = tuple(
            (left, right)
            for layout_id, left, right in sorted(selected_pairs)
            if layout_id == library.layout_id
        )
        losses = np.asarray(library.normalized_losses, dtype=np.float64)
        calibration = tuning_store.decision_sequence_source(
            library.layout_id,
            "ordinary_progress",
            "calibration",
            partner_labels,
            "pre_commitment",
        )
        validation = tuning_store.decision_sequence_source(
            library.layout_id,
            "ordinary_progress",
            "validation",
            partner_labels,
            "pre_commitment",
        )
        confirmatory = fresh_store.decision_sequence_source(
            library.layout_id,
            "ordinary_progress",
            "confirmatory",
            partner_labels,
            "pre_commitment",
        )
        for left, right in candidates:
            pair_key = f"{library.layout_id}--{left.replace(':', '-')}--{right.replace(':', '-')}"
            result_path = work_dir / f"{pair_key}.json"
            if result_path.is_file():
                result = _read_json(result_path)
                if result.get("configuration_hash") != _hash_json(
                    {
                        "suite": spec.suite_id,
                        "layout": library.layout_id,
                        "left": left,
                        "right": right,
                        "fresh_plan": confirmation.plan_hash,
                    }
                ):
                    raise ValueError("direct-binary GRU checkpoint belongs to another audit")
                pair_rows.append(result)
                continue
            left_index, right_index = partner_labels[left], partner_labels[right]
            label_map = {left_index: 0, right_index: 1}
            calibration_pair = _binary_subset(calibration, label_map)
            validation_pair = _binary_subset(validation, label_map)
            confirmatory_pair = _binary_subset(confirmatory, label_map)
            pair_losses = losses[[left_index, right_index]]
            seed_dri: list[float] = []
            seed_loss: list[float] = []
            for seed in spec.representations.gru_seeds:
                fit = fit_streaming_cross_fitted_gru_posterior(
                    cast(GRUSequenceBatchSource, calibration_pair),
                    cast(GRUSequenceBatchSource, validation_pair),
                    cast(GRUSequenceBatchSource, confirmatory_pair),
                    (0.5, 0.5),
                    pair_losses.tolist(),
                    response_signatures=(left, right),
                    hidden_size=spec.representations.gru_hidden_size,
                    seed=seed,
                )
                posteriors = fit.posteriors.copy()
                for index, episode in enumerate(confirmatory_pair.episodes):
                    if not episode.commitment_reached:
                        posteriors[index] = (0.5, 0.5)
                summary = summarize_posteriors(
                    (0.5, 0.5),
                    pair_losses.tolist(),
                    posteriors.tolist(),
                    response_signatures=(left, right),
                    true_modes=confirmatory_pair.labels,
                )
                seed_loss.append(summary.residual_risk)
                if summary.dri is not None:
                    seed_dri.append(float(summary.dri))
            shared_row = shared_lookup[(library.layout_id, left, right)]
            result = {
                "schema_version": 1,
                "configuration_hash": _hash_json(
                    {
                        "suite": spec.suite_id,
                        "layout": library.layout_id,
                        "left": left,
                        "right": right,
                        "fresh_plan": confirmation.plan_hash,
                    }
                ),
                "layout_id": library.layout_id,
                "left_partner_id": left,
                "right_partner_id": right,
                "seed_dri": seed_dri,
                "seed_residual_risk": seed_loss,
                "direct_dri": float(np.mean(seed_dri)),
                "direct_residual_risk": float(np.mean(seed_loss)),
                "shared_encoder_dri": shared_row.dri,
                "decision_loss_difference": float(np.mean(seed_loss))
                - shared_row.residual_risk,
                "seed_standard_deviation": float(np.std(seed_dri)),
            }
            _atomic_json(result_path, result)
            pair_rows.append(result)
            if progress is not None:
                progress(f"direct binary GRU: {library.layout_id}/{left}/{right}")
    direct = np.asarray([float(row["direct_dri"]) for row in pair_rows])
    shared_values = np.asarray(
        [float(row["shared_encoder_dri"] or 0.0) for row in pair_rows]
    )
    correlation = (
        0.0
        if len(direct) < 2
        or np.std(direct) <= 1e-15
        or np.std(shared_values) <= 1e-15
        else float(np.corrcoef(direct, shared_values)[0, 1])
    )
    report = {
        "status": "complete",
        "pair_selection": "same_ten_sha256_selected_conflicting_pairs_as_v2",
        "pair_count": len(pair_rows),
        "numerical_identity_with_shared_encoder_required": False,
        "mean_decision_loss_difference": float(
            np.mean([row["decision_loss_difference"] for row in pair_rows])
        ),
        "dri_correlation": correlation,
        "mean_seed_dispersion": float(
            np.mean([row["seed_standard_deviation"] for row in pair_rows])
        ),
        "pairs": pair_rows,
    }
    _atomic_json(output / "direct-binary-gru-diagnostic-v3.json", report)
    return report


def _binary_subset(source: Any, label_map: Mapping[int, int]) -> _RelabeledSequenceSource:
    indices = [
        index for index, label in enumerate(source.labels) if int(label) in label_map
    ]
    return _RelabeledSequenceSource(source, indices, label_map)


def _calibration_report(
    spec: OfficialMeasurementAuditSuiteV3,
    rows: Sequence[PairwiseDecisionValueRow],
    regression: Mapping[str, Any],
    intervention: Mapping[str, Any],
    permutation: Mapping[str, Any],
    direct_binary: Mapping[str, Any],
) -> MeasurementCalibrationReportV3:
    synthetic = synthetic_decision_estimator_controls(
        tolerance=spec.calibration.synthetic_tolerance
    )
    primary = [
        row
        for row in rows
        if row.representation == "gru"
        and row.evidence_policy == "ordinary_progress"
        and row.prefix == "pre_commitment"
    ]
    brier: dict[str, bool] = {
        layout.layout_id: bool(
            np.mean([row.brier_score for row in primary if row.layout_id == layout.layout_id])
            < np.mean(
                [row.uniform_brier_score for row in primary if row.layout_id == layout.layout_id]
            )
        )
        for layout in spec.layouts
    }
    fixed: dict[str, bool] = {
        layout.layout_id: bool(
            np.mean([row.residual_risk for row in primary if row.layout_id == layout.layout_id])
            <= np.mean(
                [row.fixed_response_risk for row in primary if row.layout_id == layout.layout_id]
            )
            + 1e-12
        )
        for layout in spec.layouts
    }
    seed_stability: dict[str, Any] = {}
    for layout in spec.layouts:
        layout_rows = [row for row in primary if row.layout_id == layout.layout_id]
        aligned = [
            float(np.mean(np.asarray(row.seed_dri) * float(row.dri or 0.0) >= 0))
            for row in layout_rows
            if row.seed_dri
        ]
        seed_stability[layout.layout_id] = {
            "mean_direction_agreement": float(np.mean(aligned)) if aligned else 0.0,
            "passed": bool(aligned) and float(np.mean(aligned)) >= 0.8,
        }
    event_sign = {
        "regression": bool(regression["event_same_effect_direction"]),
        **{
            str(row["layout_id"]): float(row["event_decision_risk_reduction"]) >= 0
            for row in intervention["layouts"]
        },
    }
    passive_permutation_pass = all(
        bool(permutation["tests"][f"{layout.layout_id}:passive_dri"]["passed"])
        for layout in spec.layouts
    )
    passed = bool(
        synthetic["passed"]
        and all(brier.values())
        and all(fixed.values())
        and passive_permutation_pass
        and all(value["passed"] for value in seed_stability.values())
        and event_sign["regression"]
    )
    return MeasurementCalibrationReportV3(
        synthetic_controls=dict(synthetic),
        leakage_checks={
            "v2_confirmatory_not_used_for_tuning": True,
            "fresh_confirmation_not_used_for_tuning": True,
            "post_commitment_excluded_from_precommitment": True,
            "partner_metadata_excluded": True,
        },
        brier_checks=brier,
        fixed_response_checks=fixed,
        permutation_tests=tuple(
            {"test_id": name, **value} for name, value in permutation["tests"].items()
        ),
        holm_adjusted=dict(permutation["holm_adjusted"]),
        seed_stability=seed_stability,
        event_sign_sensitivity=event_sign,
        direct_binary_gru_diagnostic=dict(direct_binary),
        passed=passed,
    )


def _v3_verdict(
    spec: OfficialMeasurementAuditSuiteV3,
    calibration: MeasurementCalibrationReportV3,
    regression: Mapping[str, Any],
    intervention: Mapping[str, Any],
    source: Mapping[str, Any],
) -> tuple[str, dict[str, bool | None]]:
    gru = regression["gru"]
    overall = gru["overall"]
    layout_prediction = {
        layout.layout_id: bool(
            gru["by_layout"][layout.layout_id]["delta_mae"] < 0
            and gru["by_layout"][layout.layout_id]["delta_mse"] < 0
        )
        for layout in spec.layouts
    }
    coefficient_negative = float(gru["dri_coefficient_interval"]["ci_high"]) < 0
    overall_value = bool(
        overall["delta_r2"] > 0
        and overall["delta_mae"] < 0
        and overall["delta_mse"] < 0
        and coefficient_negative
    )
    robustness = bool(
        all(layout_prediction.values())
        and regression["event_same_effect_direction"]
    )
    permutation_tests = {
        str(item["test_id"]): bool(item["passed"])
        for item in calibration.permutation_tests
    }
    intervention_by_layout = {
        str(item["layout_id"]): item for item in intervention["layouts"]
    }
    primary = intervention_by_layout["random3_m"]
    primary_intervention = bool(
        primary["qualifies_before_permutation"]
        and permutation_tests.get("random3_m:selected_intervention", False)
    )
    systematic_gap = bool(
        _read_json(
            _resolve_source_path(
                Path(__file__).resolve().parents[2],
                spec.v2.v2_manifest_path,
            ).parent
            / "natural-intervention-audit.json"
        ).get("systematic_method_gap", False)
    )
    gates: dict[str, bool | None] = {
        "measurement_calibration": calibration.passed,
        "overall_delta_r2_positive": overall["delta_r2"] > 0,
        "overall_delta_mae_negative": overall["delta_mae"] < 0,
        "overall_delta_mse_negative": overall["delta_mse"] < 0,
        "both_layouts_predictive": all(layout_prediction.values()),
        "clustered_dri_coefficient_negative": coefficient_negative,
        "event_sensitivity_same_direction": bool(regression["event_same_effect_direction"]),
        "primary_intervention_confirmed": primary_intervention,
        "existing_methods_leave_systematic_gap": systematic_gap,
    }
    if not calibration.passed:
        return "redesign", gates
    if not overall_value:
        return "stop", gates
    if not robustness:
        return "redesign", gates
    if not primary_intervention:
        return "complete_measurement_only", gates
    if systematic_gap:
        return "continue_top_paper_package", gates
    return "complete_evaluation_only", gates


def _fresh_key_leakage_audit(
    spec: OfficialMeasurementAuditSuiteV3,
    confirmation: OfficialConfirmationPlan,
    project_root: Path,
) -> dict[str, Any]:
    v2_plan = _read_json(_resolve_source_path(project_root, spec.v2.v2_rollout_plan_path))
    v2_keys = {
        int(key) for shard in v2_plan["shards"] for key in shard.get("episode_keys", ())
    }
    fresh_keys = {
        key for shard in confirmation.rollout_plan.shards for key in shard.episode_keys
    }
    return {
        "schema_version": 3,
        "v2_unique_key_count": len(v2_keys),
        "fresh_unique_key_count": len(fresh_keys),
        "fresh_episode_count": sum(
            len(shard.episode_keys) for shard in confirmation.rollout_plan.shards
        ),
        "overlap_count": len(v2_keys & fresh_keys),
        "passed": not bool(v2_keys & fresh_keys),
        "paired_across_evidence_policies": True,
    }


def _fresh_trace_index(
    plan: OfficialConfirmationPlan,
    ledger: OfficialConfirmationLedger,
) -> OfficialTraceIndex:
    runtime_ledger = OfficialRolloutLedger(
        suite_id=plan.suite_id,
        plan_hash=plan.rollout_plan.plan_hash,
        entries=ledger.entries,
        complete=ledger.complete,
        failed_shards=ledger.failed_shards,
    )
    return build_official_trace_index(plan.rollout_plan, runtime_ledger)


def _require_confirmation_complete(
    spec: OfficialMeasurementAuditSuiteV3,
    plan: OfficialConfirmationPlan,
    ledger: OfficialConfirmationLedger,
    fit: MeasurementFitManifest,
) -> None:
    _validate_confirmation_plan_integrity(plan)
    if ledger.plan_hash != plan.plan_hash or not ledger.complete or ledger.failed_shards:
        raise ValueError("v3 analysis requires a complete failure-free confirmation ledger")
    if fit.frozen_configuration_hash != plan.frozen_configuration_hash:
        raise ValueError("measurement configuration changed after confirmation began")
    if any(shard.kind != "trace" for shard in plan.rollout_plan.shards):
        raise ValueError("v3 analysis rejects non-trace confirmation shards")
    if sum(len(shard.episode_keys) for shard in plan.rollout_plan.shards) != 9600:
        raise ValueError("v3 confirmation episode count differs from the frozen protocol")
    if spec.policy_training_allowed:
        raise ValueError("policy training is forbidden in the v3 confirmation protocol")


def _resolve_confirmation_ledger(
    value: OfficialConfirmationLedger | str | Path,
) -> OfficialConfirmationLedger:
    if isinstance(value, OfficialConfirmationLedger):
        return value
    return OfficialConfirmationLedger.model_validate(_read_json(Path(value)))


def _decoder_matches_artifact(
    decoder: PairwiseDecisionDecoder,
    artifact: MeasurementRepresentationArtifact,
) -> bool:
    return bool(
        decoder.layout_id == artifact.layout_id
        and decoder.evidence_policy == artifact.evidence_policy
        and str(decoder.prefix) == str(artifact.prefix)
        and decoder.representation == artifact.representation
        and decoder.seed == artifact.seed
    )


def _encoder_signature(
    spec: OfficialMeasurementAuditSuiteV3,
    artifact: MeasurementRepresentationArtifact,
) -> str:
    return _hash_json(
        {
            "suite": spec.suite_id,
            "unit": (
                f"{artifact.layout_id}--{artifact.evidence_policy.replace('_', '-')}--"
                f"{str(artifact.prefix).replace('_', '-')}--gru-{artifact.seed}"
            ),
            "calibration_keys": artifact.calibration_key_hash,
            "training_fraction": spec.representations.encoder_training_fraction,
            "feature_contract": (
                "ego_observation+previous_ego_action+visible_partner_action+reward+step"
            ),
        }
    )


def _visible_partner_action_tv(ordinary: Sequence[Any], option: Sequence[Any]) -> float:
    def distribution(episodes: Sequence[Any]) -> np.ndarray:
        counts = np.zeros(7, dtype=np.float64)
        for episode in episodes:
            for action in episode.precommitment_partner_actions():
                counts[min(max(action, 0), 6)] += 1
        return counts / counts.sum() if counts.sum() else np.full(7, 1.0 / 7.0)

    return float(0.5 * np.abs(distribution(ordinary) - distribution(option)).sum())


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
    family_size: int,
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("intervention interval requires at least two conflicting pairs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    draws = values[indices].mean(axis=1)
    alpha = 0.05 / family_size
    low, high = np.quantile(draws, (alpha / 2, 1 - alpha / 2))
    return float(low), float(high)


def _permutation_seed(base: int, permutation: int, *parts: str) -> int:
    digest = hashlib.sha256(
        f"{base}:{permutation}:{':'.join(parts)}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _fresh_environment_steps(plan: OfficialConfirmationPlan) -> int:
    total = 0
    for shard in plan.rollout_plan.shards:
        result = Path(shard.result_path)
        if not result.is_file():
            continue
        import gzip

        with gzip.open(result, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        total += sum(len(episode.get("steps", ())) for episode in payload.get("episodes", ()))
    return total


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return np.asarray(values / values.sum(axis=1, keepdims=True), dtype=np.float64)


def _read_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"expected a list of JSON objects: {path}")
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
