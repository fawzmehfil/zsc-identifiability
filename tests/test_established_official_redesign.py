from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from zsc_identifiability import established_official_redesign as redesign_module
from zsc_identifiability import established_official_redesign_analysis as analysis_module
from zsc_identifiability.cli import _parser
from zsc_identifiability.established_dri import summarize_posteriors
from zsc_identifiability.established_official_decision import (
    PairwiseDecoderDataset,
    decoder_probabilities,
    evaluate_pairwise_decision,
    fit_ridge_logistic,
    holm_correct_permutation_tests,
    one_sided_permutation_p_value,
    select_global_decoder_configuration,
    signed_hash_event_features,
    synthetic_decision_estimator_controls,
)
from zsc_identifiability.established_official_models import (
    OfficialRolloutPlan,
    OfficialRolloutShard,
    OfficialTraceIndex,
    OfficialTraceIndexEntry,
)
from zsc_identifiability.established_official_redesign import (
    _fresh_environment_key,
    _hash_json,
    _suite_hash,
    _tuning_trace_index,
    prepare_fresh_confirmation,
    run_fresh_confirmation,
)
from zsc_identifiability.established_official_redesign_analysis import _write_v3_figures
from zsc_identifiability.established_official_redesign_models import (
    MeasurementFitManifest,
    OfficialConfirmationPlan,
    OfficialMeasurementAuditSuiteV3,
    PairwiseDecisionDecoder,
    load_official_measurement_suite,
)
from zsc_identifiability.established_official_representation import (
    deterministic_stratified_calibration_split,
    fit_gru_identity_representation,
)

ROOT = Path(__file__).resolve().parents[1]
V2_SUITE = ROOT / "phase-6-established-validation/suites/official-checkpoint-v2.json"
V3_SUITE = ROOT / "phase-6-established-validation/suites/official-measurement-v3.json"


def test_analysis_preserves_file_suite_hash_for_nested_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinel = object()
    monkeypatch.setattr(
        analysis_module, "_resolve_confirmation_plan", lambda _value: sentinel
    )
    monkeypatch.setattr(
        analysis_module, "_resolve_confirmation_ledger", lambda _value: sentinel
    )
    monkeypatch.setattr(
        analysis_module, "_resolve_fit_manifest", lambda _value: (sentinel, None)
    )
    monkeypatch.setattr(analysis_module, "_validate_fit_manifest", lambda *_args: None)
    monkeypatch.setattr(
        analysis_module, "_require_confirmation_complete", lambda *_args: None
    )

    def stop_after_suite_check(
        suite: object,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[object, ...]:
        assert Path(suite).resolve() == V3_SUITE.resolve()  # type: ignore[arg-type]
        raise RuntimeError("nested suite path preserved")

    monkeypatch.setattr(
        analysis_module, "evaluate_fresh_decision_value", stop_after_suite_check
    )
    with pytest.raises(RuntimeError, match="nested suite path preserved"):
        analysis_module.analyze_measurement_redesign(
            V3_SUITE,
            "plan.json",
            "ledger.json",
            "fit.json",
            tmp_path,
        )


def test_identity_softmax_normalizes_float32_logits_in_float64() -> None:
    logits = np.zeros((4, 30), dtype=np.float32)
    probabilities = analysis_module._softmax(logits)
    assert probabilities.dtype == np.float64
    assert np.max(np.abs(probabilities.sum(axis=1) - 1.0)) <= 1e-12


def test_v2_is_archived_byte_for_byte_and_v3_is_frozen() -> None:
    canonical = ROOT / "phase-6-established-validation/suites/canonical.json"
    assert V2_SUITE.read_bytes() == canonical.read_bytes()
    suite = load_official_measurement_suite(V3_SUITE)
    assert suite.policy_training_allowed is False
    assert suite.fresh_confirmation_salt == "zsc-stage6-v3-confirmatory-9d41"
    assert sum(
        layout.expected_partner_count
        * len(layout.evidence_policies)
        * layout.fresh_episodes_per_partner_policy
        for layout in suite.layouts
    ) == 9600
    assert {layout.layout_id: layout.frozen_intervention for layout in suite.layouts} == {
        "random3_m": "temporary_role_takeover",
        "small_corridor": "corridor_yield",
    }


def test_v3_schema_rejects_policy_training_and_protocol_drift() -> None:
    raw = json.loads(V3_SUITE.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError, match="policy training is structurally forbidden"):
        OfficialMeasurementAuditSuiteV3.model_validate({**raw, "policy_training_budget": 1})
    changed = json.loads(V3_SUITE.read_text(encoding="utf-8"))
    changed["layouts"][0]["fresh_episodes_per_partner_policy"] = 62
    with pytest.raises(ValidationError, match="9,600 episodes"):
        OfficialMeasurementAuditSuiteV3.model_validate(changed)
    changed = json.loads(V3_SUITE.read_text(encoding="utf-8"))
    changed["representations"]["encoder_training_fraction"] = 0.8
    with pytest.raises(ValidationError, match="0.75"):
        OfficialMeasurementAuditSuiteV3.model_validate(changed)
    changed = json.loads(V3_SUITE.read_text(encoding="utf-8"))
    changed["representations"]["gru_seeds"][0] = 1
    with pytest.raises(ValidationError, match="frozen GRU"):
        OfficialMeasurementAuditSuiteV3.model_validate(changed)
    changed = json.loads(V3_SUITE.read_text(encoding="utf-8"))
    changed["decoder"]["temperatures"][1] = 0.3
    with pytest.raises(ValidationError, match="log-spaced"):
        OfficialMeasurementAuditSuiteV3.model_validate(changed)


def test_cli_exposes_every_registered_redesign_operation() -> None:
    parser = _parser()
    for operation in (
        "validate",
        "fit",
        "prepare-confirmation",
        "run-confirmation",
        "status",
        "analyze",
    ):
        required = {
            "validate": ["--suite", "suite.json"],
            "fit": ["--suite", "suite.json", "--output", "out"],
            "prepare-confirmation": [
                "--suite",
                "suite.json",
                "--fit-manifest",
                "fit.json",
                "--workspace",
                "run",
            ],
            "run-confirmation": ["--plan", "plan.json"],
            "status": ["--plan", "plan.json"],
            "analyze": [
                "--suite",
                "suite.json",
                "--plan",
                "plan.json",
                "--ledger",
                "ledger.json",
                "--fit-manifest",
                "fit.json",
                "--output",
                "out",
            ],
        }[operation]
        args = parser.parse_args(["established", "official", "redesign", operation, *required])
        assert args.redesign_command == operation


def test_event_hashing_is_deterministic_temporal_and_fixed_width() -> None:
    kwargs = {
        "observed_length": 12,
        "cumulative_reward": -2.0,
        "partner_visibility_rate": 0.25,
    }
    first = signed_hash_event_features(((2, "partner_action:1"),), **kwargs)
    second = signed_hash_event_features(((2, "partner_action:1"),), **kwargs)
    later = signed_hash_event_features(((22, "partner_action:1"),), **kwargs)
    assert first.shape == (512,)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, later)
    assert first[-3:].tolist() == [12.0, -2.0, 0.25]


def test_ridge_logistic_and_loss_aware_pairwise_decision() -> None:
    features = np.asarray([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    coefficient, intercept, mean, scale = fit_ridge_logistic(
        features, labels, ridge_strength=0.1
    )
    decoder = PairwiseDecisionDecoder(
        decoder_id="decoder",
        layout_id="random3_m",
        evidence_policy="ordinary_progress",
        prefix="pre_commitment",
        representation="event",
        left_partner_id="left",
        right_partner_id="right",
        ridge_strength=0.1,
        temperature=1.0,
        prior_shrinkage=0.0,
        coefficient=tuple(coefficient),
        intercept=intercept,
        feature_mean=tuple(mean),
        feature_scale=tuple(scale),
        configuration_id="config",
        calibration_examples=6,
        validation_examples=6,
    )
    probabilities = decoder_probabilities(decoder, features)
    assert np.all(probabilities[:3] < 0.5)
    assert np.all(probabilities[3:] > 0.5)
    evaluation = evaluate_pairwise_decision(
        probabilities,
        labels,
        ((0.0, 40.0), (40.0, 0.0)),
    )
    assert evaluation.dri == 1.0

    asymmetric = evaluate_pairwise_decision(
        (0.1, 0.55),
        (0, 1),
        ((0.0, 1.0, 4.0), (5.0, 0.0, 1.0)),
    )
    assert asymmetric.decisions.tolist() == [0, 1]


def test_prior_shrinkage_can_revert_unreliable_evidence_to_uniform() -> None:
    decoder = PairwiseDecisionDecoder(
        decoder_id="decoder",
        layout_id="random3_m",
        evidence_policy="ordinary_progress",
        prefix="pre_commitment",
        representation="event",
        left_partner_id="left",
        right_partner_id="right",
        ridge_strength=1.0,
        temperature=1.0,
        prior_shrinkage=1.0,
        coefficient=(100.0,),
        intercept=100.0,
        feature_mean=(0.0,),
        feature_scale=(1.0,),
        configuration_id="uniform",
        calibration_examples=4,
        validation_examples=4,
    )
    assert decoder_probabilities(decoder, np.asarray(((-10.0,), (10.0,)))).tolist() == [
        0.5,
        0.5,
    ]


def test_global_decoder_selection_uses_one_configuration_across_seeds() -> None:
    train = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.asarray([0, 0, 1, 1])
    datasets = tuple(
        PairwiseDecoderDataset(
            seed=seed,
            left_partner_id="left",
            right_partner_id="right",
            calibration_features=train,
            calibration_labels=labels,
            validation_features=train,
            validation_labels=labels,
            loss_matrix=np.asarray(((0.0, 1.0), (1.0, 0.0))),
        )
        for seed in (1, 2)
    )
    selection = select_global_decoder_configuration(
        datasets,
        layout_id="random3_m",
        evidence_policy="ordinary_progress",
        prefix="pre_commitment",
        representation="gru",
        seed=None,
        ridge_strengths=(0.1, 1.0),
        temperatures=(0.5, 1.0),
        prior_shrinkages=(0.0, 1.0),
    )
    assert {decoder.configuration_id for decoder in selection.decoders} == {
        selection.configuration_id
    }
    assert {decoder.seed for decoder in selection.decoders} == {1, 2}


def test_phase3_noisy_signal_trajectory_fixture_recovers_registered_dri() -> None:
    features = np.asarray([[-1.0]] * 8 + [[1.0]] * 2 + [[-1.0]] * 2 + [[1.0]] * 8)
    labels = np.asarray([0] * 10 + [1] * 10)
    dataset = PairwiseDecoderDataset(
        seed=6173,
        left_partner_id="expects_role_a",
        right_partner_id="expects_role_b",
        calibration_features=features,
        calibration_labels=labels,
        validation_features=features,
        validation_labels=labels,
        loss_matrix=np.asarray(((0.0, 40.0), (40.0, 0.0))),
    )
    selection = select_global_decoder_configuration(
        (dataset,),
        layout_id="random3_m",
        evidence_policy="ordinary_progress",
        prefix="pre_commitment",
        representation="gru",
        seed=6173,
        ridge_strengths=(0.01, 0.1, 1.0, 10.0),
        temperatures=(0.25, 0.5, 1.0, 2.0, 4.0),
        prior_shrinkages=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    evaluation = evaluate_pairwise_decision(
        decoder_probabilities(selection.decoders[0], features),
        labels,
        dataset.loss_matrix,
    )
    assert evaluation.dri == pytest.approx(0.6)


def test_permutation_control_is_one_sided_and_allows_negative_null_dri() -> None:
    null = [-0.4] * 100
    assert one_sided_permutation_p_value(0.1, null) == pytest.approx(1 / 101)
    assert one_sided_permutation_p_value(-0.5, null) == 1.0
    adjusted = holm_correct_permutation_tests(
        {"primary": 1 / 101, "robustness": 2 / 101, "option_a": 0.5, "option_b": 0.8}
    )
    assert adjusted["primary"] == pytest.approx(4 / 101)
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_response_signature_information_is_not_partner_identity_information() -> None:
    summary = summarize_posteriors(
        (0.25, 0.25, 0.25, 0.25),
        ((0.0, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 0.0)),
        (
            (0.5, 0.0, 0.5, 0.0),
            (0.0, 0.5, 0.0, 0.5),
        ),
        response_signatures=("A", "A", "B", "B"),
    )
    assert summary.identity_mutual_information_nats > 0
    assert summary.response_signature_mutual_information_nats == pytest.approx(0.0)


def test_v3_synthetic_controls_cover_late_identity_and_censoring() -> None:
    report = synthetic_decision_estimator_controls()
    assert report["passed"] is True
    assert report["informative_dri"] == pytest.approx(0.6)
    assert report["identity_only_dri"] == 0.0
    assert report["late_precommitment_dri"] == 0.0
    assert report["late_eventual_dri"] == 1.0
    assert report["censored_dri"] == 0.0


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_pairwise_decision_value_is_invariant_to_partner_label_swap(
    probability_right: float,
) -> None:
    probabilities = np.asarray([probability_right, 1.0 - probability_right])
    labels = np.asarray([0, 1])
    losses = np.asarray(((0.0, 3.0, 7.0), (5.0, 0.0, 2.0)))
    original = evaluate_pairwise_decision(probabilities, labels, losses)
    swapped = evaluate_pairwise_decision(
        1.0 - probabilities,
        1 - labels,
        losses[::-1],
    )
    assert swapped.prior_risk == pytest.approx(original.prior_risk)
    assert swapped.residual_risk == pytest.approx(original.residual_risk)
    assert swapped.dri == pytest.approx(original.dri)


def test_v3_tuning_boundary_excludes_v2_confirmatory_entries() -> None:
    entries = tuple(
        OfficialTraceIndexEntry(
            trace_id=split,
            layout_id="random3_m",
            partner_id="partner",
            evidence_policy="ordinary_progress",
            split=split,  # type: ignore[arg-type]
            path=f"/{split}.json.gz",
            content_hash="a" * 64,
            episodes=2,
        )
        for split in ("calibration", "validation", "confirmatory")
    )
    tuning = _tuning_trace_index(OfficialTraceIndex(suite_id="v2", entries=entries))
    assert {entry.split for entry in tuning.entries} == {"calibration", "validation"}
    assert all(entry.trace_id != "confirmatory" for entry in tuning.entries)


@dataclass(frozen=True)
class _Episode:
    environment_key: int


class _SplitSource:
    def __init__(self) -> None:
        self.labels = (0, 0, 0, 0, 1, 1, 1, 1)
        self.episodes = tuple(_Episode(index) for index in range(8))

    def subset(self, indices: Any) -> tuple[int, ...]:
        return tuple(indices)


class _ArraySequenceSource:
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._features = features.astype(np.float32)
        self._labels = labels.astype(np.int64)
        self.size = len(labels)
        self.feature_width = features.shape[-1]

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Any:
        order = np.arange(self.size)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for offset in range(0, self.size, batch_size):
            indices = order[offset : offset + batch_size]
            yield (
                self._features[indices],
                np.full(len(indices), self._features.shape[1], dtype=np.int64),
                self._labels[indices],
                indices,
            )


def test_calibration_internal_split_is_stratified_deterministic_and_disjoint() -> None:
    source = _SplitSource()
    first_train, first_stop = deterministic_stratified_calibration_split(
        source, salt="fixed"  # type: ignore[arg-type]
    )
    second_train, second_stop = deterministic_stratified_calibration_split(
        source, salt="fixed"  # type: ignore[arg-type]
    )
    assert first_train == second_train
    assert first_stop == second_stop
    assert not set(first_train) & set(first_stop)
    assert len(first_train) == 6
    assert len(first_stop) == 2
    assert {source.labels[index] for index in first_train} == {0, 1}
    assert {source.labels[index] for index in first_stop} == {0, 1}


def test_identity_representation_checkpoint_resumes_without_refitting(tmp_path: Path) -> None:
    features = np.asarray(
        [[[-1.0], [-1.0]], [[-0.8], [-1.0]], [[1.0], [1.0]], [[0.8], [1.0]]]
    )
    source = _ArraySequenceSource(features, np.asarray([0, 0, 1, 1]))
    checkpoint = tmp_path / "encoder.pt"
    first = fit_gru_identity_representation(
        source,  # type: ignore[arg-type]
        source,  # type: ignore[arg-type]
        mode_count=2,
        hidden_size=4,
        seed=6173,
        signature="frozen-unit",
        checkpoint_path=checkpoint,
        max_epochs=4,
        patience=2,
        batch_size=2,
    )
    modified_time = checkpoint.stat().st_mtime_ns
    second = fit_gru_identity_representation(
        source,  # type: ignore[arg-type]
        source,  # type: ignore[arg-type]
        mode_count=2,
        hidden_size=4,
        seed=6173,
        signature="frozen-unit",
        checkpoint_path=checkpoint,
        max_epochs=4,
        patience=2,
        batch_size=2,
    )
    assert second.checkpoint_hash == first.checkpoint_hash
    assert checkpoint.stat().st_mtime_ns == modified_time


def test_fresh_key_is_paired_across_policies_and_seeded_by_frozen_salt() -> None:
    first = _fresh_environment_key(
        "zsc-stage6-v3-confirmatory-9d41", "random3_m", "partner", 0
    )
    repeated = _fresh_environment_key(
        "zsc-stage6-v3-confirmatory-9d41", "random3_m", "partner", 0
    )
    next_key = _fresh_environment_key(
        "zsc-stage6-v3-confirmatory-9d41", "random3_m", "partner", 1
    )
    assert first == repeated
    assert first != next_key
    assert 0 <= first < 2**63


def test_confirmation_preparation_is_trace_only_exact_and_performs_no_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = load_official_measurement_suite(V3_SUITE)
    v2_workspace = tmp_path / "v2"
    requests = v2_workspace / "requests"
    requests.mkdir(parents=True)
    partners: list[SimpleNamespace] = []
    shards: list[OfficialRolloutShard] = []
    for layout in suite.layouts:
        for partner_number in range(layout.expected_partner_count):
            partner_id = f"{layout.layout_id}:partner:{partner_number:02d}"
            partners.append(SimpleNamespace(partner_id=partner_id, layout_id=layout.layout_id))
            for policy in layout.evidence_policies:
                request = {
                    "schema_version": 1,
                    "suite_id": "v2",
                    "runtime": "zsceval_py39",
                    "operation": "official_trace_rollout",
                    "policy_training_allowed": False,
                    "layout_id": layout.layout_id,
                    "max_episode_steps": 400,
                    "repository_commit": "a" * 40,
                    "policy_pool_revision": "b" * 40,
                    "payload": {
                        "partner_checkpoint_path": "/partner.pt",
                        "reference_checkpoint_path": "/reference.pt",
                        "evidence_policy": policy,
                        "split": "calibration",
                        "episode_keys": [partner_number + 1],
                        "balanced_seats": True,
                        "deterministic": True,
                        "maximum_option_steps": 16,
                        "prefix_steps": [0, 8, 16, 32],
                    },
                }
                request_hash = _hash_json(request)
                request["request_hash"] = request_hash
                request_path = requests / f"{layout.layout_id}-{partner_number}-{policy}.json"
                request_path.write_text(json.dumps(request), encoding="utf-8")
                shards.append(
                    OfficialRolloutShard(
                        shard_id=f"{layout.layout_id}-{partner_number}-{policy}",
                        kind="trace",
                        layout_id=layout.layout_id,
                        request_path=str(request_path),
                        result_path=str(tmp_path / "unused.json.gz"),
                        request_hash=request_hash,
                        partner_id=partner_id,
                        evidence_policy=policy,
                        split="calibration",
                        episode_keys=(partner_number + 1,),
                        deterministic=True,
                    )
                )
    rollout_payload = {
        "schema_version": 1,
        "suite_id": "v2",
        "suite_path": str(V2_SUITE),
        "suite_hash": "c" * 64,
        "inventory_path": str(tmp_path / "inventory.json"),
        "inventory_hash": suite.v2.v2_inventory_hash,
        "workspace": str(v2_workspace),
        "shards": [shard.to_dict() for shard in shards],
    }
    v2_plan = OfficialRolloutPlan.model_validate(
        {**rollout_payload, "plan_hash": _hash_json(rollout_payload)}
    )
    decoder_path = tmp_path / "decoders.json"
    decoder_path.write_text("{}\n", encoding="utf-8")
    fit = MeasurementFitManifest(
        suite_id=suite.suite_id,
        suite_hash=_suite_hash(suite, None),
        source_v2_plan_hash=suite.v2.v2_rollout_plan_hash,
        source_v2_trace_index_hash=suite.v2.v2_trace_index_hash,
        calibration_data_hash="d" * 64,
        validation_data_hash="e" * 64,
        artifacts=(),
        decoder_manifest_path=str(decoder_path),
        decoder_manifest_hash=hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        frozen_configuration_hash="f" * 64,
        complete=True,
    )
    fit_path = tmp_path / "measurement-fit-manifest.json"
    fit_path.write_text(json.dumps(fit.to_dict()), encoding="utf-8")
    monkeypatch.setattr(
        redesign_module,
        "_validate_v2_source_lock",
        lambda _spec, _root: {"rollout_plan": v2_plan.to_dict()},
    )
    monkeypatch.setattr(
        redesign_module,
        "load_official_asset_inventory",
        lambda _path: SimpleNamespace(partners=tuple(partners)),
    )
    monkeypatch.setattr(redesign_module, "_link_runtime_assets", lambda *_args: None)
    plan = prepare_fresh_confirmation(suite, fit_path, tmp_path / "confirmation")
    assert len(plan.rollout_plan.shards) == 180
    assert {shard.kind for shard in plan.rollout_plan.shards} == {"trace"}
    assert sum(len(shard.episode_keys) for shard in plan.rollout_plan.shards) == 9600
    assert all(shard.split == "confirmatory" for shard in plan.rollout_plan.shards)
    assert all(
        json.loads(Path(shard.request_path).read_text())["policy_training_allowed"] is False
        for shard in plan.rollout_plan.shards
    )


def test_trace_only_confirmation_resume_skips_hash_valid_results(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json.gz"
    request = {
        "operation": "official_trace_rollout",
        "policy_training_allowed": False,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    shard = OfficialRolloutShard(
        shard_id="trace",
        kind="trace",
        layout_id="random3_m",
        request_path=str(request_path),
        result_path=str(result_path),
        request_hash=hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        partner_id="partner",
        evidence_policy="ordinary_progress",
        split="confirmatory",
        episode_keys=(1, 2),
        deterministic=True,
    )
    rollout_payload = {
        "schema_version": 1,
        "suite_id": "v3",
        "suite_path": str(tmp_path / "v2-suite.json"),
        "suite_hash": "a" * 64,
        "inventory_path": str(tmp_path / "inventory.json"),
        "inventory_hash": "b" * 64,
        "workspace": str(tmp_path),
        "shards": [shard.to_dict()],
    }
    rollout = OfficialRolloutPlan.model_validate(
        {**rollout_payload, "plan_hash": _hash_json(rollout_payload)}
    )
    decoder_path = tmp_path / "decoders.json"
    decoder_path.write_text("{}\n", encoding="utf-8")
    fit = MeasurementFitManifest(
        suite_id="v3",
        suite_hash="a" * 64,
        source_v2_plan_hash="c" * 64,
        source_v2_trace_index_hash="d" * 64,
        calibration_data_hash="e" * 64,
        validation_data_hash="f" * 64,
        artifacts=(),
        decoder_manifest_path=str(decoder_path),
        decoder_manifest_hash=hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
        frozen_configuration_hash="1" * 64,
        complete=True,
    )
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps(fit.to_dict()), encoding="utf-8")
    public_payload = {
        "schema_version": 1,
        "suite_id": "v3",
        "suite_path": str(tmp_path / "v3-suite.json"),
        "suite_hash": "a" * 64,
        "frozen_configuration_hash": "1" * 64,
        "fit_manifest_path": str(fit_path),
        "fit_manifest_hash": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        "source_v2_plan_hash": "c" * 64,
        "source_v2_environment_key_hash": "2" * 64,
        "fresh_environment_key_hash": "3" * 64,
        "workspace": str(tmp_path),
        "rollout_plan": rollout.to_dict(),
    }
    plan = OfficialConfirmationPlan.model_validate(
        {**public_payload, "plan_hash": _hash_json(public_payload)}
    )
    calls: list[str] = []

    def executor(item: Any, _workspace: Path, _logs: Path) -> str:
        calls.append(item.shard_id)
        with gzip.open(item.result_path, "wt", encoding="utf-8") as handle:
            json.dump({"operation": "official_trace_rollout"}, handle)
        return hashlib.sha256(Path(item.result_path).read_bytes()).hexdigest()

    first = run_fresh_confirmation(plan, workers=1, executor=executor)
    second = run_fresh_confirmation(plan, workers=1, executor=executor)
    assert first.complete and second.complete
    assert calls == ["trace"]
    assert all(entry.status == "complete" for entry in second.entries)

    fit_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        run_fresh_confirmation(plan, workers=1, executor=executor)


def test_v3_registered_figures_are_generated_from_compact_results(tmp_path: Path) -> None:
    from zsc_identifiability.established_official_redesign_models import (
        PairwiseDecisionValueRow,
    )

    suite = load_official_measurement_suite(V3_SUITE)
    rows = tuple(
        PairwiseDecisionValueRow(
            layout_id=layout.layout_id,
            left_partner_id=f"{layout.layout_id}:left",
            right_partner_id=f"{layout.layout_id}:right",
            left_scheme_id="left",
            right_scheme_id="right",
            evidence_policy="ordinary_progress",
            representation=representation,  # type: ignore[arg-type]
            prefix=prefix,
            prior_risk=0.5,
            residual_risk=0.25,
            fixed_response_risk=0.5,
            dri=0.5,
            brier_score=0.2,
            uniform_brier_score=0.25,
            commitment_rate=1.0,
            sample_count=4,
        )
        for layout in suite.layouts
        for representation in ("gru", "event")
        for prefix in suite.representations.prefixes
    )
    regression = {
        representation: {
            "overall": {"delta_mae": -0.1, "delta_mse": -0.05},
            "by_layout": {
                layout.layout_id: {"delta_mae": -0.1, "delta_mse": -0.05}
                for layout in suite.layouts
            },
        }
        for representation in ("gru", "event")
    }
    intervention = {
        "layouts": [
            {
                "layout_id": layout.layout_id,
                "gru_decision_risk_reduction": 0.1,
                "normalized_task_cost": 0.02,
            }
            for layout in suite.layouts
        ]
    }
    permutation = {
        "tests": {
            f"{layout.layout_id}:{kind}": {
                "observed": 0.1,
                "null_values": [-0.1, 0.0, 0.01],
            }
            for layout in suite.layouts
            for kind in ("passive_dri", "selected_intervention")
        }
    }
    paths = _write_v3_figures(
        suite, rows, regression, intervention, permutation, tmp_path / "figures"
    )
    assert len(paths) == 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
