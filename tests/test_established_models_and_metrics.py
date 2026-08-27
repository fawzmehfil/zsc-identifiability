from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from zsc_identifiability.established_commitment import extract_work_units, prefix_history
from zsc_identifiability.established_diagnostics import audit_diagnostic_options
from zsc_identifiability.established_divergence import estimate_prefix_tv_curves
from zsc_identifiability.established_dri import (
    estimate_event_dri,
    summarize_posteriors,
    synthetic_dri_calibration,
)
from zsc_identifiability.established_io import load_trace_jsonl, write_trace_jsonl
from zsc_identifiability.established_matching import (
    audit_confirmatory_population_pair,
    select_matched_population_pair,
)
from zsc_identifiability.established_models import (
    CandidatePartnerMetrics,
    CommitmentTraceStep,
    EstablishedMethodAssetsManifest,
    EstablishedPolicyArtifact,
    EstablishedPolicyComponent,
    EstablishedTrainingManifest,
    EstablishedValidationSuite,
    PartnerCheckpoint,
    TaskEvent,
    load_established_suite_file,
)
from zsc_identifiability.established_partners import (
    enumerate_reward_vectors,
    reward_vector_hash,
    split_for_reward_vector,
)
from zsc_identifiability.established_predictability import (
    estimate_lobp_action_oracle_from_trace_files,
)
from zsc_identifiability.established_response import (
    best_fixed_response_value,
    build_response_library_from_values,
)
from zsc_identifiability.established_runtime import validate_upstreams, write_runtime_request
from zsc_identifiability.established_statistics import (
    leave_one_reward_vector_out_regression,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "phase-6-established-validation/suites/canonical.json"


def test_canonical_stage6_suite_locks_protocol() -> None:
    suite = load_established_suite_file(SUITE_PATH)
    assert suite.schema_version == 1
    assert {item.layout_id for item in suite.layouts} == {
        "demo_cook_simple",
        "test_time_simple",
        "grounded_coord_simple",
        "demo_cook_wide",
    }
    assert all(item.max_steps == 400 and item.agent_view_size == 2 for item in suite.layouts)
    assert suite.commitment.event_name == "successful_pot_ingredient_placement"
    assert suite.matching.minimum_dri_separation == 0.15
    assert len(suite.training.confirmatory_seeds) == 10


def test_method_assets_require_tbs_cross_play_and_content_hashes() -> None:
    common = {
        "method_id": "tbs_style",
        "train_pool_path": "train.json",
        "train_pool_hash": "a" * 64,
        "validation_pool_path": "validation.json",
        "validation_pool_hash": "b" * 64,
    }
    with pytest.raises(ValidationError, match="cross-play"):
        EstablishedMethodAssetsManifest.model_validate(common)
    manifest = EstablishedMethodAssetsManifest.model_validate(
        {
            **common,
            "cross_play_values_path": "cross-play.json",
            "cross_play_values_hash": "c" * 64,
        }
    )
    assert manifest.compute_allocation == "per-specialist"


def test_composite_policy_artifact_locks_csp_reconnaissance_protocol() -> None:
    component = EstablishedPolicyComponent(
        component_id="probe", role="probe_policy", path="probe-checkpoint", content_hash="d" * 64
    )
    with pytest.raises(ValidationError, match="reconnaissance"):
        EstablishedPolicyArtifact(
            policy_kind="csp_reconnaissance",
            method_id="csp_style_reconnaissance",
            layout_id="demo_cook_simple",
            seed=1,
            backbone_config={},
            components=(component,),
            partner_ids=("partner",),
            source_configuration_hash="e" * 64,
            aggregate_training_transitions=1,
        )
    components = (
        component,
        EstablishedPolicyComponent(
            component_id="encoder",
            role="trajectory_encoder",
            path="encoder",
            content_hash="d" * 64,
        ),
        EstablishedPolicyComponent(
            component_id="decoder",
            role="response_decoder",
            path="decoder",
            content_hash="d" * 64,
        ),
        EstablishedPolicyComponent(
            component_id="specialist",
            role="specialist",
            path="specialist",
            content_hash="d" * 64,
            cluster_id=0,
        ),
    )
    artifact = EstablishedPolicyArtifact(
        policy_kind="csp_reconnaissance",
        method_id="csp_style_reconnaissance",
        layout_id="demo_cook_simple",
        seed=1,
        backbone_config={},
        components=components,
        partner_ids=("partner",),
        centroids=((0.0,) * 32,),
        reconnaissance_episodes=1,
        source_configuration_hash="e" * 64,
        aggregate_training_transitions=1,
    )
    assert artifact.reconnaissance_episodes == 1
    with pytest.raises(ValidationError, match="one specialist per centroid"):
        EstablishedPolicyArtifact.model_validate(
            {
                **artifact.to_dict(),
                "centroids": ((0.0,) * 32, (1.0,) * 32),
            }
        )


def test_training_manifest_v2_requires_resume_lineage() -> None:
    base = {
        "suite_id": "suite",
        "method_id": "pace_aux",
        "layout_id": "demo_cook_simple",
        "split": "smoke",
        "seed": 1,
        "requested_transitions": 10,
        "completed_transitions": 10,
        "checkpoint_path": "checkpoint",
        "checkpoint_hash": "f" * 64,
        "upstream_commit": "0" * 40,
        "configuration_hash": "1" * 64,
        "dataset_hashes": (),
        "python_version": "3.10",
        "jax_version": "0.4",
        "xla_version": "0.4",
        "device": "cpu",
        "resumed": True,
        "policy_kind": "pace",
    }
    with pytest.raises(ValidationError, match="lineage"):
        EstablishedTrainingManifest.model_validate(base)
    manifest = EstablishedTrainingManifest.model_validate(
        {**base, "parent_checkpoint_hash": "2" * 64}
    )
    assert manifest.schema_version == 2 and manifest.resumed


def test_partner_checkpoint_requires_paired_full_training_state() -> None:
    payload = {
        "partner_id": "partner",
        "reward_vector_id": "reward",
        "reward_vector_hash": "a" * 64,
        "split": "train",
        "seed": 1,
        "layout_id": "demo_cook_simple",
        "checkpoint_path": "compact-policy",
        "normalized_checkpoint_hash": "b" * 64,
        "transitions": 5,
        "validation_correct_delivery_rate": 1.0,
        "competent": True,
        "training_state_checkpoint_path": "full-state",
    }
    with pytest.raises(ValidationError, match="training-state checkpoint"):
        PartnerCheckpoint.model_validate(payload)
    checkpoint = PartnerCheckpoint.model_validate(
        {**payload, "training_state_checkpoint_hash": "c" * 64}
    )
    assert checkpoint.training_state_checkpoint_path == "full-state"


def test_suite_rejects_short_upstream_pin() -> None:
    data = json.loads(SUITE_PATH.read_text())
    data["upstreams"][0]["commit"] = "5ce1707"
    with pytest.raises(ValidationError, match="full lowercase 40-character"):
        EstablishedValidationSuite.model_validate(data)


def test_recipe_button_cannot_be_declared_partner_probe() -> None:
    data = json.loads(SUITE_PATH.read_text())
    button = next(
        item for item in data["diagnostics"] if item["option_id"] == "recipe_button_control"
    )
    button["partner_diagnostic_candidate"] = True
    with pytest.raises(ValidationError, match="environment-information control"):
        EstablishedValidationSuite.model_validate(data)


def test_commitment_history_excludes_irreversible_event_and_late_evidence() -> None:
    steps = (
        _step(0, high_level_events=("partner:movement",)),
        _step(
            1,
            events=(TaskEvent(name="successful_pot_ingredient_placement", actor="partner"),),
        ),
        _step(2, high_level_events=("partner:delivery_approach",)),
        _step(3, events=(TaskEvent(name="successful_delivery", actor="ego"),)),
        _step(4, high_level_events=("post_delivery_secret",)),
    )
    history = extract_work_units(steps)[0]
    assert tuple(item.step for item in history.pre_commitment) == (0,)
    assert history.commitment_step is not None and history.commitment_step.step == 1
    assert tuple(item.step for item in history.eventual) == (0, 1, 2, 3)
    assert tuple(item.step for item in prefix_history(history, "pre_commitment")) == (0,)
    assert all("post_delivery_secret" not in item.high_level_events for item in history.eventual)


def test_no_commitment_history_is_censored_without_dropping_episode() -> None:
    history = extract_work_units((_step(0), _step(1)))[0]
    assert not history.commitment_reached
    assert tuple(item.step for item in history.pre_commitment) == (0, 1)


def test_dri_recovers_phase3_analytical_controls() -> None:
    calibration = synthetic_dri_calibration()
    assert calibration["passed"] is True
    assert calibration["informative_dri"] == pytest.approx(0.6)
    assert calibration["informative_residual_risk"] == pytest.approx(8.0)
    assert calibration["identity_only_dri"] == pytest.approx(0.0)
    assert calibration["identity_only_identity_mi_nats"] > 0
    assert calibration["late_precommitment_dri"] == pytest.approx(0.0)
    assert calibration["late_eventual_dri"] == pytest.approx(1.0)


def test_identity_information_can_leave_response_risk_unchanged() -> None:
    summary = summarize_posteriors(
        (0.25, 0.25, 0.25, 0.25),
        ((0, 40), (0, 40), (40, 0), (40, 0)),
        ((0.5, 0, 0.5, 0), (0, 0.5, 0, 0.5)),
        response_signatures=("A", "A", "B", "B"),
    )
    assert summary.dri == pytest.approx(0.0)
    assert summary.identity_mutual_information_nats > 0
    assert summary.response_signature_mutual_information_nats == pytest.approx(0.0)


def test_event_estimator_uses_disjoint_calibration_table() -> None:
    calibration_histories = (("left",),) * 20 + (("right",),) * 20
    labels = (0,) * 20 + (1,) * 20
    estimate = estimate_event_dri(
        calibration_histories,
        labels,
        (("left",), ("right",)),
        (0.5, 0.5),
        ((0, 40), (40, 0)),
        response_signatures=("A", "B"),
        confirmatory_labels=(0, 1),
    )
    assert estimate.cross_fitted
    assert estimate.points[0].dri is not None
    assert estimate.points[0].dri > 0.9


def test_confidently_wrong_posterior_cannot_fake_dri() -> None:
    summary = summarize_posteriors(
        (0.5, 0.5),
        ((0, 40), (40, 0)),
        ((0.0, 1.0), (1.0, 0.0)),
        true_modes=(0, 1),
    )
    assert summary.residual_risk == pytest.approx(40.0)
    assert summary.dri == pytest.approx(-1.0)


def test_response_library_uses_approximate_oracle_and_conflicts() -> None:
    library = build_response_library_from_values(
        {
            "partner_a": {"response_a": 1.0, "response_b": 0.4},
            "partner_b": {"response_a": 0.2, "response_b": 1.0},
        },
        adequacy_margin=0.02,
        response_clusters={"partner_a": 0, "partner_b": 1},
    )
    assert library.approximate_oracle_label == "response_library_oracle"
    assert library.loss_matrix == ((0.0, 0.6), (0.8, 0.0))
    assert library.response_conflicts == (("partner_a", "partner_b"),)
    assert best_fixed_response_value(library) == pytest.approx(0.7)


def test_reward_vectors_are_sparse_hash_ordered_and_split_deterministically() -> None:
    suite = load_established_suite_file(SUITE_PATH)
    vectors = enumerate_reward_vectors(suite)
    assert vectors
    assert all(1 <= len(vector) <= 3 for vector in vectors)
    assert [reward_vector_hash(vector) for vector in vectors] == sorted(
        reward_vector_hash(vector) for vector in vectors
    )
    assert split_for_reward_vector(vectors[0], suite) == split_for_reward_vector(vectors[0], suite)


def test_upstream_audit_reports_missing_assets_without_installing(tmp_path: Path) -> None:
    suite = load_established_suite_file(SUITE_PATH)
    audit = validate_upstreams(suite, tmp_path)
    assert not audit.passed
    assert {item.repository_id for item in audit.repositories if not item.passed} == {
        "overcookedv2",
        "zsceval",
        "tomzsc",
    }


def test_canonical_suite_rejects_changed_upstream_pin() -> None:
    payload = json.loads(SUITE_PATH.read_text())
    payload["upstreams"][0]["commit"] = "0" * 40
    with pytest.raises(ValueError, match="canonical upstream pin changed"):
        EstablishedValidationSuite.model_validate(payload)


def test_runtime_request_is_content_hashed_and_runtime_scoped(tmp_path: Path) -> None:
    suite = load_established_suite_file(SUITE_PATH)
    path = write_runtime_request(
        suite,
        "tomzsc_py310",
        "validate",
        {"probe": "value"},
        tmp_path / "request.json",
    )
    payload = json.loads(path.read_text())
    observed_hash = payload.pop("request_hash")
    expected_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert observed_hash == expected_hash
    assert tuple(payload["upstreams"]) == ("tomzsc",)


def test_milp_matching_freezes_disjoint_high_low_dri_pair() -> None:
    suite = load_established_suite_file(SUITE_PATH)
    candidates = tuple(_candidate(index, high=index < 8) for index in range(16))
    audit = select_matched_population_pair(candidates, suite.matching, "passive_dri")
    assert audit.discovery_passed
    assert audit.frozen
    assert len(audit.left_partner_ids) == len(audit.right_partner_ids) == 8
    assert not set(audit.left_partner_ids) & set(audit.right_partner_ids)
    outcomes = {item.partner_id: (True,) * 9 + (False,) for item in candidates}
    confirmatory = audit_confirmatory_population_pair(
        audit,
        candidates,
        suite.matching,
        commitment_outcomes=outcomes,
    )
    assert confirmatory.confirmatory_passed is True


def test_diagnostic_audit_rejects_recipe_control_and_selects_task_option() -> None:
    audit = audit_diagnostic_options(
        "demo_cook_simple",
        (
            {
                "option_id": "ordinary_progress",
                "completes_before_commitment": True,
                "conflicting_mode_response_tv": 0.0,
                "passive_dri": 0.1,
                "option_dri": 0.1,
                "recipe_prediction_only": False,
                "expected_cost": 0.0,
                "expected_risk_reduction": 0.0,
                "universal_response_succeeds": False,
            },
            {
                "option_id": "stage_candidate_ingredient",
                "completes_before_commitment": True,
                "conflicting_mode_response_tv": 0.4,
                "passive_dri": 0.1,
                "option_dri": 0.35,
                "recipe_prediction_only": False,
                "expected_cost": 1.0,
                "expected_risk_reduction": 3.0,
                "universal_response_succeeds": False,
            },
            {
                "option_id": "recipe_button_control",
                "completes_before_commitment": True,
                "conflicting_mode_response_tv": 0.8,
                "passive_dri": 0.1,
                "option_dri": 0.8,
                "recipe_prediction_only": True,
                "expected_cost": 5.0,
                "expected_risk_reduction": 10.0,
                "universal_response_succeeds": False,
            },
        ),
    )
    assert audit.passed
    assert audit.selected_option == "stage_candidate_ingredient"
    recipe = next(item for item in audit.results if item.option_id == "recipe_button_control")
    assert not recipe.qualifying_partner_diagnostic


def test_leave_one_vector_out_regression_detects_incremental_dri_value() -> None:
    rows = []
    for vector in range(4):
        for method_index, method in enumerate(("rnn_ippo", "fcp")):
            for level in (0.1, 0.8):
                rows.append(
                    {
                        "reward_vector_id": f"rv{vector}",
                        "method_id": method,
                        "regret": 10.0 + method_index - 5.0 * level,
                        "competence": 1.0,
                        "br_div": 0.5,
                        "br_prox": 0.8,
                        "predictability": -0.4,
                        "trajectory_divergence": 0.3,
                        "dri": level,
                    }
                )
    report = leave_one_reward_vector_out_regression(rows)
    assert report["incremental_value"]
    assert report["full_mse"] < report["baseline_mse"]
    assert report["delta_r2"] > 0
    assert report["dri_coefficient_min"] == pytest.approx(-5.0)
    assert report["dri_coefficient_max"] == pytest.approx(-5.0)


def test_trace_jsonl_round_trip_is_versioned_and_ordered(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    digest = write_trace_jsonl(path, (_step(0), _step(1)))
    assert len(digest) == 64
    loaded = load_trace_jsonl(path)
    assert tuple(item.step for item in loaded) == (0, 1)


def test_predictability_control_uses_disjoint_visible_action_targets(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.jsonl"
    confirmatory = tmp_path / "confirmatory.jsonl"
    calibration_steps = tuple(
        _step(index).model_copy(
            update={
                "episode_id": "calibration",
                "visible_partner_action": index % 2,
            }
        )
        for index in range(16)
    )
    confirmatory_steps = tuple(
        item.model_copy(update={"episode_id": "confirmatory"}) for item in calibration_steps
    )
    write_trace_jsonl(calibration, calibration_steps)
    write_trace_jsonl(confirmatory, confirmatory_steps)
    report = estimate_lobp_action_oracle_from_trace_files(
        calibration,
        confirmatory,
    )
    assert report["precommit_target_count"] == 16
    assert report["full_episode_target_count"] == 16
    assert float(report["full_episode_score_nats"]) <= 0


def test_prefix_tv_reports_response_conflicting_partner_divergence(tmp_path: Path) -> None:
    path = tmp_path / "divergence.jsonl"
    steps = tuple(
        _step(step).model_copy(
            update={
                "episode_id": f"{partner}-{episode}",
                "partner_id": partner,
                "visible_partner_action": action,
            }
        )
        for partner, action in (("partner-a", 0), ("partner-b", 1))
        for episode in range(4)
        for step in range(8)
    )
    write_trace_jsonl(path, steps)
    report = estimate_prefix_tv_curves(
        path,
        response_signatures={"partner-a": "A", "partner-b": "B"},
    )
    assert report["mean_prefix_tv"]["pre_commitment"] == pytest.approx(1.0)
    assert report["mean_conflicting_prefix_tv"]["pre_commitment"] == pytest.approx(1.0)


def _step(
    step: int,
    *,
    events: tuple[TaskEvent, ...] = (),
    high_level_events: tuple[str, ...] = (),
) -> CommitmentTraceStep:
    return CommitmentTraceStep(
        episode_id="episode-0",
        partner_id="partner-0",
        reward_vector_id="reward-0",
        layout_id="demo_cook_simple",
        environment_key="0",
        work_unit=0,
        step=step,
        ego_action="stay",
        events=events,
        high_level_events=high_level_events,
    )


def _candidate(index: int, *, high: bool) -> CandidatePartnerMetrics:
    features = [0.0] * 8
    features[index % 8] = 1.0
    return CandidatePartnerMetrics(
        partner_id=f"partner-{index:02d}",
        reward_vector_id=f"reward-{index:02d}",
        response_cluster=index % 8,
        competence=0.95,
        best_fixed_response_value=0.75,
        br_prox=0.8,
        lobp_score_nats=-0.5,
        commitment_reached_rate=0.9,
        passive_dri=0.8 if high else 0.1,
        active_dri=0.9 if high else 0.2,
        trajectory_divergence=0.4,
        br_event_features=tuple(features),
    )
