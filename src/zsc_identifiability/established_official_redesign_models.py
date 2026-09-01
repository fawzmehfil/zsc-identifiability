"""Versioned models for the Stage 6 v3 decision-risk measurement redesign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zsc_identifiability.established_official_models import (
    OfficialLayoutId,
    OfficialRolloutLedgerEntry,
    OfficialRolloutPlan,
    OfficialRuntimeSpec,
    OfficialVerdict,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"

MeasurementRepresentation = Literal["gru", "event"]
MeasurementPrefix = int | Literal["pre_commitment", "eventual"]


class FrozenRedesignModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OfficialV2SourceLock(FrozenRedesignModel):
    """Immutable references to the completed, scientifically failed v2 audit."""

    v2_suite_path: str
    v2_suite_hash: str = Field(pattern=SHA256_PATTERN)
    v2_rollout_plan_path: str
    v2_rollout_plan_file_hash: str = Field(pattern=SHA256_PATTERN)
    v2_rollout_plan_hash: str = Field(pattern=SHA256_PATTERN)
    v2_rollout_ledger_path: str
    v2_rollout_ledger_hash: str = Field(pattern=SHA256_PATTERN)
    v2_asset_inventory_path: str
    v2_inventory_hash: str = Field(pattern=SHA256_PATTERN)
    v2_manifest_path: str
    v2_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    v2_source_hash: str = Field(pattern=SHA256_PATTERN)
    v2_trace_index_path: str
    v2_trace_index_hash: str = Field(pattern=SHA256_PATTERN)
    response_value_matrices_path: str
    response_value_matrices_hash: str = Field(pattern=SHA256_PATTERN)
    official_method_evaluation_path: str
    official_method_evaluation_hash: str = Field(pattern=SHA256_PATTERN)
    exclusions_path: str
    exclusions_hash: str = Field(pattern=SHA256_PATTERN)
    v2_verdict: Literal["redesign"] = "redesign"


class OfficialMeasurementLayoutV3(FrozenRedesignModel):
    layout_id: OfficialLayoutId
    role: Literal["primary", "robustness"]
    expected_partner_count: int = Field(ge=1)
    evidence_policies: tuple[str, ...]
    fresh_episodes_per_partner_policy: int = Field(ge=2)
    frozen_intervention: str
    max_episode_steps: Literal[400] = 400

    @model_validator(mode="after")
    def validate_layout(self) -> OfficialMeasurementLayoutV3:
        if self.fresh_episodes_per_partner_policy % 2:
            raise ValueError("fresh confirmation episodes must balance the two player seats")
        if not self.evidence_policies or self.evidence_policies[0] != "ordinary_progress":
            raise ValueError("ordinary_progress must be the first evidence policy")
        if self.frozen_intervention not in self.evidence_policies:
            raise ValueError("the frozen intervention must remain in the reported option set")
        return self


class OfficialMeasurementRepresentationSpecV3(FrozenRedesignModel):
    primary: Literal["gru"] = "gru"
    sensitivity: Literal["event"] = "event"
    gru_hidden_size: Literal[64] = 64
    gru_seeds: tuple[int, ...] = (6173, 6174, 6175, 6176, 6177)
    encoder_training_fraction: float = Field(default=0.75, gt=0, lt=1)
    event_feature_width: Literal[512] = 512
    event_hash_salt: Literal["zsc-dri-v3-event-features"] = "zsc-dri-v3-event-features"
    temporal_bins: tuple[str, ...] = ("0-7", "8-15", "16-31", "32+")
    prefixes: tuple[MeasurementPrefix, ...] = (0, 8, 16, 32, "pre_commitment", "eventual")

    @model_validator(mode="after")
    def validate_representation(self) -> OfficialMeasurementRepresentationSpecV3:
        if self.gru_seeds != (6173, 6174, 6175, 6176, 6177):
            raise ValueError("v3 uses the five frozen GRU estimator seeds")
        if self.temporal_bins != ("0-7", "8-15", "16-31", "32+"):
            raise ValueError("v3 event features use the registered absolute temporal bins")
        if abs(self.encoder_training_fraction - 0.75) > 1e-12:
            raise ValueError("v3 freezes the encoder-training fraction at 0.75")
        return self


class OfficialPairwiseDecoderSpecV3(FrozenRedesignModel):
    ridge_strengths: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    temperatures: tuple[float, ...]
    prior_shrinkages: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    maximum_iterations: int = Field(default=100, ge=1)
    convergence_tolerance: float = Field(default=1e-9, gt=0)

    @model_validator(mode="after")
    def validate_decoder(self) -> OfficialPairwiseDecoderSpecV3:
        if self.ridge_strengths != (0.01, 0.1, 1.0, 10.0):
            raise ValueError("v3 uses the frozen pairwise ridge grid")
        if len(self.temperatures) != 81:
            raise ValueError("v3 requires 81 registered temperature candidates")
        expected_temperatures = tuple(0.25 * (2.0 ** (index / 20.0)) for index in range(81))
        if any(
            abs(observed - expected) > 1e-12
            for observed, expected in zip(
                self.temperatures, expected_temperatures, strict=True
            )
        ):
            raise ValueError("v3 temperatures must be log-spaced from 0.25 to 4")
        if self.prior_shrinkages != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("v3 uses the frozen prior-shrinkage grid")
        if any(value <= 0 for value in self.ridge_strengths + self.temperatures):
            raise ValueError("ridge strengths and temperatures must be positive")
        if any(value < 0 or value > 1 for value in self.prior_shrinkages):
            raise ValueError("prior shrinkage must lie in [0, 1]")
        return self


class OfficialMeasurementCalibrationSpecV3(FrozenRedesignModel):
    synthetic_tolerance: float = Field(default=0.03, ge=0)
    permutation_repeats: Literal[100] = 100
    correction: Literal["holm"] = "holm"
    permutation_seed: Literal[9317] = 9317
    direct_binary_pair_count: Literal[10] = 10
    brier_improvement_required: Literal[True] = True
    event_sign_consistency_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_tolerance(self) -> OfficialMeasurementCalibrationSpecV3:
        if abs(self.synthetic_tolerance - 0.03) > 1e-12:
            raise ValueError("v3 freezes the synthetic tolerance at 0.03")
        return self


class OfficialMeasurementStatisticsSpecV3(FrozenRedesignModel):
    ridge_strengths: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0)
    bootstrap_resamples: Literal[10000] = 10000
    correction: Literal["holm"] = "holm"
    held_out_unit: Literal["hsp_scheme"] = "hsp_scheme"
    intervention_completion_threshold: float = Field(default=0.8, ge=0, le=1)

    @model_validator(mode="after")
    def validate_intervention_threshold(self) -> OfficialMeasurementStatisticsSpecV3:
        if self.ridge_strengths != (0.0, 0.01, 0.1, 1.0, 10.0):
            raise ValueError("v3 uses the frozen scheme-held-out ridge grid")
        if abs(self.intervention_completion_threshold - 0.8) > 1e-12:
            raise ValueError("v3 freezes the intervention-completion threshold at 0.8")
        return self


class OfficialMeasurementAuditSuiteV3(FrozenRedesignModel):
    schema_version: Literal[3] = 3
    audit_mode: Literal["official_measurement_redesign"] = "official_measurement_redesign"
    suite_id: str
    state_directory: str
    policy_training_allowed: Literal[False] = False
    measurement_model_training_only: Literal[True] = True
    fresh_confirmation_salt: Literal["zsc-stage6-v3-confirmatory-9d41"] = (
        "zsc-stage6-v3-confirmatory-9d41"
    )
    v2: OfficialV2SourceLock
    runtime: OfficialRuntimeSpec
    layouts: tuple[OfficialMeasurementLayoutV3, ...]
    representations: OfficialMeasurementRepresentationSpecV3
    decoder: OfficialPairwiseDecoderSpecV3
    calibration: OfficialMeasurementCalibrationSpecV3
    statistics: OfficialMeasurementStatisticsSpecV3

    @model_validator(mode="before")
    @classmethod
    def reject_policy_training(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = {
                "policy_training",
                "policy_training_budget",
                "transition_budget",
                "partner_generation",
                "train_partners",
                "train_methods",
            }

            def visit(value: Any, path: tuple[str, ...] = ()) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in forbidden:
                            raise ValueError(
                                "policy training is structurally forbidden in Stage 6 v3: "
                                + ".".join((*path, key))
                            )
                        visit(child, (*path, key))
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        visit(child, (*path, str(index)))

            visit(data)
        return data

    @model_validator(mode="after")
    def validate_contract(self) -> OfficialMeasurementAuditSuiteV3:
        if len(self.layouts) != 2 or {item.role for item in self.layouts} != {
            "primary",
            "robustness",
        }:
            raise ValueError("v3 requires exactly one primary and one robustness layout")
        if len({item.layout_id for item in self.layouts}) != 2:
            raise ValueError("v3 layout identifiers must be unique")
        episode_total = sum(
            item.expected_partner_count
            * len(item.evidence_policies)
            * item.fresh_episodes_per_partner_policy
            for item in self.layouts
        )
        if episode_total != 9600:
            raise ValueError("the frozen v3 confirmation design must contain 9,600 episodes")
        expected = {
            "random3_m": (
                "primary",
                30,
                (
                    "ordinary_progress",
                    "stage_onion",
                    "stage_tomato",
                    "temporary_role_takeover",
                ),
                64,
                "temporary_role_takeover",
            ),
            "small_corridor": (
                "robustness",
                20,
                ("ordinary_progress", "stage_onion", "corridor_yield"),
                32,
                "corridor_yield",
            ),
        }
        for layout in self.layouts:
            role, count, options, episodes, intervention = expected[layout.layout_id]
            if (
                layout.role,
                layout.expected_partner_count,
                layout.evidence_policies,
                layout.fresh_episodes_per_partner_policy,
                layout.frozen_intervention,
            ) != (role, count, options, episodes, intervention):
                raise ValueError(f"{layout.layout_id} differs from the frozen v3 protocol")
        return self


class MeasurementRepresentationArtifact(FrozenRedesignModel):
    artifact_id: str
    layout_id: OfficialLayoutId
    evidence_policy: str
    prefix: MeasurementPrefix
    representation: MeasurementRepresentation
    seed: int | None = None
    identity_temperature: float | None = Field(default=None, gt=0)
    encoder_checkpoint_path: str | None = None
    encoder_checkpoint_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    calibration_embeddings_path: str
    calibration_embeddings_hash: str = Field(pattern=SHA256_PATTERN)
    validation_embeddings_path: str
    validation_embeddings_hash: str = Field(pattern=SHA256_PATTERN)
    calibration_key_hash: str = Field(pattern=SHA256_PATTERN)
    validation_key_hash: str = Field(pattern=SHA256_PATTERN)
    feature_width: int = Field(ge=1)


class MeasurementRepresentationManifest(FrozenRedesignModel):
    schema_version: Literal[1] = 1
    suite_id: str
    suite_hash: str = Field(pattern=SHA256_PATTERN)
    source_v2_plan_hash: str = Field(pattern=SHA256_PATTERN)
    source_v2_trace_index_hash: str = Field(pattern=SHA256_PATTERN)
    calibration_data_hash: str = Field(pattern=SHA256_PATTERN)
    validation_data_hash: str = Field(pattern=SHA256_PATTERN)
    artifacts: tuple[MeasurementRepresentationArtifact, ...]
    policy_training_performed: Literal[False] = False
    complete: bool


class PairwiseDecisionDecoder(FrozenRedesignModel):
    decoder_id: str
    layout_id: OfficialLayoutId
    evidence_policy: str
    prefix: MeasurementPrefix
    representation: MeasurementRepresentation
    seed: int | None = None
    left_partner_id: str
    right_partner_id: str
    ridge_strength: float = Field(gt=0)
    temperature: float = Field(gt=0)
    prior_shrinkage: float = Field(ge=0, le=1)
    coefficient: tuple[float, ...]
    intercept: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    configuration_id: str
    calibration_examples: int = Field(ge=2)
    validation_examples: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_dimensions(self) -> PairwiseDecisionDecoder:
        width = len(self.coefficient)
        if width < 1 or len(self.feature_mean) != width or len(self.feature_scale) != width:
            raise ValueError("decoder coefficient and standardization widths differ")
        if any(value <= 0 for value in self.feature_scale):
            raise ValueError("decoder feature scales must be positive")
        return self


class MeasurementFitManifest(FrozenRedesignModel):
    schema_version: Literal[1] = 1
    suite_id: str
    suite_hash: str = Field(pattern=SHA256_PATTERN)
    source_v2_plan_hash: str = Field(pattern=SHA256_PATTERN)
    source_v2_trace_index_hash: str = Field(pattern=SHA256_PATTERN)
    calibration_data_hash: str = Field(pattern=SHA256_PATTERN)
    validation_data_hash: str = Field(pattern=SHA256_PATTERN)
    artifacts: tuple[MeasurementRepresentationArtifact, ...]
    decoder_manifest_path: str
    decoder_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    frozen_configuration_hash: str = Field(pattern=SHA256_PATTERN)
    policy_training_performed: Literal[False] = False
    complete: bool


class OfficialConfirmationPlan(FrozenRedesignModel):
    schema_version: Literal[1] = 1
    suite_id: str
    suite_path: str
    suite_hash: str = Field(pattern=SHA256_PATTERN)
    frozen_configuration_hash: str = Field(pattern=SHA256_PATTERN)
    fit_manifest_path: str
    fit_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    source_v2_plan_hash: str = Field(pattern=SHA256_PATTERN)
    source_v2_environment_key_hash: str = Field(pattern=SHA256_PATTERN)
    fresh_environment_key_hash: str = Field(pattern=SHA256_PATTERN)
    workspace: str
    rollout_plan: OfficialRolloutPlan
    plan_hash: str = Field(pattern=SHA256_PATTERN)


class OfficialConfirmationLedger(FrozenRedesignModel):
    schema_version: Literal[1] = 1
    suite_id: str
    plan_hash: str = Field(pattern=SHA256_PATTERN)
    frozen_configuration_hash: str = Field(pattern=SHA256_PATTERN)
    entries: tuple[OfficialRolloutLedgerEntry, ...]
    complete: bool
    failed_shards: tuple[str, ...] = ()


class PairwiseDecisionValueRow(FrozenRedesignModel):
    layout_id: OfficialLayoutId
    left_partner_id: str
    right_partner_id: str
    left_scheme_id: str
    right_scheme_id: str
    evidence_policy: str
    representation: MeasurementRepresentation
    prefix: MeasurementPrefix
    prior_risk: float = Field(ge=0)
    residual_risk: float = Field(ge=0)
    fixed_response_risk: float = Field(ge=0)
    dri: float | None
    brier_score: float = Field(ge=0)
    uniform_brier_score: float = Field(ge=0)
    commitment_rate: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=1)
    seed_dri: tuple[float, ...] = ()
    identity_mi_nats: float | None = Field(default=None, ge=0)
    decision_signature_mi_nats: float | None = Field(default=None, ge=0)


class MeasurementCalibrationReportV3(FrozenRedesignModel):
    schema_version: Literal[3] = 3
    synthetic_controls: dict[str, Any]
    leakage_checks: dict[str, bool]
    brier_checks: dict[str, bool]
    fixed_response_checks: dict[str, bool]
    permutation_tests: tuple[dict[str, Any], ...]
    holm_adjusted: dict[str, float]
    seed_stability: dict[str, Any]
    event_sign_sensitivity: dict[str, bool]
    direct_binary_gru_diagnostic: dict[str, Any]
    passed: bool


class OfficialMeasurementAuditManifestV3(FrozenRedesignModel):
    schema_version: Literal[3] = 3
    suite_id: str
    status: Literal["complete", "incomplete", "invalid"]
    verdict: OfficialVerdict
    v2_preserved: bool
    policy_training_performed: Literal[False] = False
    confirmation_complete: bool
    calibration_passed: bool
    scientific_gates: dict[str, bool | None]
    source_hashes: dict[str, str]
    generated_files: tuple[str, ...]
    total_fresh_episodes: int = Field(ge=0)
    total_fresh_environment_steps: int = Field(ge=0)
    peak_workers: int = Field(ge=0, le=4)


def load_official_measurement_suite(
    path: str | Path,
) -> OfficialMeasurementAuditSuiteV3:
    with Path(path).open(encoding="utf-8") as handle:
        return OfficialMeasurementAuditSuiteV3.model_validate(json.load(handle))
