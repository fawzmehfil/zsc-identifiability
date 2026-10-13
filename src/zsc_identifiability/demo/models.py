"""Versioned contracts for curated BeforeCommit replay artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DemoLayoutId = Literal["random3_m", "small_corridor"]
DemoCategory = Literal[
    "clear_before_commitment",
    "late_evidence",
    "evidence_outcome_gap",
    "robust_success",
    "intervention_control",
    "intervention_treatment",
]
EvidencePolicy = Literal[
    "ordinary_progress",
    "temporary_role_takeover",
    "corridor_yield",
]


class FrozenDemoModel(BaseModel):
    """Strict immutable base model used by public replay contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DemoCurationEntry(FrozenDemoModel):
    replay_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*-v[0-9]+$")
    title: str = Field(min_length=1)
    short_title: str = Field(min_length=1)
    category: DemoCategory
    layout_id: DemoLayoutId
    partner_id: str = Field(min_length=1)
    left_partner_id: str = Field(min_length=1)
    right_partner_id: str = Field(min_length=1)
    evidence_policy: EvidencePolicy
    environment_key: int = Field(ge=0)
    ego_seat: Literal[0, 1]
    representative: bool
    featured: bool
    selection_note: str = Field(min_length=1)
    caption: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_partner_pair(self) -> DemoCurationEntry:
        if self.left_partner_id == self.right_partner_id:
            raise ValueError("curated comparison partners must be distinct")
        if self.partner_id not in {self.left_partner_id, self.right_partner_id}:
            raise ValueError("recorded partner must belong to the comparison pair")
        if any(
            not partner.startswith(f"{self.layout_id}:")
            for partner in (
                self.partner_id,
                self.left_partner_id,
                self.right_partner_id,
            )
        ):
            raise ValueError("curated partners must match the selected layout")
        if self.layout_id == "small_corridor" and self.evidence_policy == (
            "temporary_role_takeover"
        ):
            raise ValueError("temporary role takeover belongs to random3_m")
        if self.layout_id == "random3_m" and self.evidence_policy == "corridor_yield":
            raise ValueError("corridor yield belongs to small_corridor")
        return self


class DemoCuration(FrozenDemoModel):
    schema_version: Literal[1] = 1
    collection_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    entries: tuple[DemoCurationEntry, ...]

    @model_validator(mode="after")
    def validate_collection(self) -> DemoCuration:
        identifiers = [entry.replay_id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("curated replay identifiers must be unique")
        if len(self.entries) < 6:
            raise ValueError("the public collection requires at least six replays")
        if sum(entry.featured for entry in self.entries) != 1:
            raise ValueError("the public collection requires exactly one featured replay")
        return self


class ReplaySource(FrozenDemoModel):
    suite_id: str
    layout_id: DemoLayoutId
    environment_key: int = Field(ge=0)
    ego_seat: Literal[0, 1]
    ego_policy_id: str
    partner_id: str
    evidence_policy: EvidencePolicy
    checkpoint_hashes: dict[str, str]
    source_artifact_hashes: dict[str, str]
    policy_training_performed: Literal[False] = False


class CommitmentRecord(FrozenDemoModel):
    definition: Literal["first_successful_ingredient_placement"]
    step: int | None = Field(default=None, ge=0)
    frame: int | None = Field(default=None, ge=0)
    reached: bool


class ReplayResponse(FrozenDemoModel):
    response_id: str
    alias: str
    optimal_for: tuple[Literal["left", "right"], ...]


class DiagnosticPoint(FrozenDemoModel):
    key: str
    prefix: int | Literal["pre_commitment", "eventual"]
    frame: int = Field(ge=0)
    phase: Literal["before", "commitment", "after"]
    probability_left: float = Field(ge=0, le=1)
    probability_right: float = Field(ge=0, le=1)
    recommended_response_id: str
    recommended_response_alias: str
    estimated_expected_risk: float = Field(ge=0)
    selected_response_loss_for_true_mode: float = Field(ge=0)
    aggregate_event_dri: float | None = None


class ReplayEvent(FrozenDemoModel):
    frame: int = Field(ge=0)
    step: int = Field(ge=0)
    kind: str
    label: str
    phase: Literal["before", "commitment", "after"]


class KitchenItem(FrozenDemoModel):
    kind: str
    ingredients: tuple[str, ...] = ()
    cooking_tick: int | None = None
    ready: bool = False


class KitchenPlayer(FrozenDemoModel):
    physical_index: Literal[0, 1]
    role: Literal["ego", "partner"]
    x: int
    y: int
    facing: Literal["north", "south", "east", "west"]
    held: KitchenItem | None = None


class KitchenObject(FrozenDemoModel):
    x: int
    y: int
    item: KitchenItem


class KitchenState(FrozenDemoModel):
    timestep: int = Field(ge=0)
    players: tuple[KitchenPlayer, KitchenPlayer]
    objects: tuple[KitchenObject, ...] = ()


class ReplayFrame(FrozenDemoModel):
    frame: int = Field(ge=0)
    action_step: int | None = Field(default=None, ge=0)
    state: KitchenState
    ego_action: int | None = None
    partner_action: int | None = None
    sparse_reward: float = 0.0
    event_kinds: tuple[str, ...] = ()
    phase: Literal["before", "commitment", "after"]


class ReplaySummary(FrozenDemoModel):
    sparse_return: float
    first_delivery_step: int | None = Field(default=None, ge=0)
    steps_elapsed: int = Field(ge=1)
    prior_risk: float = Field(ge=0)
    aggregate_dri_primary: float | None
    aggregate_dri_event: float | None
    aggregate_residual_risk_primary: float = Field(ge=0)
    selected_response_loss_at_commitment: float = Field(ge=0)
    eventual_dri_primary: float | None


class ReplayManifest(FrozenDemoModel):
    schema_version: Literal["beforecommit_replay_1.0.0"] = (
        "beforecommit_replay_1.0.0"
    )
    replay_id: str
    collection_id: str
    title: str
    short_title: str
    category: DemoCategory
    caption: str
    selection_note: str
    representative: bool
    featured: bool
    status: Literal["complete"] = "complete"
    source: ReplaySource
    commitment: CommitmentRecord
    terrain: tuple[tuple[str, ...], ...]
    partner_modes: dict[Literal["left", "right"], str]
    responses: tuple[ReplayResponse, ...]
    summary: ReplaySummary
    frame_count: int = Field(ge=2)
    diagnostic_count: int = Field(ge=1)
    event_count: int = Field(ge=0)
    artifacts: dict[str, str]


class ReplayCatalogEntry(FrozenDemoModel):
    replay_id: str
    title: str
    short_title: str
    category: DemoCategory
    caption: str
    layout_id: DemoLayoutId
    evidence_policy: EvidencePolicy
    featured: bool
    representative: bool
    sparse_return: float
    commitment_step: int | None
    aggregate_dri_primary: float | None
    eventual_dri_primary: float | None
    frame_count: int


class ReplayCatalog(FrozenDemoModel):
    schema_version: Literal[1] = 1
    collection_id: str
    items: tuple[ReplayCatalogEntry, ...]
    total: int = Field(ge=1)


class BenchmarkSummary(FrozenDemoModel):
    schema_version: Literal[1] = 1
    title: str
    partner_count: Literal[50] = 50
    method_family_count: Literal[6] = 6
    layout_count: Literal[2] = 2
    seats_per_layout: Literal[2] = 2
    confirmation_episode_count: Literal[9600] = 9600
    held_out_delta_r2: float
    adjusted_dri_coefficient: float
    coefficient_interval: tuple[float, float]
    verdict: Literal["complete_measurement_only"]
    intervention_result: Literal["neither_confirmed"]
    source_hash: str
