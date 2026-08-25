"""Immutable schema for version-one finite convention games."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zsc_identifiability.numeric import parse_rational

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]*$")
FORBIDDEN_ACTION_WORDS = {"identify", "query", "ask_type", "reveal_type", "diagnose"}


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WeightedId(FrozenModel):
    id: str
    probability: str


class Availability(FrozenModel):
    state: str
    times: tuple[int, ...]


class ActionSpec(FrozenModel):
    id: str
    description: str
    task_semantics: str = Field(min_length=4)
    passive: bool = False
    kind: Literal["task"] = "task"
    available: tuple[Availability, ...]


class OutcomeSpec(FrozenModel):
    next_state: str
    observation: str
    probability: str
    cost: str = "0"


class KernelRow(FrozenModel):
    time: int = Field(ge=0)
    state: str
    action: str
    mode: str
    outcomes: tuple[OutcomeSpec, ...]


class LossSpec(FrozenModel):
    mode: str
    decision: str
    loss: str


class PostObservationSpec(FrozenModel):
    mode: str
    observations: tuple[WeightedId, ...]


class AnalyticalExpectation(FrozenModel):
    name: str
    value: str | None = None
    text: str | None = None


class FiniteConventionGame(FrozenModel):
    """Validated serial representation of a finite convention game."""

    schema_version: Literal[1]
    game_id: str
    description: str
    horizon: int = Field(ge=1)
    modes: tuple[WeightedId, ...]
    states: tuple[str, ...]
    initial_state: str
    observations: tuple[str, ...]
    actions: tuple[ActionSpec, ...]
    decisions: tuple[str, ...]
    kernels: tuple[KernelRow, ...]
    decision_losses: tuple[LossSpec, ...]
    post_commitment_observations: tuple[PostObservationSpec, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)
    analytical_expectations: tuple[AnalyticalExpectation, ...] = ()

    @model_validator(mode="after")
    def validate_semantics(self) -> FiniteConventionGame:
        self._validate_identifiers()
        mode_ids = tuple(item.id for item in self.modes)
        action_ids = tuple(item.id for item in self.actions)
        self._require_unique("mode", mode_ids)
        self._require_unique("state", self.states)
        self._require_unique("observation", self.observations)
        self._require_unique("action", action_ids)
        self._require_unique("decision", self.decisions)
        if self.initial_state not in self.states:
            raise ValueError("initial_state is unknown")
        self._require_distribution("prior", tuple(item.probability for item in self.modes), True)
        for action in self.actions:
            lower = f"{action.id} {action.task_semantics}".lower()
            if any(word in lower for word in FORBIDDEN_ACTION_WORDS):
                raise ValueError(f"action {action.id!r} has special query/identification semantics")
            if not action.available:
                raise ValueError(f"action {action.id!r} is never available")
            for availability in action.available:
                if availability.state not in self.states:
                    raise ValueError(f"action {action.id!r} uses unknown state")
                if not availability.times:
                    raise ValueError(f"action {action.id!r} has empty availability")
                if any(time < 0 or time >= self.horizon for time in availability.times):
                    raise ValueError(
                        f"action {action.id!r} is available outside pre-commitment horizon"
                    )
        self._validate_kernels(set(mode_ids), set(action_ids))
        self._validate_losses(set(mode_ids))
        self._validate_post_observations(set(mode_ids))
        return self

    def _validate_identifiers(self) -> None:
        values = [
            self.game_id,
            *(item.id for item in self.modes),
            *self.states,
            *self.observations,
            *(item.id for item in self.actions),
            *self.decisions,
        ]
        for value in values:
            if not IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid stable identifier: {value!r}")

    @staticmethod
    def _require_unique(label: str, values: tuple[str, ...]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} identifier")

    @staticmethod
    def _require_distribution(label: str, values: tuple[str, ...], positive: bool) -> None:
        probabilities = tuple(parse_rational(value) for value in values)
        if any(value < 0 for value in probabilities):
            raise ValueError(f"{label} contains a negative probability")
        if positive and any(value == 0 for value in probabilities):
            raise ValueError(f"{label} must have positive mass on every mode")
        if sum(probabilities) != 1:
            raise ValueError(f"{label} probabilities do not sum to one")

    def _validate_kernels(self, modes: set[str], actions: set[str]) -> None:
        observed_keys: set[tuple[int, str, str, str]] = set()
        for row in self.kernels:
            if row.mode not in modes or row.action not in actions or row.state not in self.states:
                raise ValueError("kernel row uses an unknown mode, action, or state")
            key = (row.time, row.state, row.action, row.mode)
            if key in observed_keys:
                raise ValueError(f"duplicate kernel row: {key}")
            observed_keys.add(key)
            if not row.outcomes:
                raise ValueError(f"kernel row {key} has no outcomes")
            self._require_distribution(
                f"kernel row {key}", tuple(x.probability for x in row.outcomes), False
            )
            for outcome in row.outcomes:
                if (
                    outcome.next_state not in self.states
                    or outcome.observation not in self.observations
                ):
                    raise ValueError(f"kernel row {key} uses an unknown outcome identifier")
                if parse_rational(outcome.cost) < 0:
                    raise ValueError(f"kernel row {key} has negative intervention cost")
        for action in self.actions:
            for availability in action.available:
                for time in availability.times:
                    for mode in modes:
                        key = (time, availability.state, action.id, mode)
                        if key not in observed_keys:
                            raise ValueError(f"missing kernel for available action and mode: {key}")

    def _validate_losses(self, modes: set[str]) -> None:
        loss_map: dict[tuple[str, str], Fraction] = {}
        for item in self.decision_losses:
            if item.mode not in modes or item.decision not in self.decisions:
                raise ValueError("loss row uses unknown mode or decision")
            key = (item.mode, item.decision)
            if key in loss_map:
                raise ValueError(f"duplicate loss row: {key}")
            value = parse_rational(item.loss)
            if value < 0:
                raise ValueError("decision losses must be nonnegative")
            loss_map[key] = value
        for mode in modes:
            values: list[Fraction] = []
            for decision in self.decisions:
                key = (mode, decision)
                if key not in loss_map:
                    raise ValueError(f"missing decision loss: {key}")
                values.append(loss_map[key])
            if min(values) != 0:
                raise ValueError(f"mode {mode!r} has no zero-loss decision")

    def _validate_post_observations(self, modes: set[str]) -> None:
        seen: set[str] = set()
        for row in self.post_commitment_observations:
            if row.mode not in modes or row.mode in seen:
                raise ValueError("invalid or duplicate post-commitment mode")
            seen.add(row.mode)
            self._require_distribution(
                f"post-commitment row {row.mode}",
                tuple(item.probability for item in row.observations),
                False,
            )
        if seen and seen != modes:
            raise ValueError("post-commitment kernel must cover every mode")

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.modes)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.actions)

    def prior_exact(self) -> tuple[Fraction, ...]:
        return tuple(parse_rational(item.probability) for item in self.modes)

    def loss_exact(self, mode: str, decision: str) -> Fraction:
        for item in self.decision_losses:
            if item.mode == mode and item.decision == decision:
                return parse_rational(item.loss)
        raise KeyError((mode, decision))

    def action(self, action_id: str) -> ActionSpec:
        return next(action for action in self.actions if action.id == action_id)

    def available_actions(
        self, state: str, time: int, passive_only: bool = False
    ) -> tuple[str, ...]:
        result = []
        for action in self.actions:
            if passive_only and not action.passive:
                continue
            if any(item.state == state and time in item.times for item in action.available):
                result.append(action.id)
        return tuple(sorted(result))

    def kernel(self, time: int, state: str, action: str, mode: str) -> KernelRow:
        for row in self.kernels:
            if (row.time, row.state, row.action, row.mode) == (time, state, action, mode):
                return row
        raise KeyError((time, state, action, mode))


def load_game_file(path: str | Path) -> FiniteConventionGame:
    with Path(path).open(encoding="utf-8") as handle:
        return FiniteConventionGame.model_validate(json.load(handle))
