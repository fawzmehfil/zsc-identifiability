"""Deterministic generators for matched finite convention-game populations."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

from zsc_identifiability.benchmark_models import (
    BinaryCellSpec,
    BinaryFamilySpec,
    FactorizedCellSpec,
    FactorizedFamilySpec,
    GeneratedBenchmarkSet,
    GeneratedPopulation,
    GeneratedPopulationDescriptor,
    MatchedBenchmarkSuite,
    SignalTarget,
    SweepSpec,
)
from zsc_identifiability.models import (
    ActionSpec,
    AnalyticalExpectation,
    Availability,
    FiniteConventionGame,
    KernelRow,
    LossSpec,
    OutcomeSpec,
    PostObservationSpec,
    WeightedId,
)

BINARY_SYMMETRIES = (
    ("identity", False, False),
    ("role_swap", True, False),
    ("signal_swap", False, True),
    ("role_signal_swap", True, True),
)
FACTORIZED_SYMMETRIES = tuple(
    (
        "_".join(
            part
            for part, enabled in (
                ("role_swap", role_flip),
                ("subtype_swap", subtype_flip),
                ("signal_swap", signal_flip),
            )
            if enabled
        )
        or "identity",
        role_flip,
        subtype_flip,
        signal_flip,
    )
    for role_flip in (False, True)
    for subtype_flip in (False, True)
    for signal_flip in (False, True)
)


def generate(spec: MatchedBenchmarkSuite, backend: str = "fraction") -> GeneratedBenchmarkSet:
    """Generate all canonical, symmetry, and one-factor-at-a-time populations."""
    if backend not in {"fraction", "float"}:
        raise ValueError(f"unknown backend: {backend!r}")
    suite_hash = _content_hash(spec.model_dump(mode="json"))
    populations: list[GeneratedPopulation] = []
    for family in spec.families:
        if isinstance(family, BinaryFamilySpec):
            populations.extend(_generate_binary_family(spec, family, suite_hash))
        elif isinstance(family, FactorizedFamilySpec):
            populations.extend(_generate_factorized_family(spec, family, suite_hash))
        else:  # pragma: no cover - discriminated Pydantic union prevents this
            raise TypeError(f"unsupported family: {type(family)!r}")
    identifiers = [item.descriptor.population_id for item in populations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("generated duplicate population identifiers")
    result = GeneratedBenchmarkSet(spec, tuple(populations), suite_hash)
    known = set(result.by_id())
    for contract in spec.matching_contracts:
        missing = {contract.left_population_id, contract.right_population_id} - known
        if missing:
            raise ValueError(
                f"matching contract {contract.contract_id!r} references missing populations: "
                f"{sorted(missing)}"
            )
    return result


def _generate_binary_family(
    suite: MatchedBenchmarkSuite,
    family: BinaryFamilySpec,
    suite_hash: str,
) -> list[GeneratedPopulation]:
    populations: list[GeneratedPopulation] = []
    symmetries = BINARY_SYMMETRIES if family.generate_symmetries else BINARY_SYMMETRIES[:1]
    for cell in family.cells:
        for symmetry_id, role_flip, signal_flip in symmetries:
            populations.append(
                _binary_population(
                    suite,
                    family,
                    cell,
                    family.reliability,
                    cell.intervention_cost or family.intervention_cost,
                    symmetry_id,
                    role_flip,
                    signal_flip,
                    suite_hash,
                )
            )
    for sweep in family.sweeps:
        base = next(cell for cell in family.cells if cell.cell_id == sweep.cell_id)
        for value in sweep.values:
            cell, reliability, cost = _apply_binary_sweep(base, family, sweep, value)
            populations.append(
                _binary_population(
                    suite,
                    family,
                    cell,
                    reliability,
                    cost,
                    f"sweep_{sweep.sweep_id}_{_slug(value)}",
                    False,
                    False,
                    suite_hash,
                )
            )
    return populations


def _apply_binary_sweep(
    cell: BinaryCellSpec,
    family: BinaryFamilySpec,
    sweep: SweepSpec,
    value: str,
) -> tuple[BinaryCellSpec, str, str]:
    reliability = family.reliability
    cost = cell.intervention_cost or family.intervention_cost
    if sweep.parameter == "reliability":
        reliability = value
    elif sweep.parameter == "intervention_cost":
        cost = value
    elif sweep.parameter == "evidence_slot":
        cell = cell.model_copy(update={"passive_evidence_slot": value})
    elif sweep.parameter == "active_signal_target":
        cell = cell.model_copy(update={"active_signal_target": value})
    else:
        raise ValueError(f"sweep parameter {sweep.parameter!r} is invalid for binary family")
    return cell, reliability, cost


def _binary_population(
    suite: MatchedBenchmarkSuite,
    family: BinaryFamilySpec,
    cell: BinaryCellSpec,
    reliability: str,
    intervention_cost: str,
    symmetry_id: str,
    role_flip: bool,
    signal_flip: bool,
    suite_hash: str,
) -> GeneratedPopulation:
    suffix = "" if symmetry_id in {item[0] for item in BINARY_SYMMETRIES} else ""
    population_id = f"{family.family_id}--{cell.cell_id}--{symmetry_id}{suffix}"
    modes = ("partner_alpha", "partner_beta")
    decisions = ("take_role_a", "take_role_b")
    correct_bit = {
        modes[0]: 1 if role_flip else 0,
        modes[1]: 0 if role_flip else 1,
    }
    if cell.shared_response:
        correct_bit = {mode: 0 for mode in modes}
    actions = (
        ActionSpec(
            id="advance_shared_task",
            description="Advance the shared task along its default route.",
            task_semantics="Make ordinary progress on the shared preparation.",
            passive=True,
            available=(Availability(state="start", times=(0,)),),
        ),
        ActionSpec(
            id="stage_shared_item",
            description="Place a shared item where either role may claim it.",
            task_semantics="Stage an ordinary shared resource before role allocation.",
            passive=False,
            available=(Availability(state="start", times=(0,)),),
        ),
        ActionSpec(
            id="complete_preparation",
            description="Finish the preparation before assigning roles.",
            task_semantics="Complete the ordinary preparation sequence.",
            passive=True,
            available=(Availability(state="preparing", times=(1,)),),
        ),
    )
    kernels: list[KernelRow] = []
    for mode in modes:
        passive_target: SignalTarget = "response" if cell.passive_evidence_slot == "0" else "null"
        kernels.append(
            _kernel_row(
                0,
                "start",
                "advance_shared_task",
                mode,
                "preparing",
                passive_target,
                correct_bit[mode],
                None,
                reliability,
                "0",
                signal_flip,
            )
        )
        kernels.append(
            _kernel_row(
                0,
                "start",
                "stage_shared_item",
                mode,
                "preparing",
                cell.active_signal_target,
                correct_bit[mode],
                None,
                reliability,
                intervention_cost,
                signal_flip,
            )
        )
        second_target: SignalTarget = "response" if cell.passive_evidence_slot == "1" else "null"
        kernels.append(
            _kernel_row(
                1,
                "preparing",
                "complete_preparation",
                mode,
                "commit_ready",
                second_target,
                correct_bit[mode],
                None,
                reliability,
                "0",
                signal_flip,
            )
        )
    post_target: SignalTarget = (
        "response" if cell.passive_evidence_slot == "post_commitment" else "null"
    )
    post = tuple(
        PostObservationSpec(
            mode=mode,
            observations=_signal_distribution(
                post_target,
                correct_bit[mode],
                None,
                "1" if post_target == "response" else reliability,
                signal_flip,
            ),
        )
        for mode in modes
    )
    losses = tuple(
        LossSpec(
            mode=mode,
            decision=decision,
            loss=("0" if decisions[correct_bit[mode]] == decision else family.mismatch_loss),
        )
        for mode in modes
        for decision in decisions
    )
    game = FiniteConventionGame(
        schema_version=1,
        game_id=population_id,
        description=(
            "Matched binary role-allocation population with ordinary task evidence and an "
            "auditable commitment boundary."
        ),
        horizon=2,
        modes=tuple(WeightedId(id=mode, probability="1/2") for mode in modes),
        states=("start", "preparing", "commit_ready"),
        initial_state="start",
        observations=("signal_zero", "signal_one"),
        actions=actions,
        decisions=decisions,
        kernels=tuple(kernels),
        decision_losses=losses,
        post_commitment_observations=post,
        metadata={
            "phase": "3",
            "family": family.family_id,
            "cell": cell.cell_id,
            "symmetry": symmetry_id,
            "ordinary_task_intervention": "stage_shared_item",
        },
        analytical_expectations=(
            AnalyticalExpectation(name="prior_risk", value="0" if cell.shared_response else "20"),
            AnalyticalExpectation(name="reliability", value=reliability),
        ),
    )
    response_signatures = {mode: decisions[correct_bit[mode]] for mode in modes}
    features: dict[str, tuple[str, ...]] = {
        mode: ("1", "0") if correct_bit[mode] == 0 else ("0", "1") for mode in modes
    }
    descriptor = _descriptor(
        suite,
        game,
        suite_hash,
        family.family_id,
        family.kind,
        cell.cell_id,
        cell.matching_group,
        symmetry_id,
        response_signatures,
        features,
        ("advance_shared_task", "complete_preparation"),
        ("start", "preparing", "commit_ready"),
        {
            "passive_evidence_slot": cell.passive_evidence_slot,
            "active_signal_target": cell.active_signal_target,
            "reliability": reliability,
        },
        {
            "intervention_cost": intervention_cost,
            "mismatch_loss": family.mismatch_loss,
            "horizon": "2",
        },
        {
            "prior_risk": "0" if cell.shared_response else "20",
            "response_signal_dri": "null" if cell.shared_response else _binary_dri(reliability),
        },
    )
    return GeneratedPopulation(descriptor, game)


def _generate_factorized_family(
    suite: MatchedBenchmarkSuite,
    family: FactorizedFamilySpec,
    suite_hash: str,
) -> list[GeneratedPopulation]:
    populations: list[GeneratedPopulation] = []
    symmetries = FACTORIZED_SYMMETRIES if family.generate_symmetries else FACTORIZED_SYMMETRIES[:1]
    for cell in family.cells:
        for symmetry_id, role_flip, subtype_flip, signal_flip in symmetries:
            populations.append(
                _factorized_population(
                    suite,
                    family,
                    cell,
                    family.reliability,
                    cell.intervention_cost or family.intervention_cost,
                    cell.distractor_steps
                    if cell.distractor_steps is not None
                    else family.distractor_steps,
                    symmetry_id,
                    role_flip,
                    subtype_flip,
                    signal_flip,
                    suite_hash,
                )
            )
    for sweep in family.sweeps:
        base = next(cell for cell in family.cells if cell.cell_id == sweep.cell_id)
        for value in sweep.values:
            cell, reliability, cost, distractors = _apply_factorized_sweep(
                base, family, sweep, value
            )
            populations.append(
                _factorized_population(
                    suite,
                    family,
                    cell,
                    reliability,
                    cost,
                    distractors,
                    f"sweep_{sweep.sweep_id}_{_slug(value)}",
                    False,
                    False,
                    False,
                    suite_hash,
                )
            )
    return populations


def _apply_factorized_sweep(
    cell: FactorizedCellSpec,
    family: FactorizedFamilySpec,
    sweep: SweepSpec,
    value: str,
) -> tuple[FactorizedCellSpec, str, str, int]:
    reliability = family.reliability
    cost = cell.intervention_cost or family.intervention_cost
    distractors = (
        cell.distractor_steps if cell.distractor_steps is not None else family.distractor_steps
    )
    if sweep.parameter == "reliability":
        reliability = value
    elif sweep.parameter == "intervention_cost":
        cost = value
    elif sweep.parameter == "distractor_steps":
        distractors = int(value)
    elif sweep.parameter == "active_signal_target":
        cell = cell.model_copy(update={"active_signal_target": value})
    else:
        raise ValueError(f"sweep parameter {sweep.parameter!r} is invalid for factorized family")
    return cell, reliability, cost, distractors


def _factorized_population(
    suite: MatchedBenchmarkSuite,
    family: FactorizedFamilySpec,
    cell: FactorizedCellSpec,
    reliability: str,
    intervention_cost: str,
    distractor_steps: int,
    symmetry_id: str,
    role_flip: bool,
    subtype_flip: bool,
    signal_flip: bool,
    suite_hash: str,
) -> GeneratedPopulation:
    population_id = f"{family.family_id}--{cell.cell_id}--{symmetry_id}"
    modes = ("partner_a0", "partner_a1", "partner_b0", "partner_b1")
    raw_bits = {
        "partner_a0": (0, 0),
        "partner_a1": (0, 1),
        "partner_b0": (1, 0),
        "partner_b1": (1, 1),
    }
    bits = {
        mode: (response ^ role_flip, subtype ^ subtype_flip)
        for mode, (response, subtype) in raw_bits.items()
    }
    decisions = ("take_role_a", "take_role_b")
    states = ["start"]
    for index in range(1, distractor_steps + 1):
        states.append(f"preparing_{index}")
    states.append("commit_ready")
    first_next = "commit_ready" if distractor_steps == 0 else "preparing_1"
    availability = tuple(
        Availability(state=f"preparing_{index}", times=(index,))
        for index in range(1, distractor_steps + 1)
    )
    actions = [
        ActionSpec(
            id="advance_shared_task",
            description="Advance the shared role-allocation task normally.",
            task_semantics="Make ordinary progress before assigning complementary roles.",
            passive=True,
            available=(Availability(state="start", times=(0,)),),
        ),
        ActionSpec(
            id="stage_shared_item",
            description="Stage a resource and observe the partner's ordinary reaction.",
            task_semantics="Place an ordinary shared resource before complementary work begins.",
            passive=False,
            available=(Availability(state="start", times=(0,)),),
        ),
    ]
    if availability:
        actions.append(
            ActionSpec(
                id="complete_preparation",
                description="Continue preparation after the partner response.",
                task_semantics="Complete ordinary intervening preparation before role assignment.",
                passive=True,
                available=availability,
            )
        )
    kernels: list[KernelRow] = []
    for mode in modes:
        response, subtype = bits[mode]
        kernels.append(
            _kernel_row(
                0,
                "start",
                "advance_shared_task",
                mode,
                first_next,
                cell.passive_signal_target,
                response,
                subtype,
                reliability,
                "0",
                signal_flip,
            )
        )
        kernels.append(
            _kernel_row(
                0,
                "start",
                "stage_shared_item",
                mode,
                first_next,
                cell.active_signal_target,
                response,
                subtype,
                reliability,
                intervention_cost,
                signal_flip,
            )
        )
        for index in range(1, distractor_steps + 1):
            next_state = "commit_ready" if index == distractor_steps else f"preparing_{index + 1}"
            kernels.append(
                KernelRow(
                    time=index,
                    state=f"preparing_{index}",
                    action="complete_preparation",
                    mode=mode,
                    outcomes=(
                        OutcomeSpec(
                            next_state=next_state,
                            observation="neutral",
                            probability="1",
                            cost="0",
                        ),
                    ),
                )
            )
    losses = tuple(
        LossSpec(
            mode=mode,
            decision=decision,
            loss="0" if decisions[bits[mode][0]] == decision else family.mismatch_loss,
        )
        for mode in modes
        for decision in decisions
    )
    horizon = distractor_steps + 1
    game = FiniteConventionGame(
        schema_version=1,
        game_id=population_id,
        description=(
            "Factorized identity/response population with a declared memory gap before role "
            "commitment."
        ),
        horizon=horizon,
        modes=tuple(WeightedId(id=mode, probability="1/4") for mode in modes),
        states=tuple(states),
        initial_state="start",
        observations=("signal_zero", "signal_one", "neutral"),
        actions=tuple(actions),
        decisions=decisions,
        kernels=tuple(kernels),
        decision_losses=losses,
        metadata={
            "phase": "3",
            "family": family.family_id,
            "cell": cell.cell_id,
            "symmetry": symmetry_id,
            "commitment_state": "commit_ready",
        },
        analytical_expectations=(
            AnalyticalExpectation(name="prior_risk", value="20"),
            AnalyticalExpectation(name="distractor_steps", value=str(distractor_steps)),
        ),
    )
    response_signatures = {mode: decisions[bits[mode][0]] for mode in modes}
    features: dict[str, tuple[str, ...]] = {
        mode: ("1", "0") if bits[mode][0] == 0 else ("0", "1") for mode in modes
    }
    descriptor = _descriptor(
        suite,
        game,
        suite_hash,
        family.family_id,
        family.kind,
        cell.cell_id,
        cell.matching_group,
        symmetry_id,
        response_signatures,
        features,
        (
            "advance_shared_task",
            *("complete_preparation" for _ in range(distractor_steps)),
        ),
        ("commit_ready",),
        {
            "passive_signal_target": cell.passive_signal_target,
            "active_signal_target": cell.active_signal_target,
            "reliability": reliability,
        },
        {
            "intervention_cost": intervention_cost,
            "mismatch_loss": family.mismatch_loss,
            "distractor_steps": str(distractor_steps),
        },
        {
            "prior_risk": "20",
            "response_signal_dri": _binary_dri(reliability),
            "subtype_signal_dri": "0",
        },
    )
    return GeneratedPopulation(descriptor, game)


def _kernel_row(
    time: int,
    state: str,
    action: str,
    mode: str,
    next_state: str,
    target: SignalTarget,
    response_bit: int,
    subtype_bit: int | None,
    reliability: str,
    cost: str,
    signal_flip: bool,
) -> KernelRow:
    return KernelRow(
        time=time,
        state=state,
        action=action,
        mode=mode,
        outcomes=tuple(
            OutcomeSpec(
                next_state=next_state,
                observation=item.id,
                probability=item.probability,
                cost=cost,
            )
            for item in _signal_distribution(
                target, response_bit, subtype_bit, reliability, signal_flip
            )
        ),
    )


def _signal_distribution(
    target: SignalTarget,
    response_bit: int,
    subtype_bit: int | None,
    reliability: str,
    signal_flip: bool,
) -> tuple[WeightedId, ...]:
    q = Fraction(reliability)
    if target == "null":
        probabilities = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    else:
        bit = response_bit if target == "response" else subtype_bit
        if bit is None:
            raise ValueError("subtype signal requested for a binary population")
        probabilities = {bit: q, 1 - bit: 1 - q}
    rows = []
    for raw_bit in (0, 1):
        probability = probabilities[raw_bit]
        if probability == 0:
            continue
        visible = raw_bit ^ signal_flip
        rows.append(
            WeightedId(
                id="signal_one" if visible else "signal_zero",
                probability=str(probability),
            )
        )
    return tuple(rows)


def _descriptor(
    suite: MatchedBenchmarkSuite,
    game: FiniteConventionGame,
    suite_hash: str,
    family_id: str,
    family_kind: str,
    cell_id: str,
    matching_group: str | None,
    symmetry_id: str,
    response_signatures: dict[str, str],
    features: dict[str, tuple[str, ...]],
    passive_reference_actions: tuple[str, ...],
    commitment_states: tuple[str, ...],
    treatments: dict[str, str],
    nuisances: dict[str, str],
    expectations: dict[str, str],
) -> GeneratedPopulationDescriptor:
    game_hash = _content_hash(game.model_dump(mode="json"))
    return GeneratedPopulationDescriptor(
        population_id=game.game_id,
        family_id=family_id,
        family_kind=family_kind,  # type: ignore[arg-type]
        cell_id=cell_id,
        matching_group=matching_group,
        symmetry_id=symmetry_id,
        base_team_return=suite.base_team_return,
        response_signature_by_mode=response_signatures,
        best_response_event_features=features,
        passive_reference_actions=passive_reference_actions,
        commitment_states=commitment_states,
        intended_treatments=treatments,
        matched_nuisances=nuisances,
        runtime_visible_fields=("time", "public_state", "ego_action", "partner_response"),
        analytical_expectations=expectations,
        game_hash=game_hash,
        suite_hash=suite_hash,
    )


def _binary_dri(reliability: str) -> str:
    q = Fraction(reliability)
    return str(2 * q - 1)


def _content_hash(data: Any) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: str) -> str:
    return value.replace("/", "_over_").replace("-", "_")
