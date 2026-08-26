"""Deterministic replanning controllers for ordinary diagnostic task options."""

from __future__ import annotations

from collections import deque

import numpy as np
from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject, StaticObject


class DiagnosticGoalController:
    """Produces legal low-level actions for at most the declared option horizon."""

    def __init__(self, option_id, candidate_ingredient=0):
        self.option_id = option_id
        self.candidate_ingredient = int(candidate_ingredient)
        self.yielded = False

    def action(self, state):
        if self.option_id == "ordinary_progress":
            return None
        if self.option_id == "corridor_yield":
            return self._corridor_yield(state)
        if self.option_id == "recipe_button_control":
            return self._approach_and_interact(
                state,
                state.grid[:, :, 0] == StaticObject.BUTTON_RECIPE_INDICATOR,
            )
        if self.option_id in {"stage_candidate_ingredient", "temporary_role_takeover"}:
            inventory = int(np.asarray(state.agents.inventory)[0])
            if inventory == DynamicObject.EMPTY:
                pile = StaticObject.ingredient_pile(self.candidate_ingredient)
                return self._approach_and_interact(state, state.grid[:, :, 0] == pile)
            if self.option_id == "stage_candidate_ingredient":
                empty_counter = (state.grid[:, :, 0] == StaticObject.WALL) & (
                    state.grid[:, :, 1] == DynamicObject.EMPTY
                )
                return self._approach_and_interact(state, empty_counter)
            pot_mask = state.grid[:, :, 0] == StaticObject.POT
            action = self._approach(state, pot_mask)
            return Actions.stay if action is None else action
        raise ValueError(f"unknown diagnostic goal option: {self.option_id!r}")

    def _corridor_yield(self, state):
        move_area = np.asarray(state.grid[:, :, 0] == StaticObject.EMPTY)
        degrees = np.zeros_like(move_area, dtype=np.int32)
        degrees[1:, :] += move_area[:-1, :]
        degrees[:-1, :] += move_area[1:, :]
        degrees[:, 1:] += move_area[:, :-1]
        degrees[:, :-1] += move_area[:, 1:]
        corridor = move_area & (degrees <= 2)
        action = self._move_to_open_target(state, corridor)
        if action is None:
            self.yielded = True
            return Actions.stay
        return action

    def _approach_and_interact(self, state, target_mask):
        approach = self._approach(state, target_mask)
        return Actions.interact if approach is None else approach

    def _approach(self, state, target_mask):
        targets = np.argwhere(np.asarray(target_mask))
        if len(targets) == 0:
            return Actions.stay
        position = np.asarray(state.agents.pos.to_array())[0]
        direction = int(np.asarray(state.agents.dir)[0])
        for target_y, target_x in targets:
            delta = (int(target_x - position[0]), int(target_y - position[1]))
            if abs(delta[0]) + abs(delta[1]) == 1:
                required_direction = _direction_from_delta(delta)
                return None if direction == required_direction else _action_from_delta(delta)
        move_area = np.asarray(state.grid[:, :, 0] == StaticObject.EMPTY)
        approach_cells = np.zeros_like(move_area)
        for target_y, target_x in targets:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, y = target_x + dx, target_y + dy
                if 0 <= y < move_area.shape[0] and 0 <= x < move_area.shape[1]:
                    approach_cells[y, x] |= move_area[y, x]
        return self._move_to_open_target(state, approach_cells)

    def _move_to_open_target(self, state, target_mask):
        start_xy = tuple(int(value) for value in np.asarray(state.agents.pos.to_array())[0])
        start = (start_xy[1], start_xy[0])
        targets = {tuple(item) for item in np.argwhere(np.asarray(target_mask))}
        if start in targets:
            return None
        move_area = np.asarray(state.grid[:, :, 0] == StaticObject.EMPTY)
        queue = deque([start])
        predecessor = {start: None}
        found = None
        while queue and found is None:
            current = queue.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dy, current[1] + dx)
                y, x = neighbor
                if not (0 <= y < move_area.shape[0] and 0 <= x < move_area.shape[1]):
                    continue
                if not move_area[y, x] or neighbor in predecessor:
                    continue
                predecessor[neighbor] = current
                if neighbor in targets:
                    found = neighbor
                    break
                queue.append(neighbor)
        if found is None:
            return Actions.stay
        step = found
        while predecessor[step] != start:
            parent = predecessor[step]
            if parent is None:
                return Actions.stay
            step = parent
        delta = (step[1] - start[1], step[0] - start[0])
        return _action_from_delta(delta)


def _action_from_delta(delta):
    return {
        (1, 0): Actions.right,
        (-1, 0): Actions.left,
        (0, 1): Actions.down,
        (0, -1): Actions.up,
    }[tuple(delta)]


def _direction_from_delta(delta):
    # Direction encoding in the pinned environment is up, down, right, left.
    return {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}[tuple(delta)]
