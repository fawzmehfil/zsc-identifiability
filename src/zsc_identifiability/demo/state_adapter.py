"""Normalize the two pinned Overcooked state formats for the public viewer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zsc_identifiability.demo.models import (
    KitchenItem,
    KitchenObject,
    KitchenPlayer,
    KitchenState,
)

_DIRECTIONS = {
    (0, -1): "north",
    (0, 1): "south",
    (1, 0): "east",
    (-1, 0): "west",
}


def normalize_terrain(raw: Sequence[Any]) -> tuple[tuple[str, ...], ...]:
    """Return a rectangular tuple of single-character terrain codes."""

    rows: list[tuple[str, ...]] = []
    for raw_row in raw:
        values = tuple(str(value) for value in raw_row)
        rows.append(values)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("visual terrain must be a non-empty rectangle")
    return tuple(rows)


def normalize_state(raw: Mapping[str, Any], *, ego_seat: int) -> KitchenState:
    """Translate old and new official state dictionaries to one stable schema."""

    players_raw = raw.get("players")
    objects_raw = raw.get("objects", ())
    if not isinstance(players_raw, Sequence) or len(players_raw) != 2:
        raise ValueError("official visual state must contain exactly two players")
    if not isinstance(objects_raw, Sequence):
        raise ValueError("official visual state objects must be a sequence")
    players = tuple(
        _normalize_player(player, physical_index=index, ego_seat=ego_seat)
        for index, player in enumerate(players_raw)
    )
    objects = tuple(_normalize_object(item) for item in objects_raw)
    return KitchenState(
        timestep=max(0, int(raw.get("timestep", 0))),
        players=players,  # type: ignore[arg-type]
        objects=objects,
    )


def _normalize_player(
    raw: Any,
    *,
    physical_index: int,
    ego_seat: int,
) -> KitchenPlayer:
    if not isinstance(raw, Mapping):
        raise ValueError("official player state must be an object")
    position = _position(raw.get("position"))
    orientation_raw = raw.get("orientation")
    if not isinstance(orientation_raw, Sequence) or len(orientation_raw) != 2:
        raise ValueError("official player orientation must have two coordinates")
    orientation = (int(orientation_raw[0]), int(orientation_raw[1]))
    facing = _DIRECTIONS.get(orientation)
    if facing is None:
        raise ValueError(f"unsupported official player orientation: {orientation!r}")
    held_raw = raw.get("held_object")
    return KitchenPlayer(
        physical_index=physical_index,  # type: ignore[arg-type]
        role="ego" if physical_index == ego_seat else "partner",
        x=position[0],
        y=position[1],
        facing=facing,  # type: ignore[arg-type]
        held=None if held_raw is None else _normalize_item(held_raw),
    )


def _normalize_object(raw: Any) -> KitchenObject:
    if not isinstance(raw, Mapping):
        raise ValueError("official object state must be an object")
    x, y = _position(raw.get("position"))
    return KitchenObject(x=x, y=y, item=_normalize_item(raw))


def _normalize_item(raw: Any) -> KitchenItem:
    if not isinstance(raw, Mapping):
        raise ValueError("official item state must be an object")
    kind = str(raw.get("name", "unknown"))
    if kind != "soup":
        return KitchenItem(kind=kind)
    ingredients: list[str] = []
    ingredient_rows = raw.get("_ingredients")
    if isinstance(ingredient_rows, Sequence):
        ingredients = [
            str(item.get("name", "ingredient"))
            for item in ingredient_rows
            if isinstance(item, Mapping)
        ]
    elif isinstance(raw.get("state"), Sequence) and len(raw["state"]) == 3:
        soup_kind, item_count, _cook_time = raw["state"]
        ingredients = [str(soup_kind)] * int(item_count)
    tick_raw = raw.get("cooking_tick", raw.get("_cooking_tick"))
    if tick_raw is None and isinstance(raw.get("state"), Sequence):
        tick_raw = raw["state"][2]
    cooking_tick = None if tick_raw is None else int(tick_raw)
    ready = bool(raw.get("is_ready", cooking_tick is not None and cooking_tick >= 20))
    return KitchenItem(
        kind="soup",
        ingredients=tuple(ingredients),
        cooking_tick=cooking_tick,
        ready=ready,
    )


def _position(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, Sequence) or len(raw) != 2:
        raise ValueError("official position must have two coordinates")
    return int(raw[0]), int(raw[1])
