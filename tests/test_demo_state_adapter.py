from zsc_identifiability.demo.state_adapter import normalize_state


def test_state_adapter_normalizes_legacy_soup_and_physical_seat() -> None:
    state = normalize_state(
        {
            "timestep": 4,
            "players": [
                {"position": [1, 2], "orientation": [1, 0], "held_object": None},
                {
                    "position": [3, 2],
                    "orientation": [0, -1],
                    "held_object": {
                        "name": "soup",
                        "position": [3, 2],
                        "state": ["onion", 2, 7],
                    },
                },
            ],
            "objects": [],
        },
        ego_seat=1,
    )
    assert state.players[0].role == "partner"
    assert state.players[1].role == "ego"
    assert state.players[1].held is not None
    assert state.players[1].held.ingredients == ("onion", "onion")
    assert state.players[1].facing == "north"


def test_state_adapter_normalizes_new_soup() -> None:
    state = normalize_state(
        {
            "players": [
                {"position": [1, 1], "orientation": [0, 1], "held_object": None},
                {"position": [2, 1], "orientation": [-1, 0], "held_object": None},
            ],
            "objects": [
                {
                    "name": "soup",
                    "position": [3, 0],
                    "_ingredients": [
                        {"name": "tomato", "position": [3, 0]},
                        {"name": "onion", "position": [3, 0]},
                    ],
                    "cooking_tick": 20,
                    "is_ready": True,
                }
            ],
        },
        ego_seat=0,
    )
    assert state.objects[0].item.ingredients == ("tomato", "onion")
    assert state.objects[0].item.ready
