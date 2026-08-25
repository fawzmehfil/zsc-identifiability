from pathlib import Path

import pytest

from zsc_identifiability.models import FiniteConventionGame, load_game_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMES = PROJECT_ROOT / "phase-2-exact-model" / "games"


@pytest.fixture
def game_loader():
    def load(name: str) -> FiniteConventionGame:
        return load_game_file(GAMES / f"{name}.json")

    return load
