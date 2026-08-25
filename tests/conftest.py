from pathlib import Path

import pytest

from zsc_identifiability.benchmark_generator import generate
from zsc_identifiability.benchmark_models import (
    GeneratedBenchmarkSet,
    MatchedBenchmarkSuite,
    load_benchmark_suite_file,
)
from zsc_identifiability.models import FiniteConventionGame, load_game_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAMES = PROJECT_ROOT / "phase-2-exact-model" / "games"
BENCHMARK_SUITE = PROJECT_ROOT / "phase-3-matched-benchmarks" / "suites" / "canonical.json"


@pytest.fixture
def game_loader():
    def load(name: str) -> FiniteConventionGame:
        return load_game_file(GAMES / f"{name}.json")

    return load


@pytest.fixture(scope="session")
def benchmark_suite() -> MatchedBenchmarkSuite:
    return load_benchmark_suite_file(BENCHMARK_SUITE)


@pytest.fixture(scope="session")
def benchmark_set(benchmark_suite: MatchedBenchmarkSuite) -> GeneratedBenchmarkSet:
    return generate(benchmark_suite)
