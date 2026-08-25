"""Command-line interface for exact-model validation and reproduction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from zsc_identifiability.api import (
    compute_frontier,
    evaluate_policy,
    load_game,
    run_suite,
    solve_bayes,
)
from zsc_identifiability.benchmark_generator import generate as generate_benchmarks
from zsc_identifiability.benchmark_models import load_benchmark_suite_file
from zsc_identifiability.benchmark_runner import (
    execute_benchmark_suite,
    materialize_benchmark_set,
)
from zsc_identifiability.runner import _theory_checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zsc-identifiability")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a v1 finite convention game")
    validate.add_argument("--game", required=True)
    solve = commands.add_parser("solve", help="solve and evaluate a game")
    solve.add_argument("--game", required=True)
    solve.add_argument(
        "--class",
        dest="action_class",
        choices=("passive", "task", "reconnaissance"),
        default="task",
    )
    solve.add_argument("--objective", choices=("net_regret", "information"), default="net_regret")
    solve.add_argument("--backend", choices=("fraction", "float"), default="fraction")
    frontier = commands.add_parser(
        "frontier", help="compute a deterministic and convexified frontier"
    )
    frontier.add_argument("--game", required=True)
    frontier.add_argument(
        "--class",
        dest="action_class",
        choices=("passive", "task", "reconnaissance"),
        default="task",
    )
    frontier.add_argument("--backend", choices=("fraction", "float"), default="fraction")
    verify = commands.add_parser("verify-theory", help="run executable theorem checks")
    verify.add_argument("--games-dir", default="phase-2-exact-model/games")
    suite = commands.add_parser("run-suite", help="reproduce the complete Phase 2 artifact suite")
    suite.add_argument("--suite", required=True)
    suite.add_argument("--output", required=True)
    benchmark = commands.add_parser(
        "benchmark", help="validate, generate, audit, or reproduce Phase 3 benchmarks"
    )
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_validate = benchmark_commands.add_parser(
        "validate", help="validate and instantiate a matched benchmark suite"
    )
    benchmark_validate.add_argument("--suite", required=True)
    benchmark_generate = benchmark_commands.add_parser(
        "generate", help="materialize generated v1 games and descriptors"
    )
    benchmark_generate.add_argument("--suite", required=True)
    benchmark_generate.add_argument("--output", required=True)
    for name in ("audit", "run"):
        item = benchmark_commands.add_parser(
            name,
            help=(
                "run matching and shortcut audits"
                if name == "audit"
                else "reproduce the complete Phase 3 artifact suite"
            ),
        )
        item.add_argument("--suite", required=True)
        item.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            game = load_game(args.game)
            print(json.dumps({"valid": True, "game_id": game.game_id}, indent=2))
        elif args.command == "solve":
            game = load_game(args.game)
            solution = solve_bayes(game, args.action_class, args.objective, args.backend)
            payload = solution.to_dict()
            payload["evaluation"] = evaluate_policy(game, solution.policy, args.backend).to_dict()
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.command == "frontier":
            game = load_game(args.game)
            print(
                json.dumps(
                    compute_frontier(game, args.action_class, args.backend).to_dict(),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "verify-theory":
            directory = Path(args.games_dir)
            games = {path.stem: load_game(path) for path in sorted(directory.glob("*.json"))}
            checks = _theory_checks(games)
            print(json.dumps([check.__dict__ for check in checks], indent=2, sort_keys=True))
            return 0 if all(check.passed for check in checks) else 1
        elif args.command == "run-suite":
            print(
                json.dumps(run_suite(args.suite, args.output).to_dict(), indent=2, sort_keys=True)
            )
        elif args.command == "benchmark":
            if args.benchmark_command == "validate":
                spec = load_benchmark_suite_file(args.suite)
                generated = generate_benchmarks(spec)
                print(
                    json.dumps(
                        {
                            "valid": True,
                            "suite_id": spec.suite_id,
                            "population_count": len(generated.populations),
                            "matching_contract_count": len(spec.matching_contracts),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.benchmark_command == "generate":
                spec = load_benchmark_suite_file(args.suite)
                generated = generate_benchmarks(spec)
                files = materialize_benchmark_set(generated, args.output)
                print(
                    json.dumps(
                        {
                            "suite_id": spec.suite_id,
                            "population_count": len(generated.populations),
                            "generated_files": list(files),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                manifest = execute_benchmark_suite(args.suite, args.output)
                print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
                return 0 if manifest.scientific_audit_passed else 3
    except (ValidationError, ValueError, RuntimeError, AssertionError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    return 0
