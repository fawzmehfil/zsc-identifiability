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
    except (ValidationError, ValueError, RuntimeError, AssertionError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    return 0
