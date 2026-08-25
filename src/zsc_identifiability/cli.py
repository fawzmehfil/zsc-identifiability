"""Command-line interface for exact-model validation and reproduction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

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
    learn = commands.add_parser(
        "learn", help="validate, train, evaluate, or audit Stage 4 learned agents"
    )
    learn_commands = learn.add_subparsers(dest="learn_command", required=True)
    learn_validate = learn_commands.add_parser(
        "validate", help="validate the learning suite and its disjoint partner pools"
    )
    learn_validate.add_argument("--suite", required=True)
    learn_generate = learn_commands.add_parser(
        "generate", help="materialize Stage 4 train/validation/test games"
    )
    learn_generate.add_argument("--suite", required=True)
    learn_generate.add_argument("--output", required=True)
    learn_train = learn_commands.add_parser("train", help="train one method matrix")
    learn_train.add_argument("--suite", required=True)
    learn_train.add_argument("--method", required=True)
    learn_train.add_argument("--output", required=True)
    learn_train.add_argument(
        "--gate",
        choices=("smoke", "development", "confirmatory", "rescue"),
        default="smoke",
    )
    learn_train.add_argument(
        "--selection",
        help="validation-only hyperparameter-selection report for this method",
    )
    learn_tune = learn_commands.add_parser(
        "tune", help="run the preregistered global validation-only method search"
    )
    learn_tune.add_argument("--suite", required=True)
    learn_tune.add_argument("--method", required=True)
    learn_tune.add_argument("--output", required=True)
    learn_rescue = learn_commands.add_parser(
        "rescue", help="run the prespecified rescue for all comparators in one failed cell"
    )
    learn_rescue.add_argument("--suite", required=True)
    learn_rescue.add_argument("--cell", required=True)
    learn_rescue.add_argument("--output", required=True)
    learn_symmetry = learn_commands.add_parser(
        "symmetry", help="independently retrain one method on selected relabelings"
    )
    learn_symmetry.add_argument("--suite", required=True)
    learn_symmetry.add_argument("--method", required=True)
    learn_symmetry.add_argument("--output", required=True)
    learn_symmetry.add_argument("--selection")
    learn_train.add_argument("--cell")
    learn_train.add_argument("--seed", type=int)
    learn_train.add_argument(
        "--resume",
        help="resume an exact batch-boundary checkpoint; requires --cell and --seed",
    )
    learn_evaluate = learn_commands.add_parser(
        "evaluate", help="exactly evaluate one learned checkpoint"
    )
    learn_evaluate.add_argument("--run", required=True)
    learn_evaluate.add_argument("--suite", default="phase-4-learned-audit/suites/canonical.json")
    learn_smoke_audit = learn_commands.add_parser(
        "smoke-audit", help="apply aggregate capability gates to smoke checkpoints"
    )
    learn_smoke_audit.add_argument("--suite", required=True)
    learn_smoke_audit.add_argument("--runs-dir", required=True)
    learn_smoke_audit.add_argument("--output", required=True)
    for name in ("audit", "run"):
        item = learn_commands.add_parser(
            name,
            help=(
                "audit an existing confirmatory run matrix"
                if name == "audit"
                else "train and audit the complete confirmatory matrix"
            ),
        )
        item.add_argument("--suite", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--runs-dir")
        item.add_argument("--rescue-runs-dir")
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
        elif args.command == "learn":
            from zsc_identifiability.learning_models import load_learning_suite_file
            from zsc_identifiability.learning_pools import generate_learning_pools
            from zsc_identifiability.learning_runner import (
                execute_learning_audit,
                execute_training_matrix,
                materialize_learning_pools,
            )

            if args.learn_command == "validate":
                learning_spec = load_learning_suite_file(args.suite)
                pools = generate_learning_pools(learning_spec, suite_path=args.suite)
                print(
                    json.dumps(
                        {
                            "valid": True,
                            "suite_id": learning_spec.suite_id,
                            "cell_count": len(pools.cells),
                            "train_game_count": sum(len(item.train) for item in pools.cells),
                            "validation_game_count": sum(
                                len(item.validation) for item in pools.cells
                            ),
                            "test_game_count": sum(len(item.test) for item in pools.cells),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.learn_command == "generate":
                learning_spec = load_learning_suite_file(args.suite)
                pools = generate_learning_pools(learning_spec, suite_path=args.suite)
                files = materialize_learning_pools(pools, args.output)
                print(
                    json.dumps(
                        {
                            "suite_id": learning_spec.suite_id,
                            "cell_count": len(pools.cells),
                            "generated_files": list(files),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            elif args.learn_command == "train":
                manifests = execute_training_matrix(
                    args.suite,
                    args.output,
                    method_id=args.method,
                    gate=args.gate,
                    cell_id=args.cell,
                    seed_override=args.seed,
                    resume_checkpoint=args.resume,
                    selection_report=args.selection,
                )
                print(json.dumps(list(manifests), indent=2, sort_keys=True))
            elif args.learn_command == "tune":
                from zsc_identifiability.learning_tuning import run_development_search

                selection = run_development_search(args.suite, args.output, method_id=args.method)
                print(json.dumps(selection, indent=2, sort_keys=True))
            elif args.learn_command == "rescue":
                from zsc_identifiability.learning_runner import execute_rescue_matrix

                manifests = execute_rescue_matrix(args.suite, args.output, cell_id=args.cell)
                print(json.dumps(list(manifests), indent=2, sort_keys=True))
            elif args.learn_command == "symmetry":
                from zsc_identifiability.learning_runner import execute_symmetry_matrix

                manifests = execute_symmetry_matrix(
                    args.suite,
                    args.output,
                    method_id=args.method,
                    selection_report=args.selection,
                )
                print(json.dumps(list(manifests), indent=2, sort_keys=True))
            elif args.learn_command == "evaluate":
                from zsc_identifiability.learning_evaluation import (
                    evaluate_neural_policy_exact,
                )
                from zsc_identifiability.learning_trainer import load_checkpoint

                run_dir = Path(args.run).resolve()
                run_manifest = json.loads((run_dir / "manifest.json").read_text())
                learning_spec = load_learning_suite_file(args.suite)
                pools = generate_learning_pools(learning_spec, suite_path=args.suite)
                cell = pools.by_cell()[run_manifest["cell_id"]]
                model, payload = load_checkpoint(run_manifest["checkpoint_path"])
                method_id = str(payload["method"]["method_id"])
                action_class = "passive" if method_id == "gru_ppo_passive" else "task"
                evaluation_modes: tuple[Literal["greedy", "stochastic"], ...] = (
                    "greedy",
                    "stochastic",
                )
                results = [
                    evaluate_neural_policy_exact(
                        model,
                        cell.test[0],
                        method_id=method_id,
                        mode=mode,
                        action_class=action_class,
                        identity_label_response_classes=tuple(
                            int(item) for item in payload["partner_response_classes"]
                        ),
                    ).to_dict()
                    for mode in evaluation_modes
                ]
                print(json.dumps(results, indent=2, sort_keys=True))
            elif args.learn_command == "smoke-audit":
                from zsc_identifiability.learning_runner import audit_smoke_matrix

                report = audit_smoke_matrix(args.suite, args.runs_dir, args.output)
                print(json.dumps(report, indent=2, sort_keys=True))
                if report["status"] == "incomplete":
                    return 4
                if not report["passed"]:
                    return 3
            else:
                runs_dir = args.runs_dir or str(Path(args.output).resolve().parent / "runs")
                if args.learn_command == "run":
                    learning_spec = load_learning_suite_file(args.suite)
                    from zsc_identifiability.learning_tuning import run_development_search

                    development_dir = Path(args.output).resolve().parent / "development"
                    for method in learning_spec.methods:
                        if method.enabled:
                            selection_path = development_dir / f"{method.method_id}-selection.json"
                            if not selection_path.exists():
                                run_development_search(
                                    args.suite,
                                    development_dir,
                                    method_id=method.method_id,
                                )
                            execute_training_matrix(
                                args.suite,
                                runs_dir,
                                method_id=method.method_id,
                                gate="confirmatory",
                                selection_report=selection_path,
                            )
                            from zsc_identifiability.learning_runner import (
                                execute_symmetry_matrix,
                            )

                            execute_symmetry_matrix(
                                args.suite,
                                runs_dir,
                                method_id=method.method_id,
                                selection_report=selection_path,
                            )
                learning_manifest = execute_learning_audit(
                    args.suite,
                    args.output,
                    runs_dir=runs_dir,
                    rescue_runs_dir=args.rescue_runs_dir,
                )
                if (
                    args.learn_command == "run"
                    and learning_manifest.status == "incomplete"
                    and learning_manifest.missing_runs
                    and all(item.startswith("rescue:") for item in learning_manifest.missing_runs)
                ):
                    rescue_dir = args.rescue_runs_dir or str(
                        Path(args.output).resolve().parent / "rescue-runs"
                    )
                    from zsc_identifiability.learning_runner import execute_rescue_matrix

                    execute_rescue_matrix(args.suite, rescue_dir, cell_id="active_only")
                    learning_manifest = execute_learning_audit(
                        args.suite,
                        args.output,
                        runs_dir=runs_dir,
                        rescue_runs_dir=rescue_dir,
                    )
                print(json.dumps(learning_manifest.to_dict(), indent=2, sort_keys=True))
                if learning_manifest.status == "incomplete":
                    return 4
                if learning_manifest.scientific_verdict in {"redesign", "stop"}:
                    return 3
    except (ValidationError, ValueError, RuntimeError, AssertionError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    return 0
