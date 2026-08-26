"""CLI wiring for the isolated Stage 6 established-environment workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from zsc_identifiability.established_diagnostics import audit_diagnostic_options
from zsc_identifiability.established_divergence import estimate_prefix_tv_curves
from zsc_identifiability.established_matching import (
    audit_confirmatory_population_pair,
    select_matched_population_pair,
)
from zsc_identifiability.established_models import (
    CandidatePartnerMetrics,
    EstablishedPolicyEvaluation,
    EstablishedTrainingManifest,
    MatchedPopulationAudit,
    PartnerCheckpoint,
    ResponseLibrary,
    TraceManifest,
    load_established_suite_file,
)
from zsc_identifiability.established_partners import (
    generate_partner_pool_manifest,
    load_partner_checkpoints,
    reward_vector_hash,
    vectors_for_split,
)
from zsc_identifiability.established_predictability import (
    estimate_lobp_action_oracle_from_trace_files,
)
from zsc_identifiability.established_response import build_response_library_from_values
from zsc_identifiability.established_runner import (
    execute_established_audit,
    validate_established_configuration,
)
from zsc_identifiability.established_runtime import (
    bootstrap_isolated_runtimes,
    bootstrap_upstreams,
    dispatch_runtime_request,
    write_runtime_request,
)
from zsc_identifiability.established_statistics import (
    hierarchical_dri_coefficient_interval,
    leave_one_reward_vector_out_regression,
)
from zsc_identifiability.established_trace_estimation import (
    estimate_dri_curve_from_trace_files,
)


def add_established_parser(commands: argparse._SubParsersAction[Any]) -> None:
    established = commands.add_parser(
        "established", help="bootstrap, measure, train, or audit Stage 6 validation"
    )
    subcommands = established.add_subparsers(dest="established_command", required=True)
    bootstrap = subcommands.add_parser("bootstrap", help="clone pins and sync isolated runtimes")
    bootstrap.add_argument("--suite", required=True)
    bootstrap.add_argument("--project-root")
    validate = subcommands.add_parser("validate", help="validate schema, pins, runtimes, and DRI")
    validate.add_argument("--suite", required=True)
    validate.add_argument("--project-root")

    partners = subcommands.add_parser(
        "train-partners", help="prepare or execute hash-split partner training jobs"
    )
    partners.add_argument("--suite", required=True)
    partners.add_argument("--split", choices=("train", "validation", "evaluation"), required=True)
    partners.add_argument("--layout", default="demo_cook_simple")
    partners.add_argument("--output", required=True)
    partners.add_argument("--gate", choices=("screen", "finalist"), default="screen")
    partners.add_argument("--offset", type=int, default=0)
    partners.add_argument("--vector-count", type=int)
    partners.add_argument("--execute", action="store_true")
    partners.add_argument("--checkpoint-index")

    responses = subcommands.add_parser(
        "build-responses", help="freeze the empirical response-loss matrix"
    )
    responses.add_argument("--suite", required=True)
    responses.add_argument("--values", required=True)
    responses.add_argument("--clusters")
    responses.add_argument("--output", required=True)

    collect = subcommands.add_parser("collect", help="collect ego-visible commitment traces")
    collect.add_argument("--suite", required=True)
    collect.add_argument("--payload", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--execute", action="store_true")

    estimate = subcommands.add_parser(
        "estimate-dri", help="fit disjoint event/GRU estimators and emit DRI curves"
    )
    estimate.add_argument("--suite", required=True)
    estimate.add_argument("--library", required=True)
    estimate.add_argument("--calibration", required=True)
    estimate.add_argument("--validation", required=True)
    estimate.add_argument("--confirmatory", required=True)
    estimate.add_argument("--estimator", choices=("event", "gru", "both"), default="both")
    estimate.add_argument("--output", required=True)

    predictability = subcommands.add_parser(
        "estimate-predictability",
        help="score the disjoint visible partner-action prediction control",
    )
    predictability.add_argument("--calibration", required=True)
    predictability.add_argument("--confirmatory", required=True)
    predictability.add_argument("--output", required=True)

    divergence = subcommands.add_parser(
        "estimate-divergence",
        help="estimate mode-conditioned ego-visible prefix-TV curves",
    )
    divergence.add_argument("--traces", required=True)
    divergence.add_argument("--response-signatures")
    divergence.add_argument("--output", required=True)

    match = subcommands.add_parser("match", help="freeze and audit matched partner populations")
    match.add_argument("--suite", required=True)
    match.add_argument("--candidates", required=True)
    match.add_argument("--contrast", choices=("passive_dri", "active_dri"), required=True)
    match.add_argument("--confirmatory-candidates")
    match.add_argument("--commitment-outcomes")
    match.add_argument("--frozen-selection")
    match.add_argument("--output", required=True)

    method = subcommands.add_parser("train-method", help="prepare or execute one method run")
    method.add_argument("--suite", required=True)
    method.add_argument("--method", required=True)
    method.add_argument("--layout", required=True)
    method.add_argument("--seed", type=int, required=True)
    method.add_argument(
        "--gate", choices=("smoke", "development", "confirmatory"), required=True
    )
    method.add_argument("--learning-rate", type=float, required=True)
    method.add_argument("--entropy-coefficient", type=float, required=True)
    method.add_argument("--population-path")
    method.add_argument("--output", required=True)
    method.add_argument("--execute", action="store_true")

    evaluate = subcommands.add_parser("evaluate", help="validate a compact policy evaluation")
    evaluate.add_argument("--evaluation", required=True)

    diagnostic = subcommands.add_parser(
        "audit-diagnostics", help="audit natural task options and empirical frontier"
    )
    diagnostic.add_argument("--layout", required=True)
    diagnostic.add_argument("--measurements", required=True)
    diagnostic.add_argument("--output", required=True)

    regression = subcommands.add_parser(
        "regress", help="run leave-one-reward-vector-out incremental DRI analysis"
    )
    regression.add_argument("--rows", required=True)
    regression.add_argument("--output", required=True)

    secondary = subcommands.add_parser(
        "secondary-audit", help="audit availability of official ZSC-Eval assets"
    )
    secondary.add_argument("--suite", required=True)
    secondary.add_argument("--policy-pool", required=True)
    secondary.add_argument("--output", required=True)
    secondary.add_argument("--execute", action="store_true")

    for name in ("audit", "run"):
        item = subcommands.add_parser(
            name,
            help=(
                "assemble a Stage 6 verdict from frozen artifacts"
                if name == "audit"
                else "reproduce available compact Stage 6 analyses and assemble the verdict"
            ),
        )
        item.add_argument("--suite", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--state-dir")
        item.add_argument("--project-root")


def dispatch_established(args: argparse.Namespace) -> int:
    command = args.established_command
    root = _root(getattr(args, "project_root", None))
    if command == "bootstrap":
        suite = load_established_suite_file(args.suite)
        bootstrap_upstreams(suite, root)
        upstream_audit = bootstrap_isolated_runtimes(suite, root)
        print(json.dumps(upstream_audit.to_dict(), indent=2, sort_keys=True))
        return 0
    if command == "validate":
        report = validate_established_configuration(args.suite, project_root=root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["runtime_ready"] and report["scientific_analysis_ready"] else 4
    if command == "train-partners":
        return _partner_jobs(args, root)
    if command == "build-responses":
        suite = load_established_suite_file(args.suite)
        values = _read_object(args.values)
        clusters = None if args.clusters is None else _read_object(args.clusters)
        library = build_response_library_from_values(
            values,
            adequacy_margin=suite.response_library.adequacy_margin,
            response_clusters=clusters,
        )
        _write_json(Path(args.output), library.to_dict())
        print(json.dumps(library.to_dict(), indent=2, sort_keys=True))
        return 0
    if command == "collect":
        suite = load_established_suite_file(args.suite)
        payload = _read_object(args.payload)
        output = Path(args.output).resolve()
        request = write_runtime_request(
            suite,
            "overcookedv2_py310",
            "collect",
            payload,
            output / "request.json",
        )
        if not args.execute:
            print(json.dumps({"status": "prepared", "request": str(request)}, indent=2))
            return 0
        result = dispatch_runtime_request(
            suite,
            "overcookedv2_py310",
            request,
            output / "result.json",
            root,
        )
        trace_manifest = TraceManifest(suite_id=suite.suite_id, **result["payload"])
        _write_json(output / "trace-manifest.json", trace_manifest.to_dict())
        print(json.dumps(trace_manifest.to_dict(), indent=2, sort_keys=True))
        return 0
    if command == "estimate-dri":
        suite = load_established_suite_file(args.suite)
        library = ResponseLibrary.model_validate(_read_object(args.library))
        estimators = ("event", "gru") if args.estimator == "both" else (args.estimator,)
        estimates = {
            estimator: estimate_dri_curve_from_trace_files(
                args.calibration,
                args.validation,
                args.confirmatory,
                library,
                suite.dri_estimator,
                estimator=estimator,
            ).to_dict()
            for estimator in estimators
        }
        payload = {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "estimates": estimates,
        }
        _write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if command == "estimate-predictability":
        payload = estimate_lobp_action_oracle_from_trace_files(
            args.calibration,
            args.confirmatory,
        )
        _write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if command == "estimate-divergence":
        signatures = (
            None
            if args.response_signatures is None
            else _read_object(args.response_signatures)
        )
        payload = estimate_prefix_tv_curves(
            args.traces,
            response_signatures=signatures,
        )
        _write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if command == "match":
        return _match(args)
    if command == "train-method":
        return _method_job(args, root)
    if command == "evaluate":
        evaluation = EstablishedPolicyEvaluation.model_validate(_read_object(args.evaluation))
        print(json.dumps(evaluation.to_dict(), indent=2, sort_keys=True))
        return 0
    if command == "audit-diagnostics":
        measurements = _read_list(args.measurements)
        diagnostic_audit = audit_diagnostic_options(args.layout, measurements)
        _write_json(Path(args.output), diagnostic_audit.to_dict())
        print(json.dumps(diagnostic_audit.to_dict(), indent=2, sort_keys=True))
        return 0 if diagnostic_audit.passed else 3
    if command == "regress":
        rows = _read_list(args.rows)
        report = leave_one_reward_vector_out_regression(rows)
        if all(
            {"training_seed", "partner_id", "episode_id"}.issubset(row) for row in rows
        ):
            report["dri_coefficient_interval"] = hierarchical_dri_coefficient_interval(
                rows
            )
        _write_json(Path(args.output), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["incremental_value"] else 3
    if command == "secondary-audit":
        return _secondary(args, root)
    if command in {"audit", "run"}:
        stage6_manifest = execute_established_audit(
            args.suite,
            args.output,
            state_dir=args.state_dir,
            project_root=root,
        )
        print(json.dumps(stage6_manifest.to_dict(), indent=2, sort_keys=True))
        if stage6_manifest.status == "incomplete":
            return 4
        return 3 if stage6_manifest.scientific_verdict in {"redesign", "stop"} else 0
    raise ValueError(f"unknown established command: {command!r}")


def _partner_jobs(args: argparse.Namespace, root: Path) -> int:
    suite = load_established_suite_file(args.suite)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_index:
        checkpoints = load_partner_checkpoints(args.checkpoint_index)
        manifest = generate_partner_pool_manifest(suite, args.split, checkpoints)
        _write_json(output / f"{args.split}-pool-manifest.json", manifest.to_dict())
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return 0 if manifest.quota_met else 3
    vectors = vectors_for_split(suite, args.split)
    default_vectors = {
        "train": (suite.partner_generation.training_partner_quota + 1) // 2,
        "validation": (suite.partner_generation.validation_partner_quota + 1) // 2,
        "evaluation": (suite.partner_generation.evaluation_candidate_quota + 1) // 2,
    }[args.split]
    count = default_vectors if args.vector_count is None else args.vector_count
    selected = vectors[args.offset : args.offset + count]
    if not selected:
        raise ValueError("selected partner reward-vector slice is empty")
    transition_budget = (
        suite.partner_generation.screen_transitions
        if args.gate == "screen"
        else suite.partner_generation.finalist_transitions
    )
    jobs: list[dict[str, Any]] = []
    completed_checkpoints: list[PartnerCheckpoint] = []
    for vector_index, vector in enumerate(selected, start=args.offset):
        vector_hash = reward_vector_hash(vector)
        for replicate in range(suite.partner_generation.seeds_per_reward_vector):
            seed = 41001 + 2 * vector_index + replicate
            job_id = f"{args.split}-{vector_hash[:12]}-seed{seed}-{args.gate}"
            job_dir = output / "jobs" / job_id
            payload = {
                "method_id": "partner_ippo",
                "layout_id": args.layout,
                "seed": seed,
                "transitions": transition_budget,
                "learning_rate": 0.00025,
                "entropy_coefficient": 0.01,
                "behavior_preferences": vector,
                "reward_vector_id": vector_hash,
                "split": args.split,
                "output_dir": str(job_dir / "checkpoints"),
            }
            request = write_runtime_request(
                suite,
                "overcookedv2_py310",
                "train_partner",
                payload,
                job_dir / "request.json",
            )
            job = {"job_id": job_id, "request": str(request), "payload": payload}
            if args.execute:
                training_result = dispatch_runtime_request(
                    suite,
                    "overcookedv2_py310",
                    request,
                    job_dir / "result.json",
                    root,
                )
                checkpoint_path = training_result["payload"]["checkpoint_paths"][-1]
                competence_payload = {
                    "ego_checkpoint": checkpoint_path,
                    "partner_checkpoint": checkpoint_path,
                    "layout_id": args.layout,
                    "environment_keys": list(
                        range(1_729_000, 1_729_000 + suite.partner_generation.validation_rollouts)
                    ),
                }
                competence_request = write_runtime_request(
                    suite,
                    "overcookedv2_py310",
                    "evaluate_pair",
                    competence_payload,
                    job_dir / "competence-request.json",
                )
                competence_result = dispatch_runtime_request(
                    suite,
                    "overcookedv2_py310",
                    competence_request,
                    job_dir / "competence-result.json",
                    root,
                )
                delivery_rate = float(
                    competence_result["payload"]["correct_delivery_episode_rate"]
                )
                checkpoint = PartnerCheckpoint(
                    partner_id=f"{args.split}-{vector_hash[:12]}-seed{seed}",
                    reward_vector_id=vector_hash,
                    reward_vector_hash=vector_hash,
                    split=args.split,
                    seed=seed,
                    layout_id=args.layout,
                    checkpoint_path=str(checkpoint_path),
                    normalized_checkpoint_hash=str(
                        training_result["payload"]["checkpoint_parameter_hashes"][
                            checkpoint_path
                        ]
                    ),
                    transitions=transition_budget,
                    validation_correct_delivery_rate=delivery_rate,
                    competent=(
                        delivery_rate
                        >= suite.partner_generation.minimum_correct_delivery_rate
                    ),
                )
                completed_checkpoints.append(checkpoint)
                job["result"] = training_result
                job["competence"] = competence_result
                job["checkpoint"] = checkpoint.to_dict()
            jobs.append(job)
    report = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "split": args.split,
        "gate": args.gate,
        "executed": bool(args.execute),
        "jobs": jobs,
    }
    _write_json(output / f"{args.split}-{args.gate}-jobs.json", report)
    if completed_checkpoints:
        _write_json(
            output / f"{args.split}-{args.gate}-checkpoint-index.json",
            {"checkpoints": [item.to_dict() for item in completed_checkpoints]},
        )
        pool = generate_partner_pool_manifest(suite, args.split, completed_checkpoints)
        _write_json(output / f"{args.split}-{args.gate}-pool-manifest.json", pool.to_dict())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _match(args: argparse.Namespace) -> int:
    suite = load_established_suite_file(args.suite)
    candidates = tuple(
        CandidatePartnerMetrics.model_validate(item) for item in _read_list(args.candidates)
    )
    if args.frozen_selection:
        frozen = MatchedPopulationAudit.model_validate(_read_object(args.frozen_selection))
    else:
        frozen = select_matched_population_pair(candidates, suite.matching, args.contrast)
    result = frozen
    if args.confirmatory_candidates:
        confirmatory = tuple(
            CandidatePartnerMetrics.model_validate(item)
            for item in _read_list(args.confirmatory_candidates)
        )
        commitment_outcomes = (
            None if args.commitment_outcomes is None else _read_object(args.commitment_outcomes)
        )
        result = audit_confirmatory_population_pair(
            frozen,
            confirmatory,
            suite.matching,
            commitment_outcomes=commitment_outcomes,
        )
    _write_json(Path(args.output), result.to_dict())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    passed = result.confirmatory_passed if args.confirmatory_candidates else result.discovery_passed
    return 0 if passed else 3


def _method_job(args: argparse.Namespace, root: Path) -> int:
    suite = load_established_suite_file(args.suite)
    method = next(
        (item for item in suite.training.methods if item.method_id == args.method),
        None,
    )
    if method is None:
        raise ValueError(f"method is not declared in the suite: {args.method!r}")
    if args.learning_rate not in suite.training.learning_rates:
        raise ValueError("learning rate is outside the preregistered search")
    if args.entropy_coefficient not in suite.training.entropy_coefficients:
        raise ValueError("entropy coefficient is outside the preregistered search")
    valid_seeds = (
        suite.training.confirmatory_seeds
        if args.gate == "confirmatory"
        else suite.training.development_seeds
    )
    if args.seed not in valid_seeds:
        raise ValueError("seed is outside the declared gate schedule")
    transitions = (
        suite.training.smoke_transitions
        if args.gate == "smoke"
        else method.development_transitions
        if args.gate == "development"
        else method.confirmatory_transitions
    )
    output = Path(args.output).resolve()
    payload = {
        "method_id": args.method,
        "layout_id": args.layout,
        "seed": args.seed,
        "split": args.gate,
        "transitions": transitions,
        "learning_rate": args.learning_rate,
        "entropy_coefficient": args.entropy_coefficient,
        "smoke": args.gate == "smoke",
        "population_path": args.population_path,
        "output_dir": str(output / "checkpoints"),
    }
    request = write_runtime_request(
        suite,
        "overcookedv2_py310",
        "train_method",
        payload,
        output / "request.json",
    )
    if not args.execute:
        print(json.dumps({"status": "prepared", "request": str(request)}, indent=2))
        return 0
    result = dispatch_runtime_request(
        suite,
        "overcookedv2_py310",
        request,
        output / "runtime-result.json",
        root,
    )
    runtime_payload = result["payload"]
    checkpoint_path = runtime_payload["checkpoint_paths"][-1]
    manifest = EstablishedTrainingManifest(
        suite_id=suite.suite_id,
        method_id=args.method,
        layout_id=args.layout,
        split=args.gate,
        seed=args.seed,
        requested_transitions=transitions,
        completed_transitions=int(runtime_payload["completed_transitions"]),
        checkpoint_path=checkpoint_path,
        checkpoint_hash=str(runtime_payload["checkpoint_parameter_hashes"][checkpoint_path]),
        upstream_commit=next(
            item.commit for item in suite.upstreams if item.repository_id == "overcookedv2"
        ),
        configuration_hash=str(runtime_payload["configuration_hash"]),
        dataset_hashes=(
            ()
            if args.population_path is None
            else (_path_hash(Path(args.population_path).resolve()),)
        ),
        python_version=str(result["python_version"]),
        jax_version=str(result["dependency_versions"]["jax"]),
        xla_version=str(result["dependency_versions"]["jaxlib"]),
        device=str(runtime_payload["device"]),
        resumed=False,
    )
    _write_json(output / "manifest.json", manifest.to_dict())
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


def _secondary(args: argparse.Namespace, root: Path) -> int:
    suite = load_established_suite_file(args.suite)
    output = Path(args.output).resolve()
    payload = {
        "policy_pool_path": args.policy_pool,
        "minimum_partners": suite.secondary_zsceval.minimum_partners,
        "minimum_algorithms": suite.secondary_zsceval.minimum_algorithms,
    }
    request = write_runtime_request(
        suite,
        "zsceval_py39",
        "audit_assets",
        payload,
        output / "request.json",
    )
    if not args.execute:
        print(json.dumps({"status": "prepared", "request": str(request)}, indent=2))
        return 0
    result = dispatch_runtime_request(
        suite, "zsceval_py39", request, output / "result.json", root
    )
    _write_json(output / "secondary-zsceval-audit.json", result["payload"])
    print(json.dumps(result["payload"], indent=2, sort_keys=True))
    return 0 if result["payload"]["status"] == "complete" else 4


def _read_object(path: str | Path) -> dict[str, Any]:
    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_list(path: str | Path) -> list[dict[str, Any]]:
    value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"expected a list of JSON objects: {path}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _root(value: str | None) -> Path:
    return Path(value).resolve() if value else Path(__file__).resolve().parents[2]


def _path_hash(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ValueError(f"cannot hash missing runtime asset: {path}")
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()
