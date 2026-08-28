"""CLI wiring for the isolated Stage 6 established-environment workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from zsc_identifiability.established_diagnostics import audit_diagnostic_options
from zsc_identifiability.established_divergence import estimate_prefix_tv_curves
from zsc_identifiability.established_matching import (
    audit_confirmatory_population_pair,
    select_matched_population_pair,
)
from zsc_identifiability.established_models import (
    CandidatePartnerMetrics,
    EstablishedMethodAssetsManifest,
    EstablishedPolicyArtifact,
    EstablishedPolicyEvaluation,
    EstablishedPolicyKind,
    EstablishedTrainingManifest,
    MatchedPopulationAudit,
    PartnerCheckpoint,
    PartnerPoolBuildStatus,
    ResponseLibrary,
    SplitName,
    TraceManifest,
    load_established_suite_file,
)
from zsc_identifiability.established_official_assets import (
    load_official_asset_inventory,
    load_official_asset_lock,
    prepare_official_asset_lock,
    sync_official_assets,
)
from zsc_identifiability.established_official_models import (
    OfficialRolloutLedger,
    load_official_checkpoint_suite,
)
from zsc_identifiability.established_official_reporting import (
    run_complete_official_checkpoint_analysis,
)
from zsc_identifiability.established_official_rollouts import (
    get_official_rollout_status,
    prepare_official_rollouts,
    run_official_rollouts,
)
from zsc_identifiability.established_partner_pools import (
    freeze_partner_pools,
    get_partner_pool_status,
    load_partner_pool_build_plan,
    partner_seed,
    prepare_partner_pool_build,
    run_partner_pool_build,
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
    partners.add_argument(
        "--resume-index",
        help="screen checkpoint index used to continue finalist jobs to their total budget",
    )

    pools = subcommands.add_parser(
        "partner-pools", help="prepare, resume, inspect, or freeze partner pools"
    )
    pool_commands = pools.add_subparsers(dest="partner_pools_command", required=True)
    pool_prepare = pool_commands.add_parser(
        "prepare", help="write an immutable deterministic build plan without training"
    )
    pool_prepare.add_argument("--suite", required=True)
    pool_prepare.add_argument("--layout", default="demo_cook_simple")
    pool_prepare.add_argument("--workspace", required=True)
    pool_prepare.add_argument("--project-root")
    pool_run = pool_commands.add_parser("run", help="run or resume the planned queue")
    pool_run.add_argument("--plan", required=True)
    pool_run.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "evaluation"),
        default=("train", "validation", "evaluation"),
    )
    pool_run.add_argument("--workers", type=int, default=1)
    pool_run.add_argument("--freeze-on-success", action="store_true")
    pool_status = pool_commands.add_parser("status", help="inspect a build without training")
    pool_status.add_argument("--plan", required=True)
    pool_freeze = pool_commands.add_parser(
        "freeze", help="verify and immutably publish completed pools"
    )
    pool_freeze.add_argument("--plan", required=True)

    official = subcommands.add_parser(
        "official", help="run the inference-only official ZSC-Eval checkpoint audit"
    )
    official_commands = official.add_subparsers(dest="official_command", required=True)
    official_prepare = official_commands.add_parser(
        "prepare", help="lock official assets and materialize rollout shards"
    )
    official_prepare.add_argument("--suite", required=True)
    official_prepare.add_argument("--workspace", required=True)
    official_prepare.add_argument(
        "--inventory", help="prepare rollouts immediately from an existing synced inventory"
    )
    official_sync = official_commands.add_parser(
        "sync", help="download only the pinned minimal policy assets"
    )
    official_sync.add_argument("--suite", required=True)
    official_sync.add_argument("--lock", required=True)
    official_smoke = official_commands.add_parser(
        "smoke", help="run only official evaluator parity shards"
    )
    official_smoke.add_argument("--plan", required=True)
    official_smoke.add_argument("--workers", type=int, default=2)
    official_run = official_commands.add_parser(
        "run", help="run or resume all official partner-policy shards"
    )
    official_run.add_argument("--plan", required=True)
    official_run.add_argument("--workers", type=int, default=2)
    official_status = official_commands.add_parser(
        "status", help="inspect the official rollout ledger without execution"
    )
    official_status.add_argument("--plan", required=True)
    official_analyze = official_commands.add_parser(
        "analyze", help="build response conflicts, DRI, and the Stage 6 v2 verdict"
    )
    official_analyze.add_argument("--suite", required=True)
    official_analyze.add_argument("--plan", required=True)
    official_analyze.add_argument("--ledger", required=True)
    official_analyze.add_argument("--output", required=True)

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
    method.add_argument("--gate", choices=("smoke", "development", "confirmatory"), required=True)
    method.add_argument("--learning-rate", type=float, required=True)
    method.add_argument("--entropy-coefficient", type=float, required=True)
    method.add_argument("--population-path")
    method.add_argument("--train-pool")
    method.add_argument("--validation-pool")
    method.add_argument("--cross-play-values")
    method.add_argument("--resume")
    method.add_argument(
        "--compute-allocation",
        choices=("per-specialist", "split-total"),
        default="per-specialist",
    )
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
    if command == "partner-pools":
        return _partner_pool_command(args, root)
    if command == "official":
        return _official_command(args)
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
            None if args.response_signatures is None else _read_object(args.response_signatures)
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
        if all({"training_seed", "partner_id", "episode_id"}.issubset(row) for row in rows):
            report["dri_coefficient_interval"] = hierarchical_dri_coefficient_interval(rows)
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


def _official_command(args: argparse.Namespace) -> int:
    operation = args.official_command
    if operation == "prepare":
        suite = load_official_checkpoint_suite(args.suite)
        lock = prepare_official_asset_lock(args.suite, args.workspace)
        payload: dict[str, Any] = {"asset_lock": lock.to_dict()}
        if args.inventory:
            inventory = load_official_asset_inventory(args.inventory)
            plan = prepare_official_rollouts(args.suite, inventory, args.workspace)
            payload["rollout_plan"] = plan.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if suite.policy_training_allowed is False else 2
    if operation == "sync":
        lock = load_official_asset_lock(args.lock)
        inventory = sync_official_assets(lock, args.suite)
        if inventory.complete:
            prepare_official_rollouts(args.suite, inventory, lock.workspace)
        print(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
        return 0 if inventory.complete else 4
    if operation in {"smoke", "run"}:
        ledger = run_official_rollouts(
            args.plan,
            workers=args.workers,
            resume=True,
            kinds=("parity",) if operation == "smoke" else None,
        )
        print(json.dumps(_official_status_payload(ledger), indent=2, sort_keys=True))
        if ledger.failed_shards:
            return 2
        if operation == "smoke":
            plan_payload = _read_object(args.plan)
            parity_ids = {
                str(item["shard_id"]) for item in plan_payload["shards"] if item["kind"] == "parity"
            }
            parity_complete = all(
                entry.status == "complete"
                for entry in ledger.entries
                if entry.shard_id in parity_ids
            )
            return 0 if parity_ids and parity_complete else 4
        return 0 if ledger.complete else 4
    if operation == "status":
        ledger = get_official_rollout_status(args.plan)
        print(json.dumps(_official_status_payload(ledger), indent=2, sort_keys=True))
        if ledger.failed_shards:
            return 2
        return 0 if ledger.complete else 4
    if operation == "analyze":
        manifest = run_complete_official_checkpoint_analysis(
            args.suite, args.plan, args.ledger, args.output
        )
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        if manifest.status == "incomplete":
            return 4
        return 3 if manifest.verdict in {"redesign", "stop"} else 0
    raise ValueError(f"unknown official-checkpoint command: {operation!r}")


def _official_status_payload(ledger: OfficialRolloutLedger) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for entry in ledger.entries:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return {
        "suite_id": ledger.suite_id,
        "complete": ledger.complete,
        "counts": counts,
        "failed_shards": list(ledger.failed_shards),
    }


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
    resume_by_key: dict[tuple[str, int], PartnerCheckpoint] = {}
    if args.resume_index:
        if args.gate != "finalist":
            raise ValueError("--resume-index is only valid for finalist continuation")
        resume_checkpoints = load_partner_checkpoints(args.resume_index)
        resume_by_key = {(item.reward_vector_hash, item.seed): item for item in resume_checkpoints}
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
            seed = partner_seed(cast(SplitName, args.split), vector_index, replicate)
            job_id = f"{args.split}-{vector_hash[:12]}-seed{seed}-{args.gate}"
            job_dir = output / "jobs" / job_id
            payload = {
                "method_id": "partner_ippo",
                "layout_id": args.layout,
                "seed": seed,
                "transitions": transition_budget,
                "schedule_target_transitions": suite.partner_generation.finalist_transitions,
                "learning_rate": 0.00025,
                "entropy_coefficient": 0.01,
                "behavior_preferences": vector,
                "reward_vector_id": vector_hash,
                "split": args.split,
                "output_dir": str(job_dir / "checkpoints"),
            }
            resume = resume_by_key.get((vector_hash, seed))
            if args.gate == "finalist":
                if resume is None:
                    raise ValueError(
                        "every finalist job requires its matching competent screen checkpoint"
                    )
                if not resume.competent:
                    raise ValueError("cannot continue an incompetent screen checkpoint")
                if resume.training_state_checkpoint_path is None:
                    raise ValueError("finalist continuation requires a full screen training state")
                resume_path = Path(resume.training_state_checkpoint_path).resolve()
                resume_hash = _path_hash(resume_path)
                if resume_hash != resume.training_state_checkpoint_hash:
                    raise ValueError("screen training-state checkpoint hash mismatch")
                payload["resume_checkpoint"] = str(resume_path)
                payload["resume_checkpoint_hash"] = resume_hash
                payload["resume_completed_transitions"] = resume.transitions
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
                delivery_rate = float(competence_result["payload"]["correct_delivery_episode_rate"])
                checkpoint = PartnerCheckpoint(
                    partner_id=f"{args.split}-{vector_hash[:12]}-seed{seed}",
                    reward_vector_id=vector_hash,
                    reward_vector_hash=vector_hash,
                    split=args.split,
                    seed=seed,
                    layout_id=args.layout,
                    checkpoint_path=str(checkpoint_path),
                    normalized_checkpoint_hash=str(
                        training_result["payload"]["checkpoint_parameter_hashes"][checkpoint_path]
                    ),
                    training_state_checkpoint_path=(
                        None
                        if not training_result["payload"].get("training_state_paths")
                        else training_result["payload"]["training_state_paths"][-1]
                    ),
                    training_state_checkpoint_hash=(
                        None
                        if not training_result["payload"].get("training_state_paths")
                        else training_result["payload"]["training_state_hashes"][
                            training_result["payload"]["training_state_paths"][-1]
                        ]
                    ),
                    transitions=int(training_result["payload"]["completed_transitions"]),
                    validation_correct_delivery_rate=delivery_rate,
                    competent=(
                        delivery_rate >= suite.partner_generation.minimum_correct_delivery_rate
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


def _partner_pool_command(args: argparse.Namespace, root: Path) -> int:
    operation = args.partner_pools_command
    if operation == "prepare":
        suite = load_established_suite_file(args.suite)
        plan = prepare_partner_pool_build(
            suite,
            suite_path=args.suite,
            layout=args.layout,
            workspace=args.workspace,
            project_root=root,
        )
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0
    plan = load_partner_pool_build_plan(args.plan)
    if operation == "run":
        ledger = run_partner_pool_build(
            plan,
            splits=tuple(cast(SplitName, item) for item in args.splits),
            workers=args.workers,
            freeze_on_success=args.freeze_on_success,
        )
        status = get_partner_pool_status(plan, ledger=ledger)
        print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        return _partner_pool_status_code(status)
    if operation == "status":
        status = get_partner_pool_status(plan)
        print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        return _partner_pool_status_code(status)
    if operation == "freeze":
        bundle = freeze_partner_pools(plan)
        print(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unknown partner-pool command: {operation!r}")


def _partner_pool_status_code(status: PartnerPoolBuildStatus) -> int:
    if status.complete:
        return 0
    if status.unresolved_failures:
        return 2
    if any(item.cap_exhausted and not item.quota_met for item in status.splits):
        return 3
    return 4


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
    ported = {
        "tbs_style",
        "pace_aux",
        "pace_style",
        "csp_style_reconnaissance",
    }
    train_pool_hash: str | None = None
    validation_pool_hash: str | None = None
    cross_play_hash: str | None = None
    if args.method in ported:
        if not args.train_pool or not args.validation_pool:
            raise ValueError(f"{args.method} requires --train-pool and --validation-pool")
        train_pool_hash = _path_hash(Path(args.train_pool).resolve())
        validation_pool_hash = _path_hash(Path(args.validation_pool).resolve())
        if args.method == "tbs_style" and not args.cross_play_values:
            raise ValueError("tbs_style requires --cross-play-values")
        if args.cross_play_values:
            cross_play_hash = _path_hash(Path(args.cross_play_values).resolve())
        assets = EstablishedMethodAssetsManifest(
            method_id=args.method,
            train_pool_path=str(Path(args.train_pool).resolve()),
            train_pool_hash=train_pool_hash,
            validation_pool_path=str(Path(args.validation_pool).resolve()),
            validation_pool_hash=validation_pool_hash,
            cross_play_values_path=(
                None
                if args.cross_play_values is None
                else str(Path(args.cross_play_values).resolve())
            ),
            cross_play_values_hash=cross_play_hash,
            compute_allocation=args.compute_allocation,
        )
        _write_json(output / "method-assets.json", assets.to_dict())
    payload = {
        "method_id": args.method,
        "layout_id": args.layout,
        "seed": args.seed,
        "split": args.gate,
        "transitions": transitions,
        "schedule_target_transitions": transitions,
        "learning_rate": args.learning_rate,
        "entropy_coefficient": args.entropy_coefficient,
        "smoke": args.gate == "smoke",
        "population_path": args.population_path,
        "train_pool_path": args.train_pool,
        "validation_pool_path": args.validation_pool,
        "cross_play_values_path": args.cross_play_values,
        "compute_allocation": args.compute_allocation,
        "resume_checkpoint": args.resume,
        "dataset_hashes": [
            item
            for item in (train_pool_hash, validation_pool_hash, cross_play_hash)
            if item is not None
        ],
        "tomzsc_path": next(
            item.local_directory for item in suite.upstreams if item.repository_id == "tomzsc"
        ),
        "tomzsc_commit": next(
            item.commit for item in suite.upstreams if item.repository_id == "tomzsc"
        ),
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
    deployment_artifact_path = runtime_payload.get("deployment_artifact_path")
    deployment_artifact_hash = runtime_payload.get("deployment_artifact_hash")
    artifact: EstablishedPolicyArtifact | None = None
    if deployment_artifact_path is not None:
        artifact_path = Path(str(deployment_artifact_path)).resolve()
        observed_artifact_hash = _path_hash(artifact_path)
        if observed_artifact_hash != deployment_artifact_hash:
            raise ValueError("runtime deployment artifact hash does not match its contents")
        artifact = EstablishedPolicyArtifact.model_validate(_read_object(artifact_path))
        if (
            artifact.method_id != args.method
            or artifact.layout_id != args.layout
            or artifact.seed != args.seed
        ):
            raise ValueError("runtime deployment artifact does not match the training request")
    if args.method in ported:
        if artifact is None or train_pool_hash is None or validation_pool_hash is None:
            raise ValueError("ported methods must emit a validated deployment artifact")
        assets = EstablishedMethodAssetsManifest(
            method_id=args.method,
            train_pool_path=str(Path(args.train_pool).resolve()),
            train_pool_hash=train_pool_hash,
            validation_pool_path=str(Path(args.validation_pool).resolve()),
            validation_pool_hash=validation_pool_hash,
            cross_play_values_path=(
                None
                if args.cross_play_values is None
                else str(Path(args.cross_play_values).resolve())
            ),
            cross_play_values_hash=cross_play_hash,
            cluster_assignments=artifact.cluster_assignments,
            concept_schema=artifact.concept_schema,
            component_paths={item.component_id: item.path for item in artifact.components},
            component_hashes={
                item.component_id: item.content_hash for item in artifact.components
            },
            compute_allocation=args.compute_allocation,
        )
        _write_json(output / "method-assets.json", assets.to_dict())
    parent_checkpoint_hash = runtime_payload.get("parent_checkpoint_hash")
    dataset_hashes = tuple(
        item
        for item in (
            None
            if args.population_path is None
            else _path_hash(Path(args.population_path).resolve()),
            train_pool_hash,
            validation_pool_hash,
            cross_play_hash,
        )
        if item is not None
    )
    manifest = EstablishedTrainingManifest(
        schema_version=2,
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
        dataset_hashes=dataset_hashes,
        python_version=str(result["python_version"]),
        jax_version=str(result["dependency_versions"]["jax"]),
        xla_version=str(result["dependency_versions"]["jaxlib"]),
        device=str(runtime_payload["device"]),
        resumed=bool(runtime_payload.get("resumed", False)),
        policy_kind=cast(EstablishedPolicyKind, str(runtime_payload.get("policy_kind", "ppo"))),
        deployment_artifact_path=deployment_artifact_path,
        deployment_artifact_hash=deployment_artifact_hash,
        resume_checkpoint_path=runtime_payload.get("resume_checkpoint_path"),
        parent_checkpoint_hash=parent_checkpoint_hash,
        component_transitions={
            str(key): int(value)
            for key, value in runtime_payload.get("component_transitions", {}).items()
        },
        aggregate_training_transitions=int(
            runtime_payload.get("aggregate_training_transitions", transitions)
        ),
        best_validation_checkpoint_path=runtime_payload.get("best_validation_checkpoint_path"),
        best_validation_checkpoint_hash=runtime_payload.get("best_validation_checkpoint_hash"),
        best_validation_metric=runtime_payload.get("best_validation_metric"),
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
    result = dispatch_runtime_request(suite, "zsceval_py39", request, output / "result.json", root)
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
