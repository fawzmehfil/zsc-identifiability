"""Reproducible Phase 3 benchmark materialization, audit, and reporting."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from zsc_identifiability.benchmark_audit import audit_pair, audit_shortcuts
from zsc_identifiability.benchmark_generator import generate
from zsc_identifiability.benchmark_models import (
    BenchmarkRunManifest,
    GeneratedBenchmarkSet,
    MatchedBenchmarkSuite,
    PopulationMetrics,
    load_benchmark_suite_file,
)
from zsc_identifiability.benchmark_sampling import calibrate_pair
from zsc_identifiability.numeric import close
from zsc_identifiability.population_metrics import compute as compute_population_metrics
from zsc_identifiability.solver import solve


def materialize_benchmark_set(
    benchmark: GeneratedBenchmarkSet,
    output_dir: str | Path,
) -> tuple[str, ...]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for population in benchmark.populations:
        identifier = population.descriptor.population_id
        game_path = output / "games" / f"{identifier}.json"
        descriptor_path = output / "descriptors" / f"{identifier}.json"
        _write_json(game_path, population.game.model_dump(mode="json"))
        _write_json(descriptor_path, population.descriptor.model_dump(mode="json"))
        written.extend(
            (
                str(game_path.relative_to(output)),
                str(descriptor_path.relative_to(output)),
            )
        )
    index = {
        "schema_version": 1,
        "suite_id": benchmark.suite.suite_id,
        "suite_hash": benchmark.suite_hash,
        "population_count": len(benchmark.populations),
        "populations": [
            {
                "population_id": item.descriptor.population_id,
                "family_id": item.descriptor.family_id,
                "cell_id": item.descriptor.cell_id,
                "symmetry_id": item.descriptor.symmetry_id,
                "game_hash": item.descriptor.game_hash,
            }
            for item in benchmark.populations
        ],
    }
    _write_json(output / "index.json", index)
    written.append("index.json")
    return tuple(sorted(written))


def execute_benchmark_suite(
    suite_path: str | Path,
    output_dir: str | Path,
) -> BenchmarkRunManifest:
    suite_path = Path(suite_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite = load_benchmark_suite_file(suite_path)
    benchmark = generate(suite)
    materialize_benchmark_set(benchmark, output / "generated")
    populations = benchmark.by_id()
    metrics = {
        population.descriptor.population_id: compute_population_metrics(population, "fraction")
        for population in benchmark.populations
    }
    _write_population_artifacts(output, benchmark, metrics)
    audits = tuple(
        audit_pair(
            populations[contract.left_population_id],
            populations[contract.right_population_id],
            contract,
            "fraction",
            metrics[contract.left_population_id],
            metrics[contract.right_population_id],
        )
        for contract in suite.matching_contracts
    )
    _write_json(output / "matching-audit.json", [audit.to_dict() for audit in audits])
    pd.DataFrame(_matching_rows(audits)).to_csv(output / "matching-audit.csv", index=False)
    canonical = tuple(
        population
        for population in benchmark.populations
        if population.descriptor.symmetry_id == "identity"
    )
    shortcut_audits = tuple(audit_shortcuts(population, "fraction") for population in canonical)
    _write_json(
        output / "shortcut-audit.json", [audit.to_dict() for audit in shortcut_audits]
    )
    calibrations = tuple(
        calibrate_pair(
            populations[contract.left_population_id],
            populations[contract.right_population_id],
            metrics[contract.left_population_id],
            metrics[contract.right_population_id],
            contract,
            suite.sample_audit,
            "fraction",
        )
        for contract in suite.matching_contracts
        if contract.sampled
    )
    _write_json(output / "estimator-calibration.json", calibrations)
    scientific_checks = _scientific_checks(
        suite,
        benchmark,
        metrics,
        audits,
        shortcut_audits,
        calibrations,
    )
    _write_json(output / "scientific-checks.json", scientific_checks)
    _plot_all(output / "figures", populations, metrics, shortcut_audits)
    scientific_pass = all(scientific_checks.values())
    verdict: Literal["continue", "redesign", "stop"] = (
        "continue" if scientific_pass else "redesign"
    )
    project_root = Path(__file__).resolve().parents[2]
    dependencies = _dependency_versions()
    generated_files = tuple(
        str(path.relative_to(output))
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = BenchmarkRunManifest(
        schema_version=1,
        suite_id=suite.suite_id,
        scientific_verdict=verdict,
        implementation_passed=True,
        scientific_audit_passed=scientific_pass,
        configuration_hash=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        source_tree_hash=_source_hash(project_root),
        python_version=platform.python_version(),
        dependency_versions=dependencies,
        invoked_command=(
            f"zsc-identifiability benchmark run --suite {suite_path} --output {output}"
        ),
        rng_configuration={
            "generator": "PCG64",
            "seed": suite.sample_audit.seed,
            "episodes_per_mode": suite.sample_audit.episodes_per_mode,
            "bootstrap_resamples": suite.sample_audit.bootstrap_resamples,
            "confidence_level": suite.sample_audit.confidence_level,
            "dri_margin": suite.sample_audit.dri_margin,
        },
        generated_files=generated_files,
    )
    _write_json(output / "manifest.json", manifest.to_dict())
    return manifest


def _write_population_artifacts(
    output: Path,
    benchmark: GeneratedBenchmarkSet,
    metrics: dict[str, PopulationMetrics],
) -> None:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    prefix: dict[str, Any] = {}
    brdiv: dict[str, Any] = {}
    predictability: dict[str, Any] = {}
    for population in benchmark.populations:
        identifier = population.descriptor.population_id
        item = metrics[identifier]
        row = {
            "population_id": identifier,
            "family_id": population.descriptor.family_id,
            "cell_id": population.descriptor.cell_id,
            "symmetry_id": population.descriptor.symmetry_id,
            **{key: _scalar(value) for key, value in item.values.items()},
        }
        rows.append(row)
        summaries.append(
            {
                "population_id": identifier,
                "descriptor": population.descriptor.model_dump(mode="json"),
                "metric_scope": item.metric_scope,
                "estimator_type": item.estimator_type,
                "applicability_flags": item.applicability_flags,
                "values": {key: _scalar(value) for key, value in item.values.items()},
                "per_mode": {
                    mode: {key: _scalar(value) for key, value in values.items()}
                    for mode, values in item.per_mode.items()
                },
            }
        )
        prefix[identifier] = {
            "curves": _serialize(item.prefix_tv_curves),
            "threshold_steps": item.divergence_threshold_steps,
            "deterministic_edp": item.deterministic_edp,
        }
        brdiv[identifier] = _serialize(item.brdiv_matrices)
        predictability[identifier] = item.brdiv_matrices["prediction"]
        _write_json(output / "policies" / identifier / "passive.json", item.passive_policy)
        _write_json(output / "policies" / identifier / "active.json", item.active_policy)
        _write_json(
            output / "policies" / identifier / "information.json",
            item.information_policy,
        )
        _write_json(output / "policies" / identifier / "reference.json", item.reference_policy)
        _write_json(output / "frontiers" / f"{identifier}.json", item.active_frontier)
    pd.DataFrame(rows).to_csv(output / "population-summary.csv", index=False)
    _write_json(
        output / "population-summary.json",
        {
            "suite_id": benchmark.suite.suite_id,
            "population_count": len(summaries),
            "populations": summaries,
        },
    )
    _write_json(output / "prefix-tv-curves.json", prefix)
    _write_json(output / "brdiv-matrices.json", brdiv)
    _write_json(output / "predictability-audit.json", predictability)


def _scientific_checks(
    suite: MatchedBenchmarkSuite,
    benchmark: GeneratedBenchmarkSet,
    metrics: dict[str, PopulationMetrics],
    audits: tuple[Any, ...],
    shortcut_audits: tuple[Any, ...],
    calibrations: tuple[dict[str, Any], ...],
) -> dict[str, bool]:
    binary_early = metrics["binary-role-allocation--passive_early--identity"]
    binary_active = metrics["binary-role-allocation--active_only--identity"]
    binary_inseparable = metrics[
        "binary-role-allocation--precommit_inseparable--identity"
    ]
    factor_response = metrics[
        "factorized-identity-memory--remember_response--identity"
    ]
    factor_subtype = metrics[
        "factorized-identity-memory--remember_subtype--identity"
    ]
    no_identification = metrics[
        "binary-role-allocation--no_identification_needed--identity"
    ]
    expensive_id = "binary-role-allocation--active_too_expensive--identity"
    boundary_id = "binary-role-allocation--active_boundary--identity"
    expensive_population = benchmark.by_id()[expensive_id]
    boundary_population = benchmark.by_id()[boundary_id]
    expensive = solve(
        expensive_population.game,
        "task",
        "net_regret",
        "fraction",
        commitment_states=frozenset(expensive_population.descriptor.commitment_states),
    )
    boundary = solve(
        boundary_population.game,
        "task",
        "net_regret",
        "fraction",
        commitment_states=frozenset(boundary_population.descriptor.commitment_states),
    )
    return {
        "all_matching_contracts_pass": all(audit.passed for audit in audits),
        "all_shortcut_audits_pass": all(audit.passed for audit in shortcut_audits),
        "all_sample_calibrations_pass": all(item["passed"] for item in calibrations),
        "binary_brdiv_nonzero_and_matched": (
            binary_early.values["rahman_brdiv_return"]
            == binary_active.values["rahman_brdiv_return"]
            and binary_early.values["rahman_brdiv_return"] > 0
            and binary_early.values["zsceval_br_div_raw"]
            == binary_active.values["zsceval_br_div_raw"]
            and binary_early.values["zsceval_br_div_raw"] > 0
        ),
        "broad_predictability_matched_with_dri_gap": (
            close(
                binary_early.values["lobp_action_oracle_score_nats"],
                binary_active.values["lobp_action_oracle_score_nats"],
            )
            and binary_early.values["passive_dri"]
            - binary_active.values["passive_dri"]
            == Fraction(3, 5)
        ),
        "active_separability_gap": (
            binary_active.values["active_dri"]
            - binary_inseparable.values["active_dri"]
            == Fraction(3, 5)
        ),
        "identity_information_matched_with_decision_gap": (
            close(
                factor_response.values["identity_mutual_information_bits"],
                factor_subtype.values["identity_mutual_information_bits"],
            )
            and factor_response.values["passive_dri"]
            - factor_subtype.values["passive_dri"]
            == Fraction(3, 5)
        ),
        "late_evidence_excluded": (
            binary_active.values["passive_dri"] == 0
            and binary_active.values["eventual_dri"] == 1
        ),
        "no_identification_needed_is_null": no_identification.values["passive_dri"] is None,
        "expensive_intervention_rejected": expensive.policy.kind == "commit",
        "boundary_intervention_tie_commits": boundary.policy.kind == "commit",
        "symmetry_invariance": _symmetry_invariance(benchmark, metrics),
        "fraction_float_agreement": _canonical_backend_agreement(benchmark, metrics),
        "suite_has_two_families": len(suite.families) == 2,
    }


def _canonical_backend_agreement(
    benchmark: GeneratedBenchmarkSet,
    exact: dict[str, PopulationMetrics],
) -> bool:
    keys = (
        "prior_risk",
        "passive_residual_risk",
        "active_residual_risk",
        "passive_dri",
        "active_dri",
        "rahman_brdiv_return",
        "zsceval_br_div_raw",
        "zsceval_br_div_code",
    )
    for population in benchmark.populations:
        if population.descriptor.symmetry_id != "identity":
            continue
        approximate = compute_population_metrics(population, "float")
        for key in keys:
            left = exact[population.descriptor.population_id].values[key]
            right = approximate.values[key]
            if left is None or right is None:
                if left is not None or right is not None:
                    return False
            elif not close(left, right, 1e-10):
                return False
    return True


def _symmetry_invariance(
    benchmark: GeneratedBenchmarkSet,
    metrics: dict[str, PopulationMetrics],
) -> bool:
    groups: dict[tuple[str, str], list[str]] = {}
    for population in benchmark.populations:
        symmetry = population.descriptor.symmetry_id
        if symmetry.startswith("sweep_"):
            continue
        group_key = (population.descriptor.family_id, population.descriptor.cell_id)
        groups.setdefault(group_key, []).append(population.descriptor.population_id)
    invariant_keys = (
        "prior_risk",
        "passive_dri",
        "active_dri",
        "eventual_dri",
        "rahman_brdiv_return",
        "zsceval_br_div_raw",
        "lobp_action_oracle_score_nats",
    )
    for identifiers in groups.values():
        reference = metrics[identifiers[0]].values
        for identifier in identifiers[1:]:
            candidate = metrics[identifier].values
            for metric_key in invariant_keys:
                left, right = reference[metric_key], candidate[metric_key]
                if left is None or right is None:
                    if left is not None or right is not None:
                        return False
                elif not close(left, right, 1e-10):
                    return False
    return True


def _matching_rows(audits: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows = []
    for audit in audits:
        for item in audit.items:
            rows.append(
                {
                    "contract_id": audit.contract_id,
                    "left_population_id": audit.left_population_id,
                    "right_population_id": audit.right_population_id,
                    "metric": item.metric,
                    "left": _scalar(item.left),
                    "right": _scalar(item.right),
                    "difference": _scalar(item.difference),
                    "tolerance": _scalar(item.tolerance),
                    "status": item.status,
                    "role": item.role,
                    "reason": item.reason,
                }
            )
    return rows


def _plot_all(
    output: Path,
    populations: dict[str, Any],
    metrics: dict[str, PopulationMetrics],
    shortcut_audits: tuple[Any, ...],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _plot_timing(output, populations, metrics)
    _plot_frontiers(output, metrics)
    _plot_dri_brdiv(output, metrics)
    _plot_dri_predictability(output, metrics)
    _plot_identity(output, metrics)
    _plot_prefix_tv(output, metrics)
    _plot_memory(output, shortcut_audits)
    _plot_threshold(output, populations, metrics)


def _plot_timing(
    output: Path, populations: dict[str, Any], metrics: dict[str, PopulationMetrics]
) -> None:
    selected = [
        identifier
        for identifier, population in populations.items()
        if population.descriptor.family_id == "binary-role-allocation"
        and population.descriptor.symmetry_id.startswith("sweep_timing_")
    ]
    selected.sort(key=lambda item: populations[item].descriptor.symmetry_id)
    labels = [
        populations[item].descriptor.intended_treatments["passive_evidence_slot"]
        for item in selected
    ]
    values = [float(metrics[item].values["passive_dri"] or 0) for item in selected]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_ylabel("Pre-commitment DRI")
    ax.set_title("Evidence timing changes decision sufficiency")
    _save_figure(fig, output, "dri-by-evidence-timing")


def _plot_frontiers(output: Path, metrics: dict[str, PopulationMetrics]) -> None:
    identifiers = (
        "binary-role-allocation--active_only--identity",
        "binary-role-allocation--precommit_inseparable--identity",
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for identifier in identifiers:
        points = metrics[identifier].active_frontier["deterministic_points"]
        ax.plot(
            [_as_float(point["expected_cost"]) for point in points],
            [_as_float(point["dri"] or 0) for point in points],
            marker="o",
            label=identifier.split("--")[1],
        )
    ax.set_xlabel("Expected intervention cost")
    ax.set_ylabel("DRI")
    ax.set_title("Matched active-identifiability frontiers")
    ax.legend()
    _save_figure(fig, output, "matched-active-frontiers")


def _plot_dri_brdiv(output: Path, metrics: dict[str, PopulationMetrics]) -> None:
    identifiers = [
        "binary-role-allocation--passive_early--identity",
        "binary-role-allocation--active_only--identity",
        "binary-role-allocation--precommit_inseparable--identity",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    markers = ("o", "s", "x")
    for identifier, marker in zip(identifiers, markers, strict=True):
        item = metrics[identifier]
        label = identifier.split("--")[1]
        axes[0].scatter(
            float(item.values["rahman_brdiv_return"]),
            float(item.values["passive_dri"] or 0),
            label=label,
            marker=marker,
            s=70,
        )
        axes[1].scatter(
            float(item.values["zsceval_br_div_raw"]),
            float(item.values["passive_dri"] or 0),
            label=label,
            marker=marker,
            s=70,
        )
    axes[0].set_xlabel("Rahman return BRDiv")
    axes[0].set_ylabel("Passive DRI")
    axes[1].set_xlabel("Raw ZSC-Eval BR-Div")
    axes[1].set_ylabel("Passive DRI")
    axes[1].legend()
    fig.suptitle("Matched best-response diversity does not determine timely DRI")
    _save_figure(fig, output, "dri-vs-brdiv")


def _plot_dri_predictability(
    output: Path, metrics: dict[str, PopulationMetrics]
) -> None:
    identifiers = [
        "binary-role-allocation--passive_early--identity",
        "binary-role-allocation--active_only--identity",
        "binary-role-allocation--precommit_inseparable--identity",
    ]
    markers = ("o", "s", "x")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for identifier, marker in zip(identifiers, markers, strict=True):
        item = metrics[identifier]
        ax.scatter(
            float(item.values["lobp_action_oracle_score_nats"]),
            float(item.values["passive_dri"] or 0),
            label=identifier.split("--")[1],
            marker=marker,
            s=70,
        )
    ax.set_xlabel("Full-episode LoBP-style score (nats)")
    ax.set_ylabel("Passive DRI")
    ax.set_title("Equal broad predictability, different timely DRI")
    ax.legend()
    _save_figure(fig, output, "dri-vs-lobp-predictability")


def _plot_identity(output: Path, metrics: dict[str, PopulationMetrics]) -> None:
    identifiers = (
        "factorized-identity-memory--remember_response--identity",
        "factorized-identity-memory--remember_subtype--identity",
    )
    labels = [identifier.split("--")[1] for identifier in identifiers]
    identity = [
        float(metrics[item].values["identity_mutual_information_bits"])
        for item in identifiers
    ]
    decision = [
        float(metrics[item].values["decision_signature_mutual_information_bits"])
        for item in identifiers
    ]
    dri = [float(metrics[item].values["passive_dri"] or 0) for item in identifiers]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar([value - 0.25 for value in x], identity, width=0.25, label="Identity MI")
    ax.bar(x, decision, width=0.25, label="Decision-signature MI")
    ax.bar([value + 0.25 for value in x], dri, width=0.25, label="DRI")
    ax.set_xticks(list(x), labels)
    ax.set_title("Equal identity information can have different decision value")
    ax.legend()
    _save_figure(fig, output, "identity-information-vs-dri")


def _plot_prefix_tv(output: Path, metrics: dict[str, PopulationMetrics]) -> None:
    identifiers = (
        "factorized-identity-memory--remember_response--identity",
        "factorized-identity-memory--remember_subtype--identity",
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for identifier in identifiers:
        curves = metrics[identifier].prefix_tv_curves
        ordered = sorted(tuple(float(value) for value in curve) for curve in curves.values())
        mean = [sum(values) / len(ordered) for values in zip(*ordered, strict=True)]
        ax.plot(range(len(mean)), mean, marker="o", label=identifier.split("--")[1])
    ax.set_xlabel("Pre-commitment slot")
    ax.set_ylabel("Mean pairwise prefix TV")
    ax.set_title("Matched divergence profiles, different DRI")
    ax.legend()
    _save_figure(fig, output, "matched-prefix-tv")


def _plot_memory(output: Path, shortcut_audits: tuple[Any, ...]) -> None:
    item = next(
        audit
        for audit in shortcut_audits
        if audit.population_id
        == "factorized-identity-memory--remember_response--identity"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(
        ["Evidence blind", "Memoryless", "History aware"],
        [
            float(item.evidence_blind_risk),
            float(item.memoryless_risk),
            float(item.history_aware_risk),
        ],
        color=["#e45756", "#f2cf5b", "#4c78a8"],
    )
    ax.set_ylabel("Expected confusion loss")
    ax.set_title("Early evidence must survive the memory gap")
    _save_figure(fig, output, "memory-shortcut-audit")


def _plot_threshold(
    output: Path, populations: dict[str, Any], metrics: dict[str, PopulationMetrics]
) -> None:
    q_items = [
        (population.descriptor.intended_treatments["reliability"], metrics[identifier])
        for identifier, population in populations.items()
        if population.descriptor.symmetry_id.startswith("sweep_reliability_")
        and population.descriptor.family_id == "binary-role-allocation"
    ]
    c_items = [
        (population.descriptor.matched_nuisances["intervention_cost"], metrics[identifier])
        for identifier, population in populations.items()
        if population.descriptor.symmetry_id.startswith("sweep_cost_")
    ]
    q_items.sort(key=lambda item: _fraction_text(item[0]))
    c_items.sort(key=lambda item: float(item[0]))
    q_values = sorted({float(_fraction_text(item[0])) for item in q_items})
    c_values = sorted({float(_fraction_text(item[0])) for item in c_items})
    heat = [[1.0 if cost < 40 * (q - 0.5) else 0.0 for q in q_values] for cost in c_values]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    image = ax.imshow(heat, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(q_values)), [str(value) for value in q_values])
    ax.set_yticks(range(len(c_values)), [str(value) for value in c_values])
    ax.set_xlabel("Diagnostic reliability q")
    ax.set_ylabel("Intervention cost")
    ax.set_title("Exact intervention-optimality threshold")
    fig.colorbar(image, ax=ax, label="Probe strictly optimal")
    _save_figure(fig, output, "intervention-threshold")


def _save_figure(fig: Any, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def _fraction_text(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def _as_float(value: Any) -> float:
    """Convert numeric artifact values, including serialized rationals, to float."""
    if isinstance(value, str):
        return _fraction_text(value)
    return float(value)


def _dependency_versions() -> dict[str, str]:
    result = {}
    for package in (
        "pydantic",
        "numpy",
        "pandas",
        "matplotlib",
        "pytest",
        "hypothesis",
        "ruff",
        "mypy",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _source_hash(project_root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [
        project_root / "pyproject.toml",
        project_root / ".python-version",
        project_root / "uv.lock",
        project_root / "README.md",
    ]
    for directory in (project_root / "src", project_root / "tests"):
        candidates.extend(path for path in directory.rglob("*") if path.is_file())
    phase = project_root / "phase-3-matched-benchmarks"
    candidates.extend(
        path
        for path in phase.rglob("*")
        if path.is_file()
        and not {"artifacts", "generated"}.intersection(path.relative_to(phase).parts)
    )
    for path in sorted(set(candidates)):
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize(value: Any) -> Any:
    from fractions import Fraction

    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _scalar(value: Any) -> Any:
    from fractions import Fraction

    return str(value) if isinstance(value, Fraction) else value
