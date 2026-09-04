# ZSC Identifiability

Research infrastructure for asking a missing question in zero-shot
coordination: **can an agent identify the response an unfamiliar teammate
requires before the task makes that information useless?**

[Final Stage 6 results](phase-6-established-validation/results/README.md) ·
[Statistical claim audit](phase-6-established-validation/statistical-claim-audit.md) ·
[Exact model](phase-2-exact-model/README.md) ·
[Matched benchmark](phase-3-matched-benchmarks/README.md) ·
[Learned-agent audit](phase-4-learned-audit/README.md) ·
[Established validation](phase-6-established-validation/README.md) ·
[Related work](docs/related-work.md) ·
[MIT license](LICENSE)

## Overview

Partner diversity does not guarantee that a partner is identifiable in time.
Two partners can require incompatible responses while behaving alike until
after an irreversible decision. Conversely, two visibly different partners
may admit the same response, making identity information irrelevant.

This project measures the missing axis with **decision-relevant
identifiability (DRI)**: the reduction in best-response decision risk provided
by the evidence available before a task-defined commitment point.

```text
unfamiliar partner
        |
        v
pre-commitment evidence --> residual response risk --> coordination regret
        ^
        |
ordinary task interaction, if one is informative and worth its cost
```

DRI is normalized Bayes-risk reduction, not new information theory. The
contribution is its commitment-timed ZSC operationalization, exact controlled
benchmark, learned-agent decomposition, and validation on an externally
selected official partner population.

## Main result

The final official-checkpoint audit used 50 ZSC-Eval partners, six published
ZSC method families, two layouts, and a fresh 9,600-episode confirmation set.
Adding pre-commitment DRI to the registered controls improved held-out regret
prediction in both layouts. The independent event representation preserved the
effect direction.

| Representation | Scope | Held-out ΔR² | ΔMAE | ΔMSE |
|---|---:|---:|---:|---:|
| GRU | Overall | +0.0205 | -0.00339 | -0.00137 |
| GRU | `random3_m` | +0.0243 | -0.00220 | -0.00090 |
| GRU | `small_corridor` | +0.00745 | -0.00123 | -0.00060 |
| Event | Overall | +0.0118 | -0.00215 | -0.00079 |

The adjusted GRU coefficient relating DRI to regret was -0.180, with a
clustered 95% interval of [-0.295, -0.079]. Passive DRI exceeded the registered
permutation null in both layouts after Holm correction (`p = 0.0396`).

![Held-out predictive value of DRI](phase-6-established-validation/results/figures/dri-predictive-value.png)

The natural-intervention result was negative. Neither
`temporary_role_takeover` nor `corridor_yield` produced a corrected,
permutation-supported improvement in decision risk. The final Stage 6 verdict
is therefore **`complete_measurement_only`**: established-environment evidence
supports DRI as a measurement, not an active-probing claim.

## Research package

| Component | What it establishes | Status |
|---|---|---|
| Exact finite games | Belief updates, Bayes risk, DRI, TV identities, regret bounds, and active-identifiability frontiers | Complete |
| Matched benchmark | DRI varies while response diversity, competence, task structure, and broad predictability remain fixed | Complete |
| Learned-agent audit | Memory, evidence acquisition, evidence relevance, and evidence use are separated against exact oracles | Complete |
| Official-checkpoint validation | DRI adds held-out predictive value in two ZSC-Eval layouts | Complete: measurement only |
| New repair method | Existing methods closed the synthetic active gap, so a new method was not justified | Intentionally skipped |

Stage 4 trained compact agents only in the exact synthetic benchmark. Stage 6
performed no partner or coordination-policy training: it used official frozen
checkpoints and fitted small measurement models only.

## Quick start

Install the exact package and reproduce the finite-game results:

```bash
git clone https://github.com/fawzmehfil/zsc-identifiability.git
cd zsc-identifiability
uv sync --dev

uv run zsc-identifiability run-suite \
  --suite phase-2-exact-model/suites/canonical.json \
  --output phase-2-exact-model/artifacts

uv run zsc-identifiability benchmark run \
  --suite phase-3-matched-benchmarks/suites/canonical.json \
  --output phase-3-matched-benchmarks/artifacts
```

Run the repository checks:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

The official-checkpoint audit uses pinned external assets in isolated runtimes.
Its complete workflow is documented in
[Stage 6](phase-6-established-validation/README.md). If the frozen row-level
analysis is present locally, regenerate the committed publication package with:

```bash
uv run zsc-identifiability established official redesign publish \
  --input phase-6-established-validation/artifacts/official-measurement-v3 \
  --output phase-6-established-validation/results
```

## Technical shape

| Layer | Implementation |
|---|---|
| Formal model | Finite hidden partner modes with early or forced commitment |
| Exact control | Fraction-backed belief-state dynamic programming and policy-tree enumeration |
| Benchmarking | Structurally matched populations, shortcut audits, exact frontiers |
| Learned audit | PyTorch PPO/GRU baselines with exact neural-policy evaluation |
| Established validation | Pinned ZSC-Eval checkpoints, CPU inference, cross-fitted decision decoders |
| Reproducibility | Versioned schemas, content hashes, disjoint keys, resumable shards, compact artifacts |

## Scientific boundary

The project does not claim that active teammate probing, Bayes-risk reasoning,
or online partner adaptation is new. It does not claim that the empirical
response library is globally optimal, that every held-out scheme improves, or
that the audited task interventions work. See the
[claim audit](phase-6-established-validation/statistical-claim-audit.md) for
the exact permissible wording and limitations.
