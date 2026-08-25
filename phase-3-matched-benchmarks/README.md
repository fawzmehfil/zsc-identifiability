# Matched Identifiability Benchmark

This package tests whether decision-sufficient identifiability is a distinct
evaluation axis in zero-shot coordination. It constructs partner populations with
the same task, competence, loss geometry, best-response diversity, and broad
behavioral predictability, then changes only whether useful partner evidence is
available before an irreversible decision.

The implementation is training-free. Every population is a finite convention game
solved exactly by the Phase 2 belief-state tools.

## Benchmark families

`Binary Role Allocation` contains two equally competent partner modes that require
opposite role choices. Its cells isolate passive evidence, ordinary task actions
that reveal evidence, late evidence, intervention cost, and complete
inseparability. Conflicting cells have prior risk 20, known-mode return 100, best
fixed return 80, Rahman return BRDiv 40, and raw ZSC-Eval BR-Div 1.

`Factorized Identity and Memory` contains four modes with a response bit and an
irrelevant subtype bit. Response and subtype signals carry equal identity
information and have the same aggregate prefix-TV profile, but only the response
signal reduces decision loss. A neutral intervening step makes the response cell a
strict memory test.

## Reproduce

```bash
uv sync --dev

uv run zsc-identifiability benchmark validate \
  --suite phase-3-matched-benchmarks/suites/canonical.json

uv run zsc-identifiability benchmark generate \
  --suite phase-3-matched-benchmarks/suites/canonical.json \
  --output phase-3-matched-benchmarks/generated

uv run zsc-identifiability benchmark run \
  --suite phase-3-matched-benchmarks/suites/canonical.json \
  --output phase-3-matched-benchmarks/artifacts
```

Command exit codes are `0` for a passing implementation and scientific audit, `2`
for schema/runtime/numerical failure, and `3` when generation succeeds but a
scientific matching requirement fails.

## Canonical result

The current exact run returns `continue`. In particular:

- passive evidence yields DRI `3/5`, while a matched active-only cell has passive
  DRI `0` despite identical Rahman BRDiv, both ZSC-Eval determinants, known-mode
  competence, and full-episode LoBP-style predictability;
- the active-only and pre-commitment-inseparable cells have identical passive
  histories, but active DRI differs by `3/5`;
- response and subtype signals carry the same `0.2780719051` bits of identity
  information and the same aggregate divergence profile, while their DRI values
  are `3/5` and `0`;
- in the memory cell, evidence-blind and memoryless policies retain risk `20`,
  while the history-aware oracle reduces risk to `8`;
- an informative intervention costs `5`, leaves risk `8`, and achieves net regret
  `13`; costs `12` and `15` trigger immediate commitment under the declared
  tie-breaking rule.

The authoritative gate is
[`scientific-checks.json`](artifacts/scientific-checks.json). The complete exact
values are in [`population-summary.json`](artifacts/population-summary.json), and
the pairwise controls are in
[`matching-audit.json`](artifacts/matching-audit.json).

## Artifact map

- `suites/canonical.json`: versioned benchmark source.
- `generated/`: canonical v1 games and population descriptors.
- `artifacts/generated/`: generated inputs captured by the complete run.
- `artifacts/population-summary.*`: exact population metrics.
- `artifacts/matching-audit.*`: control and treatment checks per contract.
- `artifacts/shortcut-audit.json`: evidence-blind, fixed, memoryless, and leakage
  checks.
- `artifacts/estimator-calibration.json`: paired Monte Carlo and bootstrap audit.
- `artifacts/frontiers/` and `artifacts/policies/`: exact frontiers and oracle
  policy trees.
- `artifacts/figures/`: report figures in PDF and PNG.
- `artifacts/manifest.json`: source/configuration hashes and runtime provenance.

See [`schema.md`](schema.md) for the suite and descriptor contracts and
[`benchmark-card.md`](benchmark-card.md) for intended use and limitations.
