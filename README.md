# ZSC Identifiability

This repository contains exact research infrastructure for auditing whether an
agent can obtain decision-useful evidence about an unfamiliar partner before an
irreversible coordination choice. Phase 2 supplies finite-game solvers and
metrics. Phase 3 supplies matched populations that vary timely identifiability
while holding standard controls fixed. Stage 4 adds a controlled learned-agent
audit over those exact populations. Stage 6 adds an isolated, pinned
OvercookedV2 validation pipeline for testing whether the same measurement is
useful in an established environment.

Stage 4's confirmatory audit is complete. Existing methods reach the exact active
oracle in the canonical games, the preregistered strict ranking-reversal count is
zero, and the current scientific verdict is `continue_without_repair`. The next
research step is established-environment validation rather than a new repair
algorithm.

The Stage 6 core environment and measurement platform is implemented and
verified, but its method-specific TBS/PACE/CSP ports and large partner and
confirmatory matrices remain open. No established-environment scientific result
is claimed yet.

The exact package uses finite, static hidden partner modes and belief-state dynamic
programming. Neural baselines are isolated behind the optional `learning` extra;
the exact Phase 2/3 package remains usable without PyTorch.

## Reproduce the research package

```bash
uv sync --dev
uv run python -m zsc_identifiability run-suite \
  --suite phase-2-exact-model/suites/canonical.json \
  --output phase-2-exact-model/artifacts
uv run zsc-identifiability benchmark run \
  --suite phase-3-matched-benchmarks/suites/canonical.json \
  --output phase-3-matched-benchmarks/artifacts
uv sync --extra learning --dev
uv run --extra learning zsc-identifiability learn validate \
  --suite phase-4-learned-audit/suites/canonical.json
uv run pytest
uv run ruff check .
uv run mypy src
uv run zsc-identifiability established validate \
  --suite phase-6-established-validation/suites/canonical.json
```

The Phase 3 canonical suite generates 94 exact populations across two benchmark
families, checks four matching contracts, runs restricted-policy shortcut audits,
calibrates sampled estimators, and emits all tables and figures. Its current
machine-readable verdict is `continue`.

See `phase-2-exact-model/README.md` for the exact model and
`phase-3-matched-benchmarks/README.md` for the matched benchmark. The Stage 4
protocol and execution status are in `phase-4-learned-audit/README.md`. The
Stage 6 environment protocol, runtime isolation, and execution gates are in
`phase-6-established-validation/README.md`.
