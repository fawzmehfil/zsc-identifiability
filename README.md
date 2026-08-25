# ZSC Identifiability

This repository contains exact research infrastructure for auditing whether an
agent can obtain decision-useful evidence about an unfamiliar partner before an
irreversible coordination choice. Phase 2 supplies the finite-game solvers and
metrics. Phase 3 supplies matched benchmark populations that vary timely
identifiability while holding standard controls fixed.

The package uses finite, static hidden partner modes and exact belief-state dynamic
programming. It deliberately contains no learned policies, neural dependencies, or
large environments.

## Reproduce the research package

```bash
uv sync --dev
uv run python -m zsc_identifiability run-suite \
  --suite phase-2-exact-model/suites/canonical.json \
  --output phase-2-exact-model/artifacts
uv run zsc-identifiability benchmark run \
  --suite phase-3-matched-benchmarks/suites/canonical.json \
  --output phase-3-matched-benchmarks/artifacts
uv run pytest
uv run ruff check .
uv run mypy src
```

The Phase 3 canonical suite generates 94 exact populations across two benchmark
families, checks four matching contracts, runs restricted-policy shortcut audits,
calibrates sampled estimators, and emits all tables and figures. Its current
machine-readable verdict is `continue`.

See `phase-2-exact-model/README.md` for the exact model and
`phase-3-matched-benchmarks/README.md` for the matched benchmark.
