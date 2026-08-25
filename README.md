# ZSC Identifiability

This repository contains exact research infrastructure for auditing whether an
agent can obtain decision-useful evidence about an unfamiliar partner before an
irreversible coordination choice. Phase 2 supplies finite-game solvers and
metrics. Phase 3 supplies matched populations that vary timely identifiability
while holding standard controls fixed. Stage 4 adds a controlled learned-agent
audit over those exact populations.

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
```

The Phase 3 canonical suite generates 94 exact populations across two benchmark
families, checks four matching contracts, runs restricted-policy shortcut audits,
calibrates sampled estimators, and emits all tables and figures. Its current
machine-readable verdict is `continue`.

See `phase-2-exact-model/README.md` for the exact model and
`phase-3-matched-benchmarks/README.md` for the matched benchmark. The Stage 4
protocol and execution status are in `phase-4-learned-audit/README.md`.
