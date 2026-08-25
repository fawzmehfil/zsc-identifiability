# ZSC Identifiability

This repository contains the exact-model research package for auditing whether an
agent can obtain decision-useful evidence about an unfamiliar partner before an
irreversible coordination choice.

The package uses finite, static hidden partner modes and exact belief-state dynamic
programming. It deliberately contains no learned policies, neural dependencies, or
large environments.

## Reproduce Phase 2

```bash
uv sync --dev
uv run python -m zsc_identifiability run-suite \
  --suite phase-2-exact-model/suites/canonical.json \
  --output phase-2-exact-model/artifacts
uv run pytest
uv run ruff check .
uv run mypy src
```

See `phase-2-exact-model/README.md` for the model, schema, commands, and artifact map.

