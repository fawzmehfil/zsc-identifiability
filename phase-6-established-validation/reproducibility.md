# Reproducibility

## Pinned sources

| Source | Commit |
|---|---|
| OvercookedV2 experiments | `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e` |
| ZSC-Eval | `f940869afc42b688332a385892d8dbb57a190f95` |
| ToMZSC | `a4b41d53fc77e452cdfca8edc95fd153d51d13cd` |

Bootstrap verifies the complete checked-out hash and origin URL. Runtime requests
repeat the expected hash and refuse to execute against another checkout.
The OvercookedV2 and ToMZSC runtimes lock JAX 0.4.38 with its compatible Flax
stack; the legacy ZSC-Eval asset audit remains isolated on Python 3.9.

## Compact validation

```bash
uv sync --extra established --dev
uv run pytest
uv run ruff check src tests phase-6-established-validation/runtime-overcookedv2/src
uv run mypy src
uv run zsc-identifiability established validate \
  --suite phase-6-established-validation/suites/canonical.json
```

`established validate` may return exit code 4 before external runtimes are
bootstrapped. The JSON still distinguishes schema validity and analytical DRI
calibration from missing external assets.

## Artifact boundary

Committed files contain schemas, suite configuration, compact manifests, tables,
and final PDF/PNG figures. The following stay untracked:

- `.external/` upstream repositories;
- isolated `.venv/` directories;
- checkpoints and optimizer state;
- raw trace JSONL and rollout data;
- policy pools, logs, and videos.

Every runtime request and result carries a schema version and content hash. Trace
manifests record checkpoint hashes, partner identifiers, evaluation keys, and
whether post-commitment evidence was excluded.

Notebook output is not an accepted source for a reported number. Compact results
must be reproduced by a registered CLI command.
