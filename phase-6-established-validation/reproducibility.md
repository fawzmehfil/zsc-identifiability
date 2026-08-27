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

## Frozen partner-pool queue

Create the deterministic plan first. This command performs validation and file
generation only:

```bash
uv run zsc-identifiability established partner-pools prepare \
  --suite phase-6-established-validation/suites/canonical.json \
  --layout demo_cook_simple \
  --workspace phase-6-established-validation/runs/partner-pools/demo-cook-simple
```

The queue is safe to detach and invoke again after interruption. The transition
targets are totals, and completed hash-valid work is skipped:

```bash
nohup caffeinate -is uv run zsc-identifiability established partner-pools run \
  --plan phase-6-established-validation/runs/partner-pools/demo-cook-simple/build-plan.json \
  --workers 1 \
  --freeze-on-success \
  > phase-6-established-validation/runs/partner-pools/demo-cook-simple/queue.log 2>&1 &
```

Read-only status inspection:

```bash
uv run zsc-identifiability established partner-pools status \
  --plan phase-6-established-validation/runs/partner-pools/demo-cook-simple/build-plan.json
```

Exit code `2` identifies unresolved job failures, `3` means a split exhausted
its cap without meeting quota, and `4` means the requested subset is valid but
the complete three-split pool is not ready. A successful immutable freeze
returns `0`.

## Resumable training

`established train-method` accepts `--resume` with either a full update-boundary
checkpoint or a multi-component `pipeline-state.json`. The target transition
count is total, never additive. Resume validates the method, layout, seed,
partner-pool hashes, architecture, hyperparameters, suite, upstream commit, and
local runner source before restoring optimizer, environment, recurrent,
partner-index, schedule, auxiliary-model, and PRNG state.

TBS additionally requires `--cross-play-values`. TBS and CSP accept
`--compute-allocation per-specialist|split-total`; reports retain both component
and aggregate transition counts. A resumed multi-component pipeline verifies
completed artifacts by content hash and skips them.

Full checkpoints remain untracked. Ported methods retain the latest two and the
best fixed-key validation snapshot. `latest.json` and `best.json` are published
only after the new state has been restored and hash-verified. Compact deployment
artifacts contain frozen inference components only.

The low-level compatibility command can still continue qualifying partner
screens manually:

```bash
uv run zsc-identifiability established train-partners \
  --suite phase-6-established-validation/suites/canonical.json \
  --split train \
  --layout demo_cook_simple \
  --gate finalist \
  --resume-index SCREEN_CHECKPOINT_INDEX.json \
  --output phase-6-established-validation/runs/partner-finalists \
  --execute
```

Notebook output is not an accepted source for a reported number. Compact results
must be reproduced by a registered CLI command.
