# Reproducibility

## Locked sources

| Source | Revision |
|---|---|
| ZSC-Eval | `f940869afc42b688332a385892d8dbb57a190f95` |
| Official ZSC-Eval policy pool | `a39b45a326c6fb9c4aee79550903a7de702c6974` |

Both sources use the MIT license. The asset lock records every selected path,
revision, size when known, file hash, and normalized tensor hash. Duplicate
weights remain visible and cannot count as independent seeds.

## Verify the implementation

```bash
uv sync --all-extras --group dev
uv run pytest
uv run ruff check .
uv run mypy src
uv lock --check
uv lock --project phase-6-established-validation/runtime-zsceval --check
git diff --check
```

The isolated runtime is synchronized separately:

```bash
uv sync --project phase-6-established-validation/runtime-zsceval
```

Its pinned compatibility stack uses Python 3.9, PyTorch 2.2, NumPy 1.23, and
Gym 0.22. The main package remains on Python 3.12.

## Resumption and storage

The rollout plan partitions work at partner-policy shard boundaries. A ledger
is atomically replaced after each boundary. A completed shard is skipped only
when its result file still matches the recorded SHA-256 hash. An interrupted
`running` entry becomes pending on resume; failures are retained and do not
erase completed work.

CPU workers are capped at four and default to two. Every worker disables CUDA
and limits numerical-library threads. Result shards are gzip-compressed. Full
reference observations are retained only for fixed evidence policies; method
evaluation retains compact visible event histories.

Raw checkpoints, source checkouts, rollout shards, logs, and measurement-model
states remain ignored. Committed artifacts contain the suite, asset hashes,
compact tables, reports, and publication figures.

## Statistical contract

All response-conflicting partner pairs are analyzed. Mid/final checkpoints from
one HSP scheme are grouped together. The primary prediction model holds out
every pair involving one HSP scheme and selects ridge strength using nested
scheme-level folds. BR-Prox is a secondary outcome and is never used to predict
response-library regret.

Sensitivity analyses retain every official partner and report any exclusions.
They vary response adequacy, estimator family, deployment stochasticity,
commitment definition, checkpoint stage, player seat, competence filtering,
identity information, and prefix-TV substitution. Null and negative folds are
reported.
