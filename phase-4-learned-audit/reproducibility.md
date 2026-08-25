# Reproducibility

Primary training runs on CPU with deterministic PyTorch algorithms. NumPy partner
and transition sampling uses `PCG64`; Python, NumPy, PyTorch, optimizer, model, and
environment states are captured in checkpoints. Canonical values are computed by
exact finite-tree traversal, and sampled evaluation is used only as a calibration
check.

An interrupted run can resume from a numbered `checkpoints/step-*.pt` boundary
with `learn train --resume CHECKPOINT --cell CELL --seed SEED`. The command rejects
suite, method, cell, seed, or configuration-hash mismatches.

Development seeds are `101`, `102`, and `103`. Confirmatory seeds are `2001`
through `2010`. Test mechanisms never select hyperparameters or checkpoints.

To reproduce the software gates:

```bash
uv sync --extra learning --dev
uv run ruff check .
uv run mypy src
uv run pytest
uv run --extra learning zsc-identifiability learn validate \
  --suite phase-4-learned-audit/suites/canonical.json
```

Configurations, exact summaries, compact tables, figures, and manifests are
versioned. Checkpoints, optimizer states, and raw rollouts remain outside Git.

The suite declares two structural relabelings. `learn symmetry --method METHOD`
re-trains them independently with the confirmatory seed schedule; contradictory
signal codebooks are never mixed into one policy's training distribution.

Rescue runs live under `phase-4-learned-audit/rescue-runs/` and are evaluated only
after the frozen primary analysis finds an active-oracle gap. The final
`continue_to_repair` verdict cannot be emitted while those runs are missing.
