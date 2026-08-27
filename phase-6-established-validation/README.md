# Stage 6: Established-Environment Validation

Stage 6 tests whether pre-commitment decision-relevant identifiability (DRI)
explains zero-shot coordination performance beyond competence, BR-Div, BR-Prox,
behavioral predictability, and trajectory divergence in OvercookedV2. It skips a
standalone repair-method phase because Stage 4 found that existing mechanisms
already reach the active oracle in the exact games.

The environment, measurement platform, method-specific TBS/PACE/CSP ports,
exact checkpoint-resumption layer, and frozen partner-pool orchestrator are
implemented. Partner generation and the development and confirmatory matrices
have not been executed, so the current scientific verdict remains `pending`.

## Fixed protocol

- The primary environment is the official `demo_cook_simple` OvercookedV2
  layout. `test_time_simple` audits evidence timing,
  `grounded_coord_simple` is the recipe-information control, and
  `demo_cook_wide` is the geometry replication.
- Episodes use 400 steps, view radius 2, randomized agent positions, negative
  incorrect-delivery reward, and recipe resampling after delivery.
- The commitment point is an actual increase in pot ingredient count. The event
  itself is excluded from pre-commitment evidence.
- The recipe button is always classified as environment information, never as a
  teammate probe.
- Hidden partner modes are frozen checkpoints. A fixed empirical response
  library defines approximate response loss; it is not called a globally
  optimal oracle.
- GRU and high-level-event DRI estimators train on calibration traces, calibrate
  on validation traces, and report only on disjoint confirmatory traces. Their
  selected response is scored against the held-out partner's frozen loss row,
  so confident classifier error cannot masquerade as useful information.
- Partner populations are selected on discovery data and then frozen. Failed
  confirmatory matching cannot be repaired by changing partners or margins.

## Runtime isolation

The exact Python 3.12 package never imports JAX or the legacy repositories.
Versioned JSON requests, trace JSONL, and compact result manifests cross the
runtime boundary.

| Runtime | Purpose |
|---|---|
| Python 3.12 | DRI, response loss, matching, statistics, reports |
| Python 3.10 | pinned OvercookedV2/JAX environment and policies |
| Python 3.9 | pinned ZSC-Eval asset audit |
| Python 3.10 | pinned ToMZSC reference tooling (JAX 0.4.38) |

The suite pins full upstream commit hashes. `.external/`, checkpoints, raw
traces, runtime environments, and logs are ignored by Git.

## Commands

Validate the schema, local pins, isolated runtimes, and analytical DRI bridge:

```bash
uv run zsc-identifiability established validate \
  --suite phase-6-established-validation/suites/canonical.json
```

Bootstrap the three pinned repositories and isolated runtimes:

```bash
uv run zsc-identifiability established bootstrap \
  --suite phase-6-established-validation/suites/canonical.json
```

Prepare the complete deterministic partner-pool plan without launching training:

```bash
uv run zsc-identifiability established partner-pools prepare \
  --suite phase-6-established-validation/suites/canonical.json \
  --layout demo_cook_simple \
  --workspace phase-6-established-validation/runs/partner-pools/demo-cook-simple
```

The plan materializes every candidate through the registered caps and activates
only the first quota-sized batch. Run the resumable queue separately when
compute is available:

```bash
nohup caffeinate -is uv run zsc-identifiability established partner-pools run \
  --plan phase-6-established-validation/runs/partner-pools/demo-cook-simple/build-plan.json \
  --workers 1 \
  --freeze-on-success \
  > phase-6-established-validation/runs/partner-pools/demo-cook-simple/queue.log 2>&1 &
```

Inspect it without starting or modifying training:

```bash
uv run zsc-identifiability established partner-pools status \
  --plan phase-6-established-validation/runs/partner-pools/demo-cook-simple/build-plan.json
```

If the queue was run without `--freeze-on-success`, freeze the verified pools:

```bash
uv run zsc-identifiability established partner-pools freeze \
  --plan phase-6-established-validation/runs/partner-pools/demo-cook-simple/build-plan.json
```

`established train-partners` remains available as the low-level compatibility
interface for manually prepared job slices. It is not the recommended complete
pool workflow.

Run the registered 100k-transition trainer/checkpoint smoke for one method:

```bash
uv run zsc-identifiability established train-method \
  --suite phase-6-established-validation/suites/canonical.json \
  --method rnn_ippo \
  --layout demo_cook_simple \
  --seed 5101 \
  --gate smoke \
  --learning-rate 0.00025 \
  --entropy-coefficient 0.01 \
  --output phase-6-established-validation/runs/smoke/rnn-ippo \
  --execute
```

Ported methods additionally receive frozen training and validation pools. TBS
also receives a training-only cross-play matrix; CSP is always labelled as a
two-episode reconnaissance protocol:

```bash
uv run zsc-identifiability established train-method \
  --suite phase-6-established-validation/suites/canonical.json \
  --method pace_style \
  --layout demo_cook_simple \
  --seed 5101 \
  --gate smoke \
  --learning-rate 0.00025 \
  --entropy-coefficient 0.01 \
  --train-pool TRAIN_POOL.json \
  --validation-pool VALIDATION_POOL.json \
  --output phase-6-established-validation/runs/smoke/pace-style \
  --execute
```

Build the frozen response-loss matrix from cross-play values:

```bash
uv run zsc-identifiability established build-responses \
  --suite phase-6-established-validation/suites/canonical.json \
  --values phase-6-established-validation/runs/response-values.json \
  --clusters phase-6-established-validation/runs/response-clusters.json \
  --output phase-6-established-validation/runs/response-library.json
```

The `collect`, `match`, `train-method`, `evaluate`, `audit-diagnostics`,
`estimate-dri`, `regress`, `secondary-audit`, `audit`, and `run` subcommands
expose the remaining registered stages. Heavy runtime commands write a request
first and execute only when `--execute` is supplied.

Exit code `4` means required external assets are not available. Exit code `3`
means execution completed but a scientific contract failed. Neither is silently
converted into a successful result.

## Current status

- Suite, trace, response-library, partner-pool, DRI, matching, training, and
  evaluation schemas: implemented.
- Commitment extraction and leakage controls: implemented and tested.
- Event and GRU posterior-to-DRI estimators: implemented.
- Disjoint visible-action LoBP-style predictability control: implemented.
- Mode-conditioned ego-visible prefix-TV divergence curves: implemented.
- Sparse reward-vector generation and SHA-based splits: implemented.
- Frozen mixed-integer matching and confirmatory audit: implemented.
- Natural diagnostic goal controllers and restricted empirical frontier:
  implemented.
- Cross-fitted incremental regression and hierarchical bootstrap utilities:
  implemented.
- Pinned upstream bootstrap and file-based runtime boundary: implemented.
- Official environment, commitment, 100k-transition training/checkpoint, and
  checkpoint-to-trace integration smokes: passed.
- PACE auxiliary, PACE-style, TBS-style, and CSP-style method ports: implemented
  and smoke-verified with reloadable deployment artifacts.
- Exact update-boundary continuation, method-pipeline resumption, best-validation
  retention, and screening-to-finalist partner continuation: implemented.
- Deterministic partner-pool planning, queue expansion, per-stage atomic ledger,
  streamed logs, recovery-only compact export, leakage audit, and immutable
  freezing: implemented and tested without training.
- Established-environment partner pools, trained methods, matched contrasts, and
  scientific verdict: pending execution.

See `stage-6-exit-memo.md` for the exact gate that remains open.
