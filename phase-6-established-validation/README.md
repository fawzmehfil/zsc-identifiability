# Stage 6: Official-Checkpoint Validation

Stage 6 is an inference-only audit of the official ZSC-Eval benchmark. It asks
whether pre-commitment decision-relevant identifiability (DRI) explains
coordination regret beyond competence, BR-Div, visible-action predictability,
and trajectory divergence.

The canonical study does not train partners or coordination policies. It uses:

- all 30 officially selected `random3_m` partners;
- all 20 officially selected `small_corridor` partners;
- each partner's official co-trained response counterpart;
- five published checkpoints for FCP, MEP, TrajeDi, HSP, COLE, and E3T;
- CPU inference plus small cross-fitted event and GRU measurement models.

The previous custom OvercookedV2 partner-generation study remains available as
the optional `suites/full-scale-overcookedv2.json` extension. It is not part of
the default workflow and its partial run is not reused.

Asset provenance and redistribution boundaries are documented in
`official-assets-and-license-card.md`.

## Fixed scientific boundary

Partner membership comes only from the official benchmark YAML files. No
checkpoint may be selected or removed using measured DRI. Both player seats and
paired environment keys are evaluated. The primary commitment point is the
first successful ingredient placement into a pot; delivery feedback is
post-commitment evidence and cannot change pre-commitment DRI.

The empirical response library is made from official co-trained counterparts.
It is an approximate response-library oracle, not a globally optimal oracle.
Response conflict is reported at adequacy margins `0.01`, `0.02`, and `0.05`.

The passive reference is FCP seed 1 in greedy mode. The event estimator and a
five-seed 64-unit GRU ensemble fit on calibration keys, calibrate on validation
keys, and score only untouched confirmatory keys. GRU fitting is measurement,
not policy training.

## Runtime and asset boundary

The main Python 3.12 package handles asset locks, response loss, DRI, statistics,
and reports. A Python 3.9 runtime loads the pinned official environment and
checkpoints. File-based, content-hashed requests cross this boundary.

Only the two benchmark YAMLs, four policy configs, 50 partner checkpoints, 50
response counterparts, and 60 method checkpoints are downloaded. The complete
policy pool is never synchronized. Confirmatory inference works offline after
asset synchronization.

The runtime imports official environment and policy classes directly, disables
CUDA, does not import an upstream trainer, and rejects any request that does not
declare `policy_training_allowed: false`. Historical seat-0 integration smokes
pass for both layouts. The canonical two-seat parity smoke is deliberately
pending until the locked official assets are synchronized.

Every official method checkpoint is evaluated using its published stochastic
sampling semantics and a greedy sensitivity deployment under identical
environment keys.

## Canonical workflow

Create the immutable asset lock:

```bash
uv run --extra established zsc-identifiability established official prepare \
  --suite phase-6-established-validation/suites/canonical.json \
  --workspace phase-6-established-validation/runs/official-checkpoints
```

Synchronize only the locked assets. This also creates the rollout plan:

```bash
uv run --extra established zsc-identifiability established official sync \
  --suite phase-6-established-validation/suites/canonical.json \
  --lock phase-6-established-validation/runs/official-checkpoints/official-asset-lock.json
```

Run evaluator parity before the full audit:

```bash
uv run --extra established zsc-identifiability established official smoke \
  --plan phase-6-established-validation/runs/official-checkpoints/official-rollout-plan.json \
  --workers 2
```

Run or resume the CPU rollout queue:

```bash
caffeinate -is uv run --extra established zsc-identifiability established official run \
  --plan phase-6-established-validation/runs/official-checkpoints/official-rollout-plan.json \
  --workers 2
```

Inspect status without running inference:

```bash
uv run --extra established zsc-identifiability established official status \
  --plan phase-6-established-validation/runs/official-checkpoints/official-rollout-plan.json
```

Build the response-conflict and pairwise-identifiability artifacts:

```bash
uv run --extra established zsc-identifiability established official analyze \
  --suite phase-6-established-validation/suites/canonical.json \
  --plan phase-6-established-validation/runs/official-checkpoints/official-rollout-plan.json \
  --ledger phase-6-established-validation/runs/official-checkpoints/official-rollout-ledger.json \
  --output phase-6-established-validation/artifacts/official-checkpoint-audit
```

Analysis first converts each completed trace shard to an ignored sparse cache,
then trains the measurement GRUs with bounded mini-batches. Cache construction,
every layout/policy/prefix work unit, and every GRU seed are checkpointed under
the output directory's `.analysis-state/`. Repeating the same command resumes
automatically and never reruns policy inference.

Exit code `2` denotes an engineering or integrity failure, `3` a completed
scientific-gate failure, and `4` missing assets or incomplete shards.

## Current status

- v2 schema and inference-only policy: implemented and tested;
- official asset discovery, minimal synchronization, hashing, and duplicate
  detection: implemented;
- atomic CPU rollout plan, ledger, recovery, and offline boundary: implemented;
- official policy/environment loading and two-seat parity harness: implemented;
- historical seat-0 integration parity on both layouts: verified;
- canonical two-seat parity and complete rollout matrix: complete;
- response-library, conflict, event/GRU DRI, and scheme-held-out statistical
  components: implemented;
- custom full-compute extension: preserved but retired from the default path;
- bounded-memory, resumable scientific analysis: implemented and under execution;
- final established-environment verdict: pending completed analysis.

No established-environment scientific finding is claimed before the complete
official audit and sensitivity analyses pass.
