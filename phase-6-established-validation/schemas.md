# Versioned Schemas

The completed inference configuration is `OfficialCheckpointAuditSuiteV2`,
preserved byte-for-byte in `suites/official-checkpoint-v2.json` and represented
by `schemas/official-checkpoint-suite-v2.schema.json`. Unknown fields are
rejected. The Pydantic validator additionally rejects policy training budgets
at any nesting level, unpinned revisions, DRI-based partner selection, missing
response counterparts, fewer than two layouts or four published methods, and
overlapping evidence keys.

The official-checkpoint pipeline uses immutable versioned records for:

- the content-addressed official asset lock and synchronized inventory;
- selected partners, co-trained responses, published method seeds, and tensor
  duplicate groups;
- atomic rollout plans, shards, ledgers, and compressed trace indexes;
- raw and normalized response-value matrices and adequacy-margin conflicts;
- pairwise identifiability rows and the final audit manifest.

Every asset record includes source revision, relative and local path, size,
file hash, normalized tensor hash when applicable, architecture, layout,
algorithm, seed, and provenance. Runtime requests include a content hash and
`policy_training_allowed: false`.

The measurement redesign uses `OfficialMeasurementAuditSuiteV3`, represented by
`suites/official-measurement-v3.json` and
`schemas/official-measurement-suite-v3.schema.json`. It content-locks the v2
suite, rollout plan and ledger, asset inventory, completed manifest, trace
index, response values, method outcomes, and exclusions. It fixes the
representation seeds, decoder search grid, permutation test, fresh salt,
intervention choices, and 9,600-episode confirmation design.

Versioned v3 artifacts separately model frozen representations and direct
pairwise decoders, the trace-only confirmation plan and ledger, pairwise
decision-value rows, calibration report, and final audit manifest. Tuning code
can address only v2 calibration and validation entries; confirmation code can
request only `official_trace_rollout` work.

The v1 `EstablishedValidationSuite` and its response, trace, training,
checkpoint, and partner-pool schemas remain supported by
`suites/full-scale-overcookedv2.json`. They are an optional full-compute
extension rather than the canonical Stage 6 path.

Schema metadata never becomes part of a policy observation.
