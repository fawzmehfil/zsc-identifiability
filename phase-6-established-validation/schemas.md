# Versioned Schemas

The canonical configuration is `OfficialCheckpointAuditSuiteV2`, represented by
`suites/canonical.json` and `schemas/official-checkpoint-suite-v2.schema.json`.
Unknown fields are rejected. The Pydantic validator additionally rejects policy
training budgets at any nesting level, unpinned revisions, DRI-based partner
selection, missing response counterparts, fewer than two layouts or four
published methods, and overlapping evidence keys.

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

The v1 `EstablishedValidationSuite` and its response, trace, training,
checkpoint, and partner-pool schemas remain supported by
`suites/full-scale-overcookedv2.json`. They are an optional full-compute
extension rather than the canonical Stage 6 path.

Schema metadata never becomes part of a policy observation.
