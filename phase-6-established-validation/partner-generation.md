# Partner Generation and Population Freezing

Behavior-preferring partners retain the original shared task reward and add at
most three nonzero event preferences. Registered events cover ingredient type,
counter use, pot and cooking activity, plates and delivery, recipe-button use,
task regions, corridor behavior, yielding, idling, movement, and successful task
reward.

Canonical JSON serialization and SHA-256 determine reward-vector identity.
A salted SHA-256 assignment fixes train, validation, and evaluation splits before
training. Reward vectors, seeds, or normalized checkpoint hashes may not cross
splits.

Each reward vector has two independent partner seeds. The split-specific seed
bands are disjoint: training begins at `41001`, validation at `141001`, and
evaluation at `241001`; vector index and replicate determine the exact seed.
The candidate quotas/caps are `24/48`, `8/16`, and `32/64`. Expansion always
activates eight complete candidates, preserving the two-seed reward-vector
grouping. A failed quota is reported; competence or matching thresholds cannot
be relaxed.

`partner-pools prepare` validates the suite, pinned upstream commits, isolated
runtime, layout, and source trees. It enumerates every possible candidate
through each cap in SHA-256 reward-vector order, followed by replicate `0` and
`1`. It writes an immutable build plan and an initially inactive/pending ledger;
it never invokes the training runtime.

`partner-pools run` processes training, validation, then evaluation by default.
Every active candidate is trained to the screening target and receives the
registered greedy 100-rollout self-play evaluation on fixed environment keys.
Competent screens continue from their full optimizer/environment/RNG state to
the finalist total target; they do not restart. Finalists are evaluated again.
When eligible finalists remain below quota, the next eight candidates activate
until quota or cap is reached.

The ledger is atomically published after every screening and finalist boundary.
Each job has dedicated streamed runtime and competence logs. Requests, results,
compact policies, and full states are content-hashed. Interrupted work resumes
from `latest.json`; a completed full state whose compact export is missing uses
a recovery-only export operation that performs no optimizer update. A workspace
lock prevents duplicate queue runners, and termination signals are forwarded to
active isolated-runtime processes.

`partner-pools freeze` re-verifies every finalist, competence key, threshold,
attainable transition target, plan/source/upstream hash, request/result hash,
and compact/full checkpoint hash. It also rejects partner-ID, reward-vector,
seed, normalized-parameter, or checkpoint-content leakage across splits.
Training and validation retain the first 24 and 8 competent finalists in
candidate order. Evaluation retains every competent finalist from all processed
batches after at least 32 qualify.

Operational manifests are immutable and remain under the ignored workspace:

```text
frozen/train-pool.json
frozen/validation-pool.json
frozen/evaluation-candidates.json
frozen/frozen-pool-bundle.json
frozen/leakage-audit.json
```

The same directory contains `publication-summary.json`, which omits local
machine paths and can be promoted into the presentation package after execution.

Response clustering and stopping decisions use validation partners. Evaluation
partners are selected in two disjoint groups of eight through a mixed-integer
contract. The solver enforces group size, disjointness, identical response-cluster
counts, competence, best-fixed response, BR-Prox, predictability, trajectory
divergence, and commitment-rate discovery margins. BR-Div log determinant is
audited afterward without diagonal jitter.

The selected partner identifiers are frozen before confirmatory trajectories are
read. Confirmatory matching additionally requires episode-level commitment
outcomes and a nonsignificant two-sided Fisher exact test. Failure remains a
failed construction.
