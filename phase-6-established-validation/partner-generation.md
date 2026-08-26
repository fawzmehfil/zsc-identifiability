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

Each reward vector has two independent partner seeds. Candidate screening and
finalist budgets, competence thresholds, quotas, expansion blocks, and the
64-candidate cap are fixed in `suites/canonical.json`. A failed quota is reported;
competence or matching thresholds cannot be relaxed.

Every executed partner job is followed by the registered 100-rollout greedy
checkpoint-pair evaluation on common environment keys. The runner emits a
checkpoint index with correct-delivery episode rate, competence status, source
reward-vector hash, seed, transition budget, and normalized checkpoint hash.

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
