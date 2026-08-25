# Benchmark Card

## Intended scientific use

This benchmark audits whether a coordination method can acquire and retain the
partner evidence needed for the correct response before commitment. It is designed
for controlled comparison of fixed responses, passive adaptation, active task
interventions, information-only policies, and history-dependent policies.

It is not an environment-complexity benchmark and not evidence that a method will
work with humans, learned adaptive partners, or high-dimensional observations.

## Controlled factors

Within declared matching contracts, populations keep fixed the task, partner
competence, priors, loss matrix, action and observation spaces, commitment rule,
known-mode return, best-response descriptors, passive reference behavior, and
observation budget. Contracts additionally match the appropriate BR-diversity,
LoBP-style predictability, and divergence statistics.

The intended treatments are evidence timing, passive versus active availability,
whether a signal concerns the required response or irrelevant identity, memory
gap, signal reliability, and intervention cost.

## Oracles and metrics

All canonical claims are exact. The package reports fixed, passive, task-active,
information-only, known-mode, evidence-blind, memoryless, and history-aware
solutions. DRI is based on reduction in Bayes confusion loss; it is unavailable
when prior risk is zero.

Identity mutual information, decision-signature mutual information, pairwise
prefix total variation, Rahman return BRDiv, two ZSC-Eval determinant variants,
BR-Prox, LoBP-style predictability, intervention cost, and net regret are reported
separately.

## Shortcut resistance

The audit collapses observations to test evidence-blind solutions, enumerates
latest-observation policies to test memorylessness, removes post-commitment kernels
to detect timing leakage, checks that public states do not encode old signals,
rejects hidden-mode identifiers in runtime symbols, rules out universal responses
in conflicting populations, and enforces commit-first tie-breaking for valueless
probes.

## Statistical calibration

Exact results are authoritative. Rollout estimators use paired common random
numbers, NumPy `PCG64`, 10,000 episodes per mode and audited reference policy, and
2,000 paired percentile-bootstrap resamples. Equivalence passes only when the
complete 95% interval lies inside the declared margin. Sampled passive DRI must
also lie within `0.02` of its exact value.

## Known limitations

- Partners are scripted, static, finite, and selected from a known hypothesis set.
- The observations are symbolic and low-dimensional.
- Commitment boundaries and loss tables are declared by construction.
- The LoBP score is an exact action-prediction analogue, not trained Theory of Mind.
- The benchmark diagnoses a capability; it does not provide a learned repair.
- Gridworld, OvercookedV2, ZSC-Eval partner populations, learned agents, and mutual
  adaptation are outside this phase.

These limits are deliberate: Phase 3 establishes that the proposed evaluation axis
can be isolated before Phase 4 asks whether learned coordination methods fail on it.
