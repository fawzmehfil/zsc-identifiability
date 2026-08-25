# Preregistered Comparisons

Primary comparisons use ten paired confirmatory seeds and greedy exact policy-tree
evaluation. Stochastic policies are reported as a diagnostic. Hyperparameters and
checkpoint selection use validation partners only.

The `learn tune` command constructs the declared global grid for one method,
trains every candidate on the four selection cells with development seeds, and
writes a selection report. Confirmatory `learn train --selection REPORT` validates
the method ID and the report's `test_data_used=false` certificate before applying
the chosen configuration.

If a capacity sanity gate fails, `learn rescue --cell CELL` applies the same
hidden-size-128, learning-rate-`1e-4`, two-million-transition rescue to every
central comparator applicable to that cell. Rescue outputs remain separate from
the primary tables.

The principal matched contrasts are:

1. `passive_early` versus `active_only`: freely observed evidence versus evidence
   requiring an ordinary costly task action.
2. `active_only` versus `precommit_inseparable`: identical passive histories but
   different active separability.
3. `remember_response` versus `remember_subtype`: matched identity information and
   divergence profile, but different decision relevance.
4. `active_response` versus `active_identity_only`: active information that changes
   the correct response versus active information about irrelevant identity.
5. `active_only`, `active_boundary`, and `active_too_expensive`: probe value below,
   at, and above the exact cost boundary.

A strict ranking reversal requires opposite paired method differences in the two
cells and Holm-adjusted 95% intervals excluding zero in both. Null reversals remain
reportable results.

The implementation sanity gates are evaluated by capability class, not by
requiring every baseline to exhibit every capability. Every applicable core
method must clear the common-response control; at least one active-capability
method must clear the deterministic zero-cost active control; at least one
recurrent method must recover the memory-dependent improvement over the
memoryless control; and ToM-selector-style must clear a freely available passive
evidence control. Failures by passive selectors to intervene in `active_only` are
preserved as diagnostics rather than misclassified as infrastructure failures.
Scientific conclusions are blocked if the aggregate capability gates do not pass.
