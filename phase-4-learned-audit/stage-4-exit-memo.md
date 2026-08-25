# Stage 4 Exit Memo

## Current verdict

**Implementation pass; scientific verdict pending.**

The learning suite, disjoint partner-pool generator, vector environment, masked
recurrent PPO implementation, style baselines, exact neural-policy evaluator,
regret decomposition, paired statistics, leakage checks, CLI, tests, and reporting
pipeline are present. Phase 2 and Phase 3 regression gates continue to pass.
The deterministic `q=1`, zero-cost active smoke game reaches exact greedy return
`100`, exceeding the preregistered `98` implementation threshold.

The complete capability-level smoke matrix also passes. All common-response
controls return `100`; the memoryless policy remains at `80` in the delayed
evidence control while every recurrent comparator returns `100`; and
ToM-selector-style returns `100` when partner evidence is freely available. Its
preserved pre-correction `active_only` checkpoint returns `80`, commits
immediately, and acquires zero DRI. That checkpoint is retained as a diagnostic
of the original training/deployment mismatch and is not treated as a scientific
finding. Exact details are recorded in `artifacts/smoke-matrix-audit.json`.

The confirmatory training matrix has not been executed. Consequently this memo
does not claim a probing failure, ranking reversal, or justification for a new
repair method.

## Decision rule after training

- `continue_to_repair` only if passive and memory controls pass and a robust active
  gap survives the prespecified optimization rescue.
- `continue_without_repair` if existing methods approach the active oracle while
  the matched audit still changes evaluation conclusions.
- `redesign` if capacity, optimization, split validity, or adaptation fidelity
  explains the apparent failure.
- `stop` if the learned audit removes the scientific premise.

Stage 5 is not authorized by the current evidence.
