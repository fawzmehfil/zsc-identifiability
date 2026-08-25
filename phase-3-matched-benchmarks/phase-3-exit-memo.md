# Phase 3 Exit Memo

## Verdict: Continue

The matched benchmark axis survived the exact construction audit. Pre-commitment
decision-sufficient identifiability changes under controls that preserve partner
competence, task structure, best-response diversity, broad predictability, and the
declared divergence profile.

## Claims that survived

- **Matched complementarity:** Binary conflicting-response populations retain
  known-mode return `100`, best fixed return `80`, prior risk `20`, Rahman return
  BRDiv `40`, raw ZSC-Eval BR-Div `1`, and normalized ZSC-Eval BR-Div
  `1000000000000/1004006004001` while passive DRI changes by `3/5`.
- **Active attainability:** `active_only` and `precommit_inseparable` have exactly
  equal passive history distributions. Their task-active DRI values are `3/5` and
  `0`.
- **Timing:** Full-episode LoBP-style predictability is exactly matched at
  `-log(2)` nats per target while useful evidence is moved across the commitment
  boundary.
- **Decision relevance:** Response and subtype signals both provide
  `0.2780719051126377` bits of identity information and the same pairwise prefix-TV
  multiset. Only the response signal provides decision-signature information and
  DRI `3/5`.
- **Memory:** In `remember_response`, the fixed, evidence-blind, and memoryless
  risks are `20`; the history-aware risk is `8`.
- **Cost sensitivity:** At reliability `4/5`, staging with cost `5` is selected and
  produces net regret `13`. At costs `12` and `15`, immediate commitment is
  selected.

## Verification result

All four exact matching contracts, shortcut audits, symmetry checks, Fraction versus
float comparisons, and sampled estimator calibrations passed. The current test
suite contains 69 passing tests, including property-based reliability checks and
all unchanged Phase 2 tests.

The machine-readable evidence is in:

- `artifacts/scientific-checks.json`;
- `artifacts/matching-audit.json`;
- `artifacts/shortcut-audit.json`;
- `artifacts/estimator-calibration.json`;
- `artifacts/manifest.json`.

## Constraint carried into Phase 4

Phase 4 should treat this benchmark as a diagnostic audit, not claim that active
probing or Bayes-risk reasoning is itself new. It must compare existing passive and
active baselines before introducing a repair. Results must remain separated by
passive DRI, active DRI, intervention cost, memory requirement, and evidence timing.
