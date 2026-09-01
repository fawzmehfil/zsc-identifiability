# Stage 6 Exit Memo

## Current verdict

**`redesign` after the completed v2 audit; v3 confirmation has not started.**

Stage 6 v2 completed 4,842 shards, 240,800 episodes, and 96.32 million
environment steps without policy training. Engineering and integrity checks
passed, but estimator calibration failed. The v2 audit therefore cannot support
paper claims. Its apparent DRI regression and intervention effects remain
exploratory and are recorded, with exact hashes, in
`v2-failed-audit-summary.json`.

## Completed v2 platform

- The canonical suite forbids partner and ZSC-policy training.
- Asset selection is locked to the complete official benchmark YAMLs.
- Minimal asset synchronization is pinned, hashed, duplicate-audited, and
  usable offline.
- All 50 partners retain their matching co-trained response counterpart.
- Six official methods retain five published seeds in both layouts.
- Rollouts are CPU-only, bounded to four workers, atomic, compressed, and
  resumable at partner-policy shard boundaries.
- Two-seat official-policy and environment parity passed in both layouts.
- Response conflict, pairwise event/GRU DRI, intervention traces, and nested
  leave-one-HSP-scheme-out regression completed.
- The retired custom partner run produced zero checkpoints; its files are
  preserved and cannot be reused by the official audit.

## Why v2 failed

- The maximum event/GRU intervention-effect disagreement was
  `0.37770475335536746`.
- The maximum restricted-posterior/direct-refit disagreement was
  `0.23671144859813092`.
- The shuffled-label gate used absolute DRI and incorrectly treated harmful
  negative random decisions as false positive evidence.

## Frozen v3 redesign

Stage 6 v3 replaces pairwise renormalization of a multiclass identity posterior
with direct pairwise decision decoders. A five-seed GRU representation is
primary; a temporally hashed event decoder is an independent sensitivity. The
new one-sided permutation test allows negative null DRI.

All representation learning and decoder selection use only v2 calibration and
validation data. A fresh, disjoint 9,600-episode confirmation set is reserved
for the final decision-value, regression, and intervention tests. Existing
official checkpoints, response matrices, and method outcomes are reused. No RL
training or policy update is allowed.

The full frozen contract is in `protocol-amendment-v3.md` and
`suites/official-measurement-v3.json`. The next operational steps are to fit and
freeze measurement representations and decoders, prepare the trace-only
confirmation plan, collect the fresh episodes, and run the registered analysis.

The top-paper framing remains unauthorized until calibrated fresh DRI improves
scheme-held-out prediction in both layouts and the registered robustness gates
pass.

Possible final verdicts are `continue_top_paper_package`,
`complete_evaluation_only`, `complete_measurement_only`, `redesign`, and `stop`.
Negative results are retained rather than converted into an unregistered method
claim.
