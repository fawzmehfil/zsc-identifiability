# Statistical Claim and Limitation Audit

## Claim status

| Proposed statement | Status | Evidence or correction |
|---|---|---|
| Pre-commitment DRI adds held-out predictive value beyond the registered controls. | Supported | Overall ΔR² is positive and ΔMAE/ΔMSE are negative for the primary GRU estimator and event sensitivity. |
| The predictive direction reproduces across layouts. | Supported | Both layouts have positive ΔR² and negative ΔMAE/ΔMSE under both representations. |
| Higher DRI is associated with lower response-library regret. | Supported as an adjusted association | GRU coefficient -0.180, clustered 95% interval [-0.295, -0.079]; event coefficient -0.158, interval [-0.284, -0.055]. |
| Passive histories contain decision-useful partner evidence. | Supported | Holm-adjusted permutation p = 0.0396 in each layout; calibration, Brier, prior-fallback, and leakage gates pass. |
| DRI causes better coordination. | Not supported | The established analysis is predictive and cross-fitted, not a causal identification design. |
| The selected ordinary task interventions raise DRI. | Not supported | Both corrected intervals cross zero and both Holm-adjusted permutation p-values are 0.9901. |
| Existing methods fail because they do not probe. | Not supported | The systematic-gap gate is false; Stage 4 methods often reach the exact active oracle. |
| DRI is new mathematics. | Rejected | It is a normalized reduction in Bayes decision risk. Novelty lies in the ZSC measurement protocol and matched audit. |
| DRI replaces BR-Div, predictability, or prefix-TV. | Rejected | These are complementary controls; the claim is incremental value, not replacement. |

## Registered model and uncertainty

The outcome is normalized response-library regret. The baseline contains
partner competence, prior confusion risk, conflict coefficient, Rahman-style
BRDiv, raw ZSC-Eval BR-Div, visible-action predictability, prefix-TV, method
effects, and layout. BR-Prox is excluded as a predictor because it is derived
from the same performance quantity.

The comparison uses nested leave-one-HSP-scheme-out ridge regression. Features
are standardized inside training folds, and ridge strength is selected in
inner scheme-level folds. The DRI coefficient interval uses 10,000 clustered
resamples over method seed, HSP scheme, and episode key. Registered
permutation tests use 100 deterministic label permutations and Holm correction
over the two passive and two intervention hypotheses.

## Heterogeneity that must remain visible

The aggregate effect is not universal across scheme folds:

| Representation | Scope | MAE-improved folds | MSE-improved folds | Positive-ΔR² folds |
|---|---:|---:|---:|---:|
| GRU | Overall | 19/25 | 19/25 | 19/25 |
| GRU | `random3_m` | 8/15 | 11/15 | 11/15 |
| GRU | `small_corridor` | 9/10 | 8/10 | 8/10 |
| Event | Overall | 18/25 | 16/25 | 16/25 |
| Event | `random3_m` | 8/15 | 8/15 | 8/15 |
| Event | `small_corridor` | 9/10 | 9/10 | 9/10 |

Paper text must say “improves aggregate held-out prediction in both layouts,”
not “improves every held-out partner family.” All negative folds remain in the
frozen report.

## Measurement limitations

- Established-environment DRI is a cross-fitted decision-value lower bound
  derived from a finite empirical response library, not exact Bayes-optimal
  environmental DRI.
- The response library uses official co-trained counterparts. It may omit a
  better response policy and therefore changes the estimated loss geometry.
- The primary GRU learns an identity representation on calibration data before
  direct pairwise decision heads are fitted. Direct binary refits support
  representation sufficiency on ten frozen pairs, not every pair.
- `random3_m` and `small_corridor` are two layouts from one benchmark family.
  The result does not establish domain-general validity or human-team validity.
- Mid and final checkpoints from one HSP scheme are dependent. Scheme-level
  splitting and clustered uncertainty reduce, but do not erase, this issue.
- The overall predictive improvement is modest (GRU ΔR² approximately 0.020;
  event ΔR² approximately 0.012), even though it is directionally robust under
  the registered tests.
- Stage 6 reuses official-method outcome data from v2. The v3 estimators and
  confirmation histories are fresh; the coordination-policy outcome matrix is
  not.

## Design limitations

- The commitment point—first ingredient placed in a pot—is natural and
  irreversible, but it is only one decomposition of the longer task.
- No-commitment episodes use prior residual risk. Alternative censoring choices
  should remain sensitivity analyses.
- The intervention library is deliberately restricted to legal scripted task
  options. Failure of those options does not prove that no useful natural
  intervention exists.
- `corridor_yield` had a negative measured task cost, meaning it slightly
  improved task return on average. This does not rescue its negative
  decision-risk effect.
- Stage 6 v3 followed a failed v2 estimator audit. The redesign was frozen
  before the fresh confirmation set, but the sequential history must be
  disclosed.

## Submission-safe wording

> Across two official ZSC-Eval layouts, adding cross-fitted pre-commitment DRI
> to competence, response-diversity, predictability, divergence, method, and
> layout controls modestly improved held-out prediction of response-library
> regret. The association was negative under both GRU and event
> representations. Preregistered natural interventions did not confirm, so our
> established-environment conclusion is measurement-only.

Avoid “causal,” “first,” “solves,” “proves in realistic environments,” and
“active probing works” unless a separate analysis directly supports the term.
