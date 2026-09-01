# Stage 6 v3 Protocol Amendment: Direct Decision-Risk Measurement

This amendment was frozen before any Stage 6 v3 confirmation episode was
generated. Stage 6 v2 completed 240,800 inference episodes, but its
identity-oriented estimator failed calibration. Its apparent regression and
intervention effects are exploratory only. The immutable failure record is
`v2-failed-audit-summary.json`; its exact suite is
`suites/official-checkpoint-v2.json`.

## Measurement change

The primary measurement is now a cross-fitted decision-value lower bound. A
64-unit GRU learns an ego-visible history representation by predicting training
partner identity on v2 calibration traces only. Direct binary ridge-logistic
heads are then fit for every response-conflicting partner pair. Their posterior
probabilities are combined with the empirical response-loss matrix, and the
response with the lowest predicted expected loss is evaluated on held-out data.
No multiclass posterior is renormalized to manufacture pairwise probabilities.

The existing v2 calibration split is divided deterministically into 75% encoder
training and 25% early stopping. The registered v2 validation split selects one
global ridge, temperature, and prior-shrinkage configuration for each layout,
evidence policy, prefix, and representation. Neither v2 confirmatory traces nor
fresh v3 traces may be used for fitting or selection. Confirmatory DRI is not
clipped and may be negative.

A separately implemented 512-dimensional signed-hash event representation is
the sensitivity estimator. It retains four absolute time bins plus observed
length, cumulative reward, and partner-visibility rate. Agreement of numerical
point estimates is not required, but a paper claim requires consistent effect
direction.

Full-population identity posteriors remain diagnostic only. Decision
information is computed from best-response signatures rather than partner IDs.
Ten hash-selected partner pairs per layout receive from-scratch binary-GRU
refits as a representation-sufficiency diagnostic.

## Calibration and false-positive controls

The estimator must recover the registered informative, identity-only,
late-reveal, inseparable, asymmetric-loss, and censored synthetic controls. A
uniform pair posterior is the fallback for missing pre-commitment evidence.

The v2 absolute shuffled-DRI check is retired. The v3 test uses 100 deterministic
label permutations and the one-sided statistic

\[
p = \frac{1 + \#\{\mathrm{null} \ge \mathrm{observed}\}}{101}.
\]

Negative shuffled DRI is allowed because random decisions can be harmful. Holm
correction covers passive DRI and the frozen intervention contrast in both
layouts.

## Untouched confirmation data

The frozen confirmation salt is `zsc-stage6-v3-confirmatory-9d41`. The plan
contains only trace rollouts:

- `random3_m`: 30 partners, four evidence policies, 64 episodes each;
- `small_corridor`: 20 partners, three evidence policies, 32 episodes each.

This is exactly 9,600 inference-only episodes and at most 3.84 million
environment steps. Keys are paired across evidence policies, balanced across
player seats, and disjoint from every v2 key. The selected interventions remain
`temporary_role_takeover` and `corridor_yield`; fresh results cannot change
them. Response-library and official-method outcomes are reused without reruns.

## Confirmatory decisions

The unchanged scheme-held-out regression uses fresh GRU DRI at
`ordinary_progress/pre_commitment`. A positive regression claim requires
positive overall delta R-squared, negative delta MAE and MSE overall and in each
layout, a negative clustered DRI-coefficient interval, and directionally
consistent event sensitivity.

An intervention claim additionally requires at least 0.8 completion before
commitment, changed partner-response behavior, a positive corrected GRU
decision-risk interval, Holm-adjusted permutation p below 0.05, benefit greater
than task cost, and nonnegative event direction.

The implementation and this amendment must be committed before fresh inference
starts. Stage 6 v3 performs no reinforcement learning and cannot request partner
or policy training.
