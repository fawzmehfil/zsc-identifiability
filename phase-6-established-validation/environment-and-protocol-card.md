# Environment and Protocol Card

## Population and layouts

A partner mode is one official HSP `w0` benchmark checkpoint. Its response
counterpart is the same HSP scheme and training stage's `w1` checkpoint.
`random3_m` is the primary multi-recipe task; `small_corridor` is the geometry
robustness task. Mid/final checkpoints share one HSP scheme identifier.

## Observation and commitment

Measurement receives the official ego policy observation, ego actions, rewards,
time, partner actions uniquely inferable from observed position, orientation or
held-object changes, and ego events inferable from the observed transition.
Ambiguous blocked moves are recorded as unknown. Checkpoint names, HSP weights,
partner IDs, and global hidden events are excluded from estimator features.

The commitment point is the first successful placement of an ingredient into a
pot. Pre-commitment history ends before that transition. Eventual history ends
after first-delivery feedback. No-commitment episodes retain prior residual risk
at the pre-commitment endpoint.

## Response loss

For partner `theta` and response `d`, the empirical loss is:

```text
1 - V(theta, d) / max_d' V(theta, d')
```

Response adequacy is evaluated at margins `0.01`, `0.02`, and `0.05`. Partners
conflict when they share no adequate response. The response-library maximum is
not described as globally optimal.

## Decision-risk measurement

Stage 6 v3 measures whether the visible history supports the response decision,
not whether it identifies a checkpoint. The primary 64-unit GRU learns a frozen
history representation from v2 calibration traces using partner identity only
as a representation objective. A direct binary ridge-logistic head is then fit
for every response-conflicting pair and its posterior is combined with the
pair's full empirical response-loss rows. A registered prior-shrinkage choice
can return unreliable histories to the uniform pair prior.

The independent sensitivity maps visible events into a 512-dimensional signed
hash with absolute temporal bins. Identity mutual information remains a
diagnostic; decision information groups partners by their best-response
signature. No-commitment histories are explicitly assigned prior risk.

## Evidence and interventions

FCP seed 1 greedy is fixed as the passive evidence policy; the evaluated
partner remains stochastic. `random3_m` audits
ordinary progress, onion staging, tomato staging, and temporary role takeover.
`small_corridor` audits ordinary progress, onion staging, and one-step corridor
yielding. Controllers use legal official low-level actions for at most 16 steps
and continue advancing the FCP recurrent state on the observed history.

The v3 options selected before fresh confirmation are temporary role takeover
in `random3_m` and corridor yielding in `small_corridor`; fresh results cannot
reselect them. An option qualifies only if it completes before commitment,
changes the response distribution of conflicting partners, increases calibrated
decision value under the corrected permutation test, has measurable cost, and
retains decision value after the best fixed response is considered.
The restricted empirical frontier is never called an exact Bayes frontier.

## Official method evaluation

FCP, MEP, TrajeDi, HSP, COLE, and E3T use official stochastic action sampling,
both player seats, and paired environment keys. Reports separate sparse return,
response-library regret, commitment timing, method-induced DRI, diagnostic
patterns, identity information, and response-signature information. Every
checkpoint is also evaluated greedily under the same keys as a deployment
sensitivity. Only the ego deployment changes in that sensitivity: partner
sampling remains stochastic. Stochastic ego evaluation remains primary.
