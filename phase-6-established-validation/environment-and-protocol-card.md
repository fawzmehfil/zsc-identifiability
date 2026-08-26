# Environment and Protocol Card

## Scientific unit

A partner mode is one frozen, independently trained partner checkpoint. A work
unit begins at episode reset or after a delivery. The first work unit is the
primary single-encounter unit; later work units measure within-episode
adaptation separately.

## Observation boundary

The evaluated ego receives its official radius-2 observation, its own action,
the partner action only while the partner is visible, shared reward, and time.
Partner identifiers, reward vectors, response clusters, recipes outside the
official observation, DRI, and matching metadata are excluded.

Global simulator state is permitted only for measurement: detecting a physical
pot-content change, detecting delivery, verifying task-option legality, and
constructing matched evaluation keys. It is never added to a learned policy's
input.

## Commitment

Commitment is the first successful ingredient placement into any pot during a
work unit. Pot contents cannot be removed. The pre-commitment history ends before
that transition. Eventual history ends with the first delivery result. Episodes
with no commitment are retained as censored failures and receive prior residual
risk at the pre-commitment endpoint.

## Response loss

For frozen partner `theta` and response-library policy `d`, loss is the gap from
the best response-library value for that partner. Responses within 0.02
normalized return of the library maximum are adequate. Two partners conflict
when their adequate-response sets are disjoint.

This is an approximate known-partner reference. It does not establish global
optimality.

## Held-out DRI scoring

The event and GRU posterior models fit on calibration traces and calibrate on a
separate validation split. On confirmatory traces, each posterior selects the
response-library policy with the lowest posterior expected loss. Residual risk
is then the actual frozen response loss for the held-out partner mode, not the
posterior model's self-reported uncertainty. Confirmatory mode labels are used
only for this offline score; they never enter model fitting, calibration, or a
deployed policy. Consequently, a confidently wrong or label-shuffled posterior
cannot create spurious positive DRI.

Broad behavioral predictability is reported separately as held-out visible
partner-action cross-entropy. The discrete action oracle uses only ego-visible
history and is labelled LoBP-style; it does not claim to reproduce LoBP's
trained Theory-of-Mind observer or intention labels.

## Evidence policies

The restricted option audit includes ordinary progress, staging a candidate
ingredient, temporarily beginning a contested role, yielding in a shared
passage, and activating the recipe button. A deterministic replanning controller
may control the ego for at most 16 legal low-level actions before returning
control to the response policy.

An option qualifies as a teammate diagnostic only if it finishes before
commitment, separates response-conflicting partners, improves DRI beyond passive
progress, has measurable cost, cannot be explained only by recipe information,
and remains necessary after the best universal response is considered.

The recipe button is the explicit negative control. Its native cost and recipe
information may affect task return, but it cannot qualify as partner evidence.

## Frontier scope

The active-identifiability frontier is restricted to audited task options. It
contains nondominated deterministic points and episode-level randomized mixtures
between adjacent hull points. It is empirical and is never described as an exact
Bayes frontier.
