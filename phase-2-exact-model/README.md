# Phase 2: Exact Formal Model

## Result

This package instantiates the Phase 1 definitions as finite, static-mode convention
games and solves them without sampling or learned policies. It provides two numeric
backends, exact belief-state control, policy-induced evidence metrics, pure and
convexified cost-risk frontiers, executable theorem checks, canonical regression
games, and reproducible figures and tables.

The model is a deliberately narrow POMDP specialization. It is an oracle and
measurement package for the later ZSC audit, not a claim that belief-space planning
or Bayesian value of information is new.

## Model

A game is

\[
\mathcal G=(\Theta,b_0,X,A,A_{\mathrm{pass}},O,D,H,P,k,L,P_{\mathrm{post}}).
\]

The hidden partner mode is fixed for the episode. The public state is fully
observed. Before commitment, the ego may stop and choose a decision or take an
ordinary task action. At the horizon, commitment is forced. The decision-loss table
is path-independent so task-state changes cannot manufacture DRI.

At belief \(b\), terminal risk is

\[
R(b)=\min_d\sum_\theta b(\theta)L(\theta,d),
\]

and the task oracle minimizes expected intervention cost plus terminal risk. The
information oracle minimizes terminal risk first and cost second. Passive control
restricts actions to those marked `passive`; task and reconnaissance control use all
ordinary task actions. Reconnaissance is reported separately as an information-only
upper condition even when its feasible action set matches task-active control.

## Package map

- `games/`: ten schema-versioned canonical JSON games.
- `suites/canonical.json`: complete suite and declared parameter grids.
- `theory/theorems.md`: assumptions and direct proofs.
- `artifacts/`: generated policies, histories, frontiers, theorem checks, tables,
  figures, and manifest.
- `../src/zsc_identifiability/`: schema, belief model, solvers, metrics, frontiers,
  theory checks, public API, and CLI.
- `../tests/`: schema, regression, exhaustive-oracle, theorem, and property tests.

## Commands

```bash
uv sync --dev
uv run python -m zsc_identifiability validate \
  --game phase-2-exact-model/games/active-separable.json
uv run python -m zsc_identifiability solve \
  --game phase-2-exact-model/games/active-separable.json --class task
uv run python -m zsc_identifiability frontier \
  --game phase-2-exact-model/games/active-separable.json --class task
uv run python -m zsc_identifiability verify-theory
uv run python -m zsc_identifiability run-suite \
  --suite phase-2-exact-model/suites/canonical.json \
  --output phase-2-exact-model/artifacts
```

## Reported oracles

- best fixed response with immediate commitment;
- passive net-regret oracle;
- task-active net-regret oracle;
- task-active information oracle;
- reconnaissance information oracle;
- every feasible fixed first task action followed by Bayes commitment;
- known-mode and known-response-signature scalar upper controls.

The fixed-action controls include random-or-fixed-probe comparisons without adding a
privileged query channel. Every canonical action is schema-validated as an ordinary
task action.

## Numerical contract

Canonical probabilities, costs, and losses are rational strings. The fraction
backend preserves exact arithmetic through posteriors, policy evaluation, theorem
checks, and frontier pruning. The float64 backend is used for numerical cross-checks
and plotting. Suite execution fails if canonical scalar metrics disagree by more
than \(10^{-10}\).

The pure frontier lists implementable deterministic conditional policies. The
convexified envelope lists its lower-hull vertices; line segments between adjacent
vertices represent randomization once at episode start. These randomized mixtures
are not silently substituted for deterministic policies.

## Phase boundary

This phase contains no RL training, neural partner model, adaptive partner,
matched-population generator, Overcooked integration, or ZSC-Eval integration.
Those remain downstream work.

