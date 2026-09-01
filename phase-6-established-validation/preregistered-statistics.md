# Preregistered Statistical Analysis

The primary outcome is normalized response-library regret. BR-Prox is a
secondary outcome and is not a predictor of regret because it is derived from
the same performance quantity.

The baseline predictor contains competence, prior confusion risk, conflict
coefficient, Rahman return BRDiv, ZSC-Eval event-feature BR-Div, visible-action
predictability, pre-commitment prefix-TV, method fixed effects, and layout. The
full model adds fresh cross-fitted pre-commitment decision-value DRI and
method-by-DRI interactions. The primary estimator uses direct pairwise heads on
frozen GRU history representations. The signed-hash event estimator is an
independently implemented sensitivity.

Evaluation uses nested leave-one-HSP-scheme-out regression. Every test fold
holds out all partner pairs involving one scheme. Feature standardization and
ridge selection use only the training fold. Ridge candidates are `0`, `0.01`,
`0.1`, `1`, and `10`. Reports include held-out MAE, MSE, R-squared, their
changes, per-fold predictions, and every negative fold.

Uncertainty uses 10,000 resamples clustered over official method seed, HSP
scheme, and paired episode key. The false-positive audit uses 100 deterministic
label permutations and one-sided p-values; negative shuffled DRI is permitted.
Passive and frozen-intervention tests across both layouts use Holm correction.

DRI is reported at steps 0, 8, 16, and 32, immediately before commitment, and
after delivery. Phase 3 controls must be recovered within `0.03`. Event and GRU
point estimates need not agree numerically, but their claimed regression or
intervention effects cannot have opposite signs.

The primary regression requires positive overall delta R-squared, negative
delta MAE and MSE overall and within each layout, and a clustered DRI
coefficient interval below zero. The selected intervention additionally
requires at least 0.8 pre-commitment completion, changed partner behavior, a
corrected positive decision-risk interval, Holm-adjusted permutation p below
0.05, benefit above task cost, and nonnegative event direction.

Sensitivity analyses vary response adequacy, estimator family, policy
stochasticity, commitment definition, checkpoint stage, player seat,
competence filtering, identity MI, and prefix-TV substitution. Claim direction
must remain stable. Ranking reversals are secondary and may be zero.
