# Preregistered Statistical Analysis

The primary outcome is normalized response-library regret. BR-Prox is a
secondary outcome and is not a predictor of regret because it is derived from
the same performance quantity.

The baseline predictor contains competence, prior confusion risk, conflict
coefficient, Rahman return BRDiv, ZSC-Eval event-feature BR-Div, visible-action
predictability, pre-commitment prefix-TV, method fixed effects, and layout. The
full model adds pre-commitment DRI and method-by-DRI interactions.

Evaluation uses nested leave-one-HSP-scheme-out regression. Every test fold
holds out all partner pairs involving one scheme. Feature standardization and
ridge selection use only the training fold. Ridge candidates are `0`, `0.01`,
`0.1`, `1`, and `10`. Reports include held-out MAE, MSE, R-squared, their
changes, per-fold predictions, and every negative fold.

Uncertainty uses 10,000 resamples clustered over official method seed, HSP
scheme, and paired episode key. Registered method and intervention comparisons
use Holm correction.

DRI is reported at steps 0, 8, 16, and 32, immediately before commitment, and
after delivery. Event and five-seed GRU estimates must agree within `0.05` on
the treatment effect; Phase 3 controls must be recovered within `0.03`.

Sensitivity analyses vary response adequacy, estimator family, policy
stochasticity, commitment definition, checkpoint stage, player seat,
competence filtering, identity MI, and prefix-TV substitution. Claim direction
must remain stable. Ranking reversals are secondary and may be zero.
