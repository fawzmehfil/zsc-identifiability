# Preregistered Statistical Analysis

The primary outcome is held-out explanatory and predictive value, not a forced
ranking reversal.

The registered regression predicts regret from competence, BR-Div, BR-Prox,
behavioral predictability, trajectory divergence, DRI, method, and method-by-DRI
interaction. Leave-one-reward-vector-out evaluation reports baseline and full
model mean squared error, held-out R-squared, their changes, and the DRI
coefficient.

Uncertainty resamples training seed, partner, and episode hierarchically with
10,000 bootstrap draws. Confirmatory method contrasts are paired by common
environment and recipe keys. Preregistered multi-method p-values use Holm
correction.

DRI is reported at steps 0, 8, 16, and 32, immediately before commitment, and
after the first delivery. The GRU and high-level-event estimators must agree
within 0.05 on the matched DRI treatment effect. Synthetic calibration must
recover the exact Phase 3 controls within 0.03.

Strict ranking reversal remains secondary. It requires opposite paired effects
in two matched populations and Holm-adjusted confidence intervals excluding zero
in both. Zero reversals are a valid outcome.
