# Exact-Model Theory

## Scope and status

These statements specialize established Bayesian decision theory and controlled
hypothesis testing to a static hidden partner mode, an explicit pre-commitment
evidence boundary, and a ZSC confusion-loss table. The general mathematics is not
claimed as new. The research contribution is the commitment-timed operationalization
and the matched evaluation built on it.

Let \(P_i^\rho(h)\) be the probability of pre-commitment history \(h\) under mode
\(i\) and evidence policy \(\rho\). Intervention cost is reported separately from
Bayes decision risk.

## Theorem 1: binary weighted-overlap identity

Assume two modes with priors \(p\) and \(1-p\), two conflicting commitment
decisions, and mismatch losses \(M_0\) and \(M_1\). Then

\[
R(\rho)=\sum_h\min\{pM_0P_0^\rho(h),(1-p)M_1P_1^\rho(h)\}.
\]

**Proof.** At a fixed history, choosing the decision for mode 0 incurs unnormalized
posterior loss \((1-p)M_1P_1^\rho(h)\); choosing the decision for mode 1 incurs
\(pM_0P_0^\rho(h)\). The Bayes decision takes the smaller term. Summing the minimum
over mutually exclusive histories gives the expected residual loss. ∎

## Corollary 1: total-variation form

With equal priors and symmetric mismatch loss \(M\), use
\(\min(a,b)=(a+b-|a-b|)/2\) and the normalization of both history distributions:

\[
R(\rho)=\frac{M}{2}\left(1-\operatorname{TV}(P_0^\rho,P_1^\rho)\right).
\]

Thus identical pre-commitment history distributions force risk \(M/2\). Within a
restricted policy class, minimizing residual risk is equivalent to maximizing
attainable pre-commitment total variation.

## Corollary 2: cost-budget frontier

For budget \(c\), define

\[
F(c)=\sup_{\rho:K(\rho)\le c}\operatorname{TV}(P_0^\rho,P_1^\rho).
\]

The feasible set can only expand when \(c\) increases, so \(F\) is non-decreasing.
This information frontier does not decide whether probing is useful: deployment
value is evaluated by minimizing \(K(\rho)+R(\rho)\).

## Theorem 2: multi-type pairwise lower bound

For any response-conflicting pair \(i,j\), let

\[
\kappa_{ij}=\min_d[L(i,d)+L(j,d)].
\]

For every history and decision, if \(\alpha=b_iP_i^\rho(h)\) and
\(\beta=b_jP_j^\rho(h)\), then

\[
\alpha L(i,d)+\beta L(j,d)
\ge \min(\alpha,\beta)[L(i,d)+L(j,d)]
\ge \min(\alpha,\beta)\kappa_{ij}.
\]

Dropping all other nonnegative mode losses and summing histories yields

\[
R(\rho)\ge \kappa_{ij}\sum_h\min\{b_iP_i^\rho(h),b_jP_j^\rho(h)\}.
\]

The valid aggregate bound is the maximum over pairs. Pairwise bounds must not be
summed because that can count the same decision loss multiple times.

## Corollary 3: observational-equivalence impossibility

If a response-conflicting pair has positive prior mass and identical history
distributions for every policy in a declared pre-commitment class, its overlap is
positive and Theorem 2 gives strictly positive Bayes regret for every policy in that
class. Evidence arriving only after commitment cannot change this bound.

## Theorem 3: one-intervention threshold

Assume equal binary modes, symmetric loss \(M\) on each of \(N\) remaining
coordination decisions, and a diagnostic response that identifies the correct mode
with probability \(q\ge 1/2\). Immediate guessing has risk \(NM/2\). Probing has
cost-plus-risk \(c+NM(1-q)\). Probing is strictly better exactly when

\[
c<NM(q-\tfrac12).
\]

At equality, the declared tie-break chooses immediate commitment because it has
lower intervention cost and earlier commitment.

## Metric properties

For \(R_0>0\), usable evidence cannot increase Bayes risk because the decision maker
can ignore it. Hence \(0\le\mathrm{DRI}\le1\). Expanding the policy class cannot
worsen the optimum because every old policy remains feasible. The attainable
frontier is non-decreasing in its cost budget for the same set-inclusion reason.

When \(R_0=0\), DRI is undefined: the game already admits a prior-optimal zero-loss
commitment. The implementation returns `dri = null`,
`identification_required = false`, and `decision_sufficient = true`.

