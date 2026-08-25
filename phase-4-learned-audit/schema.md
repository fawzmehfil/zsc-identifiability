# Learning Audit Schema

`LearningAuditSuite` is separate from the Phase 2 game schema and Phase 3 matched
benchmark schema. It references the Phase 3 suite and declares partner profiles,
cells, matched comparisons, selected symmetry audits, methods, budgets, seeds, and
statistical settings. Generated mechanisms remain ordinary
`FiniteConventionGame` schema-v1 objects.

## Partner pools

Each `LearningGame` records a split, a profile ID, the source population, the v1
game, commitment states, and a normalized dynamics hash. The hash excludes names
and metadata and includes priors, action/state/observation spaces, kernels, losses,
and post-commitment observations. Exact hash reuse across train, validation, and
test is rejected.

The canonical test mechanism is Phase 3's exact `q=4/5, p=1/2` population. The
third training nuisance probability is `1/3`, rather than `1/2`: using `1/2`
would exactly duplicate the canonical test kernel in cells whose observations are
entirely nuisance-driven, violating the preregistered leakage rule.

## Runtime tensors

An observation concatenates:

- public-state one-hot encoding;
- latest visible response, including a start token;
- previous ego action, including a start token;
- previous normalized external reward;
- elapsed and remaining horizon fractions;
- the legal-action mask.

Commitments are masked actions of the form `commit:<decision>`. No hidden partner
or benchmark metadata enters the tensor.

## Checkpoints and results

Training checkpoints include model and optimizer states, environment generator
state, Python/NumPy/PyTorch RNG states, architecture metadata, pool hashes, and
the configuration hash. Large checkpoints and raw rollouts are intentionally
gitignored.

`LearnedPolicyEvaluation` separates intervention cost, residual Bayes risk, and
failure to use available evidence. It also records policy DRI, evidence mutual
information, probe and commitment behavior, frontier distance, calibration
diagnostics, partner-response reconstruction loss where a baseline exposes a
predictor, and applicability flags.
