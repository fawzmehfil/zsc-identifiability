# Established-Environment Baselines

The central single-encounter comparison registers recurrent IPPO, FCP,
Other-Play, TBS-style specialist selection, PACE auxiliary prediction, and
PACE-style peer-identification exploration. CSP-style receives a separate
same-partner reconnaissance episode and is excluded from central rankings.

Official OvercookedV2 recurrent convolutional policies are the common task-policy
backbone where a method does not require a distinct structure. FCP and Other-Play
use their official OvercookedV2 mechanisms. TBS-style ports response clustering,
specialist policies, and visible-history selection to the Stage 6 partner pools.
PACE variants share architecture; only PACE-style receives a training-only peer
identification reward. Test policies receive no partner labels and perform no
gradient updates.

PACE uses the official convolutional encoder and a 128-dimensional
encounter-local GRU. `pace_aux` adds prefix identity classification only.
`pace_style` uses the same architecture and adds a decaying training-only peer
reward based on the post-interaction history. Both reset context at every
unfamiliar-partner encounter, so neither is presented as an exact reproduction
of multi-episode PACE.

TBS computes the pinned ToMZSC compatibility matrix and self-tuning spectral
clusters from training-only cross-play. It trains one recurrent specialist per
cluster plus global and cluster-specific visible-event predictors. Deployment
advances every specialist state, accumulates Bernoulli KL divergence, starts
from a seeded random specialist before evidence exists, and routes without
labels or gradients.

CSP trains a recurrent probe and a 32-dimensional visible-history encoder with
a seven-class next-partner-action decoder. Its warm-up data are balanced across
passive, random-valid, and ordinary task-active behavior. After one complete
reconnaissance episode, the environment and recurrent task state reset, the
same partner remains fixed, and only the frozen embedding selects the scored
specialist. Reconnaissance return, failures, and extra interactions are always
reported separately.

The `-style` suffix is mandatory for PACE, TBS, and CSP unless their original
runtime and protocol are exactly reproduced. CSP reconnaissance reports the
extra interaction count and combined cost separately.

Hyperparameters are selected once per method across validation layouts. External
task return is primary, then diagnostic cost, then model size. Test return cannot
select a configuration or checkpoint.

Stage 6 does not add a decision-loss-weighted repair. Such a method becomes
eligible only if a natural active-information gap survives two layouts, ten
confirmatory seeds, optimization controls, matching, and leakage checks.

## Runtime status

The isolated runtime executes pinned recurrent IPPO, FCP, Other-Play, PACE
auxiliary, PACE-style, TBS-style, and CSP-style. Ported methods emit
discriminated deployment artifacts rather than masquerading as ordinary IPPO.
Registered smokes cover PACE classification and peer reward, pinned TBS
clustering and routing, CSP trajectory clustering and two-episode selection,
artifact reload, and metadata-leakage controls. The remaining work is
experimental execution on the frozen Stage 6 partner populations, not a missing
baseline implementation.
