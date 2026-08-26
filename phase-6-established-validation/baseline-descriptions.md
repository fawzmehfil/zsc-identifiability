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

The isolated runtime currently executes the pinned official recurrent IPPO,
FCP, and Other-Play training paths. TBS-style, PACE auxiliary, PACE-style, and
CSP-style remain registered comparison protocols, but their method-specific
OvercookedV2 ports are not silently replaced with ordinary IPPO. An attempted
execution fails explicitly until the corresponding port and its required
training-pool assets exist. Therefore the core environment/measurement platform
is verified, while the complete learned-method matrix is still an open
engineering gate before confirmatory execution.
