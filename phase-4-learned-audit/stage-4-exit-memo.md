# Stage 4 Exit Memo

## Verdict

**Complete: continue without a repair method.**

The confirmatory audit supports the paper's measurement-and-evaluation direction,
but it does not support introducing a new decision-weighted learning objective in
the exact finite games. At least one existing learned method reaches the active
Bayes oracle wherever a useful intervention exists, while all methods respect the
cost boundary and the pre-commitment impossibility controls.

The next research phase is established-environment validation. Phase 5's proposed
repair remains unauthorized unless a later natural environment reveals a robust
active-oracle gap that survives capacity and optimization controls.

## Execution record

The registered matrix contains `960` canonical confirmatory jobs and `160`
independent symmetry retrainings, all with ten confirmatory seeds. All `1,120`
registered jobs completed. The runs directory also contains one preserved earlier
diagnostic checkpoint, which explains its total of `1,121` manifests.

- Hyperparameters were selected on validation mechanisms only.
- Test kernels are disjoint from train and validation kernels in all 13 cells.
- Primary values use exact greedy neural-policy tree traversal.
- Both 100,000-episode Monte Carlo calibration checks contain the exact value in
  their distribution-free 95% intervals; the maximum absolute return error is
  `0.043`.
- All four learning-level matching contracts pass.
- No confirmatory or symmetry checkpoint is missing.
- No optimization rescue was triggered because the best learned active policy
  reaches the exact active oracle.

The machine-readable result is `status=complete` and
`scientific_verdict=continue_without_repair` in `artifacts/manifest.json`.

## Main empirical findings

| Capability | Confirmatory result |
|---|---|
| Common response | Every applicable core method returns `100`, matching the oracle. |
| Free passive evidence | All eight applicable methods return `92`, with DRI `0.6` and zero utilization gap. |
| Remembered evidence | All recurrent methods return `92`; memoryless PPO returns `80` despite collecting evidence with DRI `0.6`, producing a `12`-point decision-utilization gap. |
| Valuable task intervention | In `active_only`, ODITS-style, PACE auxiliary, and TALENTS-style return `87`, exactly matching the active oracle. GRU PPO active returns `82.1` and memoryless PPO returns `81.4`; passive or non-probing selectors remain at `80`. |
| Decision-relevant active evidence | In `active_response`, ODITS-style, PACE auxiliary, PACE-style, and TALENTS-style return `87`, matching the oracle; GRU PPO active returns `84.9`. |
| Identity-only or null evidence | Every method returns `80` with DRI `0`; identity information is not misreported as coordination progress. |
| Cost calibration | Every applicable method declines to probe at the exact cost boundary and in the too-expensive cell, returning the optimal `80`. |
| Pre-commitment impossibility | Every method remains at the fixed-response value `80` in both inseparable controls; post-commitment evidence does not leak into pre-commitment DRI. |

The regret decomposition is especially useful. The memoryless policy in
`remember_response` encounters decision-sufficient evidence but cannot retain and
use it. By contrast, the `active_only` failures arise from not acquiring evidence:
their decision-utilization gap is zero after the histories they actually create.
This validates the project's separation among evidence availability, evidence
acquisition, memory, and decision use.

## Statistical conclusion

The preregistered strict ranking-reversal hypothesis is **not supported**.

Across the four matched contrasts, the paired bootstrap and Holm-adjusted
sign-flip analysis finds zero strict method-pair reversals. Several comparison
cells are exactly tied at `80` or `92`, so a descriptive ordering in those cells
would be arbitrary. The generated rank matrix therefore uses average ranks for
exact ties.

Method performance still changes meaningfully with the benchmark treatment:
active methods separate in cells where decision-relevant evidence can be elicited,
collapse together when it cannot, and receive no benefit from subtype-only
identity information. This supports the diagnostic value of decision-sufficient
identifiability, but it must not be described as a statistically confirmed ranking
reversal.

## Symmetry robustness qualification

The independent label-symmetry audit passes 14 of 16 equivalence comparisons.
Two `active_only` comparisons exceed the preregistered +/-1 return equivalence
margin:

- GRU PPO active under `role_signal_swap`: mean difference `+0.7`, 95% paired
  interval `[0.0, 2.1]`;
- memoryless PPO under `role_signal_swap`: mean difference `-0.7`, 95% paired
  interval `[-2.8, 1.4]`.

The mean shifts are small, the oracle-reaching methods pass their symmetry checks,
and no runtime identifier or hidden mode leaks into observations. Nevertheless,
the two failed intervals show seed-level optimization sensitivity to relabeling.
The paper must report this qualification. Selectively adding seeds after seeing
the result would not count as preregistered confirmation; any enlarged symmetry
study must be labelled as a separate robustness analysis.

## Claims supported by Stage 4

- Decision-sufficient identifiability separates environments with matched task
  structure and nuisance controls but different attainable coordination value.
- DRI distinguishes useful response information from equally informative but
  decision-irrelevant identity information.
- Exact decomposition can tell apart failures to acquire, remember, and use
  partner evidence.
- Existing learned methods can attain the active frontier in the canonical games.
- Costly probing is not universally desirable: learned policies correctly avoid
  it when its information value does not exceed its task cost.

## Claims not supported by Stage 4

- No strict algorithm ranking reversal was found.
- The audit does not establish that existing active-identification methods
  systematically fail.
- It does not justify a new decision-weighted repair method in the finite suite.
- The `-style` baselines are controlled adaptations, not exact reproductions of
  the original published systems.
- Results from scripted, static partner modes do not yet establish the effect in
  OvercookedV2 or ZSC-Eval populations.

## Phase decision

Skip standalone Phase 5 method development for now. Preserve the exact benchmark,
metrics, and existing baselines, then proceed to Phase 6 with natural task actions
and independently trained or established partner populations. A repair method may
be reconsidered only if that validation exposes a reproducible gap against the
active oracle that is not explained by optimization, memory, capacity, or invalid
matching.
