# Stage 6 Exit Memo

## Final verdict

**`complete_measurement_only`**

The calibrated fresh-confirmation audit supports pre-commitment DRI as an
established-environment measurement. It does not support the preregistered
natural-intervention claim, and it does not justify reopening method
development.

## What survived confirmation

Adding pre-commitment DRI to the registered baseline features improved
leave-one-HSP-scheme-out prediction of normalized response-library regret:

| Representation | Scope | ΔR² | ΔMAE | ΔMSE |
|---|---:|---:|---:|---:|
| GRU, primary | Overall | +0.02047 | -0.003394 | -0.001369 |
| GRU, primary | `random3_m` | +0.02426 | -0.002199 | -0.000896 |
| GRU, primary | `small_corridor` | +0.007449 | -0.001228 | -0.000598 |
| Event, sensitivity | Overall | +0.01180 | -0.002149 | -0.000789 |
| Event, sensitivity | `random3_m` | +0.02172 | -0.001925 | -0.000803 |
| Event, sensitivity | `small_corridor` | +0.01423 | -0.002339 | -0.001143 |

The primary adjusted DRI coefficient for regret was `-0.18035`; the clustered
bootstrap mean was `-0.18290`, with a 95% interval of
`[-0.29499, -0.07918]`. The event sensitivity coefficient was `-0.15843`, with
a 95% interval of `[-0.28397, -0.05509]`.

Passive pre-commitment DRI exceeded its registered permutation null in both
layouts:

| Layout | Mean pairwise DRI | Raw p | Holm-adjusted p |
|---|---:|---:|---:|
| `random3_m` | 0.6674 | 0.00990 | 0.03960 |
| `small_corridor` | 0.5195 | 0.00990 | 0.03960 |

All synthetic controls, Brier-score checks, fixed-response non-worsening
checks, and leakage checks passed. GRU seed-direction agreement was 0.979 in
`random3_m` and 0.966 in `small_corridor`. Direct binary GRU refits on the ten
frozen diagnostic pairs correlated at 0.986 with the shared-encoder DRI.

## What did not survive confirmation

Neither frozen natural intervention qualified:

| Layout | Intervention | GRU risk reduction | Corrected 95% interval | Holm p | Verdict |
|---|---|---:|---:|---:|---|
| `random3_m` | `temporary_role_takeover` | +0.00490 | [-0.00120, 0.01110] | 0.9901 | Not confirmed |
| `small_corridor` | `corridor_yield` | -0.00487 | [-0.01707, 0.00814] | 0.9901 | Not confirmed |

The first intervention completed reliably and its point estimate exceeded its
measured task cost, but the corrected interval crossed zero and the registered
permutation test failed. The second intervention had negative GRU and event
effects. These are null results, not weak positive evidence.

The `existing_methods_leave_systematic_gap` gate was also false. Stage 4 had
already shown that several existing mechanisms reach the exact active oracle in
the canonical games. A new decision-aware repair would therefore be an
unregistered solution in search of a demonstrated problem.

## Audit scale and provenance

- Stage 6 v2 completed 4,842 shards, 240,800 episodes, and 96.32 million
  environment steps over the official checkpoint matrix. Its identity-oriented
  estimator failed calibration, so its apparent effects remain exploratory.
- Stage 6 v3 froze direct pairwise decision decoders before collecting a fresh,
  disjoint 9,600-episode, 3.84-million-step confirmation set.
- The audit covers all 30 official `random3_m` partners, all 20 official
  `small_corridor` partners, and six official ZSC method families.
- No partner or coordination-policy training occurred in Stage 6. Only the
  registered GRU measurement representations and decision heads were fitted.
- The final suite, fit configuration, confirmation plan, and v2 source plan are
  content-hashed in the compact result manifest.

## Authorized paper claims

1. DRI operationalizes how much pre-commitment evidence reduces response
   selection risk; it is normalized Bayes-risk reduction rather than new
   information theory.
2. Exact matched games show that timely decision information can vary while
   response diversity and broad behavioral predictability remain fixed.
3. Learned-agent experiments separate evidence acquisition, relevance, memory,
   and use; existing methods can solve the synthetic active cases.
4. In the official ZSC-Eval checkpoint audit, cross-fitted pre-commitment DRI
   adds modest held-out predictive value beyond the registered controls in both
   layouts, with the same direction under the event sensitivity estimator.
5. The audited natural task interventions did not confirm, so active probing is
   retained as a formal and synthetic result only.

## Claims that remain unauthorized

- DRI causes lower coordination regret.
- Natural active partner probing was demonstrated in ZSC-Eval.
- Existing ZSC methods systematically fail to acquire useful partner evidence.
- DRI is an exact environmental quantity in the established benchmark.
- The response library is globally optimal.
- DRI improves every held-out HSP-scheme fold.
- A new repair algorithm is required.

## Paper decision

Complete the project as a theory, benchmark, learned-agent audit, and
established-environment measurement paper. Frame active probing as the
controlled question that motivated the audit, not as the final established
result. Stage 5 remains skipped. Additional policy training would be post hoc
unless a new, independently preregistered study targets a genuinely different
question.

The full claim and limitation audit is in `statistical-claim-audit.md`; compact
tables, figures, and path-safe hashes are in `results/`.
