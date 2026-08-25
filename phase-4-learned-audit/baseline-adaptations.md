# Baseline Adaptations

All literature-derived names are suffixed with `-style`. The suffix is material:
these methods share the paper's relevant mechanism but run inside a common finite,
single-encounter protocol.

| Method | Audited mechanism | Central role |
|---|---|---|
| `mlp_ppo` | Current-observation PPO | Memoryless and shortcut control |
| `gru_ppo_passive` | Recurrent PPO restricted to passive task actions | Passive evidence and memory control |
| `gru_ppo_active` | Recurrent PPO with all ordinary task actions | Whether task reward alone discovers probing |
| `odits_style` | Online proxy latent aligned to a training-only full-trajectory posterior, with response reconstruction | Passive representation adaptation |
| `pace_aux` | Recurrent context plus identity classifier, external reward only | Representation versus exploration ablation |
| `pace_style` | PACE-style training-only peer-identification reward | Active information seeking |
| `tom_selector_style` | Exact training cross-play clusters, global/cluster response predictors, and cluster-weighted specialists | Strategy selection from visible behavior |
| `talents_style` | Balanced training-only trajectory collection, unsupervised sequence-VAE bootstrap, deterministic latent clustering, fixed-share updates, and a specialist mixture | Latent within-encounter adaptation |
| `csp_style_reconnaissance` | Identification-oriented probing followed by a same-partner scored encounter | Relaxed-protocol upper comparison |

PACE labels exist only during training. Validation and test rewards are purely
external. ODITS-, TALENTS-, and ToM-style agents never receive hidden mode labels
as runtime inputs. CSP-style results include reconnaissance cost, loss, and extra
interactions and are excluded from central single-encounter rankings.

The TALENTS-style bootstrap records its behavior-mixture counts and certifies
`hidden_mode_labels_used=false` in each checkpoint. The ToM-selector-style policy
uses exact training cross-play return rows to define specialist clusters during
training, then removes that routing signal and selects from visible-response
predictors at evaluation.
