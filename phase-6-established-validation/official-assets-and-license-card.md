# Official Assets and License Card

The canonical Stage 6 audit uses only two pinned upstream sources:

| Source | Role | Locked revision | License |
|---|---|---|---|
| [SJTU-MARL/ZSC-Eval](https://github.com/SJTU-MARL/ZSC-Eval) | Environment, policy definitions, evaluator semantics and benchmark YAMLs | `f940869afc42b688332a385892d8dbb57a190f95` | MIT |
| [Leoxxxxh/ZSC-Eval-policy_pool](https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool) | Published partner, response and ZSC-method checkpoints | `a39b45a326c6fb9c4aee79550903a7de702c6974` | MIT |

The asset lock is derived only from the official `benchmarks-s30.yml` and
`benchmarks-s20.yml` files plus the six method path templates declared in the
canonical suite. Measured DRI is never used to include or exclude a checkpoint.

Each synchronized inventory row records the upstream path and revision, local
path, byte size, SHA-256 file hash, normalized tensor hash where applicable,
layout, algorithm, seed, architecture and provenance. Exact or parameter-level
duplicates remain in the public inventory but are prevented from counting as
independent statistical seeds.

The repository does not redistribute upstream checkpoint files. Raw model
weights, source checkouts and rollout traces remain ignored; only hashes,
configurations, compact numerical results and figures are eligible for source
control.
