# Finite Convention Game Schema, Version 1

## Top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | Must equal `1`. |
| `game_id` | Stable lowercase identifier. |
| `description` | Human-readable scientific purpose. |
| `horizon` | Maximum number of pre-commitment interactions. |
| `modes` | Static hidden modes and strictly positive prior masses. |
| `states` / `initial_state` | Fully observed public task state. |
| `observations` | Pre-commitment visible response symbols. |
| `actions` | Ordinary task actions, passive membership, semantics, and availability. |
| `decisions` | Irreversible terminal coordination choices. |
| `kernels` | Time-, state-, action-, and mode-conditioned outcomes. |
| `decision_losses` | Fixed nonnegative confusion-loss table. |
| `post_commitment_observations` | Optional evidence excluded from pre-commitment metrics. |
| `metadata` | Non-operative experimental labels. |
| `analytical_expectations` | Human-readable regression targets. |

All operative scalars are JSON strings such as `"4/5"`, `"0"`, or `"12"`. This
prevents binary floating-point literals from entering exact canonical calculations.

## Actions and passivity

Every action has `kind: "task"`; this is the only v1 action kind. Its
`task_semantics` must describe an ordinary environment operation. Identifiers or
semantics that introduce a dedicated type query are rejected. Passive membership is
encoded as a Boolean property of a task action, so the passive class is necessarily
a subset of the task action set.

Availability is declared by public state and pre-commitment time. A kernel row must
exist for every available `(time, state, action, mode)` tuple.

## Outcome kernel

Each kernel row contains outcomes with:

- `next_state`;
- visible `observation`;
- rational `probability`;
- nonnegative rational intervention `cost`.

Rows sum to one exactly. Outcomes with the same next state and observation are
aggregated before belief updating; their conditional expected cost remains exact.

## Loss and response relations

Every `(mode, decision)` pair has one loss, and each mode has at least one zero-loss
decision. Let `Z_mode` be its zero-loss set:

- equal sets are response-equivalent;
- overlapping sets are response-compatible;
- disjoint sets are response-conflicting.

The conflict coefficient for modes `i,j` is the minimum over decisions of the sum of
their two losses.

## Validation failures

Validation rejects unknown or duplicate identifiers, malformed rational strings,
non-normalized priors or kernels, missing kernel/loss rows, negative costs or losses,
zero-prior included modes, actions outside the horizon, modes without zero-loss
decisions, incomplete post-commitment kernels, and special query semantics.

