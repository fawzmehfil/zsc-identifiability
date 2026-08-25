# Suite and Population Schemas

Phase 3 keeps the Phase 2 `FiniteConventionGame` schema at version 1. Benchmark
metadata is stored separately so hidden labels and matching annotations can never
enter runtime observations.

## `MatchedBenchmarkSuite` v1

The top-level object contains:

- `schema_version`: fixed to `1`;
- `suite_id` and rational `base_team_return`;
- `families`: discriminated `binary_role_allocation` or
  `factorized_identity_memory` specifications;
- `matching_contracts`: named population pairs and exact metric rules;
- `sample_audit`: rollout, bootstrap, seed, confidence, and equivalence settings.

Each family declares a rational reliability, intervention cost, mismatch loss,
canonical cells, one-factor sweeps, and whether structural symmetry variants are
generated. Canonical rational quantities are JSON strings such as `"4/5"`.

Validation rejects unknown fields, invalid rationals, reliability outside
`[1/2, 1]`, negative costs or losses, duplicate family/cell/sweep identifiers,
empty or dangling sweeps, and a base return below any declared maximum
cost-plus-loss.

## Generated population descriptor v1

Every population descriptor records:

- stable population, family, cell, matching-group, and symmetry identifiers;
- the corresponding validated finite game;
- hidden response signatures and best-response event vectors;
- passive reference actions and legal commitment states;
- intended treatments and matched nuisance variables;
- runtime-visible field names;
- analytical expectations;
- SHA-256 hashes of the source suite and generated game.

Only time, public task state, ego action, and partner response are runtime-visible.
The mode, response signature, family metadata, and hashes are evaluation metadata.

## Matching rules

A matching contract can require structural equality, identical passive history
distributions, equal aggregate divergence profiles, and exact or tolerance-based
metric relations. Every result is explicitly `pass`, `fail`, or
`not_applicable`; an unavailable metric never counts as a match.

Structural matching checks priors, response signatures, decision losses, base
return, horizon, state/action/observation spaces, action availability, costs,
best-response event features, passive reference policy, commitment states, and
observation budget.

## Stable Python API

```python
load_benchmark_suite(path)
generate_benchmark_suite(spec, backend="fraction")
compute_population_metrics(population, backend="fraction")
audit_population_pair(left, right, contract, backend="fraction")
audit_shortcuts(population, backend="fraction")
run_benchmark_suite(suite_config, output_dir)
```

The exact backend returns `Fraction` values internally and rational strings in JSON
artifacts. The float backend exists for sweeps and plotting and is audited against
the exact backend at `1e-10` absolute tolerance.

## Metric names

- `rahman_brdiv_return`: return-compatibility diversity.
- `zsceval_br_div_raw`: determinant of the raw best-response feature Gram matrix.
- `zsceval_br_div_code`: determinant after the official implementation's
  `column_max + 1e-3` normalization.
- `lobp_action_oracle_score_nats`: exact Bayes next-response log score, not a
  reproduction of a trained LoBP observer.
- `passive_dri`, `active_dri`, `eventual_dri`: pre-commitment passive,
  pre-commitment task-active, and post-commitment-inclusive decision-relevant
  identifiability.

The full four-mode factorized population legitimately has zero raw and normalized
ZSC-Eval determinant because two modes share each best response. The separate
response-representative determinant is diagnostic only.
