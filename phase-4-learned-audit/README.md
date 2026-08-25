# Stage 4: Learned-Agent Identifiability Audit

Stage 4 asks whether compact learned agents acquire and use timely partner
evidence in the exact matched populations produced by Phase 3. It is the first
training stage. The benchmark, policy interface, recurrent PPO trainer, exact
neural-policy evaluator, leakage checks, statistical analysis, and reporting
pipeline are implemented. The confirmatory training matrix has not been run, so
no scientific verdict is claimed yet.

The central comparison holds task structure, partner competence, best-response
diversity, and broad predictability fixed while moving decision-useful evidence
before or after commitment, making it actively available, or replacing it with
decision-irrelevant identity evidence.

## Protocol

- Train, validation, and test partner mechanisms have disjoint normalized kernel
  hashes.
- Policies see only public state, visible response, time, previous action and
  reward, remaining horizon, and legal actions.
- Test partner identities, response signatures, benchmark IDs, and DRI are never
  policy inputs.
- Primary values are obtained by enumerating every reachable neural-policy tree,
  not by relying on rollout averages.
- Published method names carry the `-style` suffix. They are controlled
  adaptations to this finite single-encounter setting, not exact reproductions of
  their original environments.
- CSP-style reconnaissance uses an extra same-partner encounter and is reported
  separately from the central single-encounter ranking.

## Commands

```bash
uv sync --extra learning --dev

uv run --extra learning zsc-identifiability learn validate \
  --suite phase-4-learned-audit/suites/canonical.json

uv run --extra learning zsc-identifiability learn generate \
  --suite phase-4-learned-audit/suites/canonical.json \
  --output phase-4-learned-audit/generated

uv run --extra learning zsc-identifiability learn train \
  --suite phase-4-learned-audit/suites/canonical.json \
  --method gru_ppo_active \
  --cell active_only \
  --gate smoke \
  --output phase-4-learned-audit/smoke-runs

uv run --extra learning zsc-identifiability learn tune \
  --suite phase-4-learned-audit/suites/canonical.json \
  --method pace_style \
  --output phase-4-learned-audit/development

uv run --extra learning zsc-identifiability learn audit \
  --suite phase-4-learned-audit/suites/canonical.json \
  --runs-dir phase-4-learned-audit/runs \
  --output phase-4-learned-audit/artifacts
```

`learn audit` exits with code `4` while required confirmatory checkpoints are
missing. This is an incomplete experiment, not a failed scientific result.
`learn run` first creates any missing validation-only selection reports and then
launches the fixed confirmatory matrix; it never selects from test performance.
If the complete primary matrix leaves an active-oracle gap, the audit remains
`pending` until the prespecified all-method rescue matrix exists. `learn run`
launches that rescue automatically; `learn rescue --cell active_only` exposes the
same step separately.

## Status

Implementation verdict: **pass**. Scientific verdict: **pending confirmatory
training**. See `stage-4-exit-memo.md` for the exact boundary.
