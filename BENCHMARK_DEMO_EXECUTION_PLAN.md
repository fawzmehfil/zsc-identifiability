# BeforeCommit: Visual Benchmark Execution Plan

## 1. Product decision

Convert the completed research codebase into a public, replay-first evaluation
tool called **BeforeCommit**.

**One-line description:**

> A visual benchmark for testing whether an agent gets enough useful evidence
> to coordinate with an unfamiliar teammate before an important decision.

The public product should lead with a concrete interaction, not the research
paper, statistical model, or the term DRI. A viewer should understand the
problem within ten seconds:

1. Two agents enter a shared cooking task.
2. The evaluated agent does not know which response its partner requires.
3. Partner behavior provides evidence over time.
4. The first successful ingredient placement commits the team to a response.
5. The benchmark checks whether useful evidence arrived before that point.

The existing research remains the methodological foundation and can be linked
from a secondary **Methodology** page. The demo is not a new research claim and
does not require new policy training.

## 2. Scope

### MVP

The first public release will contain:

- a polished landing page with a playable featured replay;
- six curated official-checkpoint replays;
- exact seeking, pause, speed control, and event markers;
- a visible commitment boundary;
- a decision panel showing the current recommended response and residual risk;
- an aggregate benchmark card showing DRI at commitment;
- a synchronized two-replay comparison view;
- a small replay library with clear diagnostic labels;
- static hosting on GitHub Pages;
- one 20 to 30 second launch video and three high-quality screenshots.

### Explicit non-goals

- No new agent, partner, or coordination-policy training.
- No new environment.
- No policy-upload service in the first release.
- No live simulation in the browser. The browser plays immutable recorded data.
- No attempt to display all 50 partners as individual replays.
- No claim that DRI is a measure of human-like understanding.
- No claim that the tested interventions create active probing.
- No redesign of the frozen Stage 6 analysis or its official results.

## 3. Intended audience and product value

The primary audience is a multi-agent evaluation or agent-infrastructure team.
The product should demonstrate four engineering abilities at once:

1. **Evaluation design:** final return is decomposed into a timed coordination
   decision rather than treated as the only signal.
2. **Trace-level diagnosis:** a viewer can inspect what happened before and
   after the irreversible event.
3. **Reproducible infrastructure:** every replay is tied to frozen policies,
   environment keys, content hashes, and a versioned schema.
4. **Communication:** a technical result is made understandable without
   weakening its meaning.

The demo should feel like a useful internal debugging tool that has been made
public, not like an interactive academic poster.

## 4. Naming and message hierarchy

### Product name

**BeforeCommit**

### Subtitle

**A visual benchmark for teammate adaptation**

### Primary question

> Did the interaction reveal the right coordination response before the team
> committed?

### Supporting language

- **Ego agent:** the agent being evaluated.
- **Partner:** the unfamiliar teammate.
- **Response:** a coordination strategy available to the ego agent.
- **Commitment:** the first point after which the focal choice cannot be
  repaired without loss.
- **Residual risk:** the expected cost of choosing a response using the evidence
  available so far.
- **DRI:** the fraction of prior response-selection risk removed across a
  distribution of histories before commitment.

The interface should use plain labels first and technical labels second. For
example:

```text
USEFUL INFORMATION AT COMMITMENT
DRI 0.61 · 61% OF PRIOR DECISION RISK REMOVED
```

## 5. Visual direction

The Voyager references are best described as **top-down, 16-bit-era-inspired
pixel art with chibi-scale characters**, presented inside a dense game-like
instrument panel. “64-bit” is not the most useful art-direction label here.

BeforeCommit should share that broader house style while using a separate
world, palette, silhouette language, and UI composition.

### Shared family resemblance with Voyager

- Top-down pixel diorama.
- Small, readable characters with oversized heads and clear facing direction.
- Nearest-neighbor rendering with no blurred scaling.
- Dark framing around a bright simulation world.
- Monospaced diagnostic text and a limited display font for major headings.
- Chunky borders, inset shadows, hard-edged buttons, and restrained animation.
- Information integrated into the replay rather than placed on a generic
  dashboard below it.
- Strong presentation mode suitable for screenshots and video capture.

### Distinct BeforeCommit identity

Voyager is an outdoor survival world dominated by grass, water, amber, and
forest green. BeforeCommit will be a compact **coordination test kitchen**:

- cream and desaturated blue floor tiles;
- charcoal and steel counters;
- copper, ceramic, and warm food colors;
- blue-violet ego agent and orange partner agent;
- cyan for evidence that reduces risk;
- amber for uncertainty;
- coral red for commitment and irreversible loss;
- subtle grid and oscilloscope motifs in the surrounding diagnostic shell.

Suggested palette:

| Use | Color | Hex |
|---|---|---|
| Page background | Midnight ink | `#0E1219` |
| Panel background | Blue charcoal | `#18212B` |
| Kitchen floor | Warm tile | `#D8C7A5` |
| Counter body | Slate | `#465463` |
| Counter highlight | Steel | `#758696` |
| Ego | Blue violet | `#6574D9` |
| Partner | Burnt orange | `#D98245` |
| Useful evidence | Cool cyan | `#55C7B8` |
| Uncertainty | Amber | `#E7B84B` |
| Commitment | Coral | `#E45E52` |
| Primary text | Warm white | `#F3ECD7` |

Blue and orange must also use different silhouettes, badges, and hatch patterns
so the interface remains legible without color.

### Pixel scale and sprite language

- Design on a 16-pixel logical tile grid and render at integer scale.
- Use 24 to 32 logical pixels for an agent sprite.
- Give each agent four facing directions and two walk frames.
- Use one clear interaction pose instead of elaborate animation.
- Use a dark one-pixel outline and one highlight shade per material.
- Keep food and cookware exaggerated enough to read in a social-media video.

### Original asset list

Create original procedural pixel assets rather than reusing Voyager island
sprites or upstream Overcooked artwork:

- two cook characters with distinct apron and hat silhouettes;
- floor, wall, counter, and pass-through tiles;
- onion or ingredient dispenser;
- plate dispenser;
- pot with empty, one-item, two-item, full, cooking, and ready states;
- serving station;
- held ingredient, dish, and soup icons;
- directional action pips;
- commitment pulse and vertical timeline marker;
- evidence spark, late-evidence fade, and response-choice badge;
- compact icons for return, risk, evidence, commitment, and regret.

Follow Voyager's successful procedural approach: draw these sprites from
rectangles and a small palette in TypeScript canvas code. This avoids a large
asset pipeline and keeps every sprite crisp, editable, and original.

## 6. Information architecture

### `/`

The landing page contains:

1. **Hero statement:** “Can an agent understand its teammate before it is too
   late?”
2. **Featured replay:** an immediately playable, self-explanatory example.
3. **Three-step explanation:** observe, commit, diagnose.
4. **Curated replay cards:** six diagnostic stories.
5. **Benchmark summary:** 50 partners, six method families, two layouts, both
   seats, and frozen policies.
6. **Links:** methodology, code, artifact provenance, and limitations.

### `/replays/{replay_id}`

Full replay viewer with the simulation, decision panel, event timeline,
diagnostic explanation, and provenance drawer.

### `/compare?left={id}&right={id}`

Synchronized comparison with a shared playhead. The first comparison should
show one trajectory where useful evidence arrives before commitment and one
where the decisive evidence arrives too late.

### `/present/{replay_id}`

Clean 16:9 view for screen recording. Hide the long explanation, provenance,
and library navigation while retaining the essential legend.

### Deferred route: `/challenge/{replay_id}`

The “Blind Partner Challenge” pauses immediately before commitment and asks the
viewer which response should be chosen. This is a strong follow-up feature but
is not required for the first release.

## 7. Replay-screen composition

Use a 16:9 composition designed for 1080p capture:

```text
┌──────────────────────────────────────────────────────────────────┐
│ BEFORECOMMIT   RANDOM3_M · PARTNER PAIR 07   STEP 014 / 400      │
├────────────────────────────────────────────┬─────────────────────┤
│                                            │ DECISION CONSOLE    │
│                                            │                     │
│          PIXEL TEST KITCHEN                │ RESPONSE A   68%    │
│                                            │ RESPONSE B   32%    │
│          EGO + PARTNER REPLAY              │                     │
│                                            │ RESIDUAL RISK  8.1  │
│                                            │ STATUS: UNCERTAIN   │
├────────────────────────────────────────────┴─────────────────────┤
│ [play] [1x]  evidence events ────────│ COMMIT ───── late events │
└──────────────────────────────────────────────────────────────────┘
```

### Main replay pane

- Render the complete kitchen state, not the model's observation tensor.
- Mark the ego with a blue-violet diamond and the partner with an orange
  circle in addition to sprite color.
- Show held items above the characters.
- Animate movement with short stepped interpolation while preserving exact
  discrete state at each recorded step.
- Briefly pulse the relevant object when a high-level event occurs.

### Decision console

At each prefix, show:

- the available response aliases;
- the frozen decoder's posterior support for each response class;
- the recommended response under the empirical loss matrix;
- the residual expected response-selection risk;
- a short human explanation such as “partner movement now favors Response A.”

Do not invent semantic names such as “left-role partner” unless a human audit
has confirmed that interpretation. Default public aliases should be Response A
and Response B.

### Commitment treatment

At the first successful ingredient placement:

- flash a coral outline around the pot;
- stamp **COMMITTED** for 500 to 700 milliseconds;
- place a permanent coral marker on the timeline;
- freeze the “information available at commitment” value;
- continue the replay so late evidence remains visible but clearly separated.

Evidence after commitment should be desaturated and labelled **TOO LATE FOR
THIS DECISION**. It remains part of the trace but cannot improve the displayed
precommitment benchmark score.

### Metric accuracy rule

Do not display “live DRI” for one episode. The following separation is
required:

- **Per replay and per step:** posterior support, recommended response, and
  residual risk for the current history prefix.
- **At commitment:** the current episode's frozen residual-risk snapshot.
- **On the benchmark card:** aggregate DRI computed over the corresponding
  history distribution.

This prevents an individual confidence trajectory from being mislabelled as a
population-level Bayes-risk reduction.

## 8. Curated replay collection

Ship six clearly labelled cases:

1. **Clear before commitment**
   Useful partner evidence arrives early and residual risk falls before the pot
   placement.
2. **Still ambiguous**
   The two response-conflicting partner modes produce similar precommitment
   histories.
3. **One step too late**
   Strong evidence appears shortly after the commitment marker.
4. **Evidence available, outcome poor**
   The measured history is informative but the recorded team outcome is weak.
   Do not automatically call this an “evidence-use failure.”
5. **Robust without identification**
   The run succeeds despite low measured identifiability, illustrating that
   final return and DRI answer different questions.
6. **Natural intervention comparison**
   Ordinary progress and the registered intervention are shown together. The
   caption states that the intervention did not reliably improve information
   across the benchmark.

Selection must be deterministic and documented in
`demo/curation/official-v1.yaml`. Each entry records:

- source suite and source artifact hashes;
- layout, partner pair, ego policy, seat, and environment key;
- selection rule and diagnostic category;
- whether the example is representative or intentionally illustrative;
- the final public caption;
- any human-authored behavioral annotation.

Curated examples are for explanation, not estimates of frequency. Aggregate
cards must continue to come from the full frozen result package.

## 9. Data and metric contract

### Replay manifest

Create a versioned `beforecommit_replay_1.0.0` schema containing:

```json
{
  "replay_id": "clear-before-commitment-v1",
  "schema_version": "beforecommit_replay_1.0.0",
  "status": "complete",
  "source": {
    "suite_id": "official-measurement-v3",
    "layout_id": "random3_m",
    "environment_key": 0,
    "ego_seat": 0,
    "ego_policy_id": "...",
    "partner_id": "...",
    "checkpoint_hashes": {},
    "source_artifact_hashes": {}
  },
  "commitment": {
    "definition": "first_successful_ingredient_placement",
    "step": 0,
    "reached": true
  },
  "summary": {
    "sparse_return": 0.0,
    "response_regret": 0.0,
    "residual_risk_at_commitment": 0.0,
    "aggregate_dri": 0.0
  },
  "artifacts": []
}
```

The example values above are placeholders. The exporter must populate them
from authoritative artifacts and reject absent or inconsistent values.

### Frame record

Each frame contains only the normalized visual and diagnostic state required by
the viewer:

- step number and replay time;
- player position, orientation, held object, and action;
- counter objects;
- pot contents and cooking status;
- dispenser and serving-station locations;
- sparse reward and high-level events;
- decoder response support at the current prefix;
- recommended response and residual risk;
- flags for precommitment, commitment, and postcommitment.

Do not expose the 1,000-dimensional observation vectors to the browser.

### File layout

Episodes are at most 400 steps, so the MVP can remain simple:

```text
web/public/static-api/
  catalog.json
  benchmark-summary.json
  replays/
    {replay_id}/
      manifest.json
      initial.json
      frames.json
      diagnostics.json
      events.json
```

Every emitted file receives a SHA-256 hash in the manifest. JSON should use
stable key ordering and compact separators. Enable Brotli or gzip at the host
when available.

## 10. Technical architecture

### Front end

- Vite
- React
- TypeScript
- Phaser for the replay world
- Vitest for presentation and controller tests
- Static JSON artifacts
- GitHub Pages deployment

No backend is needed for the first release. A static site is faster to build,
cheaper to host, easier to reproduce, and adequate for immutable curated runs.

### Python exporter

Add a small demo namespace to the existing package:

```text
src/zsc_identifiability/demo/
  __init__.py
  models.py
  curation.py
  state_adapter.py
  diagnostics.py
  exporter.py
  validator.py
  cli.py
```

Responsibilities:

- validate the curation manifest;
- run or load only the selected inference episodes;
- normalize old and new official Overcooked state variants;
- apply the frozen representation and decision decoder to every history prefix;
- compute recommended response and residual risk from the empirical loss matrix;
- attach aggregate DRI from frozen analysis artifacts;
- emit deterministic, path-safe static JSON;
- verify hashes and reject a changed commitment definition.

Add CLI commands:

```bash
uv run zsc-identifiability demo capture \
  --curation demo/curation/official-v1.yaml \
  --output demo/artifacts/replays

uv run zsc-identifiability demo export-site \
  --replays demo/artifacts/replays \
  --output web/public/static-api

uv run zsc-identifiability demo validate \
  --site-data web/public/static-api
```

### Visual-state capture

The current Stage 6 traces preserve observations, actions, rewards, and events,
but not enough global state to render a faithful kitchen. Capture six selected
episodes again using the same frozen checkpoints and environment keys.

Implement a separate demo-only inference path instead of modifying the frozen
Stage 6 result contract. It must:

- import no trainer;
- require `policy_training_allowed: false`;
- use CPU inference and the original deployment semantics;
- record `env.base_env.state` after normalization at every step;
- retain the existing physical-seat mapping;
- preserve the same commitment detector;
- write source and checkpoint hashes;
- verify that return, episode length, commitment step, and visible actions agree
  with the corresponding stored trace before accepting a replay.

The rerun is artifact generation, not a new experiment.

### Reuse from Voyager

Reuse proven architecture and interaction patterns where helpful:

- replay controller state machine;
- exact seeking and playback-speed controls;
- static API export shape;
- presentation and comparison routes;
- procedural canvas texture approach;
- responsive replay-shell structure;
- deterministic export and schema-test philosophy.

Do not directly reuse the island palette, character sprites, terrain assets,
copy, or exact HUD composition. BeforeCommit should look related through craft
and interaction quality, not through duplicated art.

## 11. Proposed repository additions

```text
demo/
  README.md
  curation/
    official-v1.yaml
  schemas/
    beforecommit-replay-v1.schema.json
  artifacts/
    replays/                  # generated curated replay packages

web/
  package.json
  vite.config.ts
  src/
    App.tsx
    paths.ts
    styles.css
    components/
      ReplayPlayer.tsx
      DecisionConsole.tsx
      CommitmentTimeline.tsx
      BenchmarkCard.tsx
      ReplayLibrary.tsx
      ProvenanceDrawer.tsx
    replay/
      types.ts
      ReplaySource.ts
      ReplayController.ts
      ReplayScene.ts
      kitchenPixelArt.ts
      presentation.ts
    pages/
      HomePage.tsx
      ReplayPage.tsx
      ComparePage.tsx
      PresentPage.tsx
  public/
    static-api/               # generated deployable data

src/zsc_identifiability/demo/
  ...

tests/
  test_demo_schema.py
  test_demo_curation.py
  test_demo_capture.py
  test_demo_export.py
  test_demo_metric_contract.py
```

Generated artifacts should be committed only after license review and size
checks. Raw checkpoints, full raw trace corpora, machine-specific paths, and
observation tensors remain uncommitted.

## 12. Execution phases

### Phase 0: Freeze the product contract

**Work**

- Approve the name, one-line description, palette, and replay composition.
- Create the v1 replay schema and curation manifest.
- Identify the authoritative sources for loss matrices, decoders, aggregate DRI,
  and commitment timing.
- Confirm redistribution and attribution requirements for derived state traces.
- Select candidate episodes without editing captions to fit their outcomes.

**Exit criteria**

- Six curation entries resolve to existing assets.
- Every displayed metric has one named authoritative source.
- No UI field ambiguously labels per-episode confidence as DRI.

**Estimate:** half a focused day.

### Phase 1: Build deterministic replay capture

**Work**

- Add the isolated visual capture operation.
- Normalize both official layout state formats.
- Record visual state and existing trace features.
- Run the frozen decoder over every prefix.
- Add hash, parity, seat, commitment, and return checks.
- Export six versioned replay packages.

**Exit criteria**

- Re-exporting a replay produces byte-identical JSON.
- Each replay matches its stored trace on return, length, actions, seat, and
  commitment step.
- No policy parameters are updated.

**Estimate:** one to one and a half focused days.

### Phase 2: Build the visual vertical slice

**Work**

- Scaffold the static React and Phaser viewer.
- Implement one layout, two cooks, counters, held items, and pot state.
- Add play, pause, seek, and 0.25x to 4x speed controls.
- Add the commitment pulse and marker.
- Add the first-pass decision console.
- Implement the clean presentation route.

**Exit criteria**

- One replay is understandable without reading the paper.
- Exact seeking produces the same visual state every time.
- The 16:9 presentation view records cleanly at 1080p.

**Estimate:** one and a half focused days.

### Phase 3: Complete the diagnostic experience

**Work**

- Add response support, recommended response, and residual-risk charts.
- Visually separate precommitment and late evidence.
- Add plain-language event captions.
- Add aggregate DRI and benchmark summary cards.
- Add provenance and limitations drawers.
- Implement the replay library.

**Exit criteria**

- Every diagnostic label has a tooltip or one-sentence explanation.
- Late evidence can never alter the frozen at-commitment card.
- Aggregate values reproduce the committed result artifacts exactly.

**Estimate:** one focused day.

### Phase 4: Add comparison and finish curation

**Work**

- Implement synchronized playback.
- Build three comparison presets:
  - clear versus ambiguous;
  - useful evidence versus late evidence;
  - ordinary progress versus intervention.
- Complete all six captions and behavioral audits.
- Add stable shareable URLs.

**Exit criteria**

- Both viewers remain within one step during synchronized playback.
- Comparison captions describe observations without implying unsupported causes.
- A visitor can move from the landing page to the central comparison in one
  click.

**Estimate:** one focused day.

### Phase 5: Release and social package

**Work**

- Run Python, TypeScript, schema, and static-build checks.
- Deploy the static site to GitHub Pages.
- Capture three screenshots at 1920 by 1080.
- Record a 20 to 30 second feature video.
- Rewrite the repository README around the live benchmark while preserving a
  clear methodology link.
- Add acknowledgements, upstream asset provenance, and limitations.

**Exit criteria**

- A clean browser loads the featured replay with no local services.
- The public site contains no machine paths, raw checkpoints, or identifying
  private artifacts.
- The launch video remains understandable without audio.
- Existing repository tests still pass.

**Estimate:** one focused day.

### Total expected effort

Approximately **five to six focused working days** for the MVP. The work is
mostly export, presentation, and curation. It does not include training or a
new large benchmark run.

## 13. Verification plan

### Scientific and numerical checks

- Response values come only from the frozen empirical response library.
- Aggregate DRI values match the frozen Stage 6 artifacts.
- The displayed commitment step equals the first successful ingredient
  placement used by the analysis.
- Postcommitment frames never contribute to the at-commitment score.
- Decoder artifacts and representation seeds match the frozen fit manifest.
- The UI says “estimated” where a decoder output is shown.
- Replay examples are labelled curated and not described as frequency
  estimates.

### Replay integrity checks

- Source hashes and replay-file hashes validate.
- Physical seat mapping is tested for both seats.
- Old and new official environment variants map into one visual schema.
- Frame count equals recorded episode length.
- Exact seeking is deterministic.
- Terminal return and event totals match the source trace.

### Visual checks

- Pixel art is rendered only at integer scale where practical.
- Text is legible in a 1080p video viewed at half size.
- Commitment remains visually dominant without obscuring the action.
- Response colors remain distinguishable in grayscale through shape and pattern.
- The replay works at desktop widths down to 1024 pixels.
- A reduced mobile layout shows the video and essential result card even if the
  full inspector is desktop-first.

### Build checks

```bash
uv run pytest
uv run ruff check .
uv run mypy src
npm --prefix web test
npm --prefix web run lint
npm --prefix web run build
uv run zsc-identifiability demo validate --site-data web/public/static-api
```

## 14. Launch video storyboard

### 0 to 4 seconds

Show both cooks entering the kitchen.

On-screen text:

> SAME TASK. UNFAMILIAR TEAMMATE.

### 4 to 10 seconds

The partner moves and interacts. Response support shifts in the decision
console.

On-screen text:

> WHICH COORDINATION RESPONSE IS RIGHT?

### 10 to 15 seconds

An ingredient enters the pot. The coral commitment pulse appears and the
timeline locks the at-commitment metric.

On-screen text:

> THE TEAM HAS COMMITTED.

### 15 to 22 seconds

Cut to synchronized replays. One becomes informative before commitment. The
other reveals the answer after commitment.

On-screen text:

> SUCCESS CAN HIDE WHEN THE ANSWER BECAME KNOWABLE.

### 22 to 28 seconds

Show the result card and replay library.

On-screen text:

> BEFORECOMMIT
> TRACE-LEVEL EVALUATION FOR TEAMMATE ADAPTATION

End with the live-demo and repository URLs.

## 15. README presentation after launch

The repository front page should use this order:

1. One-sentence product description.
2. Hero screenshot or short GIF.
3. Live demo, replay library, methodology, and results links.
4. A concrete “what the viewer shows” explanation.
5. Frozen benchmark scope.
6. Quick start for the static demo.
7. Technical architecture and reproducibility.
8. Research methodology and paper link.
9. Limitations and attribution.

Do not lead the new README with regression coefficients. Keep those values in
the methodology and results sections for readers who want them.

## 16. Deferred improvements

Only consider these after the MVP is public:

- Blind Partner Challenge.
- Full partner-pair heatmap linked to replay examples.
- A downloadable replay bundle and documented replay SDK.
- User-submitted traces using the versioned schema.
- A live policy runner isolated in a sandbox.
- Additional multi-agent environments with explicit commitment contracts.
- Video export directly from presentation mode.

The first release should stay narrow. One polished question, six trustworthy
replays, and a memorable visual explanation will be more effective than a
general platform with unfinished surfaces.

## 17. Definition of done

BeforeCommit v1 is complete when a technically informed visitor can open the
site and, without reading the paper:

1. identify the ego and partner;
2. understand that different partners can require different responses;
3. see what evidence the ego receives over time;
4. locate the commitment point;
5. distinguish useful early evidence from evidence that arrives too late;
6. understand that the per-replay diagnostic and aggregate DRI are different;
7. inspect the exact replay provenance;
8. share a stable replay or comparison URL.

The central standard is not feature count. It is whether the viewer makes the
evaluation question obvious, visually memorable, and technically honest.
