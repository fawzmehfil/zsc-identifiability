# BeforeCommit replay artifacts

BeforeCommit is the replay-first interface for the decision-sufficient partner
identifiability project. It visualizes one narrow question: did an interaction
reveal the right coordination response before the team crossed a named
commitment boundary?

The six public replays are deterministic reruns of frozen official ZSC-Eval
checkpoints. They are curated explanatory examples, not estimates of how often
each behavior occurs. No policy training is performed during capture.

## Rebuild

The full local Stage 6 asset package is required for capture:

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

The exporter runs CPU inference through the isolated Python 3.9 runtime. A
replay is rejected unless its return, episode length, physical seat,
commitment step, and visible action trace match the archived confirmation
trace. Browser artifacts contain normalized visual state rather than model
observation tensors.

## Metric contract

- A replay checkpoint shows estimated partner-mode support, the recommended
  response, and residual decision risk for one recorded history.
- The replay summary shows aggregate DRI from the frozen distribution-level
  audit.
- Evidence observed after commitment never changes the displayed
  at-commitment score.
- Response A and Response B are aliases for the pair-conditioned empirical
  response-library optima. They are not hand-authored semantic role labels.

## Provenance and licensing

The visual state is derived from
[SJTU-MARL/ZSC-Eval](https://github.com/SJTU-MARL/ZSC-Eval) at commit
`f940869afc42b688332a385892d8dbb57a190f95`. Checkpoints come from the
[official ZSC-Eval policy pool](https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool)
at revision `a39b45a326c6fb9c4aee79550903a7de702c6974`. Both sources are MIT licensed.
The upstream source and raw checkpoints are not redistributed here.

The web artwork is original procedural pixel art drawn in TypeScript. It does
not reuse Voyager sprites or upstream Overcooked artwork.
