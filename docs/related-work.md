# Related Work and Positioning

This document records the submission-facing novelty boundary. It is organized
around the distinctions used by the paper rather than around algorithm names.

## Robust conventions and population diversity

[Other-Play](https://proceedings.mlr.press/v119/hu20a.html) and the later
[label-free coordination formalism](https://proceedings.mlr.press/v139/treutlein21a.html)
study how independently trained agents can avoid incompatible conventions.
Population methods instead train against varied partners.
[BRDiv](https://openreview.net/forum?id=l5BzfQhROl) makes partner diversity
response-relevant, while
[ZSC-Eval](https://proceedings.neurips.cc/paper_files/paper/2024/hash/54a7139c548c88e288aa0fcd2bcbeceb-Abstract-Datasets_and_Benchmarks_Track.html)
selects evaluation populations by approximate best-response diversity and
reports best-response proximity.
[LoBP](https://www.ijcai.org/proceedings/2024/0019.pdf) adds general behavioral
predictability as a population-design criterion.

More recent work broadens coordination robustness in different directions.
[Cross-Environment Cooperation](https://proceedings.mlr.press/v267/jha25b.html)
trains across procedurally varied tasks;
[ScaPT](https://ojs.aaai.org/index.php/AAAI/article/view/39366) scales diverse
populations through parameter sharing; and
[Influence-Based Team Steering](https://arxiv.org/abs/2605.15400) learns and
steers toward effective coordination patterns in two- and three-agent
Overcooked-AI. None of these contributions makes timely, loss-weighted partner
identifiability its evaluation target.

This paper asks a complementary question. Once partners require incompatible
responses, is evidence for selecting the response available before a
task-defined commitment? BR-Div, competence, broad predictability, and
trajectory divergence are controls rather than substitutes for this quantity.

## Passive adaptation and active probing

Ad hoc teamwork has long updated beliefs over teammate models and chosen a
response online, from early
[pursuit-domain work](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AAMAS11-barrett.pdf)
through belief-space planning in
[POMCoP](https://ojs.aaai.org/index.php/AIIDE/article/view/12510) and scalable
partial-observation inference in
[RecBayes](https://doi.org/10.3233/FAIA260495). Learned ZSC adapters include
[ODITS](https://iclr.cc/virtual/2022/poster/7013),
[TALENTS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/020973f2093a6053261da93cac30ab71-Abstract-Conference.html),
and
[Theory-of-Mind Guided Strategy Adaptation](https://arxiv.org/abs/2602.12458).
They infer teammate behavior, latent strategy, or intention and condition or
select a response during interaction.

Active information acquisition is also established.
[PACE](https://proceedings.mlr.press/v235/ma24n.html) rewards peer
identification through context-aware exploration.
[Coordination Scheme Probing](https://openreview.net/forum?id=PAKkOriJBd) and
its published
[Team Probing](https://pubmed.ncbi.nlm.nih.gov/40327481/) continuation use a
separate probing phase before expert selection. In human-robot interaction,
[information-gathering task actions](https://people.eecs.berkeley.edu/~anca/papers/IROS16_active.pdf)
already elicit diagnostic human responses.

Accordingly, this paper does not claim active teammate probing or online
partner adaptation as new. It audits whether evidence is decision-sufficient
by the task deadline and whether its value exceeds the interaction cost.

## Identity information and decision information

Exact partner identity can be unnecessary when several partners admit the same
response. Conversely, a small ambiguity can matter greatly when it spans
incompatible responses. This is established decision-theoretic territory:
[Targeted Active Learning](https://openreview.net/forum?id=KxPjuiMgmm) selects
information for a downstream decision,
[Expected Value of Communication](https://ojs.aaai.org/index.php/AAAI/article/download/17346/17153)
prices a teammate query through plan value, and POMCoP optimizes expected task
return in belief space.

DRI instantiates normalized Bayes-risk reduction using an empirical
best-response loss and a ZSC commitment boundary. Its equation is not the
novelty. The contribution is the combination of this loss, pre-commitment
timing, matched populations, learned-agent diagnostics, and an externally
selected official-checkpoint audit.

## Global predictability and timely identifiability

A partner may be easy to predict over a complete episode but ambiguous before
the consequential choice. Conversely, evidence can distinguish partner
identities without changing the correct response. Earlier work already shows
that timing matters: the
[Expected Divergence Point](https://ojs.aaai.org/index.php/AAAI/article/download/17346/17153)
measures how long teammate policies behave alike, and
[Goal Recognition Design](https://ojs.aaai.org/index.php/ICAPS/article/view/13617)
studies how environment design changes when a goal becomes distinguishable.

The retained novelty claim is narrower: we did not identify prior work that
holds required-response diversity and broad predictability fixed, varies
attainable pre-commitment decision-risk reduction, and validates the resulting
measurement against coordination regret in an official ZSC benchmark.

## Fixed and mutually adaptive partners

The core model treats a partner mode as fixed during an encounter.
[Human-robot mutual adaptation](https://personalrobotics.cs.washington.edu/publications/nikolaidis2017mutual.pdf)
and [NestRL](https://arxiv.org/abs/2602.17737) study partners that change in
response to one another. Those settings alter the object being inferred: an
intervention may reveal and change the partner simultaneously. Mutual
adaptation is therefore related future work, not part of the paper’s main
claim.

## Final positioning

The paper is not a new active-probing algorithm, a new general POMDP formalism,
or a replacement for BR-Div. It is a measurement and evaluation paper about
whether unfamiliar partners can be distinguished **in the way that matters for
the next coordination decision, before that decision becomes irreversible**.

The established-environment evidence is deliberately measurement-only. The
preregistered natural interventions did not confirm, while passive DRI added
modest held-out predictive value in both layouts. This negative boundary is
part of the contribution rather than a result to be hidden.
