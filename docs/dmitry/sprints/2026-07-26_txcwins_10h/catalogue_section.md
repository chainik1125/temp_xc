---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - reference
  - in-progress
---

## Status

The literature sweep's section for `summary.md`, lifted out of `literature_catalogue.md`
(1561 lines) so it is not buried. Written to be pasted with light editing.

**Why it ships regardless of tonight's measurements:** it is a claim about which behaviours
*can* favour a window dictionary, not about any single result. If every experiment tonight
comes back flat, this section is still the sprint's finding.

---

**A literature sweep says where a window code can help, and it is a narrow place.**

The open objection to the previous sprint's result was relevance: the crosscoder's steering
advantage appeared on a construct — two orderings of one multiset of sentences — with nothing
connecting it to a behaviour anyone wants to steer. A sweep across reasoning, refusal, deception,
degeneration, agentic and evaluation literatures answers that objection in an unexpected way. It
did not find a long list of temporally extended behaviours a crosscoder might win on. It found
that **almost every behaviour that looks temporally extended already has a per-token handle**:

| behaviour | the per-token handle that already works |
| --- | --- |
| refusal | a single direction at a single position near-saturates |
| backtracking | reasoning behaviours are "controlled by linear directions" |
| repetition | a single sign-inverted neuron; repetition-neuron edits; KV-cache pruning cutting loops by >90pp |
| emergent misalignment | a transferable misalignment direction |
| MCQ option order | token bias over option-ID tokens, and a content-invariant position attractor |
| entity / state tracking | the model reinstates bindings at readout rather than tracking incrementally |
| planning / lookahead | ~90% of planning capacity at one token, via five attention heads |

Seven literatures, seven per-token handles, each found while trying to build a case *for* the
behaviour. Set against that, the tasks that survive share exactly one structural property: the two
conditions are the **same multiset in a different arrangement**, so no bag-of-positions statistic
can separate them and no constant write can exploit one.

This repo had already established the necessity of that property empirically without naming it.
A textbook position-dependent template (passphrase verification) *lost* to a broadcast write
because it is a conjunction; ordered generation lost by 10–50× because a broadcastable mode
carries the behaviour; and only the multiset-matched trajectory tasks won, with the template
growing linearly in window length while broadcast sat at zero or below. The sweep supplies the
general form of that finding: **a window code helps exactly when the target has no DC component,
and matched multisets are how you guarantee that.**

That is a more useful claim than "the crosscoder beats the SAE on task X". It says where to look,
explains why seven other directions are dead ends, and is directly falsifiable — find a behaviour
with no matched-multiset structure where a window code wins on steering, and it is wrong.

---
