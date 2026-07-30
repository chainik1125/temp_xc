## Temporal feature-formation screen

The useful object is not a language-level decay constant. It is the joint
profile of:

1. when behavior-relevant information becomes available;
2. whether the measured representation causally mediates the behavior; and
3. whether that mediation is local to the current token or supported by prior
   sequence positions.

This separates temporal dependence from incoherence because the primary
outcome is the task's own behavior metric under causal interventions. Entropy
and teacher-forced cross-entropy remain diagnostic secondary curves.

## Two protocols, not one fictitious onset

### Event profile

Use when a rollout has a defensible first behavior-bearing event, such as a
refusal clause, backtrack, tool call, payload, or leaked canary.

- Define token zero using an exact parser, simulator, or frozen event judge.
- Measure a prespecified or cross-fitted representation over negative token
  offsets.
- Select the confirmatory neutral before observing whether the rollout later
  contains the event.
- Report both the feature-availability curve and the eventual task metric.
- Validate the feature with held-out ablation, addition, or patching.

For prompt-triggered behaviors, also sweep how much of the prompt has been
revealed. Prompt reveal and output-event offset are different coordinates and
must not be pooled.

### State profile

Use when the behavior is an endpoint propensity without a natural first token,
such as emergent misalignment, evaluation awareness, or a persistent hidden
goal.

- Label complete frozen rollouts.
- Use preregistered normalized response positions.
- Cross-fit by prompt or scenario family.
- Fit a terminal direction and transport it backward.
- Fit a fresh direction at each position as a trajectory-rotation control.
- Report the complete curve and uncertainty, not an onset token.

A flat curve from the first generated token is a valid weight-encoded,
short-inference-horizon result. A weak or unstable terminal direction is an
unresolved measurement, not evidence that the state appears late.

## Core estimands

For target prompts \(x^+\), programmatically generated neutrals \(x^-\), a
direction \(v\), representation \(h_t\), and behavior metric \(B\):

\[
A_v(t)=\operatorname{AUC}\left(v^\top h_t(x^+),
v^\top h_t(x^-)\right)
\]

is feature availability. The behavioral contrast is

\[
\Delta_B(t)=
\mathbb{E}[B(Y_t^+)-B(Y_t^-)].
\]

Neither is causal by itself. With direction ablation, define the
*causal sequence-support gap*

\[
D_{\rm seq} =
B(Y_{\rm current\text{-}token\ ablated})
-
B(Y_{\rm all\text{-}positions\ ablated}).
\]

The metric is oriented so larger means more target behavior. A positive
\(D_{\rm seq}\) says that removing the feature outside the current token has
additional causal effect. It rules out an exclusive current-token bottleneck,
but does not distinguish diffuse support from one earlier bottleneck.

The addition analogue,

\[
S_{\rm seq} =
B(Y_{\rm all\text{-}positions\ added})
-
B(Y_{\rm current\text{-}token\ added}),
\]

is useful but secondary because sequence-wide addition usually injects more
total intervention energy.

For a conventional SAE with decoder directions \(d_j\), compare the
distributed reconstruction projection with the strongest single latent:

\[
F_{\rm frag}(t) =
A_{\sum_j z_j(t)\cos(d_j,v)}(t)
-
\max_j A_{z_j}(t).
\]

A persistent positive fragmentation gap is evidence that an instantaneous
sparse basis carries the causal direction only as a distributed combination.

## Refusal calibration

The Llama-3-8B refusal pilot uses the exact held-out Arditi split, selected
layer-12 direction, chat template, response-start score, and intervention
hooks.

| Quantity | Result |
| --- | ---: |
| Direction AUC at 12.5% harmful-prompt reveal | 0.866 |
| Direction AUC at 25% reveal | 0.942 |
| Refusal-log-odds AUC first above 0.8 | 37.5% reveal |
| Prompts where direction onset preceded or matched refusal | 31/31 |
| Prompts where direction onset strictly preceded refusal | 20/31 |
| Median direction lead | 3 prompt tokens |
| Harmful refusal, baseline / current-token ablation / all-position ablation | 31/32 / 31/32 / 0/32 |
| Harmless refusal, baseline / current-token addition / all-position addition | 0/32 / 0/32 / 32/32 |
| Causal sequence-support gap \(D_{\rm seq}\) | 0.96875 |
| Harmful refusal after ablating prompt lags 1–4 / 5–8 / 9–16 | 27/32 / 31/32 / 30/32 |
| Harmful refusal after ablating prompt lags 17–32 / 33–64 | 28/32 / 30/32 |
| Harmful refusal after ablating all prior prompt positions | 0/32 |

This establishes a clean distinction:

- one direction is a sufficient sequence-wide effector and a necessary
  sequence-wide mediator;
- the direction forms before overt refusal on most prompts;
- its value at the current token is neither necessary nor sufficient.

Thus “mediated by one direction” does not imply “stored at one token.” Earlier
prompt states or their cached keys and values retain causally relevant support.
The geometric-band result does not isolate one dominant prior band: removing
any single band leaves most refusals intact, while removing all prior positions
eliminates refusal. The support is therefore redundant, conjunctive, or
distributed across bands at this resolution. Interaction tests among grouped
bands, not exhaustive single-position masks, are the next localization step.

## Backtracking calibration

The Ward direction has an observational half-peak onset at offset \(-9\), and
all norm-matched directions derived from offsets \([-13,-8]\) causally increase
backtracking. The conventional SAE reconstruction preserves the distributed
direction-level curve, while the nearest single SAE latent is close to chance.

Backtracking therefore supplies the current positive architecture-selection
calibration: pre-outcome availability, temporal stability, causal efficacy,
and instantaneous-basis fragmentation occur together.

## EM calibration

The current-paper Medical EM state pilot reuses exact-reference frozen
rollouts from the same bad-medical model. It contains 16 coherent misaligned
and 35 coherent aligned responses, with mean lengths 55.7 and 56.1 tokens.
Directions are trained leave-one-prompt-out from within-prompt contrasts.

The transported terminal direction ends at macro AUC 0.604, and its
prompt-bootstrap interval spans chance. A positionwise prefix-mean direction
has a transient AUC peak around 0.81 at 30% progress, but the curve is
non-monotone and has only six evaluable prompt groups. This is possible
trajectory rotation or small-sample instability, not a defensible EM onset.

The comparison is therefore:

- refusal: strong known mediator, pre-behavior formation, sequence-wide causal
  support;
- backtracking: pre-event causal subspace fragmented across instantaneous SAE
  features;
- EM: weak observational state readout with no natural event and no current
  causal mediation result.

## Scalable decision rule

A behavior is a strong temporal-crosscoder candidate only when all applicable
gates pass:

1. **Behavior gate:** the model organism produces a coherent, safety-relevant
   target behavior under a frozen task metric.
2. **Temporal gate:** availability rises before an event, or a state readout is
   reproducibly non-flat across progress.
3. **Causal gate:** ablation or patching changes the behavior while preserving
   coherence and generic task competence.
4. **Sequence-support gate:** \(D_{\rm seq}>0\) with uncertainty excluding
   zero, or a lag-band intervention shows causal support outside the current
   token.
5. **Factorization gate:** the distributed residual or SAE reconstruction
   carries the signal more cleanly than any single conventional SAE feature.
6. **Replication gate:** the result survives held-out prompts, seeds, length
   matching, and at least one semantically distinct task family.

The screen should return a vector of gate results, not collapse immediately to
one fitted decay coefficient. For triage, \(D_{\rm seq}\) is the most useful
single number because it is causal, expressed in the actual task metric, and
directly tests whether sequence positions beyond the current token matter.

## Avoiding combinatorial position masks

Do not enumerate all context-window subsets. Use a coarse-to-fine causal
localization:

1. compare current-token-only with sequence-wide ablation;
2. partition prior positions into geometric lag bands
   \(1, 2\!-\!4, 5\!-\!8, 9\!-\!16,\ldots\);
3. test every band separately and control false discovery across bands;
4. recursively split only bands with reproducible causal effect;
5. test interactions only among the small set of surviving bands.

This costs \(O(\log L)\) cells for the initial localization rather than
enumerating \(2^L\) masks. For cached generation, manipulate the relevant
cached states directly or recompute without cache; otherwise a
“current-token” intervention leaves earlier keys and values intact.

## Minimal adapter contract

Each new behavior supplies:

- a task sampler with target and programmatic neutral arms;
- a rollout function for the frozen subject model;
- a scalar behavior evaluator and a separate coherence/competence evaluator;
- an event locator, or an explicit declaration that the behavior is
  state-like;
- a representation hook and optional prespecified direction;
- current-token, sequence-wide, and lag-band intervention hooks.

The generic outputs are prompt-reveal, event-offset or normalized-progress
curves; cross-fitted uncertainty; causal sequence-support; and conventional
SAE fragmentation. The behavior-specific adapter defines semantics, while the
estimands and gates stay fixed.
