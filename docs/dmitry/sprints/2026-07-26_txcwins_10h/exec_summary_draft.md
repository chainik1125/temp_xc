---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - results
  - in-progress
---

## Status

Draft executive summary for `summary.md`, assembled from the theory agent's structure with
findings 5 and 6 rewritten against the completed run. Items marked ⚠ depend on measurements
still landing.

**Pending when this was written:** the rotation ladder (the only design that could support an
expressiveness claim), gradient-based rank arms at full configuration, and the held-out-content
split on recency and evidence.

## Executive summary

This sprint set out to find more tasks where a temporal crosscoder beats a per-token
dictionary. It did not find them, and in the course of not finding them it produced the
reason: a criterion, computable before any dictionary is trained, that says which tasks *could*
separate the architectures — and the finding that almost none do.

### The previous sprint's headline is withdrawn

The order-task result — crosscoder +11.29 against the SAE's +1.24 — was measured on a one-sided
dose grid. Rerun at both signs with two dictionary inits, the crosscoder does not beat the SAE
significantly in either (+6.34 and +3.41 against +4.66 and +5.16; z = 1.35 at one init, losing
at the other).

**The control that was the proof has inverted.** `txc_flat` — the same slab with its temporal
profile averaged away — was reported as *inverting* to −8.02, and that was the evidence the
profile carried the effect. It reaches **+12.10 and +18.47**, roughly double the crosscoder
itself. `txc_flat` is large and negative at positive doses and large and positive at negative
ones, so a positive-only grid recorded the negative branch and read a **sign** as an
**inversion**. Since the sign of a steering vector is a free parameter, the honest reading is
the reverse of the published one: `txc_flat` is a better constant write than the SAE's, and the
order task is steerable by a constant write.

Two lessons generalise. **A one-sided dose grid cannot distinguish a directional effect from a
magnitude artefact**, and the two failure modes point opposite ways — an arm positive at *both*
extremes is a second-order artefact, while an arm effective only at negative doses is genuinely
directional and invisible to a positive-only sweep. Both occurred here. And **selecting each arm
at its own best dose is not neutral**: it picks each arm's saturation point, which is exactly
where the linear reasoning that justifies every ratio below stops applying.

### 1. A per-token dictionary's write is rank-1, whatever its temporal machinery

Read from the decoders rather than argued: an SAE latent has one direction, and the tSAE's
attention lives entirely in its encoder while its decoder holds one direction per latent with no
position axis. Scaling a latent by its own activation — what practitioners do, and what the tSAE
does automatically — varies the *coefficient* across positions but never the direction. So a
per-token dictionary, steered well, reaches **any rank-1 write**, and the architectures can
differ only where the intervention needs genuinely different directions at different positions.

### 2. The rank of a task's optimal write is bounded by its attribute count

If activations decompose over semantic attributes, the difference slab factors as `P = S·U` —
schedules in `S`, directions in `U` — so `rank(P) ≤ A`, the number of attributes whose positional
pattern differs between conditions. **Schedule complexity lives inside `S`'s columns and cannot
raise its rank.**

This is why the search is structurally hard rather than unlucky. Language tasks almost always
manipulate one attribute — formality, intensity, refusal, which instruction applies — so however
intricate the time-course, the required write factors into one direction times a schedule. The
result arrived three ways: registered in advance, derived independently from the measurements,
and proved. It predicts the phase ladder is rank-1 at every rung (two sentence pools are one
attribute; measured `r1` 0.921 → 0.970) and an `m`-block rotation is rank `m − 1`.

The design rule that follows: **an expressiveness result needs a task where two or more distinct
attributes move in different directions at different positions.** The cheapest natural source is
content plus its own carried state, since a maintained state's schedule is the running integral
of the content's, and an integral is never proportional to its integrand.

### 3. Two numbers screen a task before any dictionary is trained

From the slab of optimal writes: `c`, the share reachable by a constant write, and `r1`, the
share reachable by any rank-1 write. Together they say whether a task can separate the
architectures at all, at a cost of one backward pass per document.

⚠ **The ordering claim is currently supported by two points, not seven, and must not ship in
its earlier form.** A table circulated earlier tonight showed `c` ordering every task from the
phase ladder through recency to the order task. Those values were computed from the
**difference-of-means** slab, which this sprint measured to be nearly orthogonal to the
gradient of the metric being reported — `cos` of +0.095 on order, +0.044 on recency, −0.037 on
evidence, +0.003 on a rotation task. A `c` computed from it is therefore not the constant share
of anything the experiment measures, and on the full set it does not order the outcomes: the
one-switch phase cell sits at `c_dom = 0.040`, essentially recency's value, while a constant
write beats the crosscoder there by 19 points.

Where `c` is computed from the **metric gradient**, it does separate the two cases available:

| task | `c` from gradient | `c` from difference-of-means | outcome |
| --- | --- | --- | --- |
| order | **0.241** | 0.039 | constant write wins |
| recency | **0.034** | 0.039 | crosscoder wins, z = 18 |

The two tasks have **near-identical** difference-of-means values and gradient values differing
by 7×. That is a clean argument for measuring on the gradient, and it is also the reason the
gate cannot yet be asserted: four ladder cells with gradient-based `c` are running, and until
they land the honest organising sentence is the empirical one —

> **the crosscoder wins under a metric that cancels constant writes, and loses under one that
> does not**

— which is the observed pattern (probe-mode tasks win, ordering-mode tasks do not) stated
without the causal claim about `c` that the data do not yet support.

The same DC-component account **retro-predicts eight of this project's own executed
experiments** — two failed language demonstrations and six successes — with no training.
Passphrase verification fails because its steering target is a validity *state*, a constant
write's natural shape, despite maximally position-dependent inputs; ordered generation fails
because a shared contextual mode is a constant write by definition; the six trajectory tasks
succeed because multiset-matched foils leave no DC component, and their broadcast arms measure
at or slightly below zero exactly as that predicts.

### 4. `c` is a property of the task *and the metric*

The same task screens differently under different metrics. An ordering metric
(`logP(A) − logP(B)`) cancels *content* when multisets match but leaves *context* exposed — the
residue a constant write rode. A difference-of-differences metric additionally cancels anything
pushing both classes the same way, driving `c` toward zero by construction. That is why the
constructed ladders reversed and the real-behaviour tasks did not.

One line: **use a difference-of-differences metric and a symmetric dose grid.** The first removes
a component the ordering metric leaves exposed; the second is the only thing that reveals which
kind of effect an arm has.

### 5. The surviving win is discovery, not expressiveness — and the ordering is measured

Instruction recency, completed configuration, every arm at matched injected norm:

| arm | Δ margin | |
| --- | --- | --- |
| `rank1_best` | +8.55 ± 0.27 | rank-1 truncation of the difference-of-means slab |
| `dom_slab` | +8.20 ± 0.22 | supervised reference |
| `sae_schedule` | +7.86 ± 0.25 | the SAE's own direction on its best schedule |
| **`txc_slab`** | **+6.48 ± 0.15** | the crosscoder |
| `tsae_broadcast` | +3.65 ± 0.14 | |
| `sae_broadcast` | +2.60 ± 0.15 | a per-token dictionary as actually deployed |
| `random_broadcast` | +1.81 ± 0.16 | |
| `txc_flat` | +1.42 ± 0.14 | profile removed |
| `random_slab` | +1.39 ± 0.07 | |
| `txc_profile_random` | +0.00 ± 0.04 | profile kept, directions randomised |

Three levels, each gap separately significant. **The crosscoder beats deployed practice by
2.5×** — +6.48 against +2.60, z = 18.3 — which is the honest, useful win. **A per-token
dictionary handed a schedule beats the crosscoder** — +7.86 and +8.55, z = 4.7 and 6.7 — so the
write was never out of reach. The crosscoder reaches **76% of the rank-1 ceiling**.

So its genuine claim is that it *found* a schedule unsupervised, from reconstruction alone, that
a per-token dictionary could have executed if handed it. That is worth having, because the
schedule is exactly what a practitioner does not possess. It is a discovery claim.

**A registered architectural prediction lands here.** The tSAE was argued from its decoder to be
rank-1 with an *automatically supplied* schedule, so it should sit strictly between a constant
write and an optimally scheduled one. Measured: +2.60 < **+3.65** < +7.86.

**The controls hold on this task**, which is what distinguishes it from the order task where they
did not: `txc_flat` at +1.42 sits *below* a random constant direction at +1.81, and
`txc_profile_random` is +0.00 ± 0.04. Neither the directions without the profile nor the profile
without the directions carries any of the effect.

⚠ The gradient-based arms exist only at smoke scale, so `rank1_best` above is the rank-1
truncation of the *difference-of-means* slab, and the `sqrt(r1)` rank law is not tested against
it — that law requires the gradient slab, and `cos(P_dom, Ḡ) = 0.044` says the two are not
proxies for each other.

### 6. The scope limit, and what the crosscoder is *not* doing

The advantage requires the factor to sit at **consistent positions across documents**. A
dictionary latent is one fixed write reused everywhere, so any fixed-write arm is bounded by the
*mean* slab; when positions vary, the per-document slab keeps its shape but slides, and the mean
is a broad ramp rather than a sharp template. Randomising the instruction positions leaves the
crosscoder retaining 10% of its effect against a fixed write's 67% — a limit on the whole
intervention class, with a crosscoder discovery gap on top.

**It is not solving the task by locating the instructions.** The supervised rank-1 write puts 97%
of its mass on the two instruction segments; the crosscoder's profile is nearly flat, its two
largest entries at positions 10 and 1 rather than 9 and 2 — and it still reaches 76% of the
supervised effect. The narrower and more interesting claim is that **there is more than one way
to move this metric, and the crosscoder found a different one from the supervised write.** That
`cos(P_dom, Ḡ) = 0.044` — the supervised and gradient routes are themselves nearly unrelated —
makes a third, differently-shaped solution unsurprising rather than anomalous.

## What was not achieved

**No expressiveness win was found.** Every measured task has a rank-1 optimal write, and on the
one task with measurable rank-2 structure the crosscoder did not approach the rank-1 ceiling
either — so the gap there is discovery, not reach. Finding 2 explains why this is structurally
hard rather than a matter of not having looked in the right place, and states the condition that
would have to hold instead.

## Limits

Two surviving tasks, both from the same rank-2 family, both designed by the same agent; one
model, one layer, one dictionary size. **Four separate results in this sprint moved materially
with learning rate or step count**, which is a caveat on every number here. The dictionaries are
trained on the same content they are asked to steer, so the current claim is "steers the ordering
of content it was trained on" until the held-out split lands. The rotation ladder, the gradient
rank arms and that split were all still running when this was written.

**The framework is in better shape than the empirical base**, and the honest reading is that the
criterion is the deliverable and the two surviving wins are its first test rather than its
confirmation.
