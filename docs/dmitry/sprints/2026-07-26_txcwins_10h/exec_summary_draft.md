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

**A number computed before any dictionary is trained orders every steering outcome in this
sprint, including the one that had to be withdrawn.** `c` — the share of the optimal write a
*constant* write can reach — is measured from 20–24 backward passes through the model, with no
dictionary involved. Across five configurations spanning both metric modes:

| task | `c` (gradient) | crosscoder | SAE | z | outcome | file |
| --- | --- | --- | --- | --- | --- | --- |
| order | 0.241 | +6.34 | +4.66 | +1.4 | no win | `order_sym_ds0.json` |
| `rotate2` | 0.163 | +2.86 | +5.27 | −1.0 | no win | `rot_m2_T.json` |
| `rotate6` | 0.102 | −0.01 | +5.92 | −7.5 | **loses** | `rot_m6_T.json` |
| `rotate12` | **0.033** | **+18.23** | +5.36 | **+9.8** | **wins** | `rot_m12_T.json` |
| recency | **0.034** | **+6.48** | +2.60 | **+18.3** | **wins** | `recency_gradsmoke.json` (c), `recency_v2.json` |

The split is clean and it falls between 0.102 and 0.034 — **every cell with `c ≥ 0.10` fails,
both cells with `c ≤ 0.034` win.** `sqrt(c)` also predicts the SAE's own level across the
rotation ladder (0.40, 0.32, 0.18 against measured +5.27, +5.92, +5.36).

**This retires the metric-mode sentence circulated earlier tonight.** "The crosscoder wins under
a metric that cancels constant writes and loses under one that does not" is refuted by
`rotate12`, which is an **ordering-mode** task — the mode that leaves constant writes exposed —
and the crosscoder wins it by z = 9.8. The property that predicts the outcome is the task's `c`,
not the metric's family. A task property beating a metric property is the stronger result, and it
is why this replaces rather than restores the earlier claim.

⚠ Two provenance notes. `c` for recency comes from a 500-step smoke run, though `c` is computed
from model gradients and does not depend on dictionary training; the outcome column is from the
completed run. An `m = 3` rung (`c` = 0.179, no win) was reported by the theory agent but its
result file was not present locally when this was written, so it is excluded from the table.

**What the earlier version got wrong.** A table circulated an hour before this one showed the
same ordering computed from the **difference-of-means** slab. Those values are near-orthogonal to
the metric gradient (`cos` = 0.057, 0.096, 0.190 across the ladder) and do **not** order the
outcomes — order and recency sit at 0.039 apart on the diff-of-means and 7× apart on the
gradient. The gate is real; the cheap proxy for it is not.

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
| `tsae_broadcast` | +3.65 ± 0.14 | **this repo's attention-based temporal SAE — not the published tSAE** |
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

**A registered architectural prediction appeared to land here and is now withdrawn as
untested.** The temporal SAE was argued from its decoder to be rank-1 with an *automatically
supplied* schedule, so it should sit strictly between a constant write and an optimally scheduled
one, and the measurement obliged: +2.60 < +3.65 < +7.86. **That arm was trained at a third of the
learning rate it wants.** At its own best recipe the attention tSAE is the *best reconstructor of
the three architectures* — FVU 0.0144 against the SAE's 0.0373 and the crosscoder's 0.0968, a
2.6× margin over the SAE — with reading AUC 1.000. Its steering number was obtained from a badly
undertrained dictionary and predicts nothing. The comparison is being rerun with each arm at its
own recipe.

⚠ **The arm measured is not the published tSAE.** `harness.py:268-274` imports `TemporalSAE`
from `temporal_crosscoders/han_tsae`, which is this repo's **attention-based** variant. The
published tSAE (Bhalla et al., ICLR 2026) is an **InfoNCE** architecture with no attention. The
rank conclusion transfers — both have one decoder direction per latent, so both are rank-1, and
the ordering prediction holds either way — but the arm's identity does not. It must be described
as "this repo's attention-based temporal SAE", never as "the tSAE". The kickoff's carried debt
on tSAE identification is therefore **resolved**, and it resolves to *we benchmarked a different
temporal SAE than the published one*.

**The controls hold on this task**, which is what distinguishes it from the order task where they
did not: `txc_flat` at +1.42 sits *below* a random constant direction at +1.81, and
`txc_profile_random` is +0.00 ± 0.04. Neither the directions without the profile nor the profile
without the directions carries any of the effect.

⚠ The gradient-based arms exist only at smoke scale, so `rank1_best` above is the rank-1
truncation of the *difference-of-means* slab, and the `sqrt(r1)` rank law is not tested against
it — that law requires the gradient slab, and `cos(P_dom, Ḡ) = 0.044` says the two are not
proxies for each other.

### 6. A single learning rate across architectures does not measure architectures

The sprint's default `lr = 3e-4` is near-optimal for the SAE and wrong for both temporal
architectures. Best FVU per arm across a 3 × 2 recipe sweep on the recency corpus, matched at 8.0
realised coefficients per segment on held-out data: **SAE 0.0373 at 3e-4, crosscoder 0.0968 at
1e-3, attention tSAE 0.0144 at 3e-3** — each arm peaking at a different recipe, spanning a 10×
range in learning rate.

This is a caveat on every cross-architecture number in both sprints, and it is the reason the
tSAE arm was reported at three different values tonight — 5× worse than the SAE, then 1.9× worse,
then 2.6× *better*. Each revision was a training-recipe artefact, not a measurement.

Two mechanical notes that follow. The crosscoder's realised coefficient spend moves with the
learning rate (10.15 at 3e-4, 8.32 at 1e-3, 8.04 at 1e-3/6000, against nominal 8), so **recipe
and budget-matching are not independent knobs**. And the crosscoder is the only one of the three
that diverges outright — FVU 0.0968 at 1e-3 against 0.3596 at 3e-3.

### 7. The scope limit, and what the crosscoder is *not* doing

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

**No expressiveness win was found, including on a design built specifically to produce one.**
The rotation ladder drives the rank-1 reachable share `r1` down to 0.177 by construction, and at
that rung the crosscoder still loses to the best rank-1 write taken from the metric's own
gradient: **+18.23 against `grad_rank1` +102.46** (z = −31.5), and against `rank1_best` +59.94
(z = −22.7). The same holds at every rung — `grad_rank1` reaches +109.98, +67.74, +102.46 while
the crosscoder reaches +2.86, −0.01, +18.23. **A rank-1 write beats the crosscoder everywhere on
a ladder designed to put the target out of rank-1 reach.**

`r1` also fails to predict its own arms: it falls 0.304 → 0.210 → 0.177 while `rank1_best` stays
flat (+50.5, +49.8, +59.9). **`r1` measures a share of the write's norm, and that share does not
determine what a rank-1 write achieves on the metric.** `c` survives this ladder; `r1` does
not. Finding 2 explains why this is structurally
hard rather than a matter of not having looked in the right place, and states the condition that
would have to hold instead.

## Limits

Two surviving tasks, both from the same rank-2 family, both designed by the same agent; one
model, one layer, one dictionary size. **Four separate results in this sprint moved materially
with learning rate or step count**, which is a caveat on every number here. The dictionaries are
trained on the same content they are asked to steer, so the current claim is "steers the ordering
of content it was trained on" until the held-out split lands. The rotation ladder, the gradient
rank arms and that split were all still running when this was written.

**The crosscoder's temporal profile is actively harmful at two of three rotation rungs.**
`txc_flat` — its own slab with the profile averaged away — reaches +10.36 against the
crosscoder's +2.86 at `rotate2` and +9.83 against −0.01 at `rotate6`, reversing only at
`rotate12` (+4.32 against +18.23). And `txc_slab` across the ladder reads +2.86, +6.43, −0.01,
+18.23 — a −0.01 followed by a +18.23 is instability, not a trend. The honest description of the
surviving win is **discovery, and unreliable discovery**.

**The framework is in better shape than the empirical base**, and the honest reading is that the
criterion is the deliverable and the two surviving wins are its first test rather than its
confirmation.
