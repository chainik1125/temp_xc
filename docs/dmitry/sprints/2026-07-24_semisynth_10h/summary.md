---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-24
tags:
  - results
  - complete
---

## Steering a shape, not a level: what the width of a steering handle buys

*10-hour sprint, 2026-07-24 21:58 → 2026-07-25 07:58 PDT. Qwen-2.5-1.5B-Instruct and
Qwen-2.5-7B-Instruct, layer 14, difference-of-means directions, Modal A10G, ~$12.*

## Executive summary

A steering vector added at a token position sets a **level**: more refusing, calmer,
more French — everywhere at once. Several things we actually want to control are
**shapes**: answer the safe part of a request and decline the step that crosses the
line; bring an escalating exchange back down; switch language for three sentences and
switch back. This sprint measures what a steering handle that *spans* W positions buys
over one that acts at a single position, using tasks where the good and bad versions
of a response are built from **the same sentences in a different order** — so a
constant write is inert by construction and only the shape is being tested.

**1. A handle that writes one constant across W segments retains exactly the part of
the target that is constant across W — predictable with no fitted parameters.**
Give the target a run length ℓ (ℓ tense sentences, ℓ calm, repeat) and limit the
handle's resolution to one constant per block of W segments. Theory says the
achievable fraction of the full per-segment effect is the mean absolute block-mean of
the profile, `R(W, ℓ) = mean_b |μ_b|`. Of the 24 (W, ℓ) cells, 13 are fixed by
construction and carry no information — 4 where a width-1 block *is* the full template,
and 9 where the balanced profile makes every block coefficient exactly zero, so no
vector is written at all. On the **11 cells that are genuine measurements** the mean
absolute error is **0.029** (1.5B) and **0.025** (7B); on the **6 cells whose
prediction is strictly between 0 and 1** — the discriminating ones — it is **0.053**
and **0.045**, with no fitted parameters. Those 6 also separate the two candidate
budget laws: a per-slot magnitude cap predicts 0.667 where an energy-matched budget
predicts 0.816, and the measurements are 0.628 and 0.641.

The sharp part is that `R` is *not monotone in W*: at ℓ=6 a width-6 handle (1.00)
beats a width-4 handle (0.64), because 6 spans whole runs and 4 straddles them. Two
further reversals (ℓ=1 and ℓ=2) are real but weaker evidence, since there the narrower
handle is one of the structurally-silent cells rather than a measured value.

- ![phase diagram](../../../../plots/2026-07-24_trajectory_steering/phase_diagram.png)

**2. The best handle width is the target's own timescale.** Counting one control
parameter per block, fidelity per knob `R·W/k` peaks at W ≈ ℓ (panel B above), and the
number of parameters needed for full fidelity is `k/ℓ`. A width-1 handle always works
and costs k knobs; a width-ℓ handle costs `k/ℓ` and gives up nothing; wider than that
collapses. This is the honest form of "performance improves with window size": **at
matched fidelity, a wider handle costs fewer control parameters, up to the timescale
of the thing being controlled.**

**3. It transfers to a safety-relevant behaviour: staged refusal.** Steering the
refuse/comply order *within a single response* — mid-response recovery, where the
target and foil contain the same sentences reordered — the scheduled handle moves the
model's own choice the intended way on **96.9%** of slots against **51.2%** for the
same direction written at constant strength (chance 50%). A random direction at
matched magnitude does nothing at any length.

- ![stance](../../../../plots/2026-07-24_trajectory_steering/stance.png)

**4. The effect is the order, not the mass.** Permuting the schedule *inside* a block
— identical coverage, contiguity and injected norm, only the arrangement changed —
collapses the effect from +29.0 to −2.3 (W=4) and +55.3 to −1.2 (W=8). Two claims we
started with did not survive their own controls, and we report the corrected versions:
what first looked like the effect *growing* with trajectory length is bookkeeping (a
permuted foil differs in more slots as k grows; with foils that differ in exactly two
slots the curve is **flat**), and our first "window" sweep turned out to vary coverage
rather than width.

- ![controls](../../../../plots/2026-07-24_trajectory_steering/controls.png)

## What problem this is, and why it is interesting

Activation steering is usually described as pushing a model along a direction: find a
refusal direction, add α times it, get more refusal. That framing fits a **level**.
The interesting safety cases are not levels. "Refuse everything" is not the useful
refusal control; "answer the safe part and decline the specific step that crosses the
line" is. "Be calmer" is not the useful de-escalation control; "come down over the
next four turns" is. Those targets are *shapes over time*, and the good and bad
versions of them are frequently built from the same material in a different
arrangement.

That observation makes a clean experimental design available. If the target and the
foil are **permutations of one another**, then every bag-of-segments statistic is
identical between them, and a constant write — which is what a per-token steering
vector applies when it is used across a span — cannot separate them except by
accident. Anything that does separate them is acting on arrangement. That is the
design used throughout, and it is what makes the constant-write arm a genuine null
rather than a weak baseline.

The prior context is a temporal-crosscoder project whose steering claim had failed to
appear in natural language behaviours: on ordered generation (days of the week,
counting), a per-token direction *broadcast* at every position matched or beat a
per-position schedule, because those behaviours ride a shared contextual mode the
broadcast reinforces everywhere. The tasks here are built specifically to remove that
mode, and they do.

## The experiments

**The resolution family (finding 1 and 2).** Profiles are square waves of run length
ℓ ∈ {1,2,3,6} at k=12 (ℓ=4 is excluded because it leaves a DC component at k=12, which
would reintroduce the broadcastable mode the design exists to remove). Two handle
classes: the full per-segment template, and a block-constant handle that writes one
scalar per block of W. Coverage is pinned at all 12 segments in every cell, so nothing
here can be a coverage effect. Predictions come from `Δ = δW Σ_b c_b μ_b` with a
per-slot magnitude cap, giving `R = mean_b |μ_b|`; the energy-matched alternative
(`R = RMS(μ_b)`) is distinguishable at three cells and the cap form fits (0.63 vs a
predicted 0.667 where RMS would predict 0.816).

**Staged refusal (finding 3).** ~40 template-generated dual-use-flavoured requests
about the user's own property, a 12-sentence declination bank and a 12-sentence
content-free procedural bank, split into disjoint halves so the direction is fit on
one half and evaluated on the other. Prompts go through the chat template. Target and
foil place the same drawn sentences in different orders. Four arms: scheduled,
constant, single-segment, and a random direction at matched magnitude. The behavioural
metric avoids a free-text classifier entirely: at each sentence boundary the model
scores two held-out candidates (one declining, one helping), and we measure whether
steering shifts that choice in the intended direction, with the model's intrinsic
preference for one class differenced out.

**Controls (finding 4).** Fixed-Hamming foils (exactly one swap, so H=2 at every k);
per-position marginals with a superadditivity index `S(B) = Δ(B) − Σ_{t∈B} Δ_t`;
within-block schedule scrambling; contiguous versus scattered writes at matched
coverage; and an SVD of the per-position template matrix.

## What we corrected, and how we found it

Three results were retracted or rescoped during the sprint, all by controls we ran on
ourselves.

- **"Effect grows with trajectory length" → "per-slot efficacy is constant in
  length."** The teacher-forced margin sums log-probs over all k segments, and a
  permuted foil differs from the target in ~k/2 slots, so a constant per-slot effect
  produces a linear-in-k curve mechanically. With fixed-Hamming foils the curve is
  flat (+71.8, +80.8, +79.6, +78.0, +77.5 for k=2…10). Constant-in-length is still a
  real property — the handle does not decay as the trajectory lengthens, while the
  constant write stays at zero — and it is what we claim.
- **"Performance improves with window size" (first version) → coverage.** Writing m
  consecutive blocks of width W occupies one contiguous span of mW segments, so that
  grid varied coverage, not width. Δ per covered slot is flat across all thirteen
  conditions (≈22 for language, ≈7.8 for intensity), and contiguous versus scattered
  writes at matched coverage are indistinguishable (+18.94 ± 2.40 vs +17.42 ± 2.81 at
  1.5B; +16.76 vs +13.79 at 7B). The resolution family replaces it.
- **"A temporal dictionary beats a per-token one" → "a schedule beats a level."** The
  per-position template is rank-1 (σ₁ = 89% of the energy), i.e. one direction with a
  sign schedule supplied externally. Everything measured here is therefore about the
  *form of the control signal*, and the dictionary-level claim needs trained
  dictionaries, which is outside a 10-hour budget.

Two further results came out inconclusive and are reported as such. **Entrainment** —
steering only the first W sentences and asking whether the model completes a
predictable pattern unaided — showed no clean threshold (unsteered-slot accuracy for
an alternating profile ran 0.440, 0.523, 0.458, 0.367 against analytic nulls of 0.400,
0.500, 0.333, 0.500). Its main yield was methodological: with *balanced* profiles the
correct null is **0.400**, not 0.500, because a French-heavy steered prefix forces an
English-heavy tail, so a model that merely persists scores below chance; with i.i.d.
profiles the null is 0.500 and the measurements sit on it (0.470–0.528 against
0.496–0.501). **Graded amplitude control** failed its pre-registered monotonicity
gate — five urgency levels do not project onto the direction in order (L1 −5.06,
L2 −7.35, L4 +3.49, L5 +2.94) — so we report sign scheduling only. The handle is
cleanly bidirectional (+11.8 forward, −16.2 with the schedule flipped).

## Map of the work

Code, all runnable as `modal run <path>`, in
`experiments/temporal_screen/trajectory_steering/`:

| file | what it does |
| --- | --- |
| `lsweep_modal.py` | the (W, ℓ) resolution family — findings 1 and 2 |
| `stance_modal.py` | staged refusal, teacher-forced, four arms |
| `stance_gen_modal.py` | menu-constrained behavioural metric + three-way pre-check |
| `controls_modal.py` | fixed-Hamming foils, SVD rank, calibrated stance shift |
| `convex_modal.py` | per-position marginals, superadditivity, schedule scrambling |
| `entrain2_modal.py` | entrainment with analytic nulls |
| `graded_modal.py` | graded amplitude with a monotonicity gate |
| `dict_modal.py` | window-spanning vs per-token dictionaries at matched knob budget |
| `wsweep_modal.py` | the retracted coverage sweep, kept for the record |

Results in `results/temporal_screen/*.json`, figures in
`plots/2026-07-24_trajectory_steering/`, plotting scripts in `scripts/plot_*.py`.
Process notes, including every dead end in the order it happened, are in
[[log]]; the task-design theory is in [[theory]], the behaviour census in
[[real_behaviors]], and the adversarial audit that forced two of the three
retractions is in [[review_audit]].

## Limitations

- **Two models, one layer, one language pair, one intensity axis.** The law replicates
  across 1.5B and 7B at layer 14; layer and attribute generality are untested.
- **Difference-of-means directions, not trained dictionaries.** This is the largest
  gap: it bounds the claim to control-signal *form*.
- **Teacher-forced margins dominate the evidence.** The behavioural results (staged
  refusal choice-shift, language-profile generation) are smaller-n and one of the two
  generation harnesses had to be rebuilt mid-sprint after its classifier proved
  artifact-prone.
- **Doses are large.** Peaks sit at 0.35–0.5 of the mean residual norm; steered free
  generation at those doses is code-mixed rather than fluent, so the behavioural
  claims are about attribute identity per slot, not about text quality.
- **The superadditivity result is modest** (S = +3.6 ± 1.3 and +4.5 ± 1.7 at W = 4, 8;
  t ≈ 2.6–2.8) and does not follow the single-constant edge-penalty law we proposed
  for it, so we report it as a small positive effect rather than a mechanism.

## What we would do next

Train a temporal crosscoder and an L0/width-matched per-token SAE on one activation
cache and rerun the resolution family with decoder rows in place of the
difference-of-means direction — that is the one experiment that converts "a schedule
beats a level" into a statement about dictionaries. Second, sweep layers and a third
attribute to see whether `W* ≈ ℓ` holds as a general law. Third, build the multi-turn
escalation version of the stance task, where the timescale ℓ is set by the
conversation rather than by us.
