---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-24
tags:
  - results
  - in-progress
---

## Steering a trajectory: what the width of a steering handle buys

*10-hour sprint, 2026-07-24 21:58 → 2026-07-25 07:58 PDT. Qwen-2.5-1.5B-Instruct and
Qwen-2.5-7B-Instruct, layer 14, difference-of-means directions, Modal A10G.*

DRAFT — sections marked ⏳ are awaiting runs still in flight.

## Executive summary

A steering vector added to one token position sets a **level**. Many behaviors we
actually want to control are **shapes**: answer the safe part of a request and decline
the specific step that crosses the line; bring an escalating conversation back down;
switch language for three sentences and switch back. This sprint asks what a steering
handle that spans **W** positions buys over one that acts at a single position, and
measures it.

**Finding 1 — A handle that writes one constant across W segments keeps exactly the
part of the target trajectory that is constant across W, and this is predictable to
within 0.013 with no free parameters.**

Write a profile of run length ℓ (a square wave: ℓ tense sentences, ℓ calm, …) and give
the handle a resolution limit: inside each block of W consecutive segments it may write
only a single constant. Theory says the achievable fraction of the full per-segment
effect is `R(W, ℓ) = mean_b |μ_b|`, the mean absolute value of the profile's block
means — 1 when W divides ℓ, 0 when a block straddles equal amounts of both phases.
Across 24 cells the measured ratios match to a mean absolute error of **0.013**.

- Figure: `plots/2026-07-24_trajectory_steering/phase_diagram.png`

The sharp test is that `R` is **not monotone in W**: at ℓ=2 a width-6 handle (0.35)
beats a width-4 handle (0.00), because 6 spans three runs and 4 spans two. All three
predicted zig-zags appear. A monotone curve could be explained many ways; a predicted
zig-zag could not.

**Finding 2 — The best handle width is the target's own timescale.** Counting one
control parameter per block, fidelity per knob `R·W/k` peaks at W ≈ ℓ, and the number
of parameters needed for full fidelity is `k/ℓ`. A width-1 handle always works but
costs k knobs; a width-ℓ handle costs k/ℓ and loses nothing; a wider handle collapses.

**Finding 3 — This transfers to a safety-relevant behavior: staged refusal.** ⏳
Steering the refuse/comply stance *within a single response* (mid-response safety
recovery), with target and foil built from the same sentences in different orders:
the scheduled handle moves the model's stance choice in the intended direction on
**93.8%** of slots (96.9% at higher dose) against **51.2%** for the same direction
broadcast at constant strength, and the teacher-forced margin grows from +20.7 to
+28.6 across k = 2…8 while broadcast stays pinned at zero and a random direction at
matched magnitude does nothing.

**Finding 4 — A negative result we take seriously: writing *more segments* is not the
same as writing a *wider window*.** Our first centerpiece — Δ improves as the window
of one knob grows at fixed knob budget — turned out to be a reparameterisation of how
many segments were written. Per-covered-slot effect is flat across all thirteen
conditions, and a matched-coverage contrast is null (contiguous +18.94 ± 2.40 vs
scattered +17.42 ± 2.81). We retract that framing and report the resolution-limited
experiment instead, where coverage is pinned at k by construction. ⏳ superadditivity
control pending.

## What problem this is

[to be written]

## Map of the work

[to be written]

## Limitations

[to be written]
