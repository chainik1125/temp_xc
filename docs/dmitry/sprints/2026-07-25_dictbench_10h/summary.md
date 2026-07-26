---
author: Dmitry Manning-Coe
date: 2026-07-25
tags:
  - results
  - in-progress
---

## What this sprint set out to do

The previous sprint produced semi-synthetic settings where windowed steering beats
per-token steering, and the honest objection to it was that nothing had been benchmarked
against a trained sparse autoencoder — the per-token arm was a broadcast proxy, and the
steering template was rank-1. This sprint trains both dictionaries on the same activations
and compares them properly, then follows the comparison to wherever it breaks.

It broke immediately, three times, and the sprint became about the benchmark rather than
the result. That turned out to be worth more.

*(Draft in progress — the attribution experiment and the T-sweep are still running. The
headline below is written against evidence in hand and is revised as those land.)*

## Executive summary

**1. Nominal k is not a budget the crosscoder spends, and matching an SAE to a crosscoder
on nominal k compares two very different amounts of capacity.**

TopK selects k latents and ReLU then zeroes any whose pre-activation was negative, so
realised sparsity is `min(k, #{pre > 0})`. For the SAE the first term binds at every budget
tested — at k=1 it has 2022 positive pre-activations to choose from, so asking for 128
gets 126.5. For the crosscoder the second term binds almost immediately, and the positive
pre-activation count *falls* as k rises: 53 → 36 → 22 → 17 → 16 as nominal k goes
12 → 240. At the configuration this project treats as standard, kper=41, the crosscoder
nominally claims 41 coefficients per segment and realises about 1.5.

**2. The comparison axis that works is realised coefficients per segment.**

It is the only quantity both architectures spend in the same units. A crosscoder's k counts
latents per *window* and an SAE's counts latents per *token*, so the two are not
comparable before dividing by T — and neither is comparable to a nominal budget the model
declines to spend. Every number in this sprint is reported on that axis.

**3. A large part of the crosscoder's shortfall is a two-line implementation asymmetry, not
a property of window codes.**

`TopKSAE.__init__` normalises its decoder; `TemporalCrosscoder.__init__` does not, so its
atoms begin at norm `sqrt(T·d_in/d_sae)` ≈ 2.12 and are rescaled only after the first
optimiser step. Normalising at init, with nothing else changed, moves the kper=4 crosscoder
from 22 positive pre-activations, 1.7 coefficients per segment and 57% of its selection
discarded, to 99 positive pre-activations, 3.98 of its nominal 4 spent, and 0.6% discarded.
FVU falls from 0.865 to 0.670.

**4. The remaining gap is not yet shown to be architectural, and I am not claiming it is.**

Even repaired, the crosscoder sits at FVU 0.670 where the SAE at the same 4 coefficients
per segment sits at 0.1175. That gap points the same way as before the fix, but a
crosscoder can in principle represent the SAE's solution exactly — atoms that are non-zero
at a single time position reduce a shared window code to a per-token code, and the shared
TopK is if anything *more* flexible than the SAE's fixed per-position allocation, since it
can spend its coefficients unevenly across the window. So there is no representational
barrier that explains 5.7×, and the honest reading is that some of what is left is also
optimisation. The T-sweep decides it: at T=1 a crosscoder **is** a TopK SAE, so a residual
gap at T=1 is implementation by construction, and a gap that grows with T is the real cost
of sharing one code across a window.

## What was measured, and where it lives

| question | script | result file |
|---|---|---|
| plumbing: cache → train both → steer from a decoder row | `smoke_modal.py` | `smoke.json` |
| m-sweep, protocols A/B, matched token budget | `bench_modal.py` | `bench.json` |
| random / shuffled / full-support nulls | `controls_modal.py` | `controls.json` |
| frozen-arm shuffle, 24 draws | `frozenshuf_modal.py` | `frozen_shuffle.json` |
| capacity sweep kper ∈ {41,100,200,341} | `health_modal.py`, `interp_modal.py` | `health.json`, `interp.json` |
| SAE k ∈ {1..128} × TXC kper ∈ {1..41} × lr | `frontier_modal.py` | `frontier.json` |
| i.i.d. vs structured corpus, 2×2 | `structured_modal.py` | `structured.json` |
| realised L0 = min(k, #{pre>0}) | `mechanism_modal.py` | — |
| centring / tied-init / aux-loss ablation | `centering_modal.py` | `centering.json` |
| init-normalisation × lr × k factorial | `initnorm_modal.py` | `initnorm.json` |
| window length T, with T=1 as an SAE control | `tsweep_modal.py` | `tsweep.json` |

All under `experiments/temporal_screen/dict_bench/` and `results/dict_bench/`. Figures from
`scripts/plot_frontier.py` and `scripts/plot_dictbench.py` into
`plots/2026-07-25_dictbench/`. The full research log, including the dead ends and the
retractions, is `log.md` in this folder.

## Three ways this benchmark broke, and what each cost

**Capacity.** The first comparison matched the two architectures on nominal k, which as
above is not a quantity the crosscoder spends. Caught by logging realised L0 alongside
nominal k.

**Data.** Every training corpus drew its per-segment label i.i.d.
(`lab = [rng.randint(0, 1) for _ in range(k_seg)]`), so the windows contained no temporal
structure for a window code to exploit. Any reading of "the crosscoder underperforms" from
those runs was measuring an absent signal. Fixed by the run-length corpus in
`structured_modal.py`, which also supplies the positive control the earlier runs lacked:
window-AUC reads exactly 0.500 on the i.i.d. corpus for all arms.

**Measurement.** Single-latent AUC was computed per *segment* against a window code holding
one shared latent vector for twelve independently-labelled segments, where chance is the
correct answer. The informative version asks a window-level question of a window code.
