---
author: Dmitry Manning-Coe
date: 2026-07-25
tags:
  - results
  - in-progress
---

## What this sprint set out to do

The previous sprint produced semi-synthetic settings where windowed steering beats
per-token steering, and the fair objection to it was that nothing had been benchmarked
against a trained sparse autoencoder — the per-token arm was a broadcast proxy and the
steering template was rank-1. This sprint trains both dictionaries on the same activations,
compares them properly, and follows the comparison to wherever it breaks.

It broke three times, and the sprint became mostly about the benchmark. That turned out to
be the more valuable outcome, because two of the three breakages affect results this
project has already published internally.

*(Draft — the T-sweep and the healthy-dictionary replication are still running. Sections
marked **open** are the ones those decide.)*

## Executive summary

**1. A temporal crosscoder can silently train into a state where almost none of its
dictionary is usable, while its configuration reports nothing wrong.**

Realised sparsity is `min(k, #{pre > 0})`: TopK selects k latents and ReLU then zeroes any
whose pre-activation was negative. That identity held exactly in every run measured. Which
of the two terms binds, however, is not a property of the architecture — it is set by the
optimiser. At lr=1e-3 the crosscoder's positive-pre-activation count collapses to ~20 and
realised capacity peaks at 2.4 coefficients per segment and then *falls* as k rises, with
ReLU discarding up to 100% of the TopK selection. At lr=3e-4, with nothing else changed, the
same model at the same nominal budgets spends 3.97 of 4 and 7.70 of 8.

| nominal kper | realised coeff/segment @ lr=1e-3 | @ lr=3e-4 |
|---|---|---|
| 4 | 2.41 | **3.97** |
| 8 | 2.04 | **7.70** |

A 3× change in learning rate moves what the model actually spends by 3.8×. Nothing in the
nominal configuration distinguishes the two runs.

**2. The comparison axis that works is realised coefficients per segment — and it has to be
measured, not assumed.**

It is the only quantity both architectures spend in the same units: a crosscoder's k counts
latents per *window*, an SAE's counts latents per *token*, and neither is comparable to a
nominal budget a model may decline to spend. Matching an SAE to a crosscoder on nominal k
is unsafe — not because crosscoders inherently cannot spend k, but because whether they do
is an unstable function of training that the configuration does not reveal.

**3. The crosscoder's learned temporal arrangement carries steering signal that a permutation
destroys.**

Taking a trained latent, permuting its decoder rows in time, and **not** refitting anything
downstream: intact fidelity is +0.242 against 24 shuffled draws at +0.002 ± 0.103, outside
the entire observed range of the null (100th percentile, one-tailed p ≤ 0.04). The effect
holds in the same direction at two refit budgets. Earlier controls that found
`shuffled ≈ intact` refit the coefficients after shuffling, which lets the fit repair the
permutation — arrangement can only be measured where it is not re-chosen.

**4. Open: whether any remaining SAE-vs-crosscoder gap is architectural.**

At matched coefficients per segment the SAE still reconstructs far better (FVU 0.118 against
0.706 at 4 coefficients per segment). But a crosscoder can represent the SAE's solution
exactly — atoms non-zero at a single time position reduce a window code to a per-token code,
and the shared TopK is if anything *more* flexible than the SAE's fixed per-position
allocation because it can spend unevenly across the window. So there is no representational
barrier that explains the gap, and after watching a single learning rate change realised
capacity by 3.8×, the prior that the remainder is also optimisation is strong. The T-sweep
settles it: at T=1 a crosscoder **is** a TopK SAE, so a gap at T=1 is implementation by
construction, and only a gap that grows with T is the real cost of sharing a code.

## Corrections made during the sprint

Recorded because several were mine and the sprint's value depends on them not standing.

| claimed | status | what actually holds |
|---|---|---|
| `b_enc` goes strongly negative and gates the dictionary | **refuted** | bias is −0.021 → −0.024 across a 20× range of k; far too small to gate |
| the crosscoder's missing input centering starves it | **refuted** | centering changes FVU 0.6696 → 0.6673 |
| decoder normalisation at init is the two-line defect | **refuted** | 3.97 coeff/seg without it vs 3.98 with, at matched lr |
| raising k destroys realised capacity | **withdrawn** | true at lr=1e-3 only; at 3e-4 capacity rises with k |
| 28× capacity overstatement at kper=41 | **withdrawn** | a fact about one training run, not the architecture |
| destroying temporal arrangement does not hurt steering | **reversed** | the earlier control refit after shuffling; frozen, it hurts |

Tied init, decoder-normalisation-at-init and a standard auxiliary dead-latent loss were all
tested and all land within noise of each other once the learning rate is right.

## Three ways this benchmark broke

**Capacity.** The first comparison matched on nominal k, which is not a quantity the
crosscoder reliably spends. Caught by logging realised L0 next to nominal k — a two-line
change that should be standard in every run.

**Data.** Every training corpus drew its per-segment label i.i.d.
(`lab = [rng.randint(0, 1) for _ in range(k_seg)]`), so the windows held no temporal
structure for a window code to exploit, and "the crosscoder underperforms" was measuring an
absent signal. Fixed by the run-length corpus in `structured_modal.py`, which also supplies
the positive control the earlier runs lacked: window-AUC reads exactly 0.500 on the i.i.d.
corpus for every arm.

**Measurement.** Single-latent AUC was computed per *segment* against a window code holding
one shared latent vector for twelve independently-labelled segments, where chance is the
correct answer and a high value would have been the surprise. The informative version asks a
window-level question: on the structured corpus, SAE 0.747 against crosscoder 0.619.

## What was run, and where it lives

| question | script | result |
|---|---|---|
| plumbing: cache → train both → steer from a decoder row | `smoke_modal.py` | `smoke.json` |
| m-sweep, protocols A/B, matched token budget | `bench_modal.py` | `bench.json` |
| random / shuffled / full-support nulls | `controls_modal.py` | `controls.json` |
| a-orthogonal profiles, frequency-matched nulls, fitted-once ceiling | `frozen2_modal.py` | `frozen2.json` |
| **frozen-arm shuffle, 24 draws, no refit** | `frozenshuf_modal.py` | `frozen_shuffle.json` |
| capacity sweep kper ∈ {41,100,200,341} | `health_modal.py`, `interp_modal.py` | `health.json`, `interp.json` |
| **SAE k ∈ {1..128} × TXC kper ∈ {1..41} × lr** | `frontier_modal.py` | `frontier.json` |
| i.i.d. vs structured corpus, 2×2 | `structured_modal.py` | `structured.json` |
| realised L0 = min(k, #{pre>0}) | `mechanism_modal.py` | — |
| centering / tied-init / aux-loss ablation | `centering_modal.py` | `centering.json` |
| **window length T, with T=1 as an SAE control** | `tsweep_modal.py` | `tsweep.json` |

Code under `experiments/temporal_screen/dict_bench/`, results under `results/dict_bench/`,
figures from `scripts/plot_frontier.py` and `scripts/plot_tsweep.py` into
`plots/2026-07-25_dictbench/`. The full log, including every dead end and retraction, is
`log.md` in this folder.

## What I would do next

1. **Re-check realised L0 on every existing crosscoder result in this project.** The
   measurement is two lines and the failure is silent. Any comparison whose crosscoder was
   trained at lr=1e-3 is suspect until its realised spend is confirmed.
2. **Finish the T-sweep.** It cleanly separates implementation from architecture, and no
   architecture claim should be made before it lands.
3. **Sweep lr jointly with k rather than fixing it.** The frontier here varies lr over two
   values and that was already enough to overturn the sprint's first conclusion.
