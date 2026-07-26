---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - proposal
  - in-progress
---

## Sprint window

10h **wall-clock**: 2026-07-25 22:33 PDT → 2026-07-26 08:33 PDT. Check elapsed time
roughly hourly against the real clock.

Branch: `dmitry-txcwins-10h`. Sprint dir: `docs/dmitry/sprints/2026-07-26_txcwins_10h/`.

## Context

The preceding sprint (`docs/dmitry/sprints/2026-07-25_dictbench_10h/`) established one
setting where a temporal crosscoder genuinely beats a TopK SAE, and — more usefully — a
*procedure* for finding such settings and a diagnosis of why most candidates fail.

What it found:

- **A crosscoder advantage exists, and it is about intervention rather than decoding.** On
  a task whose label is pure ordering, a per-token SAE latent *reads* the factor at AUC
  0.998 and the crosscoder at 0.791, but steering reverses: crosscoder +11.29 ± 0.64
  against the SAE's +1.24 ± 0.56 at matched injected norm (z = 11.8, n = 200).
- **Three controls pin it to the temporal profile.** Time-averaging the crosscoder's own
  slab inverts the effect to −8.02; a random temporal profile is flat at −0.28 (z = 14.3);
  permuting the profile's rows drops fidelity to a 24-draw null. A *random* constant
  direction (+3.01) beats the SAE's learned one (+1.24), so a constant write on that task
  is indistinguishable from noise.
- **Why reading comparisons never favour a window code.** A causal transformer has already
  written its history into every token, so pooled per-token codes recover temporal
  structure without representing anything window-level. A task built specifically to defeat
  per-token codes still gave the SAE AUC 1.000.
- **A geometric ceiling in synthetic data.** A per-token dictionary recovers an extent-L
  feature at exactly `‖largest contiguous T-chunk of p‖/‖p‖` — measured to three decimals —
  which no training can beat.
- **Methodology now fixed:** realised coefficients per segment as the comparison axis (never
  nominal k), stride-1 windowing, and `batchtopk` without ReLU as the crosscoder default
  (shipped to `src/bench/architectures/crosscoder.py` with 11 tests).

The main open objection: the winning task is a construct chosen to isolate the mechanism,
not a behaviour anyone cares about. Nothing yet connects "carried purely by ordering" to a
real steering target.

## Goal

**Find further tasks — ideally ones that correspond to real, documented model behaviours —
where the temporal crosscoder beats a TopK SAE and a tSAE.**

Repeat the procedure that worked:

1. Identify a property that structurally favours a window code (the last one was: the factor
   is invariant to any permutation-symmetric readout).
2. Build a task with that property and matched foils, so generic effects cancel.
3. Test **reading and steering separately** — they came apart last time and that was the
   result.
4. Run the controls that can kill it: time-averaged profile, random profile, random
   direction, and a supervised ceiling.

A task counts as a win when the crosscoder beats both baselines at matched realised
coefficients per segment and matched injected norm, with the temporal-profile controls
holding.

## Agent team

- **theory** — enumerate candidate task properties that should favour a window code, state
  each as a falsifiable prediction with the control that would refute it, and rank by
  (probability of a real effect) × (relevance to a behaviour anyone cares about).
- **implement** — build and run the tasks on Modal, reusing
  `experiments/temporal_screen/dict_bench/steer_order_modal.py` as the harness template.
- **review** — continuously scan the literature, read the most relevant papers, summarise
  them, and maintain a catalogue of candidate behaviours and model organisms that could be
  tested TXC vs SAE vs tSAE. Output: `literature_catalogue.md`, updated throughout.

## Key files

- Harness template: `experiments/temporal_screen/dict_bench/steer_order_modal.py`
- Architectures: `src/bench/architectures/{crosscoder,topk_sae,stacked_sae}.py`,
  `temporal_crosscoders/han_tsae/`
- Prior sprint log and summary: `docs/dmitry/sprints/2026-07-25_dictbench_10h/`
- Prior results: `results/dict_bench/`

## Carried-over debts from the last sprint

1. **The tSAE arm never produced usable numbers.** At the repo's documented `l1_coef=1e-3`
   the code is dense (2989/4096 latents active, alive 0.999) and a 100× sweep moved realised
   L0 by 0.3%. The `lam = 1/(4·d_in)` scaling means `l1` needs to be ~1–10 here. Calibrate it
   before using it as a baseline.
2. **The tSAE identification is unresolved.** This repo's `tsae_paper` is attention-based
   (ReLU + L1 on the TemporalSAE class); the description given was an InfoNCE penalty over
   nearby positions with no attention. No InfoNCE tSAE exists in this repo. Resolve or run
   both.
3. **Realised L0 must be logged in every run.** Nominal k does not bind for the crosscoder
   and the failure is silent.

## How I'll judge it

I read `summary.md` first and only look at code if the write-up sounds interesting **and**
correct. Executive summary = 2–5 findings, each ideally backed by one graph with
self-explanatory axes. Reserve the final hour for writing alone.

Report negative results plainly — the last sprint's most useful output was the diagnosis of
why a whole class of experiment was doomed.

## Autonomy & compute

Fully unsupervised for 10h; route around permission blocks rather than stopping. Modal for
GPU (A10G / L4). Keep total spend modest and comparable to the last sprint.
