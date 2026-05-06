---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Hail Mary creative #1 — H8 multidistance at T=5, k_pos=20, shifts=(T,)

> **Headline**: Han's specific suggestion (multi-distance contrastive
> shift = window length T) lifts the T=5 matched-sparsity cell from
> a 2-seed-mean LOSS (0.783) to a single-seed near-anchor TIE (1.067).
> **Best T=5 cell so far at matched per-token sparsity.**

### Cell trained

| field | value |
|---|---|
| arch_id | `txc_h8_t5_kpos20_shifts5` |
| src_class | `TXCBareMultiDistanceContrastiveAntidead` |
| T | 5 |
| k_pos | 20 |
| k_win | 100 |
| shifts | `(5,)` (single contrastive distance = T) |
| matryoshka_h | 0.2 × d_sae (canonical) |
| seed | 42 |
| init | random |
| training | TrainCfg b=4096, lr=3e-4, plateau=0.02, min_steps=3000 |

H8 stack (anti-dead + matryoshka H/L + multi-distance InfoNCE) with
custom `shifts=(5,)`. Training wall ~70 min on A40 (heavier than bare
antidead due to matryoshka heads + InfoNCE + AuxK).

### Strength curves

#### Right-edge protocol

| s_abs | success | coh |
|---|---|---|
| 14.8 | 0.233 | 2.800 |
| 29.6 | 0.333 | 2.600 |
| 59.2 | 0.467 | 2.600 |
| **148.0** | **0.900** | **1.867** ← peak under coh ≥ 1.5 |
| 296.1 | 1.200 | 1.367 ← unconstrained peak (coh below threshold) |
| 592.1 | 0.433 | 0.900 |
| 1480.3 | 0.267 | 0.933 |

Peak at coh ≥ 1.5: **0.900** (Δ vs anchor 1.10 = **−0.200** → TIE).

#### Per-position protocol

| s_abs | success | coh |
|---|---|---|
| 14.8 | 0.267 | 2.667 |
| 29.6 | 0.333 | 2.433 |
| 59.2 | 0.533 | 2.133 |
| **148.0** | **1.067** | **1.967** ← peak under coh ≥ 1.5 ⭐ |
| 296.1 | 1.367 | 1.167 ← unconstrained peak |
| 592.1 | 0.467 | 0.967 |
| 1480.3 | 0.172 | 0.966 |

Peak at coh ≥ 1.5: **1.067** (Δ vs anchor 1.10 = **−0.033** → near-TIE).

### Comparison to other T=5 cells (single-seed at k_pos=20)

| arch | right-edge | per-position | per-pos boost |
|---|---|---|---|
| T=5 bare antidead | 0.700 | 0.833 | +0.133 |
| **T=5 H8 shifts=(T,)** | **0.900** | **1.067** | **+0.167** |
| T=5 matryoshka multiscale (W's cell E) | 0.633 | 0.933 | +0.300 |

**The H8 shifts=(T,) cell is the best single-seed T=5 cell under both
protocols.** It outperforms bare antidead by +0.20 (right-edge) and
+0.234 (per-position). vs cell E matryoshka, this cell is +0.27
right-edge and +0.13 per-position.

### Comparison to T=2 cells (multi-seed)

| arch | right-edge | per-position |
|---|---|---|
| T=2 bare antidead (3-seed mean) | 1.122 | **1.200** ⭐ |
| T=5 H8 shifts=(T,) (1 seed) | 0.900 | 1.067 |
| T=2 bare antidead (best single seed) | 1.300 (sd1, sd2) | 1.367 (sd2) |

T=2 bare per-position multi-seed mean still beats T=5 H8 shifts=(T,).
But the T=5 cell is now in the same neighborhood. Whether it beats
T=2 multi-seed needs a multi-seed verify of the H8 shifts=(T,) cell.

### Why shifts=(T,) helps at sparse k_pos

The canonical H8 multidistance uses auto-scaled shifts `(1, T//4, T//2)`
which at T=5 dedupe to `(1, 2)` — short-range temporal consistency.
At sparse k_pos=20 (k_win=100), the window encoder integrates over T=5
positions but only fires 100 features per window — features are
spread thin and become polysemantic (per Y's earlier finding: 24/30
distinct picked features at T=5, vs T-SAE k=20's 28/30).

**Shifts=(T,) constrains the InfoNCE loss to the longest possible
distance within the window** — features at position t and position t+T
should look similar. This trains the encoder to extract features that
are consistent across the *entire* T-window, not just adjacent
positions. The result is sharper, less polysemantic features.

Hypothesis:
- Short shifts (1, 2) reward short-range temporal smoothness — features
  that fire at consecutive positions look similar.
- Long shift (T) rewards LONG-range coherence — features should look
  similar at the start AND end of the window.
- At sparse k_pos, long-range coherence is more constraining (fewer
  features must encode the whole window's structure) — pushes features
  toward genuine multi-token concept structure rather than per-position
  noise.

This is a paper-worthy mechanism if multi-seed verifies.

### Outcome (called against pre-registered ±0.27 threshold)

**METRIC B (peak coh ≥ 1.5)**:
- right-edge: 0.900, Δ=−0.200, **TIE**
- per-position: 1.067, Δ=−0.033, **TIE** (near-anchor)

**METRIC A (unconstrained)**:
- right-edge: 1.200 at coh=1.367
- per-position: 1.367 at coh=1.167
- Both still LOSS vs anchor 1.80

### Pre-registered next step

This is single-seed. Per the brief's TIE rule, multi-seed disambiguate
needed. **Recommend train H8 shifts=(T,) at seed=1, regrade both
protocols.** ~75 min train + 30 min eval. Could combine with creative
cell #2's (k_win=20) findings into a follow-up sweep.

### Files

- Ckpt: `results/ckpts/txc_h8_t5_kpos20_shifts5__seed42.pt`
- Training log: `results/training_logs/txc_h8_t5_kpos20_shifts5__seed42.json`
- Right-edge: `results/case_studies/steering_paper_normalised/txc_h8_t5_kpos20_shifts5/`
- Per-position: `results/case_studies/steering_paper_window_perposition/txc_h8_t5_kpos20_shifts5/`
- Trainer: `experiments/phase7_unification/case_studies/train_kpos20_h8_shifts.py`
