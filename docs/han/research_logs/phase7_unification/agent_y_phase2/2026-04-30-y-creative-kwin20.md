---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Hail Mary creative #2 — txc_bare_antidead T=5, k_win=20 (true k_win-matched)

> **Headline**: True k_win-matched cell (k_pos_avg=4 — much sparser
> per-position than k_pos=20 cells) achieves single-seed per-position
> peak success **1.167** at coh ≥ 1.5, **+0.067 above T-SAE k=20 anchor**.
> Largest right-edge → per-position boost in the matrix (+0.334).

### Cell trained

| field | value |
|---|---|
| arch_id | `txc_bare_antidead_t5_kwin20` |
| src_class | `TXCBareAntidead` |
| T | 5 |
| k_win | **20** (matches T-SAE k=20 per-window) |
| k_pos_avg | **4** (= k_win / T; per-position cap is much sparser) |
| seed | 42 |
| init | random |
| training | TrainCfg b=4096, lr=3e-4, plateau=0.02, min_steps=3000 |

This is the "true k_win-matched" definition of matched-sparsity: every
T=5 window has exactly 20 features active, identical to T-SAE k=20's
per-token 20 features. Per-position avg is 4 — 5× sparser than
k_pos=20 cells.

Training wall ~40 min on A40. Faster than k_pos=20 cells because
TopK on a much smaller k.

### Strength curves

#### Right-edge protocol

| s_abs | success | coh |
|---|---|---|
| 16.2 | 0.233 | 2.933 |
| 32.4 | 0.367 | 2.800 |
| 64.9 | 0.433 | 2.333 |
| **162.2** | **0.833** | **1.667** ← peak under coh ≥ 1.5 |
| 324.3 | 1.500 | 1.267 ← unconstrained peak |
| 648.6 | 0.600 | 1.067 |
| 1621.6 | 0.200 | 0.833 |

Peak at coh ≥ 1.5: **0.833** (Δ=−0.267 → exactly at TIE threshold)

#### Per-position protocol

| s_abs | success | coh |
|---|---|---|
| 16.2 | 0.333 | 2.900 |
| 32.4 | 0.433 | 2.500 |
| 64.9 | 0.467 | 1.767 |
| **162.2** | **1.167** | **1.567** ← peak under coh ≥ 1.5 ⭐ |
| 324.3 | 1.667 | 1.200 ← unconstrained peak |
| 648.6 | 0.655 | 0.931 |
| 1621.6 | 0.267 | 0.800 |

Peak at coh ≥ 1.5: **1.167** (Δ=**+0.067** → TIE, slightly above anchor!)

### Comparison vs other T=5 cells (single-seed at seed=42)

| arch | right-edge | per-position | per-pos boost |
|---|---|---|---|
| T=5 bare k_pos=20 | 0.700 | 0.833 | +0.133 |
| T=5 H8 shifts=(T,) | 0.900 | 1.067 | +0.167 |
| T=5 matryoshka multiscale | 0.633 | 0.933 | +0.300 |
| **T=5 bare k_win=20 (this)** | **0.833** | **1.167** | **+0.334** ⭐ |

**The k_win-matched cell has the largest per-position boost in the
matrix.** And the per-position result is the highest single-seed T=5
result yet (1.167 > prior best 1.067 at H8 shifts=(T,)).

### Combined ranking under per-position (all matched-sparsity cells)

| arch | n_seeds | mean | Δ vs anchor 1.10 |
|---|---|---|---|
| **T=2 bare k_pos=20** | 3 | **1.200** | **+0.100** ⭐ |
| **T=5 bare k_win=20** (this, single seed) | 1 | **1.167** | **+0.067** ⭐ |
| T=5 H8 shifts=(T,) (single seed) | 1 | 1.067 | −0.033 |
| T=3 bare k_pos=20 (W's, single seed) | 1 | 1.000 | −0.100 |
| T=5 matryoshka multiscale (W's, single seed) | 1 | 0.933 | −0.167 |
| T=5 bare k_pos=20 | 2 | 0.783 | −0.317 |

T=5 k_win=20 single-seed sits between T=2 bare and T=5 H8 shifts=(T,)
on the per-position metric.

### Why k_win=20 matters

The brief's "matched per-token sparsity" definition (k_pos=20) was a
*per-token* match: T-SAE k=20 fires 20 features per token; TXC k_pos=20
fires up to 20 features per position. But T-SAE has 1 position per
window and TXC has T=5 positions, so TXC at k_pos=20 fires up to
T × 20 = 100 features per window — 5× denser than T-SAE per window.

**k_win=20 is the *per-window* match**: T-SAE k=20 fires 20 features
per token = per window (T=1); TXC k_win=20 fires 20 features per window
(T=5). Same total feature budget per window. The TXC just spreads
those 20 features across 5 positions (k_pos_avg=4) rather than
concentrating them at one.

This is a stricter sparsity match. And it works! Per-position 1.167
shows that even at this stricter match, the matched-sparsity TXC is
indistinguishable from T-SAE k=20 (within tie band) — possibly slightly
better.

### Mechanism (hypothesis)

At k_win=20 (k_pos_avg=4), the encoder must be selective about which
positions get features. It can't fire blanket features at every
position. So:
- Features become more "concept-anchored" — they fire at the position
  in the window where the concept is most evident, not at every
  position.
- Per-position write-back distributes the steered concept across all
  T positions, but the *learned* feature is naturally selective. So
  per-position writing of a sharp feature gives a strong, coherent
  steering signal.
- Right-edge writing of a sharp feature only steers the last position
  of the window — losing 80% of the model's context cues — hence the
  larger per-position vs right-edge gap.

### Single-seed caveat

Single-seed result. Per the σ_seeds=0.33-0.47 we observed for
right-edge at k_pos=20, the right-edge 0.833 here could swing
substantially at seed=1. Per-position is more stable
(σ_seeds=0.07-0.23 at k_pos=20). The +0.067 above-anchor claim
needs multi-seed verify before being locked.

### Files

- Ckpt: `results/ckpts/txc_bare_antidead_t5_kwin20__seed42.pt`
- Training log: `results/training_logs/txc_bare_antidead_t5_kwin20__seed42.json`
- Right-edge: `results/case_studies/steering_paper_normalised/txc_bare_antidead_t5_kwin20/`
- Per-position: `results/case_studies/steering_paper_window_perposition/txc_bare_antidead_t5_kwin20/`
- Diagnostics: `results/case_studies/diagnostics_kwin20/z_orig_magnitudes.json`
