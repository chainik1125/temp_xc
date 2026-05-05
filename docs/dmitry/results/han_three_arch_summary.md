---
author: Dmitry Manning-Coe
date: 2026-05-05
tags:
  - results
  - complete
---

## Bill's three-arch sweep, re-run with Han's locked TXC recipe

Bill's three-arch synthetic AUC sweep ([[Synthetic-Benchmark-Report]],
`scripts/run_three_arch_sweep.py` on `bill-benchmarking-synthetic`) is
the experiment that produced the **+0.50 ΔAUC headline** for TXC vs
Stacked SAE at ρ = 0.9. This note re-runs the same DataConfig with the
two locked TXCs from `purified/configs/locked_archs.yaml` on
`origin/final` (`txc_base` and `txc_pro` / H8) and expands all four
arches to Han's `d_sae = 8 × d_in`.

## Setup

- DataConfig (Bill's three-arch, unchanged): `n_features = 128`,
  `d_model = 256`, `pi = 0.05`, deterministic emissions
  (`p_A = 0`, `p_B = 1`).
- ρ ∈ {0.0, 0.6, 0.9}.
- Han recipe applied to **all four arches**: `d_sae = 2048` (= 8 × d_in),
  `k_pos = 20`. Window length `T = 5` for the first three; `T_max = 10`,
  `t_sample = 5` for `txc_pro`.
- AuxK anti-dead (`auxk_alpha = 1/32`, dead threshold 1e7 tokens) on
  both `txc_base` and `txc_pro`. Per-atom decoder unit-norm and
  decoder-parallel grad removal on both. `txc_pro` adds matryoshka
  H+full reconstruction (`h_size = d_sae // 5 = 410`) and multi-distance
  InfoNCE at shifts (1, 2) with inverse-distance weighting.
- Training matches `scripts/run_han_txcs_10k.py` for direct comparability:
  10,000 steps, batch 64, Adam lr 3e-4, grad-clip 1, single seed (42).
- Run on `a40_txc_1` (NVIDIA A40, 46 GB), branch `bill-han-txc-10k`,
  script `scripts/run_han_three_arch_sweep.py`.

## Results

![AUC vs rho](../../../results/han_three_arch/auc_vs_rho.png)

Feature-recovery AUC by ρ:

| arch | ρ = 0.0 | ρ = 0.6 | ρ = 0.9 |
|---|---:|---:|---:|
| regular SAE (k = 20) | 0.975 | 0.971 | 0.968 |
| Stacked SAE (k = 20, T = 5) | 0.500 | 0.505 | 0.505 |
| **TXC-base (Han, T = 5)** | **0.989** | **0.990** | **0.990** |
| TXC-pro / H8 (Han, T_max = 10) | 0.885 | 0.896 | 0.985 |

NMSE (log scale) by ρ:

![NMSE vs rho](../../../results/han_three_arch/nmse_vs_rho.png)

| arch | ρ = 0.0 | ρ = 0.6 | ρ = 0.9 |
|---|---:|---:|---:|
| regular SAE | 1e-5 | 5e-5 | 1e-4 |
| Stacked SAE | 6e-5 | 9e-5 | 9e-5 |
| TXC-base | 1.1e-3 | 1.5e-3 | 2.5e-3 |
| TXC-pro | 11.5 | 1.25 | 1.05 |

## Three things to read off the plots

1. **Bill's TXC vs Stacked-SAE gap reproduces.** TXC-base (0.99) vs
   Stacked SAE (0.50) is **ΔAUC ≈ +0.49** across all three ρ. This is
   essentially Bill's headline number from the original three-arch
   sweep at very different (k, d_sae) — the gap is robust to the Han
   recipe swap. Stacked SAE's failure is structural: each per-position
   dictionary trained with per-position TopK at d_sae = 2048, k = 20
   has 1/T as much data per slot as the shared encoder and no
   cross-position pooling; it pins at AUC ≈ 0.5 regardless of ρ.

2. **TXC vs regular-SAE gap collapses.** At Bill's original setting
   (d_sae = 128, k = 2) regular SAE was the loser by a large margin.
   At d_sae = 2048, k = 20 the per-token TopK SAE has enough capacity
   to recover features from a single token without needing temporal
   pooling, and it matches TXC-base to within 0.02 AUC. So the
   "TXC-base big win over regular SAE" is **specific to the small-dict,
   low-k regime**; with a Han-style wide dictionary and high k it
   essentially disappears.

3. **TXC-pro / H8 trades reconstruction for temporal structure.** Its
   NMSE is three orders of magnitude worse than TXC-base across all ρ
   (1.05 vs 2.5e-3 at ρ = 0.9), but its AUC matches TXC-base only at
   ρ = 0.9. At ρ = 0.0 it sits at 0.885 — a real cost. This is
   consistent with the H8 design intent: the matryoshka H+full prefix
   plus multi-distance InfoNCE bias the encoder toward cross-position
   regularities; that bias is "free" only when those regularities
   exist. Note that on the toy bench `h_size = d_sae // 5 = 410` and
   `k_inference = k_pos × T_max = 200` are both spec'd around
   d_in = 2304 / d_sae = 18432 in Han's real-data setting; on the
   128-true-feature toy these are over-spec'd, which probably explains
   why even at ρ = 0.9 H8 doesn't beat the simpler TXC-base.

## Where this leaves the headline

The "TXC big win" Andre features in §1.1 of
[[tempbench_metareport]] (his +0.50 ΔAUC vs Stacked SAE at ρ = 0.9)
**holds for the TXC vs Stacked SAE comparison under Han's recipe** —
but the comparison vs regular SAE no longer holds at Han's k and
d_sae. The cleanest synthetic story is therefore narrower than Andre's
metareport phrasing implies: TXC's structural advantage over Stacked
SAE survives the recipe swap, but its advantage over a per-token SAE
is a function of the sparsity / dictionary-size regime, not of TXC
versus per-token-SAE per se.

## Caveats

- **Single seed.** Bill's original report had the same caveat. Worth
  3-seed before being load-bearing; the size of the Stacked SAE gap
  (~0.49) makes that signal robust to seed noise, but the
  TXC-pro-vs-TXC-base ranking might flip at low ρ.
- **Toy-scale H8.** As above — H8's matryoshka prefix and inference
  TopK are over-spec'd at d_in = 256, n_features = 128. A fair test of
  H8 needs either (a) scaling the toy n_features up so H8's prefix
  isn't wider than the true feature set, or (b) testing on real Gemma
  activations where Han's defaults are appropriate.
- **Window-length asymmetry.** TXC-base trains and evaluates at T = 5;
  TXC-pro at T_max = 10 with t_sample = 5. The decoder-cosine AUC
  averages decoder columns across positions for both, so the
  comparison is on a level playing field at the metric, but the *data*
  TXC-pro sees is a longer window — which is by design (subseq
  encoder), and is part of what we're testing.
- **No matched-NMSE baseline.** The four-fold NMSE blow-up of TXC-pro
  is not penalised by the AUC metric; if the downstream task were
  reconstruction-quality-sensitive, H8 would look worse than these
  numbers suggest.

## Provenance

- Sweep script: `scripts/run_han_three_arch_sweep.py`
- Plot script: `scripts/plot_han_three_arch.py`
- Raw results: `results/han_three_arch/results.json`
- Branch: `bill-han-txc-10k`
- TXC ports: `src/temporal_bench/models/txc_base.py`, `txc_pro.py`
  (faithful ports of `purified/src/temp_bench/architectures/txc_base.py`
  and `txc_pro.py` on `origin/final`).
