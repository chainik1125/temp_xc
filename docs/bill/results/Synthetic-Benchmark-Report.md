---
author: Bill
date: 2026-04-25
tags:
  - results
  - in-progress
---

## Summary

Three-architecture synthetic benchmark and HMM noisy-emission denoising
benchmark, run on RunPod. Compares **regular SAE**, **Stacked SAE**, and
**TXCDR** under a matched window-level L0 budget of `k*T`. Headline:

- TXCDR's reconstruction (NMSE) is *always worse* than regular SAE on the
  three-arch task, but its feature recovery (AUC) becomes better than
  regular SAE precisely when the data has temporal structure
  (rho > 0, low k).
- TXCDR is the only architecture that crosses the **per-token denoising
  floor** (ratio = 0.77) on the HMM benchmark — it tracks the hidden
  state better than the noisy observation it is given. The two
  position-independent baselines sit at the floor for all (T, k).

## Setups

### Three-arch sweep (Fig 5/6/7)

- `n_features = 40`, `d_model = 80`, `d_sae = 80`, `pi = 0.15`
- `rho` in {0.0, 0.6, 0.9}; `k` in {2, 5, 10, 25}; `T` in {2, 5}
- All arches matched at window L0 = `k*T`
- Decoder-cosine AUC averages decoder columns across positions for
  Stacked SAE and TXCDR
- TXCDR rows skipped where `k*T >= n_features` (degenerate TopK)

### HMM denoising sweep (Fig 8/9)

- `n_features = 40`, `d_model = 80`, `d_sae = 80`, `pi = 0.15`
- Heterogeneous `rho`: 10 features each at {0.1, 0.4, 0.7, 0.95}
- Stochastic emissions `p_A = 0`, `p_B = 0.625` (gamma ~= 0.59) — every
  token is corrupted, so a position-independent decoder cannot exceed a
  fixed denoising floor of **0.77**
- `T` in {2, 3, 4, 5, 6, 8, 10, 12}; `k` in {1, 3, 5}
- 69 of 72 cells completed; missing: (txcdr, T=8, k=5), (txcdr, T=10,
  k=5), (txcdr, T=12, k=5)

See [[Synthetic-Benchmark-Setup]] for the matched-budget framing
discussion (Framing A vs Framing B / `regular_sae_kT`).

## Three-arch results

### AUC trends across (rho, k, T)

Mean AUC by model: regular_sae **0.910**, txcdr **0.790**, stacked_sae
**0.559**.

The averaged number undersells TXCDR — the per-cell story is much more
structured:

- **rho = 0 (no temporal structure).** TXCDR loses to regular SAE in
  every cell (ΔAUC -0.09 to -0.52). Expected — there is nothing for a
  temporal model to exploit, and TXCDR's smaller per-position decoder is
  a handicap.
- **rho = 0.6 / 0.9, low k (k = 2).** TXCDR wins decisively
  (ΔAUC vs regular SAE +0.10 to +0.20 at k = 2; ΔAUC vs Stacked SAE
  +0.28 to +0.64). This is the regime the report cares about: temporal
  structure exists *and* the budget is too tight for a token-local
  decoder to recover features on its own.
- **rho = 0.6 / 0.9, large k (k = 25).** TXCDR loses again because the
  TopK collapses (`k*T = 50 + > n_features = 40`), so almost every
  latent fires every window and the dictionary is degenerate. Skip rule
  trims the most pathological points; what's left at k = 25 is mostly
  noise.
- **Stacked SAE** is consistently the worst on AUC (mean 0.56). Its T
  independent dictionaries get trained on T-times less data each, and
  the per-position-TopK design provides no temporal pooling — so it
  pays the parameter cost without the recovery benefit.

### NMSE: TXCDR is always worst on reconstruction

Across every cell, TXCDR's NMSE is at least as high as regular SAE — at
high k it is 50-700x worse (e.g. rho = 0, k = 25, T = 5: regular ~= 1e-4,
txcdr ~= 0.77). Reconstruction loss is *not* a useful proxy for what
TXCDR is doing — the model sacrifices token-level reconstruction in
exchange for cross-position information sharing.

This is the `Fig 7` story: AUC vs k and NMSE vs k tell different
stories; only AUC + a denoising metric jointly reveal the temporal
benefit.

## HMM denoising results

The single most informative table — denoising ratio (corr) vs T, all
three models:

| T  | regular_sae | stacked_sae | txcdr (k=1, 3, 5)  |
| -- | ----------- | ----------- | ------------------ |
| 2  | 0.77        | 0.78        | 0.82 / 0.89 / 0.90 |
| 4  | 0.77        | 0.75        | 0.98 / 1.01 / 1.01 |
| 6  | 0.76        | 0.77        | 1.07 / 1.11 / 1.11 |
| 8  | 0.76        | 0.76        | 1.12 / 1.12 / —    |
| 12 | 0.76        | 0.76        | 1.14 / 1.15 / —    |

Reading this table:

- The **per-token denoising floor is 0.77**: any model that decodes each
  token independently sees latent <-> observed correlation that exceeds
  latent <-> hidden correlation by exactly the emission noise gamma.
  Both regular SAE and Stacked SAE sit on this floor for every (T, k) —
  proof of concept for the bench, since both arches are by construction
  position-independent.
- TXCDR **crosses ratio = 1** by T = 4-5 and approaches 1.15 at T = 12.
  Ratio > 1 means the latents track the hidden Markov state *better
  than the noisy observation does* — the model has done genuine
  denoising along the temporal axis.
- The benefit grows monotonically in T, as it should: longer windows
  give the encoder more evidence to integrate over.
- Increasing k inside TXCDR helps slightly at small T but is roughly
  saturated by k = 3 — the temporal information sharing, not the per-
  token sparsity, is what matters.

Average corr_local / corr_global by model:

- regular_sae: local 0.72, global 0.55 (gap = noise gamma)
- stacked_sae: local 0.19, global 0.14 (worse than regular — small
  per-position dicts trained on too little data)
- txcdr: local 0.49, global 0.49 (gap closes — same correlation to
  hidden state and observation)

## Caveats / known issues

- **Single seed.** All numbers from `seed = 42`. The cell-to-cell ranking
  is consistent enough that I am willing to read direction off it, but
  small ΔAUCs (< ~0.05) are within run-to-run noise and should not be
  over-interpreted.
- **Three missing TXCDR cells** at large `T*k` in the HMM bench, all at
  k = 5 (T in {8, 10, 12}). Likely OOM or the TopK skip rule triggering;
  I have not chased this since the trend at k = 1, 3 is unambiguous.
- **Regular SAE k\*T baseline (Framing B)** not yet swept. The plot
  scripts already plumb it through; running it would tell us whether
  TXCDR's win is "temporal structure" or just "more active latents per
  token". Current ranking compares matched-window-L0; the harder
  comparison is matched-per-token-L0.
- **Stacked SAE trained from scratch each run.** With only T-times less
  data per per-position dictionary, the comparison may be unfair — the
  paper-equivalent setup is mid-training a single SAE then forking T
  copies. Not relevant for the TXCDR vs regular SAE story but worth
  flagging if Stacked SAE numbers come up later.

## Figures

- `three_arch/fig5_delta_auc_vs_regular_sae.{png,pdf}` — TXCDR ΔAUC
  heatmaps over (k, T) for each rho
- `three_arch/fig5_delta_auc_vs_stacked_sae.{png,pdf}` — same vs
  Stacked SAE
- `three_arch/fig6_delta_auc_vs_rho.{png,pdf}` — ΔAUC vs rho, one line
  per (k, T)
- `three_arch/fig7_auc_loss_vs_k.{png,pdf}` — AUC and NMSE vs k, one
  panel per rho
- `hmm_denoising/fig8_global_vs_local.{png,pdf}` — global vs local
  correlation scatter
- `hmm_denoising/fig9_denoising_ratio_corr.{png,pdf}` — denoising ratio
  vs T (corr metric)
- `hmm_denoising/fig9_denoising_ratio_r2.{png,pdf}` — same, R^2 metric
