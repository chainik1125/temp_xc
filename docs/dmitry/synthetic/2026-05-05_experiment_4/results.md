---
author: dmitry
date: 2026-05-05
tags:
  - results
  - in-progress
---

## Summary

Implemented Experiment 4 (delayed temporally-colored sources) per
[[plan|the plan]]. Stage 0 validation gates all pass; Stage 1 sweep on a40
returns a clear negative result that I think is **theoretically expected**
but **not what the proposal anticipates**: trained TXC and regular SAE both
sit flat at chance, while only the spectral oracle recovers the basis.

The infrastructure is correct and reusable. The negative result is the
finding worth understanding before scaling Stage 2 / Stage 3 effort.

![Stage 1 figure](../../../../plots/v6_colored_sources/phase_transition_stage1.png)

## What works (Stage 0)

`src/v6_colored_sources/` contains the data generator, spectral oracle,
recovery metrics, and pre-training validation gates.

Five gates pass at `N=d=64, D=2, n_seq=256, T_chain=1024, σ=0.1, ρ ∈ [0.1, 0.9]`:

- **Isotropic one-token marginal:** off-diagonal entries of `C_0` are 1% of
  the diagonal, no eigenvalue-ratio anomaly.
- **Short-lag covariance ≈ 0:** `||C_2||_op / ||C_1||_op ≈ 19`.
- **Spectral oracle recovers the basis:** `Rec(F, F̂_oracle) = 0.85`.
- **Time-shuffle destroys oracle:** post-shuffle recovery `0.12`, well below
  the chance threshold `4 log N / N ≈ 0.26`.
- **Random-dictionary chance level** matches `~2 log N / N` within MC noise.

11 unit tests pass on both local CPU and remote a40.

## Stage 1 result (clear and consistent)

Sweep config: `d = N = 128`, `D = 1`, `σ = 0.1`, `ρ ∈ [0.1, 0.9]`,
`n_seq = 256`, `T_chain = 1024`, `n_steps = 8000`, `batch = 64`, `H = N`,
`k = 8`, run on a40 in 4.4 minutes total.

Pre-training baselines:

- **Spectral oracle ceiling:** `S_adj = 0.578` (would be `0.85` with
  `n_seq=512, T=2048`).
- **Random-vector floor:** `S_adj = 0.025`.

Trained methods (final squared recovery, chance-adjusted):

| Architecture | W | S_adj |
|---|---|---|
| Regular TopKSAE (iid tokens) | 1 | 0.029 |
| TXC | 2 | 0.030 |
| TXC | 4 | 0.030 |
| TXC | 8 | 0.026 |
| TXC | 16 | 0.024 |

Every trained dictionary method sits within Monte Carlo noise of the
random-vector floor at every window length. The proposal's expected
phase transition at `W ≥ D + 1` does not appear.

## Why this happens

The TopK reconstruction objective on Gaussian sources is **rotation-invariant**:
for any orthogonal `R`, the encoder/decoder pair `(W_enc·R, R^T·W_dec)`
achieves identical loss to `(W_enc, W_dec)`. Sketch:

- One-token: `x_t = F z_t + σ ε_t`, `z_t ~ N(0, I_N)`, so `x_t ~ N(0, (1+σ²)I_d)`.
  The components of any orthogonal projection of `x_t` are iid Gaussian.
  TopK selection over iid Gaussians has the same distribution regardless
  of basis. Reconstruction loss therefore has a flat manifold of minima
  parameterized by `R`.

- Multi-token: the joint distribution `(x_t, x_{t+D})` *does* have a
  basis-privileged structure (lag-`D` covariance is diagonal in the `F`
  basis). But the TXC objective is `Σ_t ||x_t - x̂_t||²`, which only
  depends on the marginal distribution of each `x_t`, not the joint. So
  rotation invariance carries through to multi-position windows.

The **spectral oracle** breaks the symmetry by reading `Ĉ_D` directly. No
training-loss landscape connects the rotation-invariant minimum to the
basis-aligned minimum — the spectral oracle is the only method in this
regime that can identify `F`.

This matches the proposal's tightness claim: "*any* local one-token learner
outputs directions independent of `F`." But the proposal's optimism that
TXC training would somehow bridge the local / temporal gap depends on the
training objective exploiting lag-`D` covariance — and the standard TopK
reconstruction loss does not.

## What this means for the proposal

- **Spectral oracle vs. trained dictionaries** is a real and useful baseline.
  On Gaussian colored sources, the oracle is the *only* method that recovers
  `F`.

- **The "TXC > local SAE" headline** of Section 4 cannot be produced by a
  TopK reconstruction objective on Gaussian sources. The architecture-data
  match the proposal needs is one of:

  1. **Non-Gaussian sources** (sparse non-negative — e.g. Bernoulli mask
     × magnitude as in Exp 1–3). Breaks rotation invariance via
     non-Gaussianity. Gives up the clean impossibility theorem in exchange
     for an SAE-friendly setup.
  2. **A training objective that directly fits lag-`D` covariance** (e.g. a
     temporal contrastive or whitening objective). Out of scope for the
     standard TopK SAE/TXC family.

- **The "ReLU signed-pair" variant** in proposal Section 4 (`z = z⁺ - z⁻`)
  is a relabeling — the data `x_t` is unchanged, so it does not break
  rotation invariance. It only changes the chance-adjustment denominator
  (since the ground-truth dictionary doubles to `2N`).

## Stage 2 (running now)

`D ∈ {1, 2, 4, 8} × W ∈ {2, 4, 8, 16}`. The expectation is that every cell
gives the same flat-at-chance TXC curve we saw at `D = 1`. Will update
this section with the Stage 2 plot and confirmation when the sweep
finishes (~30 min).

## What's next

Three routes, in increasing effort / value:

1. **Stage 2 anyway, with Gaussian sources.** Confirms the negative result
   across `D ∈ {1, 2, 4, 8}` and `W`. Useful for the writeup ("we tested
   across delays and window lengths; result is consistent") but doesn't
   change the conclusion. Already running.

2. **Sparse non-negative variant (off-spec).** Replace Gaussian `z_t,i` with
   `(Bernoulli(p) × |Gaussian|) AR(1)` so sources are sparse-positive. Gives
   the "TXC > SAE" headline figure the proposal wants, at the cost of
   giving up the local-impossibility theorem (the marginal is no longer
   rotation-symmetric). Cost: ~1 day to implement + run.

3. **Temporal-objective TXC variant.** Augment the loss with a lag-`D`
   covariance term. Largest deviation from the existing architecture, but
   would let the colored-source theorem regime work end-to-end. Cost:
   multi-day.

Recommend option 2 for the next sprint: it gives the proposal's intended
phase-transition figure and is cheap. Option 1 is finishing in the
background as an ablation.

## Files

- Code: `src/v6_colored_sources/`
- Tests: `tests/test_v6_colored_sources.py`
- Stage 1 results: `results/v6_colored_sources/stage1.json`
- Stage 1 figure: `plots/v6_colored_sources/phase_transition_stage1.png`
- Stage 2 results / figure: pending — sweep running on a40
- Plan: [[plan]]
