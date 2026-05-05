---
author: dmitry
date: 2026-05-05
tags:
  - results
  - in-progress
---

## TL;DR

Implemented Experiment 4 (delayed temporally-colored sources) per
[[plan|the plan]]. Three architectures (regular SAE, TopK TXC, Han H8 with
multi-distance InfoNCE) all sit at chance recovery across every window
length and delay we tried. Only the spectral oracle reads `Ĉ_D` directly
and recovers the basis. The reason is structural: **rotation invariance of
Gaussian sources is preserved by reconstruction loss AND by cosine InfoNCE**,
so no dictionary-style training objective in this family can break the
symmetry. The proposal's expected `W = D + 1` phase transition is
unattainable on Gaussian sources without changing either the source
distribution or the loss function.

## Setup

- Orthonormal basis `F ∈ R^{N × d}`, `F F^T = I_N`. We use `d = N = 128`.
- Per-coordinate AR(1) latents with delay `D`:
  `z_{t+D, i} = ρ_i z_{t, i} + sqrt(1 - ρ_i²) η_{t,i}`, `η ~ N(0, I)`,
  `ρ_i` linspace on `[0.1, 0.9]`. Independent residue classes mod `D`.
- Observation: `x_t = F z_t + σ ε_t`, `σ = 0.1`.
- One-token marginal: `x_t ~ N(0, (1 + σ²) I_d)` — independent of `F`.
- Population covariances: `C_0 = (1+σ²) I_d`; `C_ℓ = 0` for `0 < ℓ < D`;
  `C_D = F diag(ρ) F^T` — eigenvectors = true basis.
- Recovery: `Rec(F, F̂) = (1/N) Σ_i max_j |⟨f_i, f̂_j⟩|²`,
  `S_adj = max(0, (Rec − log(H)/N) / (1 − log(H)/N))`.

## Stage 0 — pre-training validation (passes)

All five proposal-mandated gates pass at `N = d = 64, D = 2, n_seq = 256,
T_chain = 1024`:

| Gate | Passing value | Threshold |
|---|---|---|
| One-token isotropy | off-diag/diag = 0.010 | < 0.10 |
| Short-lag covariance ≈ 0 | `‖C_D‖_op / ‖C_short‖_op = 19×` | > 3× |
| Spectral oracle recovers basis | Rec = 0.85 | > 0.7 |
| Time shuffle destroys oracle | Rec = 0.12 | < 0.26 (4 log N / N) |
| Random-dictionary chance level | mean = 0.107 | within 2× of `log(H)/N = 0.054` |

11 unit tests pass on local CPU and on the a40.

## Stage 1 — TXC vs H8 vs regular SAE at D=1

`d = N = 128`, `D = 1`, `σ = 0.1`, `ρ ∈ [0.1, 0.9]`, `n_seq = 256`,
`T_chain = 1024`, `n_steps = 8000`, `batch = 64`, `H = N`, `k_pos = 8`.

Pre-training baselines on this data:

| Quantity | Value |
|---|---|
| Spectral oracle (`Ĉ_1` eigvecs) | **S_adj = 0.578** |
| Random unit-vector floor | S_adj = 0.025 |

Trained results (final S_adj after 8000 steps):

| Architecture | W=2 | W=4 | W=8 | W=16 |
|---|---|---|---|---|
| Regular TopKSAE (W=1, iid tokens) | 0.030 | 0.030 | 0.030 | 0.030 |
| TXC (TopK + per-pos decoder) | 0.027 | 0.030 | 0.028 | 0.025 |
| Han H8 (TXC + matryoshka + multi-dist InfoNCE) | 0.027 | 0.037 | 0.030 | 0.023 |

The SAE row is constant across W because the SAE doesn't see windows —
it's the same iid-token model regardless of W.

Every trained architecture sits within MC noise of the random-vector
floor (0.025) at every window length. The H8 W=4 cell at 0.037 is the
largest deviation; it's within run-to-run noise (other H8 cells at the
same scale give 0.023–0.030).

![Stage 1 figure](../../../../plots/v6_colored_sources/phase_transition_stage1.png)

Sweep ran in 15.3 minutes on a40 (one A40 80GB).

## Stage 2 — D × W grid (TXC vs SAE)

`D ∈ {1, 2, 4, 8} × W ∈ {2, 4, 8, 16}`, all other knobs as Stage 1. H8 was
not re-run here (Stage 1 already shows it tracks vanilla TXC; doubling
the cell budget for a confirmed-flat curve isn't a good use of compute).

Oracle ceilings:

| D | Oracle S_adj |
|---|---|
| 1 | 0.578 |
| 2 | 0.611 |
| 4 | 0.639 |
| 8 | 0.625 |

TXC results (S_adj across the 16 cells):

| D \ W | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| 1 | 0.026 | 0.028 | 0.027 | 0.021 |
| 2 | 0.023 | 0.025 | 0.026 | 0.028 |
| 4 | 0.027 | 0.024 | 0.025 | 0.021 |
| 8 | 0.029 | 0.024 | 0.025 | 0.021 |

SAE per-D (constant across W):

| D | SAE S_adj |
|---|---|
| 1 | 0.025 |
| 2 | 0.026 |
| 4 | 0.026 |
| 8 | 0.026 |

Every TXC cell is within MC noise of chance. **There is no phase
transition at `W = D + 1`** — the dotted vertical lines in the figure
mark where the proposal expects TXC to jump, and no curve responds.

![Stage 2 figure](../../../../plots/v6_colored_sources/phase_transition_stage2.png)

Sweep ran in 16.7 minutes on a40.

## Why all three trained methods fail

The training objectives in this family (TopK reconstruction + cosine
InfoNCE) are *both* rotation-invariant on Gaussian sources. For any
orthogonal `R`:

- **Reconstruction loss.** Apply `(W_enc, W_dec) ↦ (W_enc R, R^T W_dec)`.
  The encoder pre-activation rotates by `R` (so TopK indices change), but
  the decoder rotates back by `R^T`, so `x_hat` is unchanged. Loss on every
  sample is identical.
- **Lagged covariance structure.** `C_D = F diag(ρ) F^T` does have a
  preferred basis. But the TXC objective only sees the *marginal*
  distribution at each position (`Σ_t ‖x_t − x̂_t‖²`), not the joint, so
  `C_D` doesn't enter the gradient.
- **InfoNCE on TopK latents.** `sim = z_a · z_b / (‖z_a‖ ‖z_b‖)` is invariant
  under `z ↦ R^T z` (cosine similarities are rotation-invariant). So
  pulling `z(x_t)` and `z(x_{t+s})` together via cross-entropy on a
  similarity matrix gives no basis-aligning gradient on Gaussian sources
  either.

The training landscape has a flat manifold of equivalent minima
parameterized by `R ∈ O(N)`. SGD lands somewhere on that manifold,
which is "random rotation away from `F`" — exactly the chance recovery
we see. The spectral oracle works because eigendecomposition of `Ĉ_D`
*is* a basis-aligning operation.

This matches the proposal's tightness claim: "any local one-token learner
outputs directions independent of `F`." But it also extends the
impossibility to a wider class than the proposal acknowledges: any
training objective whose loss factors through marginal-position
reconstruction OR through cosine-similarity contrastive terms inherits
the rotation invariance.

## What this means for the proposal

- **Spectral oracle vs. trained dictionaries** is a useful baseline. On
  Gaussian colored sources the oracle is the *only* method we tested
  that recovers `F`.

- **The proposal's expected "TXC > local SAE" headline cannot be
  produced** by reconstruction-style or cosine-contrastive objectives
  on Gaussian sources. We need to change either the data or the loss:

  | Change | Pro | Con |
  |---|---|---|
  | **Sparse non-negative sources** (Bernoulli mask × magnitude, like Exp 1–3) | Architecture matches data; gives the expected phase transition | Gives up the local-impossibility theorem (marginal is non-Gaussian) |
  | **Add a lag-`D` covariance term to the loss** (push `W_dec` toward eigvecs of `Ĉ_D`) | Keeps the clean theorem regime; just a loss change | Custom architecture; deviates from H8 lock-in |
  | **Extra-architectural projection step** (post-train, project `W_dec` into the top-`N` eigvec subspace of `Ĉ_D`) | Cheap; uses the spectral oracle as a regularizer | Smells like bolting an oracle onto an architecture that can't find it on its own |

- **The "ReLU signed-pair" variant** in proposal Section 4 (`z = z⁺ − z⁻`,
  doubled dictionary `[F; −F]`) is a relabeling of the *evaluation* — the
  data `x_t` is identical to the Gaussian variant, so it does not break
  rotation invariance. It changes the chance-adjustment denominator
  (since `H` doubles) but does not give SAEs a foothold.

## Recommendation

For the next sprint, I'd implement the **sparse non-negative source
variant**. It gives the proposal's intended phase-transition figure and
is cheap on top of the existing v6 module — the only change is the
latent generator (`sample_ar_chains` → `sample_sparse_ar_chains`) and
the matching adjustment to the chance baseline. Estimated effort: half
a day to implement, ~30 min of a40 compute.

The clean theorem regime (Gaussian sources, rotation-invariant marginal)
is good for *rigorously bounding the local baseline* but not for
producing a "TXC wins" empirical figure. Both are useful — they answer
different questions.

## Files

| | Path |
|---|---|
| Code | `src/v6_colored_sources/` |
| Tests | `tests/test_v6_colored_sources.py` |
| Plan | [[plan]] |
| Stage 1 results | `results/v6_colored_sources/stage1.json` |
| Stage 1 figure | `plots/v6_colored_sources/phase_transition_stage1.png` |
| Stage 2 results | `results/v6_colored_sources/stage2.json` |
| Stage 2 figure | `plots/v6_colored_sources/phase_transition_stage2.png` |
