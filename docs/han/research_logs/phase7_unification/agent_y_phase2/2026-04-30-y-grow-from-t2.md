---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Y — grow window from T=2 (Han's directive)

> **Headline mixed**: Growing T=2 → T=3 via warm-start **preserves**
> the matched-sparsity advantage (per-position **1.167**, Δ=+0.067 vs
> anchor — matches T=2's 3-seed mean of 1.200 within seed-noise).
> Growing T=2 → T=5 **fails** (per-position 0.567, Δ=−0.533 — worse
> than random-init T=5's 0.783). The "duplicate last position" warm-
> start scheme has a +1-position horizon.

### Context

Han's directive (2026-04-30): *"take the BEST TXC at T=2 and see how
we can possibly grow the window without making it worse"*. T=2 is the
multi-seed champion at matched per-token sparsity (3-seed mean 1.200,
Δ=+0.10 vs T-SAE k=20 anchor 1.10).

Approach: warm-start a larger T from `txc_bare_antidead_t2_kpos20__seed42.pt`:
- W_enc[0:2] = src.W_enc[0:2] (existing 2 positions)
- W_enc[2..T-1] = src.W_enc[1] (duplicate last src position)
- W_dec, b_enc, b_dec same pattern
- Renormalize decoder unit-norm
- Train at canonical TrainCfg (b=4096, lr=3e-4, plateau=0.02)

Trainer: `experiments/phase7_unification/case_studies/train_kpos20_grow.py`.

### Results

| arch (T, init from T=2) | training | right-edge peak15 | per-position peak15 | per-pos Δ vs 1.10 |
|---|---|---|---|---|
| T=2 bare antidead (3-seed multi-seed mean) | random-init | 1.122 | **1.200** | +0.100 |
| **T=3 grown-from-T=2** (this) | warm-start T=2 | **1.000** | **1.167** | **+0.067** ⭐ |
| **T=5 grown-from-T=2** (this) | warm-start T=2 | **0.633** | **0.567** | **−0.533** ⚠ |
| T=5 bare antidead (2-seed mean) | random-init | 0.867 | 0.783 | −0.317 |

**T=3 grown achieves the same Δ=+0.07 above anchor that T=5 bare k_win=20
single-seed achieved**, *and* matches T=2's matched-sparsity advantage
(within seed-noise). The "grow by 1 position" strategy works.

**T=5 grown is dramatically worse than even random-init T=5.** The
warm-start scheme (duplicate position 1 → positions 2, 3, 4) creates
three redundant initial weights that the model doesn't escape. Training
converged via plateau (loss 19911 → 4814 at step 3600) — but to a poor
local minimum with worse downstream steering than random-init.

### Training stats

| arch | final_step | plateau_last | elapsed | loss[0] → loss[-1] |
|---|---|---|---|---|
| T=3 grown-from-T=2 | 3000 | 0.0148 | 21 min | 9377 → 4475 |
| T=5 grown-from-T=2 | 3600 | 0.0165 | 43 min | 19911 → 4814 |
| T=5 bare random-init | 3800 | 0.019 | 46 min | 20260 → 4813 |

T=5 grown reaches similar reconstruction loss (4814) as T=5 bare
random-init (4813) — training *quality* is comparable. But steering
perf differs by 0.21 (per-position): the warm-start trapped the model
in a representation that's hard to use for steering.

### Why T=5 grown fails

Hypothesis: at T=5, three of the five encoder/decoder positions are
initialized as exact copies of T=2's position 1. This creates a
3-fold redundancy — the model has 3 positions that produce identical
features at init. Gradient descent doesn't differentiate them well
because they live in the same loss-basin. The result is features that
fire similarly at positions 2, 3, 4 — so per-position write at those
positions averages a redundant signal.

T=3 doesn't have this issue: only 1 new position (position 2) is
initialized as a copy of position 1. Training differentiates them
within the available budget. Reconstruction loss drops 5377 → 4475
(steeper drop than T=5's 19911 → 4814 in absolute terms), and the
single new position can specialize.

### Updated combined matched-sparsity ranking under per-position

| arch | n_seeds | mean peak15 | σ_seeds | Δ vs anchor |
|---|---|---|---|---|
| T=2 bare k_pos=20 | 3 | **1.200** | 0.186 | +0.100 ⭐ |
| **T=3 grown-from-T=2** | 1 | **1.167** | — | **+0.067** ⭐ NEW |
| T=5 bare k_win=20 | 1 | 1.167 | — | +0.067 ⭐ |
| T=5 H8 shifts=(T,) | 2 | 1.067 | 0.000 | −0.033 |
| T=3 bare k_pos=20 (W's) | 1 | 1.000 | — | −0.100 |
| T=5 matryoshka multiscale (W's) | 1 | 0.933 | — | −0.167 |
| T=5 bare k_pos=20 | 2 | 0.783 | 0.071 | −0.317 |
| **T=5 grown-from-T=2** | 1 | **0.567** | — | **−0.533** ⚠ |

### Next steps for "grow without losing" if budget permits

1. **Sequential growth** (T=2 → T=3 → T=4 → T=5, each step warm-starts
   from the previous grown ckpt): tests whether incremental +1
   growth scales beyond +1.
2. **Different new-position init scheme**: small-noise random rather
   than exact duplicate; or fresh kaiming init for new positions only;
   or T-SAE k=20 init for new positions (since they don't have prior
   structure).
3. **Multi-seed verify of T=3 grown**: single-seed claim needs seed=1.
4. **Combine grow with shifts=(T,)**: warm-start H8 multidistance from
   T=2 ckpt, train at T=3 with shifts=(3,). Could combine T=3 grown's
   +0.07 advantage with H8 shifts=(T,)'s seed-stability.

### Files

- Trainer: `experiments/phase7_unification/case_studies/train_kpos20_grow.py`
- Ckpts: `results/ckpts/txc_bare_antidead_t{3,5}_kpos20_grownFromT2sd42__seed42.pt`
- Right-edge: `results/case_studies/steering_paper_normalised/txc_bare_antidead_t{3,5}_kpos20_grownFromT2sd42/`
- Per-position: `results/case_studies/steering_paper_window_perposition/txc_bare_antidead_t{3,5}_kpos20_grownFromT2sd42/`
