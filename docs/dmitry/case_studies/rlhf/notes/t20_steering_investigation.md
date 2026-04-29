---
author: Dmitry
date: 2026-04-29
tags:
  - results
  - complete
---

## T=20 steering investigation — probing utility ≠ steering utility

### Motivation

Han's [Phase 6.3 work](https://github.com/chainik1125/temp_xc/blob/han-phase6/docs/han/research_logs/phase6_2_autoresearch/2026-04-24-t-sweep-results.md) found that `TXCBareAntidead` (Track 2 recipe) at **T=20** Pareto-dominates T-SAE on a long-passage probing benchmark:

> "**At T=20, Track 2 beats T-SAE qualitatively (19.0 vs 13.7 semantic features) AND on probing (0.7768 vs 0.7246). Pareto-dominant.**"

We tested whether this Pareto-dominance carries over to **steering** under the paper's clamp-on-latent + error-preserve protocol on Anthropic HH-RLHF concepts.

### Three T=20 ckpt variants tested

| variant | subject model | layer | k_pos | k_win | source |
|---|---|---|---|---|---|
| `txc_bare_antidead_t20` | gemma-2-2b base | L12 | 25 | 500 | HF (Phase 7 retrain) |
| `txcdr_t20_kpos100` | gemma-2-2b base | L12 | 100 | 2000 | HF (Phase 7 retrain) |
| `phase63_track2_t20_retrain` | **gemma-2-2b-IT** | **L13** | **100** | **2000** | retrained on a40_txc_1, original Phase 6.3 setup |

The third variant is the **closest reproduction** of Han's Phase 6.3 winning ckpt — same architecture, same hyperparameters, same subject model + layer (gemma-2-2b-IT @ L13), same training data (FineWeb).

### Three injection modes tested

For each ckpt:

1. **Paper-clamp right-edge**: encode T-window → clamp z[j]=strength → decode (T, d_in) → take right-edge slice → error-preserve at right edge only. (`intervene_paper_clamp_window.py`)
2. **Paper-clamp full-window**: same as (1) but inject the full (T, d_in) steered reconstruction at all T positions of the rightmost window. Higher-layer attention reads concept-flavored residuals at multiple keys. (`intervene_paper_clamp_window_full.py`)
3. **AxBench-additive**: Han's protocol — `x_steered = x + strength · unit_norm(W_dec[:, j].mean(dim=T))` at every position. (`intervene_axbench_extended.py`)

### Headline result

**None of the 3 variants × 3 injection modes produces useful steering at T=20.**

Per-strength means under paper-clamp right-edge (the canonical paper protocol):

| config | peak success | peak coherence | at strength |
|---|---|---|---|
| L12-base k_win=500 | 0.20 | 3.00 | s=100 |
| L12-base k_win=2000 | 0.33 | 3.00 | s=150 |
| **L13-IT k_win=2000 (Phase 6.3 native)** | **0.33** | **1.93** | **s=1000** |

For comparison, `tsae_paper_k20` at the same operating point: peak (1.93, 1.37) at s=100 — **6× higher success** at comparable coherence cost.

### What rules out

The investigation tested four hypotheses sequentially. All failed to recover steering:

1. **"Right-edge attribution misses concept signal that's spread across the window."**
   → Full-window injection peak: (0.23, 1.33) at s=10000 for L12-base k_win=500. Worse, not better. Rejected.

2. **"Per-token sparsity is too low (k_win=500 with T=20 means k_pos=25 effective)."**
   → k_win=2000 (k_pos=100) variant: peak success +0.13 over k_win=500. Marginal. Not enough to close the gap to T=5 archs (1.93). Mostly rejected.

3. **"AxBench-additive's averaged decoder direction would extract concept signal that paper-clamp's right-edge slice misses."**
   → AxBench peak for L12-base k_win=500: (0.37, 3.00) at s=−10. Within noise of s=0 baseline (0.23). Rejected.

4. **"L12-base is the wrong model/layer; the original Phase 6.3 setup at L13 of Gemma-2-2b-IT recovers steering."**
   → L13-IT k_win=2000 retrain (this work): peak (0.33, 1.93) at s=1000. Same success as L12-base k_win=2000 (0.33). Coherence holds longer at higher strengths but no concept-pull. Rejected.

### What this means

**Probing utility ≠ steering utility.** Phase 6.3's "T=20 Pareto-dominates T-SAE" finding was specifically on:

- Probing AUC (a linear classifier on SAE features can predict passage class)
- Semantic-feature counts (autointerp judges the top-32 features by variance as semantic)

Both are about whether feature *patterns* discriminate semantic content over a long span. Steering requires the feature *decoder direction* to push the residual stream toward concept-aligned token outputs — a different downstream task with different requirements.

T=20 features at this width and sparsity *do* capture passage-level semantic content well enough for probing, but their decoder directions don't carry the per-token "emit this concept's tokens" signal that steering exploits in T-SAE k=20 features.

This isn't an artefact of the paper's intervention protocol — three different mechanisms (clamp+error at right edge, clamp+error full window, additive at unit-normalized averaged direction) all fail. The decoder atoms themselves don't have the structure that steering needs.

### Possible mechanism

Speculation: window encoders integrate over T tokens, producing a single z per window. The T-position decoder reconstruction `decoder(z) → (T, d_in)` distributes the "concept signal" across T positions, with each per-position decoder column carrying ~1/T of the total. Even when we inject the full window's reconstruction, the model's attention has to integrate T different views of the concept, while the per-token model just sees one strong push at every position.

For *probing*, this is fine — the linear classifier reads from the d_sae feature dimension, which is shared across positions. For *steering*, the per-token decoder atom magnitude is what matters, and at T=20 it's 4× weaker than at T=5 (decoder normalized over T*d_in jointly).

A test of this hypothesis would be: re-train T=20 with **per-position decoder atoms unit-normed independently** (instead of jointly over T*d_in). If steering recovers, the issue is decoder magnitude. If not, the issue is concept-content distribution itself.

### Files

Sources:
- `experiments/phase7_unification/case_studies/steering/retrain_phase63_t20.py` — retrain script
- `experiments/phase7_unification/case_studies/steering/steer_phase63_retrain.py` — wrapper that monkey-patches SUBJECT_MODEL + ANCHOR_LAYER
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window.py` — right-edge variant
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_full.py` — full-window variant

Outputs (on `a40_txc_1`, gitignored):
- `data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy` (7 GB)
- `experiments/phase7_unification/results/ckpts/phase63_track2_t20_retrain__seed42.pt` (6.8 GB fp32)
- `experiments/phase7_unification/results/case_studies/steering_paper/phase63_track2_t20_retrain/{generations,grades}.jsonl`

Training: 25k steps in 58.5 min on A40, loss 29147 → 9319 (~70% reduction).

### Open questions

1. Would the original Phase 6.3 ckpt (also on a40_2's local disk, not on HF) behave differently? Our retrain matches the documented config; but training noise / data-shuffle could plausibly produce a different ckpt that does steer. Low probability this matters but unverifiable without that ckpt.
2. Is there a *different* sparsity / matryoshka split that recovers per-token steering at T=20? Han's Phase 6.2 already explored this without finding one, but only for probing.
3. Does the failure transfer to other long-window archs at sufficient T (SubseqH8 T=10 partly steers under paper-clamp at peak (1.10, 1.53); SubseqH8 T=15 or T=20 untested)?
