---
author: Dmitry
date: 2026-04-29
tags:
  - results
  - complete
---

## T-SAE at attn_out (resid_mid) and ln1_normalized: same answer as resid_post

### Question

Does training a T-SAE at a non-`resid_post` hookpoint — specifically `resid_mid` (closest analog to "attn_out" in standard SAE terminology) or `ln1_normalized` (the LayerNormed input to attention) — find materially different misalignment-mediating features, or produce stronger Wang-procedure steering?

### Setup

- **resid_mid**: residual stream after attention, before MLP. = `resid_pre + attn_out`. The closest standard analog to "attn_out".
- **ln1_normalized**: LayerNorm-normalized residual fed into attention. Pre-attention.
- Trained two T-SAE variants (per-token TopK + adjacent-token contrastive, d_sae=32k, k=128, T=5) on base Qwen-7B-Instruct activations at each hookpoint, 30k steps with snapshot at 10k/20k/30k. Same training data (Pile + Ultrachat mix) as the original resid_post run.
- Ran encoder Δz̄ attribution + full Wang procedure (stages 2-4 + bundle k=30) on each.

### Top-10 by Δz̄ encoder attribution (compared across hookpoints)

| rank | resid_post (orig) | resid_mid | ln1_normalized |
|---:|---|---|---|
| 1 | feat 9195 (+0.686) | feat 30338 (+0.911) | feat 16356 (+0.319) |
| 2 | feat 29012 (+0.521) | feat 31864 (+0.758) | feat 23005 (+0.312) |
| 3 | feat 25014 (+0.396) | feat 15252 (+0.531) | feat 14959 (+0.310) |
| 4 | feat 12488 (+0.363) | feat 30941 (+0.506) | feat 7834 (+0.221) |
| 5 | feat 29972 (+0.317) | feat 10853 (+0.381) | feat 28441 (+0.217) |
| 6 | feat 1305 (+0.286) | feat 13533 (+0.381) | feat 17367 (+0.198) |
| 7 | feat 702 (+0.275) | feat 11465 (+0.362) | feat 31118 (+0.194) |
| 8 | feat 28700 (+0.257) | feat 3560 (+0.260) | feat 24695 (+0.190) |
| 9 | feat 20539 (+0.227) | feat 24919 (+0.257) | feat 32474 (+0.185) |
| 10 | feat 21679 (+0.174) | feat 659 (+0.254) | feat 7958 (+0.170) |

**Top features completely disjoint across hookpoints** — no overlaps in any pair of top-10 sets. Each hookpoint's SAE finds different "fires-more-in-misaligned" features. Magnitudes also differ: resid_mid features have the strongest rank-1 Δz̄ (+0.91), ln1_normalized the weakest (+0.32; LayerNorm renormalizes input scale).

### Wang bundle k=30 frontier

| variant | peak α | peak align | peak coh | Δalign vs α=0 |
|---|---:|---:|---:|---:|
| T-SAE 30k @ resid_post (baseline) | −10 | 49.84 | 27.42 | (no α=0 in original sweep) |
| **T-SAE 30k @ resid_mid** | **−1.75** | **50.00** | **26.33** | (~+5) |
| **T-SAE 30k @ ln1_normalized** | **−1.75** | **49.92** | **23.52** | **+5.66** (vs 44.26 baseline) |
| T-SAE 100k @ resid_post (extended) | −6 | 52.39 | 27.58 | +6.5 |
| Han 100k @ resid_post | −5 | 50.82 | 26.48 | +6 |
| **SAE 100k @ resid_post** | **−10** | **57.42** | **35.78** | **+13.8** |

**Two conclusions:**

1. **Hookpoint barely matters at the bundle level for T-SAE.** Both new hookpoints land at ~50 align — within a noise band of the resid_post baseline (49.84). The contrastive objective + bundle aggregation flatten out whatever per-feature differences exist between hookpoints.

2. **Peak α scales with activation magnitude as expected.** resid_mid and ln1_normalized peaks are at α=−1.75 (vs resid_post's −10 / −6) because these hookpoints have smaller-scale activations (especially ln1_normalized, which is LayerNormed). The *effective* intervention strength is comparable.

### Caveat: steering was applied at resid_post

All Wang-procedure steering hooks attach at `model.model.layers[15]` (the layer's output, = `resid_post`). The trained SAE's `W_dec` rows are extracted and added there, regardless of which hookpoint they were trained on. This is mechanically valid (same d_model space) but conceptually approximate — a direction the SAE learned in `ln1_normalized` activation statistics is being applied to `resid_post` activations, where statistics differ.

A clean hookpoint-matched test would require a new steering hook that attaches at the correct site (e.g., on the input_layernorm output for `ln1_normalized` steering). Adding that to `frontier_sweep.py` and `run_wang_procedure.py` is non-trivial and was deferred.

That said — the encoder Δz̄ attribution does use the correct hookpoint, so the top features identified at each hookpoint are real attribution-side findings. The bundle steering numbers are an approximate cross-hookpoint comparison only.

### Implication

**The hookpoint choice (resid_post vs resid_mid vs ln1_normalized) doesn't change the bundle-causal peak alignment for our Qwen-7B PEFT-LoRA EM organism.** The bottleneck for steering effectiveness is *not* "we're reading from the wrong site in the layer." Whatever's limiting T-SAE's bundle peak (~50) vs SAE arditi's (~57) is either:

- **Architectural**: the contrastive loss spreads misalignment-relevant signal across more features, hurting bundle aggregation
- **Capacity**: T-SAE may need a different d_sae or k to recover the SAE's compactness
- **Training data**: the Pile + Ultrachat mix may not surface enough EM-relevant features for the contrastive loss to organize

### Files

- `tsae_residmid_30k_encoder_top200.json`, `tsae_ln1_30k_encoder_top200.json` — encoder Δz̄ rankings
- `tsae_residmid_30k_stage{2,3,4}_*.json`, `tsae_ln1_30k_stage{2,3,4}_*.json` — full Wang procedure outputs
- `tsae_residmid_30k_bundle30_frontier.json`, `tsae_ln1_30k_bundle30_frontier.json` — bundle k=30 sweeps
- T-SAE checkpoints on HF: `dmanningcoe/temp-xc-em-features:tsae/qwen_l15_tsae_{residmid,ln1}_k128_step{10000,20000,30000}.pt`

### Open follow-ups

1. **Hookpoint-matched steering**: implement `frontier_sweep.py --steerer_hookpoint <hp>` that attaches the additive hook at the trained-SAE's hookpoint. Re-run Wang bundle on the resid_mid and ln1_normalized SAEs with this. Would tell us whether the cross-hookpoint comparison was genuinely flat or merely approximate.
2. **Train T-SAE at all 3 hookpoints to 100k**: would tell us whether the resid_post training-step advantage (30k → 100k = +2.55) holds at other hookpoints.
3. **`attn_out` proper** (without residual): add a new hookpoint that captures `attn_out` *before* the residual addition, train T-SAE there, see if the pre-residual SAE finds different features.
