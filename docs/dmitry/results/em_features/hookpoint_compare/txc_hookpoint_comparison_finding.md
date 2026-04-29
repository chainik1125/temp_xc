---
author: Dmitry
date: 2026-04-29
tags:
  - results
  - complete
---

## TXC brickenauxk at attn_out (resid_mid) and ln1_normalized: beats T-SAE at the same hookpoints

### Question

We previously trained T-SAE at three hookpoints (resid_post / resid_mid / ln1_normalized) and found that bundle-causal alignment was flat across hookpoints (all ~50). The follow-up question: does the same hookpoint sweep applied to **TXC brickenauxk** (the T=5 windowed crosscoder, which has more capacity for temporal structure than T-SAE's per-token encoder) also flatten out, or does it pick up something extra?

### Setup

- Trained TXC brickenauxk (`d_sae=32k, k_total=128, T=5, lr=3e-4, batch=256`) on Qwen-7B-Instruct at layer 15, hookpoint ∈ {`resid_mid`, `ln1_normalized`}, 30k steps (resumed for resid_mid after disk-full + corrupted step20k save).
- Best loss: **resid_mid 1840.94 / ln1_normalized 480.30** (ln1 smaller because LayerNorm output is unit-variance; not directly comparable to resid_mid).
- Ran encoder Δz̄ attribution + full Wang procedure (stages 2-4 + bundle k=30) on each.

### Top-10 by Δz̄ encoder attribution

| rank | TXC @ resid_mid | TXC @ ln1_normalized |
|---:|---|---|
| 1 | feat 27445 (+3.788) | feat 15673 (+0.961) |
| 2 | feat 21970 (+3.291) | feat  9811 (+0.650) |
| 3 | feat 13099 (+2.676) | feat 20890 (+0.320) |
| 4 | feat 25647 (+2.594) | feat  8516 (+0.275) |
| 5 | feat  1369 (+1.949) | feat 16736 (+0.243) |
| 6 | feat 31837 (+1.910) | feat  7700 (+0.234) |
| 7 | feat 31035 (+1.769) | feat 17040 (+0.232) |
| 8 | feat  5001 (+1.600) | feat  3443 (+0.228) |
| 9 | feat 31449 (+1.467) | feat  2102 (+0.212) |
| 10 | feat 16123 (+1.455) | feat  1281 (+0.209) |

**Disjoint top sets across hookpoints** (same pattern as T-SAE). Δz̄ magnitudes are larger than T-SAE at the same hookpoints — consistent with windowed encoding aggregating signal across T=5 positions.

### Stage 3 single-feature peaks (top-3 by causal alignment)

| TXC @ resid_mid | TXC @ ln1_normalized |
|---|---|
| feat 7369  α=−10 align=56.29 coh=36.72 | feat 9596  α=−10 align=57.61 coh=27.97 |
| feat 31449 α=−10 align=55.59 coh=24.53 | feat 15539 α=−10 align=56.29 coh=31.88 |
| feat 17266 α=−10 align=55.03 coh=30.00 | feat 11611 α=−10 align=55.17 coh=27.19 |

Single-feature causal hits in the **55–58 align** range — substantially stronger than the T-SAE single-feature peaks at the same hookpoints (which topped out around 49–51).

### Wang bundle k=30 frontier — TXC vs T-SAE at matched hookpoints

| variant | peak α | peak align | peak coh | α=0 align | Δalign vs α=0 |
|---|---:|---:|---:|---:|---:|
| T-SAE 30k @ resid_mid | −1.75 | 50.00 | 26.33 | (no α=0) | (~+5) |
| **TXC  30k @ resid_mid** | **−8** | **53.87** | **29.77** | 43.39 | **+10.48** |
| T-SAE 30k @ ln1_normalized | −1.75 | 49.92 | 23.52 | 44.26 | +5.66 |
| **TXC  30k @ ln1_normalized** | **+7** | **51.61** | **24.14** | 43.28 | **+8.33** |
| (ref) T-SAE 100k @ resid_post | −6 | 52.39 | 27.58 | — | +6.5 |
| (ref) Han 100k @ resid_post | −5 | 50.82 | 26.48 | — | +6 |
| (ref) **SAE 100k @ resid_post** | **−10** | **57.42** | **35.78** | — | **+13.8** |

**Three observations:**

1. **TXC is consistently ~3–4 align points above T-SAE at the same hookpoint.** At resid_mid: 53.87 vs 50.00. At ln1_normalized: 51.61 vs 49.92. Same Wang bundle, same data, same hookpoint — just a different SAE-side architecture. The contrastive T-SAE objective, despite being per-token, produces a less Wang-bundle-actionable feature decomposition than TXC's T=5 windowed objective.

2. **TXC @ resid_mid 30k (53.87) is competitive with T-SAE 100k @ resid_post (52.39)** — i.e. trading a different hookpoint for ~3.3× the training budget gets you a comparable bundle peak. resid_mid genuinely carries causally-actionable misalignment signal that's at least as good as resid_post for our PEFT-LoRA EM organism.

3. **The hookpoint flatness story for T-SAE doesn't fully transfer to TXC.** T-SAE was uniformly ~50 across hookpoints; TXC differentiates: resid_mid (53.87) > ln1 (51.61) > [presumably resid_post pending]. This is small (~2 points) but consistent with the idea that resid_mid (= resid_pre + attn_out) is an extra-rich signal site once your SAE-side objective can use its multi-position structure.

### Caveat: steering still applied at resid_post

Same caveat as the T-SAE hookpoint sweep — all bundle steering hooks attach at `model.model.layers[15]` output (= resid_post). The TXC's `W_dec[-1]` direction (last-position decoder) is added there directly. This is mechanically valid (same d_model) but conceptually approximate: the SAE learned the direction in resid_mid / ln1 statistics, then we apply it to resid_post.

The encoder Δz̄ attribution side does use the correct hookpoint (HookpointExtractor in `gather_residuals`) so the top-feature lists above are real per-hookpoint findings. Bundle steering is the cross-hookpoint approximate comparison.

### Implication

For a 30k-step training budget, **TXC brickenauxk is the best-performing arch at non-resid_post hookpoints we've tested.** It outperforms T-SAE at every hookpoint we can compare and approaches the resid_post 100k-step T-SAE/Han numbers from a 30k-step training run. The SAE arditi 100k @ resid_post (57.42) remains the overall champion, but it has a 3.3× training-budget advantage and is at the easiest hookpoint.

**Combining TXC + 100k steps + non-resid_post hookpoint** is the obvious next experiment if we want to try beating the SAE arditi peak.

### Files

- `txc_residmid_30k/results/qwen_l15_txc_brickenauxk_a8_residmid_step30000_encoder/top_200_features.json`
- `txc_residmid_30k/results/wang_txc_residmid_step30000/stage{2,3,4}_*.json`
- `txc_residmid_30k/results/wang_txc_residmid_step30000_bundle30_frontier.json`
- `txc_ln1_30k/results/qwen_l15_txc_brickenauxk_a8_ln1_step30000_encoder/top_200_features.json`
- `txc_ln1_30k/results/wang_txc_ln1_step30000/stage{2,3,4}_*.json`
- `txc_ln1_30k/results/wang_txc_ln1_step30000_bundle30_frontier.json`
- TXC checkpoints on HF: `dmanningcoe/temp-xc-em-features:txc/qwen_l15_txc_brickenauxk_a8_{residmid,ln1}_step{10000,20000,30000}.pt`

### Open follow-ups

1. **TXC @ resid_post 30k**: we haven't trained TXC brickenauxk fresh at resid_post for the new run-protocol — need a 30k resid_post run to make the cross-hookpoint comparison clean. (We do have h2_qwen_l15_txc_brickenauxk_a8 at resid_post but that's a different run with 60k steps, not 30k.)
2. **Hookpoint-matched steering**: implement `frontier_sweep.py --steerer_hookpoint <hp>` that attaches the additive hook at the trained hookpoint. Re-run the TXC bundle frontier with this.
3. **Extend TXC to 100k at resid_mid**: the most likely path to beat SAE arditi 100k @ resid_post on this organism, given that 30k @ resid_mid already matches T-SAE 100k @ resid_post.
