---
author: Dmitry
date: 2026-04-24
tags:
  - results
  - in-progress
---

## em_features — coherence / suppression frontier results (so far)

Replication of the `safety-research/open-source-em-features` setup on
Qwen-2.5-7B-Instruct *bad-medical*, extended with two crosscoder
variants from `temp_xc` (`TemporalCrosscoder`, `MultiLayerCrosscoder`).

Plan: [[em_features_crosscoder_frontier]].
Code: [[experiments/em_features|experiments/em_features/]].
Checkpoints: `dmanningcoe/temp-xc-em-features` on HuggingFace Hub (private).

## Setup recap

- Base: `Qwen/Qwen2.5-7B-Instruct`
- Bad-medical fine-tune: `andyrdt/Qwen2.5-7B-Instruct_bad-medical` (PEFT
  adapter merged on load)
- Baseline SAE for comparison: Andy RDT's `andyrdt/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1`
  (`d_sae=131072`, target `L0=128`, batch_top_k, ~100 M training tokens)
- Our TXC / MLC: trained on base-Qwen residuals via streaming buffer (pile + ultrachat mix)
- All steering at **layer 15**, **k=10** bundled features, coefficient grid
  `α ∈ {-10, -8, -6, -5, -4, -3, -2, -1.5, -1, +1, +2, +5}`
- Evaluation: em-features' longform (8 EM questions × 8 rollouts × OpenAI
  judge for alignment + coherence)
- Baseline (no steering) scores: **alignment 64.19, coherence 84.88**

## Headline

| method | peak align | coh @ peak | α at peak | Δalign vs baseline |
|---|---:|---:|---:|---:|
| **SAE (Andy, 131k dict)** | **85.85** | 87.78 | −6 | **+21.7** |
| MLC L=5, d_sae=32k @ 40k steps | 83.61 | 84.40 | −5 | +19.4 |
| TXC, d_sae=32k @ 200k steps (last_cos) | 74.90 | 88.40 | −5 | +10.7 |
| TXC, d_sae=32k @ 40k  steps | 74.24 | 82.65 | −10 | +10.0 |
| TXC_sum_cos (alt ranking) | 73.03 | 75.24 | +5 | +8.8 |
| TXC_encoder (alt ranking) | 67.62 | 72.90 | +5 | +3.4 |
| TXC_window (alt steering) | 67.38 | 76.46 | −6 | +3.2 |

Baseline bad-medical model @ α=0: align 64.19, coh 84.88.

## What we've learned

### 1. SAE is hard to beat at this scale

Andy RDT's public SAE clears **+22 alignment** with essentially no coherence
cost. Our TXC (at comparable parameter count, d_sae=32k × T=5 ≈ 164k
effective rows) maxes out at +10.7. MLC L=5 (d_sae=32k) gets closer (+19.4)
but still trails. See [[summary_sae_vs_txc_vs_mlc]] and
[[frontier_sae_vs_txc_vs_mlc.png|the 3-way plot]].

### 2. TXC's weakness is not fixed by any of the obvious knobs

The [TXC ranking comparison](summary_txc_ranking_comparison.md) tests three
ranking-scheme variants and two steering protocols on the same 200k-step
TXC checkpoint:

- `last_cos`  — cos(W_dec[T−1, i, :], diff) (baseline)
- `sum_cos`   — Σ_t |cos(W_dec[t, i, :], diff)| across all T slots
- `encoder`   — encode per-token-diff windows through TXC, rank by |z_i|
- window steering — add α·Σ_i W_dec[t, i, :] at *all* T positions, not just the last

None beats the baseline `last_cos` config:
- `sum_cos` and `encoder` pick features that don't track misalignment at
  negative α — both peak at **α=+5**, suggesting the features they surface
  are generic "active on medical prompts" rather than causally misalignment-mediating.
- Window steering flatlines coherence at 70–76 across all α — per-position
  writes during prefill push residuals off-manifold without helping.

### 3. Training-length scaling for TXC is largely exhausted by 100k steps

See [[summary_txc_small_scaling]]. At 40k / 100k / 200k steps:
- 40k (new run): peak align 67.3 at α=−4, coh 83.5
- 100k: peak 77.4 at α=−6, coh 87.2 ← biggest jump is 40k→100k
- 200k: peak 74.9 at α=−5, coh 88.4 ← same family, judge noise dominates

Going from 40k to 100k adds ~10 alignment at the peak. 100k to 200k is
within judge variance (±5 at n_rollouts=8). Reconstruction loss drops
1620 → 1599 → 1576 — also flattening. Extending further is unlikely to
close the gap to SAE.

### 4. MLC is the best-performing crosscoder on this task

MLC L=5 (d_sae=32k) trained for only 40k steps already reaches align 83.6
— within ~2 pts of SAE, and occasionally *beats* SAE on coherence
(at α=−4: MLC 88.7 vs SAE 86.9). See [[summary_sae_vs_txc_vs_mlc]].
At α=−10 and α=+5 MLC collapses because its effective steering magnitude
is ~5× higher than SAE (steering hits 5 layers simultaneously). A fair
comparison would rescale α by 1/L — TODO.

### 5. Compute/architecture takeaways

- TXC's temporal coupling (one sparse latent shared across T=5 positions)
  does not help here. The "write direction" structure that matters at
  generation time is concentrated at the last slot; the other 4 positions'
  decoder rows don't help when we steer a single token.
- MLC's structure (one latent shared across L residual layers) *does* help
  — writing to 5 layers simultaneously propagates steering through both
  attention and MLP path for more layers at once.
- Dict width and training length look saturated at 32k / 100k steps
  respectively for TXC. SAE's 4× wider dict (131k) likely explains the
  remaining gap.

## Currently running

- **h100_2**: MLC deepdive — small MLC (d_sae=32k, L=5) × 200k steps with
  snapshots at 40k / 100k / 200k and sweep at each. Currently around step
  35k, ETA ~10 h. Will auto-continue to all-layers MLC (L=28, d_sae=8k) ×
  200k after, unless stopped.
- **a100_1**: Idle after the TXC ranking comparison. Big-TXC (d_sae=65k)
  phase is gated by `.stop_big_txc` sentinel and skipped.

## Open questions / next steps (not yet executed)

- **α-rescale for MLC**: divide α by L before steering so the per-residual
  magnitude matches SAE. Would give a cleaner comparison and avoid the
  α=±10 collapse.
- **MLC 200k + all-layers MLC**: the queued h100_2 runs. All-layers MLC
  is the novel bit — no public SAE spans the whole model.
- **TXC ≤ SAE verified**: may not be worth throwing more compute at TXC.
  Consider instead: BatchTopK TXC (what Andy uses for SAEs), or wider TXC
  (we couldn't fit d_sae=131k in fp32 Adam — needs 8bit optim).

## Files

- [[frontier_sae_vs_txc.png]], [[summary_sae_vs_txc]] — Phase 1 result
- [[frontier_sae_vs_txc_vs_mlc.png]], [[summary_sae_vs_txc_vs_mlc]] — MLC
  L=5 @ 40k added
- [[frontier_txc_small_scaling.png]], [[summary_txc_small_scaling]] — TXC
  40k / 100k / 200k scaling
- [[frontier_txc_ranking_comparison.png]], [[summary_txc_ranking_comparison]]
  — last_cos / sum_cos / encoder / window-steering
- `sweeps/` — raw sweep JSONs (every α × every method), 428 KB total
