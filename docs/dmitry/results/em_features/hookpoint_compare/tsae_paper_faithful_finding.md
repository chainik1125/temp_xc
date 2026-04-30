---
author: Dmitry
date: 2026-04-29
tags:
  - results
  - complete
---

## T-SAE paper-faithful 30k @ resid_post: bundle peak align 56.23 (+6.4 over our settings, −1.2 vs SAE arditi 100k)

### Why this re-run

When we plotted the 8-panel TXC-vs-T-SAE hookpoint comparison, the T-SAE panels looked oddly flat across hookpoints (~50 align everywhere) and undertrained relative to the SAE arditi champion (57.42). Cross-checking the Bhalla 2025 paper ([arXiv:2511.05541](https://arxiv.org/abs/2511.05541)) and the official [AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes) training code revealed five hyperparameter mismatches between our T-SAE runs and the paper's canonical setup:

| param | Bhalla 2025 default | our T-SAE | ratio |
|---|---:|---:|---|
| **temp_alpha** (contrastive weight) | **0.1** | **1.0** | **10× too aggressive** |
| **max_steps** | 200,000 | 30k / 100k | 2–7× fewer |
| **k** (sparsity) | **20** (BatchTopK) | 128 (per-token TopK) | 6× denser |
| **batch_size** | 512 | 256 | 2× smaller |
| **d_sae** | 16,384 | 32,768 | 2× larger |
| lr | 3e-4 | 3e-4 | ✓ |

Most suspect: the 10× contrastive weight (over-aligning z(t) and z(t+1) at the expense of recon) combined with 6× denser top-K (smearing each feature's signal across many positions). Both predict that our T-SAE features should be **less specialized** and therefore **less Wang-bundle-actionable** than paper-faithful ones — exactly what we saw.

### Re-run setup

Trained T-SAE on Qwen-7B-Instruct layer-15 resid_post with paper-faithful hyperparameters and only **30k steps** (1/7 of the paper's 200k budget):

- `d_sae=16384`, `k=20` BatchTopK, `batch_size=512`, `lr=3e-4`, `contrastive_alpha=0.1`
- BatchTopK during training (global top-(N·k) across the batch); per-token TopK at inference for compatibility with our existing finder + steering pipeline
- Same andyrdt PEFT-LoRA EM organism, same streaming corpus, same Wang procedure (encoder Δz̄ → causal screen → coherence-aware strength sweep → 27-α frontier on top-3 finalists → bundle k=30)

Training run: 92 min on H100 (3 min / 1k steps). Best loss = **0.485** at step 20k. Dead-feature fraction stayed high (~92%) throughout — expected with k=20 + d_sae=16k + Bricken-style auxk loss; the surviving features specialize tightly, which is the whole point.

### Headline result

**T-SAE 30k @ resid_post, paper-faithful:**
- Peak α = −6
- Peak alignment = **56.23**
- Peak coherence = **34.84**
- α=0 baseline = 40.75 → causal lift Δ = **+15.48 align points**

**Comparison vs all other 30k-step variants on this organism:**

| variant (30k or 100k) | hookpoint | peak α | peak align | peak coh | Δ vs α=0 |
|---|---|---:|---:|---:|---:|
| SAE arditi 100k (champion) | resid_post | −10 | **57.42** | 35.78 | +15.34 |
| **T-SAE 30k paper-faithful** | resid_post | −6 | **56.23** | **34.84** | **+15.48** |
| TXC brickenauxk 30k (ours) | resid_mid | −8 | 53.87 | 29.77 | +10.48 |
| T-SAE 100k (our settings) | resid_post | −6 | 52.39 | 27.58 | +10.49 |
| TXC brickenauxk 30k (ours) | ln1_normalized | +7 | 51.61 | 24.14 | +8.34 |
| Han 100k | resid_post | −5 | 50.82 | 26.48 | — |
| T-SAE 30k (our settings) | resid_post | −10 | 49.84 | 27.42 | +5.67 |
| T-SAE 30k (our settings) | resid_mid | −1.75 | 50.00 | 26.33 | +6.61 |
| T-SAE 30k (our settings) | ln1_normalized | −1.75 | 49.92 | 23.52 | +5.65 |

**Three observations:**

1. **Paper-faithful 30k T-SAE essentially matches SAE arditi 100k.** 56.23 vs 57.42 align — within 1.2 points — at **1/3 the training budget** (30k vs 100k steps) and **half the dictionary size** (16k vs 32k). The causal lift is actually slightly larger (+15.48 vs +15.34).
2. **The +6.4 jump from "our T-SAE" to "paper T-SAE" came purely from hyperparameter changes, not training time.** Both are 30k-step runs. The difference is the 5 mismatched hyperparameters listed above. So our previous "T-SAE underperforms SAE on this organism" conclusion was largely an artifact of the 10× contrastive weight + 6× denser top-K, not a property of the Bhalla 2025 recipe.
3. **TXC at our (k=128, per-token) settings (53.87 at resid_mid) sits between paper-faithful T-SAE (56.23) and our-settings T-SAE (49.84).** This raises the obvious follow-up: TXC at paper-faithful k=20 BatchTopK might land at or above SAE arditi.

### What changed in the codebase

- `architectures/tsae_adjacent_contrastive.py`: added `batch_topk: bool = False` constructor flag and a `_encode_batch_topk` method that does a single global TopK over the (N·d_sae) flattened pre-activation matrix with budget N·k. Inference path always uses per-token TopK so the existing finder and frontier_sweep treat the saved checkpoint identically.
- `run_training_tsae.py`: `--batch_topk` flag, saves into ckpt config.
- `run_find_features_encoder.py`, `run_wang_procedure.py`, `frontier_sweep.py`: pass `batch_topk` through when reconstructing the model from a checkpoint config.

### Caveat

Our 30k re-run is still only 15% of the paper's 200k step budget. Bhalla 2025's published T-SAE numbers come from the full 200k run; ours might lose a small amount of headroom by training shorter. Conversely, the paper trains on Pythia-160m / Gemma2-2b on The Pile, while we train on Qwen-7B-Instruct on a custom mix (UltraChat + Pile slice). So none of our absolute numbers are directly the paper's numbers — what we have is a **like-for-like Wang procedure comparison** of our T-SAE-our-settings vs T-SAE-paper-settings on the same EM organism, holding everything else fixed.

### Files

- `tsae_paper_30k/results/wang_tsae_paper_k20_d16k_step30000_bundle30_frontier.json`
- `tsae_paper_30k/results/wang_tsae_paper_k20_d16k_step30000/stage{2,3,4}_*.json`
- `tsae_paper_30k/results/qwen_l15_tsae_paper_k20_d16k_step30000_encoder/top_200_features.json`
- Checkpoints on HF: `dmanningcoe/temp-xc-em-features:tsae/qwen_l15_tsae_paper_k20_d16k_a01_step{10000,20000,30000}.pt`
- Updated 9-panel figure: `txc_vs_tsae_frontier_panels.png`

### Open follow-up

Re-run **TXC brickenauxk** with paper-faithful sparsity (k_total=20 BatchTopK, d_sae=16k, batch=512, contrastive_alpha=0.1 if applicable) at resid_post for a fair comparison. Given that:

- TXC at our-settings beats T-SAE at our-settings by ~3–4 align across hookpoints
- T-SAE at paper-settings beats T-SAE at our-settings by ~6.4 align

Naive extrapolation would put TXC-paper-settings around 58–60 align, which would beat the SAE arditi champion. Worth confirming or refuting.
