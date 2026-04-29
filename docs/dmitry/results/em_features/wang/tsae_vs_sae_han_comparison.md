---
author: Dmitry
date: 2026-04-29
tags:
  - results
  - complete
---

## T-SAE vs SAE arditi vs Han champion: Wang causal attribution + bundle steering

After implementing the Bhalla et al. 2025 ([arXiv:2511.05541](https://arxiv.org/abs/2511.05541)) Temporal SAE — a per-token TopK SAE plus an adjacent-token contrastive loss — and training it on base Qwen-7B-Instruct activations to 30k and 100k steps, we ran the full Wang ([arXiv:2506.19823](https://arxiv.org/abs/2506.19823)) attribution + intervention procedure (encoder Δz̄ → causal screen → coherence-aware strength sweep → 27-α frontier on top-3 finalists) and the bundle-k=30 follow-up.

### Headline cross-arch comparison (best −α with coh ≥ 90% baseline)

| arch (step) | method | peak α | peak align | peak coh |
|---|---|---:|---:|---:|
| SAE 100k | cosine k=10 | −5 | 51.83 | 26.88 |
| SAE 100k | encoder Δz̄ k=10 | +3 | 48.56 | 24.77 |
| **SAE 100k** | **Wang bundle k=30** | **−10** | **57.42** | **35.78** |
| Han 100k | cosine k=10 | −4 | 52.38 | 26.64 |
| Han 100k | Wang bundle k=30 | −5 | 50.82 | 26.48 |
| T-SAE **30k** | Wang bundle k=30 | −10 | 49.84 | 27.42 |
| **T-SAE 100k** | **Wang bundle k=30** | **−6** | **52.39** | **27.58** |

(All on the same setup: Qwen-7B-Instruct + andyrdt bad-medical PEFT, layer-15 resid_post, deterministic CUDA + seed 42, Gemini judge with `gemini-3.1-flash-lite-preview` → `gemini-2.5-flash` fallback at temp=0.5.)

### What landed

**T-SAE 100k Wang bundle k=30**: peak align 52.39 at α=−6 with coh 27.58. That's a +2.55 alignment improvement over the T-SAE 30k bundle (49.84/27.42) — the extra 70k training steps materially helped, but not by enough to overtake the SAE bundle.

**T-SAE 100k stage 4 finalists** (full 27-α frontier per finalist):
- feat 13695 — peak α=−2, align 48.79, coh 24.45 (clean)
- **feat 30874** — peak α=+8, align **55.56**, coh **36.02** (clean *positive*-α peak; opposite direction from typical, but very high coh)
- feat 17143 — peak α=−2, align 50.42, coh 24.38

The unusual positive-α peak for feat 30874 (high align *and* high coh) suggests this T-SAE feature behaves opposite to the Wang screen prediction. The screen ranked it +21.83 (`align(α=−1) − align(α=+1)`), implying −α should be aligned, but the full frontier shows the cleanest peak is at +8.

### How it stacks up

1. **SAE arditi 100k bundle k=30 is still the headline.** 57.42 align / 35.78 coh remains the highest of any arch we've tested.
2. **T-SAE 100k Wang bundle hits 52.39** — comparable to Han 100k cosine k=10 (52.38) and Han 100k bundle (50.82). The Bhalla 2025 contrastive objective doesn't transfer the bundle-causal advantage that plain SAE gets in this setup.
3. **Training-step effect is real but small.** T-SAE 30k → 100k = +2.55 align with same coherence. Probably wouldn't close the 5-point gap to the SAE bundle even at 200k.

### Why T-SAE doesn't beat SAE here

A few possibilities:

- **Bhalla 2025's setup is Gemma-2 + matryoshka SAE**, with the contrastive loss specifically helping disentangle "high-level semantic" vs "low-level syntactic" features for *interpretability* (their headline metric is per-feature interpretability, not steering effectiveness). Their evaluation is also on factual / persona steering, not emergent misalignment.
- **The contrastive loss may flatten causally-relevant feature firing patterns**: if a misalignment-mediating feature naturally fires on a single specific token (e.g., the chat-template scaffold's `<|im_start|>` boundary), the adjacent-token contrastive penalty would *suppress* that pattern in favor of slow-varying features, making causal champions harder to recover.
- **Per-token TopK + adjacent contrastive is structurally similar to Han's matryoshka + multi-distance contrastive** — but Han trains explicitly on T=5 windows and gets a richer feature decomposition. T-SAE's T=1 encoder gives away that windowed information without using it during inference.

### What the cross-arch picture says

For our Qwen-7B + bad-medical-LoRA emergent-misalignment setup:

- **SAE arditi (k=128, d_sae=32k, T=1, no auxiliary loss)** + Wang causal pruning + bundle-k=30 = best result we've found.
- **Han champion** (T=5 windowed + multi-distance contrastive + matryoshka) gets within 5 align points of SAE bundle but doesn't beat plain cosine k=10 with the Wang screen.
- **T-SAE** (T=1 + adjacent contrastive) lands between SAE and Han at 100k. Doesn't justify its training cost vs plain SAE in this setup.

The pattern suggests: for emergent-misalignment steering on PEFT-LoRA EM organisms, **simpler architectures + better attribution (Wang causal screen) beat more complex SAE recipes**. The contrastive losses in Han and T-SAE seem to spread the misalignment signal across more features rather than concentrating it.

### Files

- `tsae_30k_*.json`, `tsae_100k_*.json` — full Wang procedure outputs at both training steps
- `sae_*_frontier.json`, `han_*_frontier.json` — existing v2 100k results for cross-comparison
- T-SAE checkpoints on HF: `dmanningcoe/temp-xc-em-features:tsae/qwen_l15_tsae_k128_step{10000,20000,30000,50000,80000,100000}.pt`
- Code: `experiments/em_features/architectures/tsae_adjacent_contrastive.py`, `run_training_tsae.py`

### Open question

Is feat 30874's positive-α peak (55.56 align, 36.02 coh) genuine misalignment-amplification-via-suppression-of-aligned-feature, or a Wang-screen mis-ranking artifact? Cheap follow-up: run the full 27-α frontier on a few more T-SAE high-screen features and check if the +α-clean-peak phenomenon repeats. If it does, T-SAE may be encoding misalignment direction with reversed sign convention vs SAE/Han, which would be a meaningful finding about how the contrastive loss reshapes feature directions.
