---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - complete
---

## Bhalla 2025 clamping on SAE 100k — comparable to additive single-feature, beaten by causal bundle

We implemented the temporal-SAE clamping procedure from [Bhalla et al. 2025](https://arxiv.org/abs/2511.05541) verbatim — encode `x` through the SAE, set one feature's activation to a constant `c`, decode, and add the reconstruction error `ε = x − decoder(z)` back. Mathematically (for a linear TopK decoder) this reduces to additive steering with context-dependent coefficient `(c − z[i])`, but the explicit round-trip is what the paper specifies, so that's what `experiments/em_features/clamp_sweep.py` does.

### Setup

- Architecture: SAE arditi v2 100k (k=128, d_sae=32k, layer 15 resid_post).
- Feature clamped: **30316** — the Wang stage-2 causal champion identified in our [Wang follow-up](../wang/causal_screen_finding.md) as the SAE feature with the largest screen score (+26.98 align swing between α=−1 and α=+1).
- Clamp grid (absolute z-magnitudes): {0, 0.5, 1, 2, 5, 10, 30, 100, 300, 1000, 3000}. The natural firing magnitude of feat 30316 on EM prompts is mean=0.10, max=3.2, frac_active=6%, so anything ≥ 30 is well above natural scale.
- Generation: identical to `frontier_sweep.py` (same chat template, `add_special_tokens=False`, default Qwen tokenizer padding, do_sample=True, temperature=1.0, no top_p/top_k); only the layer hook differs. Seed=42, deterministic CUDA.
- Judge: gemini-3.1-flash-lite-preview (with gemini-2.5-flash fallback), temp=0.5.

### Results

| c | align | coh | Δalign vs c=0 |
|---:|---:|---:|---:|
| 0 | 44.49 | 24.77 | — |
| 0.5 | 45.34 | 24.14 | +0.85 |
| 1 | 37.50 | 23.59 | −7.0 |
| 2 | 50.00 | 23.44 | +5.5 |
| 5 | 46.10 | 19.84 | +1.6 |
| 10 | 44.56 | 22.97 | +0.07 |
| **30** | **50.64** | **29.92** | **+6.15** |
| 100 | 23.44 | 19.84 | −21 |
| 300 | 6.95 | 0.00 | collapse |
| 1000 | 2.03 | 0.00 | collapse |
| 3000 | 0.00 | 0.00 | collapse |

Peak alignment is at **c=30** with **align=50.64, coh=29.92** — about 100× the feature's natural firing magnitude. Below c=10 the perturbation is too small to affect generation reliably; above c=100 the model collapses into gibberish (same shape as the α=±100 endpoints in additive sweeps).

### How this compares on SAE 100k

| procedure | peak align | peak coh | Δalign vs α=0 baseline |
|---|---:|---:|---:|
| cosine k=10 bundle | 51.83 | 26.88 | +8.2 |
| encoder Δz̄ k=10 bundle | 48.56 | 24.77 | +5.0 |
| encoder Δz̄ k=1 single feature | 53.11 | 26.02 | +9.5 |
| **Wang bundle k=30 (additive)** | **57.42** | **35.78** | **+13.8** |
| **clamp single feat 30316** | **50.64** | **29.92** | **+6.2** |

Clamping a single feature lands between encoder k=10 and encoder k=1 — comparable to additive single-feature steering, decisively beaten by the Wang causal-bundle k=30. Within statistical noise of the encoder k=1 result (53.11), which makes sense: for a linear TopK decoder, single-feature clamping with error preserved is mathematically equivalent to additive `(c − z[i]) · W_dec[i]`.

### What we got from this exercise

1. **Confirmed the math.** Clamping with error preserved reproduces the magnitude of single-feature additive steering. No surprise — the operations are equivalent up to context-dependent coefficient.

2. **Identified an experimental hazard.** Our first attempt produced suspicious "coh = 89" results because the custom generation pipeline differed from `generate_longform_completions` (`add_special_tokens=True` was double-tokenizing the chat template, no PEFT-merge details). Once we mirrored the library's generation exactly, the c=0 baseline collapsed to 44.49 (matching the additive α=0 baseline of ~43.6). **Lesson: any new intervention must reproduce the unsteered baseline first; if c=0 doesn't match the library's α=0, something is wrong with the harness, not the procedure.**

3. **Bundled approaches still dominate.** The Wang causal bundle k=30 holds its lead. If we wanted Bhalla-style clamping to compete, we'd need to clamp the entire bundle — set z[i₁]=c₁, …, z[i₃₀]=c₃₀ simultaneously. That's a natural follow-up but probably yields similar results to the additive bundle (since clamping a *previously-zero* feature is exactly additive `c · W_dec[i]`, and most of the bundle's 30 features fire in only ~5-10% of tokens).

### Files

- Sweep JSON: `clamp_sae_step100000_feat30316.json`
- Code: `experiments/em_features/clamp_sweep.py`
- HF ckpt for the SAE: `dmanningcoe/temp-xc-em-features:sae/v2_qwen_l15_sae_arditi_k128_step100000.pt`

### Open follow-up

The natural next experiment is **bundle-clamping**: take the same 30 Wang causal champions, clamp each `z[iⱼ] = c` simultaneously, sweep c. If the math intuition is right (clamping a previously-silent feature ≡ additive `c · W_dec[i]`), this should give very similar results to the Wang additive bundle k=30 peak (57.42). If it gives substantially different results, something more subtle is going on with multi-feature interactions through the encoder's TopK gate.
