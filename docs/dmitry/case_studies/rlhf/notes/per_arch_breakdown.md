---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - complete
---

## Per-arch, per-strength breakdown

Each cell is `(mean success, mean coherence)` over 30 concepts, single seed. Both grades are 0-3 Sonnet 4.6 scores (paper §B.2 prompts verbatim).

### Paper protocol — clamp-on-latent + error preserve

Strengths are absolute clamp values for `z[j]`.

| arch | s=10 | s=100 | s=150 | s=500 | s=1000 | s=1500 | s=5000 | s=10000 | s=15000 |
|---|---|---|---|---|---|---|---|---|---|
| `topk_sae` | (0.33, 2.57) | **(1.07, 1.40)** | (0.93, 0.87) | (0.13, 0.73) | (0.17, 0.83) | (0.27, 0.77) | (0.40, 0.83) | (0.37, 0.80) | (0.47, 0.80) |
| `tsae_paper_k500` | (0.27, 2.50) | **(1.33, 1.50)** | (1.20, 1.10) | (0.17, 0.80) | (0.27, 0.80) | (0.37, 0.83) | (0.27, 0.83) | (0.37, 0.87) | (0.33, 0.90) |
| `tsae_paper_k20` | (0.37, 2.77) | **(1.93, 1.37)** | (1.90, 1.07) | (0.20, 0.97) | (0.37, 0.73) | (0.30, 0.83) | (0.30, 0.87) | (0.37, 0.87) | (0.37, 0.87) |
| `agentic_txc_02` (T=5) | (0.17, 2.80) | (0.30, 2.40) | (0.43, 1.80) | **(0.97, 1.20)** | (0.63, 0.93) | (0.37, 0.97) | (0.13, 0.93) | (0.23, 0.97) | (0.20, 0.93) |
| `phase5b_subseq_h8` (T=10) | (0.27, 3.00) | (0.23, 2.30) | (0.30, 2.03) | **(1.10, 1.53)** | (0.30, 1.07) | (0.17, 1.03) | (0.20, 1.07) | (0.33, 1.00) | (0.37, 1.00) |
| `phase57_partB_h8_bare_multidistance_t5` (T=5) | (0.27, 2.97) | (0.30, 2.13) | (0.43, 1.70) | **(1.13, 1.10)** | (0.43, 0.97) | (0.30, 0.97) | (0.43, 1.00) | (0.40, 1.00) | (0.43, 1.00) |

**Per-token archs peak at s=100. Window archs peak at s=500.** Same hardware, same prompt, same concept set — the 5× shift is purely architectural.

### AxBench-extended — additive (decoder-direction multiplier)

Strengths are signed unit-norm decoder-direction multipliers; s=0 is a no-op control.

| arch | s=-100 | s=-50 | s=-25 | s=-10 | s=0 | s=10 | s=25 | s=50 | s=100 |
|---|---|---|---|---|---|---|---|---|---|
| `topk_sae` | (0.27, 0.83) | (0.13, 0.93) | (0.07, 1.27) | (0.03, 2.27) | (0.10, 2.93) | (0.40, 2.83) | (0.83, 2.30) | **(0.97, 1.33)** | (0.83, 1.10) |
| `tsae_paper_k500` | (0.43, 0.87) | (0.30, 0.97) | (0.20, 1.43) | (0.07, 2.50) | (0.10, 2.93) | (0.30, 2.93) | (0.83, 2.27) | **(1.30, 1.23)** | (1.27, 1.13) |
| `tsae_paper_k20` | (0.27, 0.93) | (0.17, 1.07) | (0.10, 1.93) | (0.10, 2.83) | (0.10, 2.93) | (0.27, 2.93) | (1.07, 2.20) | **(2.00, 1.13)** | (1.50, 1.10) |
| `agentic_txc_02` (T=5) | (0.27, 0.97) | (0.20, 1.07) | (0.07, 1.43) | (0.03, 2.30) | (0.10, 2.93) | (0.30, 2.83) | (0.97, 2.10) | **(1.53, 1.17)** | (1.30, 1.10) |
| `phase5b_subseq_h8` (T=10) | (0.30, 1.07) | (0.23, 1.10) | (0.10, 1.40) | (0.07, 2.27) | (0.10, 2.93) | (0.27, 2.97) | (0.97, 2.43) | **(1.67, 1.27)** | (1.20, 1.10) |
| `phase57_partB_h8_bare_multidistance_t5` (T=5) | (0.43, 1.10) | (0.27, 1.13) | (0.13, 1.43) | (0.03, 2.50) | (0.07, 2.93) | (0.30, 2.93) | (0.93, 2.40) | (1.37, **1.53**) | (1.37, 1.07) |

**All archs peak at s=50 or s=100 under AxBench.** The peak strength is consistent across families because the unit-norm decoder direction removes the activation-magnitude offset.

### Negative-strength reading (AxBench-extended only)

Negative strengths nominally "reverse" the steering direction. Empirically, success drops to 0.1-0.4 (low / chance) at s=-10 already, and coherence collapses to 0.8-1.1 at s=-100. So **negative steering doesn't induce an opposite concept** — it just degrades coherence in a different direction without semantic gain. This rules out a clean "anti-feature" interpretation: the SAE feature direction is meaningful in *one* sense (positive steering pushes toward concept) but not in the other (negative doesn't push toward an opposite concept; just toward incoherence).

### Peak comparison summary

| arch | paper peak | AxBench peak | match? |
|---|---|---|---|
| `topk_sae` | (1.07, 1.40) | (0.97, 1.33) | within noise |
| `tsae_paper_k500` | (1.33, 1.50) | (1.30, 1.23) | within noise |
| `tsae_paper_k20` | (1.93, 1.37) | (2.00, 1.13) | within noise on suc; coh down 0.24 |
| `agentic_txc_02` | (0.97, 1.20) | (1.53, 1.17) | **success +0.56 under AxBench** |
| `phase5b_subseq_h8` | (1.10, 1.53) | (1.67, 1.27) | **success +0.57 under AxBench** |
| `phase57_partB_h8_bare_multidistance_t5` | (1.13, 1.10) | (1.37, 1.53) | success +0.24, coh +0.43 under AxBench |

Per-token archs: protocol-invariant. Window archs: substantially better under AxBench. Hypothesis: window archs benefit from the unit-norm rescaling because their decoder atoms have larger raw norms (encoder integrates over T tokens, so the corresponding decoder needs higher norm to reconstruct the larger summed signal). Under paper-clamp, the *raw* `W_dec[:, j]` is what gets injected — and at strength=100, that injection magnitude is already past the per-arch sweet spot. Under AxBench-additive, the unit-norm step neutralizes this.
