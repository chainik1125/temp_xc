---
author: Han
date: 2026-05-02
tags:
  - design
  - in-progress
---

## Steering protocol hill-climb — iteration log

> One entry per iteration. Records what was tried, the result, and the
> next-iteration hypothesis.

### Iteration 0 — baselines

Three protocols run and graded on Galaxy 23 (T=5 SoftMaxPool) 3-seed:

| protocol | Δ ≥ 1.5 | Δ ≥ 1.75 | Δ ≥ 2.0 |
|---|---:|---:|---:|
| V1 right-edge | −0.178 | +0.233 | +0.233 |
| V7 tiled-broadcast | −0.044 | +0.678 | +0.022 |

V7 wins at the GIGABRAIN coh ≥ 1.75 metric. Sets the bar for hill-climb.

### Iteration 1 — V9 sliding tiled-broadcast (planned)

**Design**: V7 has stride T (non-overlapping) → only ⌊S/T⌋ ≈ 12 blocks.
V9 = stride 2 (or 3) sliding window with uniform δ within each window.
At stride 2, S = 64 → 31 windows → much denser steering coverage.
Overlapping writes accumulate at shared positions (mean across overlapping
windows or sum — to be decided).

**Hypothesis**: V9 retains V7's within-window attention-invariance AND
gives denser per-position coverage. Could close the V7-T=5 gap to
T=2/T=3 best cells.

**Risk**: overlapping writes may double-count. Sensitivity to stride
choice unclear.

**Implementation**:
- File: `experiments/phase7_unification/case_studies/protocol_hillclimb/intervene_v9_sliding_tb.py`
- Output dir: `steering_protocol_hillclimb_v9_stride2{,_seed1,_seed2}`
- Aggregation: average over overlapping windows for each position

### Iteration plan (subject to revision based on iter results)

1. V9 sliding tiled-broadcast (stride 2)
2. V10 encoder-magnitude weighted broadcast (within non-overlapping blocks,
   write δ in proportion to W_enc per-position magnitudes)
3. V11 decoder-magnitude weighted broadcast (symmetric to V10 using W_dec)
4. V12 hybrid PP-last + broadcast-earlier
5. V13 stride-1 totally-overlapping tiled-broadcast
6. V14 multi-scale tiled-broadcast (T=5 + T=3 + T=1 simultaneously)

### Iteration 1 — RESULTS

Three candidates run on Galaxy 23 T=5 3-seed:

| protocol | Δ≥1.75 vs V7 baseline (+0.678) | verdict |
|---|---:|---|
| **V9 sliding-TB** (stride T/2) | **+0.745** (+0.067) | ⭐ KEEP — new best |
| V13 stride1-TB | +0.689 (+0.011) | tie — too dense, signal dilutes |
| V10 encmag-TB | +0.267 (−0.411) | ❌ FAIL — encoder-mag weighting noise |

**Hypothesis confirmed**: denser per-position coverage helps at high T. V9
beats V7 by +0.067 (small but real); V13 confirms the limit (stride-1
doesn't help further beyond stride T/2).

**Surprise**: V10 (encoder-magnitude weighted) UNDERPERFORMS V7 by
−0.411. The Galaxy 23 (soft-max-pool) encoder doesn't have strong
per-position concentration, so weighting by ||W_enc[t]|| introduces
noise rather than signal.

### Iteration 2 — planned candidates

Now we have an EXPLORATION GRADIENT: stride matters, with sweet spot
around T/2. Next ideas to push past +0.745:

- **V11 — Decoder-magnitude weighted** (symmetric to V10 using W_dec):
  use ||W_dec[t]|| instead of ||W_enc[t]||. Different feature info
  (where the feature WRITES vs READS).

- **V12 — Hybrid stride T/2 + position-aware**: V9 with one extra trick —
  at the LAST T-block, apply per-position writes (V2-style); at earlier
  blocks, uniform broadcast.

- **V14 — Multi-scale tiled-broadcast** (T=5 + T=3 + T=1 simultaneously,
  each contributing its own δ summed).

- **V15 — Attention-weighted broadcast** within each window. Use the
  TRAINED encoder's position weights (learned softmax-pool weights) to
  weight the per-position decoder writes.

- **V16 — Stride T/3 sliding-TB** (variant of V9 with 1 stride finer).

Priority: V14 (multi-scale) and V15 (attention-weighted) most novel; V11
and V16 are direct extensions to confirm the stride-monotonic story.

### Iteration 2 — IN FLIGHT (V14, V15, V16)

Three protocols launched at 00:35 UTC; ETA done by ~02:23 UTC.

### Mystery investigation (parallel to iter 2)

**Content-vs-discourse trade-off (NEW finding, 2026-05-03)**:

Per-concept peak succ at coh ≥ 1.75 — Galaxy 18 (T=3) V7 vs Galaxy 23 (T=5) V7:

| concept | G18 (T=3) | G23 (T=5) | gap |
|---|:--:|:--:|:--:|
| financial, harmful_content, programming | 3 | 0 | **+3** ⭐ |
| medical, narrative | 3 | 1 | **+2** |
| formal_register, instructional, negative_emotion, technical_jargon | 2-3 | 1-2 | **+1** |
| casual_register, code_context, geographical | 1-3 | 3 | **−2 to −3** |
| imperative_form, positive_emotion | 0 | 1 | −1 |

Mean per-concept peak: G18=1.33, G23=1.03 (gap 0.30; vs aggregate
peak15 gap of 0.36 — consistent).

**Interpretation**: T=3 wins for **content-keyword** concepts (specific
terminology that fires on 1-2 tokens). T=5 wins for **discourse-style**
concepts (register, code-vs-prose context, geographical reference).

This explains why no single (arch, protocol) wins universally — the
optimal T depends on the **concept's intrinsic span**.

**Implication for paper**: instead of "best single protocol", report
**(T, protocol) selection per concept type**. Or: an ENSEMBLE of
(T=2, T=3, T=5) under their respective best protocols, voting per concept.

**Implication for hill-climb**: V14 multi-scale (T-scale + T_mid + 1-scale
combined) IS this idea — sums multiple-T deltas. If V14 wins, the
multi-scale story is the answer.

If V14 plateaus, then high-T's content-concept gap is an
**architectural** limit (per-position feature density), not a protocol
issue. Fix: train a higher-k_pos variant of Galaxy 23.
