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
