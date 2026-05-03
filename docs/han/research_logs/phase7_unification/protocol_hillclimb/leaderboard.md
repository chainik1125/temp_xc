---
author: Han
date: 2026-05-03
tags:
  - results
  - in-progress
---

## Steering protocol hill-climb leaderboard — Galaxy 23 (T=5 SoftMaxPool)

> 3-seed mean-curve Δ vs same-pod n=3 anchor (1.133 / 0.411 / 0.411 at coh ≥ 1.5/1.75/2.0).

### Final iter 2 leaderboard (sorted by Δ ≥ 1.75)

| rank | protocol | peak15 | **Δ ≥ 1.75** | Δ ≥ 2.0 | iter | status |
|:--:|---|---:|---:|---:|:--:|---|
| **1** | **V9 sliding-TB stride 2** | 1.156 | **+0.745** | −0.022 | 1 | ⭐ best (+0.067 vs V7) |
| 2 | V13 stride1-TB | 1.100 | +0.689 | −0.011 | 1 | tie |
| 2 | V14 multi-scale TB | 1.100 | +0.689 | +0.000 | 2 | NEW — no lift |
| 4 | V7 TB | 1.089 | +0.678 | +0.022 | 0 | baseline |
| 4 | V16 stride3-TB | 1.089 | +0.678 | −0.011 | 2 | exactly V7 |
| 6 | V10 encmag-TB | 1.100 | +0.267 | +0.267 | 1 | ❌ encoder-mag weight HURTS |
| 7 | V1 RE | 0.956 | +0.233 | +0.233 | 0 | baseline |
| 8 | **V15 attn-weighted-TB** | 0.922 | **−0.033** | −0.033 | 2 | ❌ FAILS BIG |

### Stride sweep — sweet spot at stride 2

| stride | protocol | Δ ≥ 1.75 |
|:--:|---|---:|
| 1 | V13 | +0.689 |
| **2** | **V9** | **+0.745** |
| 3 | V16 | +0.678 |
| 5 (= T) | V7 | +0.678 |

V9 is the unique local max. Other strides cluster at +0.678-0.689.

### Conclusion (iter 2)

Protocol space at T=5 appears EXHAUSTED. V9 stride 2 wins by +0.067 over
V7 baseline; nothing else beats V7. The +0.25 gap to T=2/T=3 best (~+1.0)
is likely **ARCHITECTURAL**, not protocol.

Next: cross-arch V9 validation + architectural fix attempt
(higher-k_pos Galaxy 23 variant) + ENSEMBLE protocol (T=3 V7 ⊕ T=5 V9
per-concept best).
