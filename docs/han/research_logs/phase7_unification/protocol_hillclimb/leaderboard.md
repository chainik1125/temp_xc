---
author: Han
date: 2026-05-03
tags:
  - results
  - in-progress
---

## Steering protocol hill-climb leaderboard — Galaxy 23 (T = 5 SoftMaxPool)

> Updated incrementally as new protocols are run. All numbers are 3-seed
> mean-curve Δ vs same-pod n=3 anchor (1.133 / 0.411 / 0.411 at
> coh ≥ 1.5/1.75/2.0).

### Current leaderboard (sorted by Δ at coh ≥ 1.75)

| rank | protocol | peak15 | **Δ ≥ 1.75** | Δ ≥ 2.0 | status |
|:--:|---|---:|---:|---:|---|
| **1** | **V9 sliding-TB (stride T/2)** | 1.156 | **+0.745** | −0.022 | ⭐ new best |
| 2 | V13 stride1-TB | 1.100 | +0.689 | −0.011 | marginal vs V7 |
| 3 | V7 tiled-broadcast | 1.089 | +0.678 | +0.022 | iter-0 baseline |
| 4 | V10 encmag-TB | 1.100 | +0.267 | +0.267 | ❌ FAILS — encmag weighting hurts |
| 5 | V1 right-edge | 0.956 | +0.233 | +0.233 | iter-0 baseline |

V9 beats V7 by **+0.067** at coh ≥ 1.75 — sliding stride T/2 + averaged
overlapping deltas. Next iteration: try other strides (T/3, T/4) and
attention-aware variants to push further.

Still short of T=2/T=3 best (~+1.0) by ~+0.25.

### Iteration history

| iter | protocol | designed-against | result | kept? |
|:--:|---|---|---|:--:|
| 0 | V1 RE | baseline | Δ=+0.233 | baseline |
| 0 | V7 TB | baseline | Δ=+0.678 | baseline |
| 1 | V9 sliding-TB | V7's sparse coverage at T=5 | **Δ=+0.745** | ⭐ **WIN** — new best |
| 1 | V13 stride1-TB | maximize coverage | Δ=+0.689 | TIE (no improvement vs V7) |
| 1 | V10 encmag-TB | encoder-position structure | Δ=+0.267 | ❌ FAIL (worse than V7) |

### Lessons learned (iter 1)

- **Stride matters monotonically up to a point**: T (V7, +0.678) → T/2 (V9, +0.745) → 1 (V13, +0.689). Sweet spot around T/2-T/3.
- **Encoder-magnitude weighting HURTS** for Galaxy 23 — soft-max-pool's encoder doesn't have strong per-position concentration; weighting introduces noise.
- The +0.067 V9-over-V7 lift suggests denser coverage helps; the V13 plateau
  shows that beyond a certain density (overlapping all positions T times),
  averaging dilutes the signal back.

### Stop criteria

- ✓ Match T=2/T=3 best cells: Δ ≥ +1.0 at coh ≥ 1.75 → STOP, paper-grade.
- 5+ failed iterations → declare the high-T ceiling.

### Cross-arch validation (run on best protocol once leaderboard converges)

V9 will be tested on:
- `txc_bare_antidead_t5_kpos20` (T=5 vanilla TXC, 2-seed)
- `txc_h8_t5_kpos20_shifts5` (T=5 H8 contrastive, 2-seed)

(Pending — once we exceed Δ ≥ +0.85 or hit iteration cap)
