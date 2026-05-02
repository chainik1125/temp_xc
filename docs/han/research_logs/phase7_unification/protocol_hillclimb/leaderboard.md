---
author: Han
date: 2026-05-02
tags:
  - results
  - in-progress
---

## Steering protocol hill-climb leaderboard — Galaxy 23 (T = 5 SoftMaxPool)

> Updated incrementally as new protocols are run. All numbers are 3-seed
> mean-curve Δ vs same-pod n=3 anchor (1.133 / 0.411 / 0.411 at
> coh ≥ 1.5/1.75/2.0).

### Current leaderboard (sorted by Δ at coh ≥ 1.75)

| rank | protocol | Δ ≥ 1.5 | **Δ ≥ 1.75** | Δ ≥ 2.0 | status |
|:--:|---|---:|---:|---:|---|
| 1 | V7 tiled-broadcast | −0.044 | **+0.678** | +0.022 | baseline (best so far) |
| 2 | V1 right-edge | −0.178 | +0.233 | +0.233 | baseline |
| – | V2 per-position | – | – | – | not run (predicted to fail at T=5) |
| – | V8 encoded-broadcast | – | – | – | not run on T=5 |

### Iteration history

(updated as iterations complete)

| iter | protocol | designed-against | result | kept? |
|:--:|---|---|---|:--:|
| 0 | V1 RE | baseline | Δ=+0.233 | baseline |
| 0 | V7 TB | baseline | Δ=+0.678 | baseline (best) |
| – | – | – | – | – |

### Stop criteria

- ✓ Match T=2/T=3 best cells: Δ ≥ +1.0 at coh ≥ 1.75 → STOP, paper-grade.
- 5+ failed iterations → declare V7 as high-T ceiling.
- Cross-arch generalization (works for T=2 H8 too) → MAJOR finding.

### Cross-arch validation (run on best protocol once leaderboard converges)

Will be filled in once a winner emerges. Test on:
- `txc_bare_antidead_t5_kpos20` (T=5 vanilla TXC, 2-seed)
- `txc_h8_t5_kpos20_shifts5` (T=5 H8 contrastive, 2-seed)
