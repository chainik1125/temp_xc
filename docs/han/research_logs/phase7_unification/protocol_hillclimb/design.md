---
author: Han
date: 2026-05-02
tags:
  - design
  - in-progress
---

## Steering protocol hill-climb — design

> Han 2026-05-02: design a controlled hill-climb on TXC steering protocols.
> Fix arch, vary protocol; iterate to find a protocol that DOMINATES at high T
> (T = 5) where V1 RE / V2 PP / V7 tiled-broadcast all leave significant gaps.

### Fixed setup

- **Primary arch**: `txc_softmaxpool_t5_kpos20` (Galaxy 23 — Galaxy 8 family at T=5, 3 seeds)
- **Secondary archs (n=2 seeds)**: `txc_bare_antidead_t5_kpos20`, `txc_h8_t5_kpos20_shifts5`
- **Concepts**: 30 AxBench-style concepts (canonical phase 7 set)
- **Strength grid**: paper-clamp normalised `s_norm ∈ {0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0}` × per-arch z_orig magnitude
- **Anchor**: T-SAE k=20 same-pod n=3 (1.133 / 0.411 / 0.411 / 0.411 / 0.411 at coh ≥ 1.5/1.75/2.0/2.25/2.5)
- **Metric**: Δ at coh ≥ 1.75 (3-seed mean-curve), with secondary tracking of Δ at coh ≥ 1.5 / 2.0
- **Sparsity**: matched per-token k_pos = 20 at all archs

### Why T = 5 specifically

The protocol-by-T pattern from the V1/V2/V7/V8 sweep:
- T = 2: V2 PP wins (Δ ≈ +1.0)
- T = 3: V7 tiled-broadcast wins (Δ ≈ +1.0)
- T = 5: V7 still best but drops to Δ ≈ +0.68 — **clear performance decay vs lower T**

T = 5 is where attention-mixing dilution dominates and the existing
protocols hit their ceiling. A protocol that closes this gap is the
target.

### Baseline numbers (already in dashboard)

For Galaxy 23 (T = 5 SoftMaxPool) 3-seed at coh ≥ 1.75:

| protocol | Δ vs anchor 0.411 | succ peak | coh at peak |
|---|---:|---:|---:|
| V1 right-edge | +0.233 | 0.644 | (across seeds) |
| V2 per-position | not run (predicted to fail like T=3) | – | – |
| **V7 tiled-broadcast** | **+0.678** | 1.089 | (best so far) |
| V8 encoded-broadcast | not run on T=5 | – | – |

### Hill-climb workflow

1. Start from V7 baseline.
2. Each iteration proposes ONE new protocol designed to address a specific
   weakness of current best.
3. Implement protocol as a new `intervene_*` script in
   `experiments/phase7_unification/case_studies/protocol_hillclimb/`.
4. Run 3-seed on Galaxy 23.
5. Grade with Sonnet.
6. Update `leaderboard.md` with Δ at coh ≥ 1.75.
7. If new protocol beats V7 by Δ > 0 (any improvement): keep + iterate.
   If Δ ≤ 0: log as failed-candidate, try a different mechanism.
8. After 3-5 iterations, sweep secondary archs to confirm
   cross-arch generalization.

### Candidate protocols (priority order)

#### V9: Sliding tiled-broadcast (stride < T)

V7 has stride T (non-overlapping). At T = 5, that's only S/T = ~12 blocks
across a 64-token prefix → relatively sparse coverage. Each block has its
own δ, but blocks are non-adjacent.

V9 = **stride T/2 sliding window** + uniform δ within each window. At T = 5
stride = 2-3, that's ~30 windows → much denser coverage. Overlapping
windows accumulate δ at shared positions.

Hypothesis: V9 should retain V7's attention-invariance within each
window AND give denser steering across positions. Risk: overlapping
writes accumulate (positions in 2-3 windows get 2-3× δ) — may need
normalization.

#### V10: Encoder-magnitude weighted broadcast

V7 broadcasts the AVERAGED δ within block. But the encoder
`W_enc[t, :, j]` tells us position-`t`'s contribution to feature `j`.
Some features have concentrated W_enc (most weight at one position);
others diffuse.

V10 = within each non-overlapping block, write δ × `||W_enc[t,:,j]||²` /
`Σ_t' ||W_enc[t',:,j]||²` (i.e., per-position encoder-magnitude weighted)
INSTEAD of uniform across positions. Each position gets δ in proportion
to how much THIS feature reads from it.

Hypothesis: matches the encoder's per-feature position structure → more
faithful to what the feature "sees".

#### V11: Decoder-magnitude weighted broadcast

Symmetric to V10 but uses `W_dec[j, t, :]` magnitudes. Each position gets
δ in proportion to how much THIS feature WRITES at that position via the
decoder.

#### V12: Hybrid — last-pos PP + earlier-pos broadcast

At T = 5, V2 PP fails because 5 different deltas scramble. V1 RE works
(only 1 of 5 positions). What about hybrid: write per-position δ at
the LAST 1-2 positions (where the feature is most discriminative for
right-edge-trained archs) + uniform broadcast at earlier positions.

#### V13: Stride-1 (totally overlapping) tiled-broadcast

Take this to the extreme: every T-window overlap (stride 1), each with
its own uniform δ. Equivalent to writing the average-decoder direction
for the WINDOW ENDING AT t, at each position t. This is essentially
"per-position-uniform" — every position t gets a δ derived from its own
T-window.

Hypothesis: dense + position-aware via window choice + attention-friendly
within each "window's contribution".

#### V14: Multi-scale tiled-broadcast

Tile with multiple block sizes simultaneously: T = 5 blocks AND T = 1
(per-position) AND T = 3 sub-blocks. Each scale contributes its own
δ, summed.

Hypothesis: captures both window-level and finer feature dynamics.

#### V15: Pre-attention steering

Hook BEFORE the L = 12 attention layer (i.e., at L = 11 output). Write
the steering signal there. Subsequent L = 12 attention then mixes the
steering as part of its normal computation. This is "let the model
process the steering naturally".

### Stop criteria

- Reach Δ ≥ +1.0 at coh ≥ 1.75 on Galaxy 23 T = 5 (matches T = 2/T = 3
  best cells).
- Or 5+ iterations with no improvement (declare V7 as the high-T best).
- Or new protocol generalizes to T = 2 H8 (which V7 fails) — would be a
  major finding.

### Files

- `experiments/phase7_unification/case_studies/protocol_hillclimb/intervene_v9_sliding_tb.py` — V9
- (more added per iteration)
- `experiments/phase7_unification/case_studies/protocol_hillclimb/run_hillclimb.sh`
- `docs/han/research_logs/phase7_unification/protocol_hillclimb/leaderboard.md`
- `docs/han/research_logs/phase7_unification/protocol_hillclimb/iteration_log.md`
