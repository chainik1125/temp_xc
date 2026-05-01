---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## Phase 7 Y — TXC steering protocol space (step-back analysis)

> Han's prompt 2026-05-01 evening: "step back and look at what steering
> protocols we have tried. It's really not clear how to do steering for
> TXCs. The key subtlety here is what happens in attention. Have we
> tried the obvious thing which is not have the TXC windows overlap
> and have the same steering vector per window?"

### The problem statement

TXC features are computed from a **window** of T residuals. T-SAE
features are computed from a **single** position's residual. The
T-SAE steering protocol is unambiguous: clamp the feature, write the
decoder direction at every token position.

For TXC: there's NO unambiguous protocol. You compute z by encoding
some T-window, clamp the feature, and the decoder gives you a (T, d_in)
block. The OPEN QUESTION: **where in the residual stream do you write
the decoder block, and how do you handle overlaps?**

This document inventories every protocol we've tried, the ones we
haven't, and the deeper attention-mixing question.

### Inventory of TXC steering protocols (as of 2026-05-01)

#### Active-window-style (encode each sliding window separately)

| name | output dir | mechanism | overlap |
|---|---|---|---|
| **V1: RE** (right-edge / paper-clamp) | `steering_paper_normalised{,_seed*}` | slide T-window stride 1; encode each; write decoder block to ONLY the right-edge position | windows overlap; only RE position written |
| **V2: PP** (per-position) | `steering_paper_window_perposition{,_seed*}` | slide T-window stride 1; encode each; write per-position decoder block to all T positions; overlapping writes get summed | heavy overlap; same position written T times |
| **V2-LE** (left-edge) | `steering_paper_window_left_edge{,_seed*}` | mirror of V1 — write at left edge | windows overlap |
| **V4: tiled** | `steering_paper_window_tiled{,_seed*}` | stride-T blocks (NO overlap); per-position writes within block | no overlap; per-position differs |
| **V7: tiled-broadcast** ⭐ NEW | `steering_paper_window_tiled_broadcast{,_seed*}` | stride-T blocks, **same single δ per block** (mean of decoded per-position); broadcast within block | no overlap; uniform within block |

#### Decoder-direction (skip encode/clamp/decode)

| name | output dir | mechanism | overlap |
|---|---|---|---|
| **V3: dec-additive** | `steering_paper_window_dec_additive{,_seed*}` | `strength × W_dec[picked, :, :]` written to last T positions only | static decoder, last T |
| **V6: dec-broadcast** | `steering_paper_window_dec_broadcast{,_seed*}` | `mean(W_dec[picked])` broadcast to ALL S positions | uniform everywhere |

#### Misc

| name | output dir | mechanism |
|---|---|---|
| **anchored** | `steering_paper_window_anchored{,_seed*}` | (W's variant) |
| **local** | `steering_paper_window_local{,_seed*}` | (W's variant — single token clamp?) |

### Protocol coverage on key cells (3-seed verified)

| arch | V1 RE | V2 PP | V4 tiled | V6 broadcast | V7 tiled-broadcast |
|---|:--:|:--:|:--:|:--:|:--:|
| T-SAE k=20 (anchor) | ✓ | n/a | n/a | n/a | n/a |
| T=2 H8 shifts=(2,) | ✓ | ✓ | – | – | – |
| T=2 bare-antidead | ✓ | ✓ | – | – | – |
| T=3 grown chain | ✓ | ✓ | – | – | – |
| T-SAE WS | ✓ | ✓ | – | – | – |
| Galaxy 4 hierarchical | ✓ | ✓ | – | – | – |
| Galaxy 6 max-pool | ✓ | ✓ | – | – | – |
| Galaxy 8 soft-max-pool | ✓ | ✓ | – | – | **launched 2026-05-01** |
| Galaxy 11 softmax+H8 | ✓ | ✓ | – | – | – |
| Galaxy 18 G8 T=3 | ✓ | ✓ | – | – | **launched 2026-05-01** |
| W's TXCMaxPoolMergeH8 | ✓ | ✓ | – | ✓ | – |
| W's TXCContrastiveMergeH8 | ✓ | ✓ | – | ✓ | – |

**Coverage gap** before today: V4 (tiled) was tested only on a few
single-seed cells; V6 only on W's contrastive arch; V7 didn't exist.

### The attention-mixing question

The reason there's no canonical TXC steering protocol is that **attention
at subsequent layers mixes features across positions**, and the right
write pattern at our intervention layer (L=12) depends on how attention
will redistribute the steering signal.

#### Mechanics

At each transformer layer L′ > 12, the attention output at position t is:
$$
\\text{attn\\_out}_t \\;=\\; \\sum_s \\alpha_{t,s} \\, W_V \\, r_s
$$
where $\\alpha_{t,s}$ is the attention weight from t to s and $r_s$ is
the residual at position s. So if we write $\\delta_s$ at position s
(by intervention at L=12), the post-attention residual at position t
gets a contribution $\\sum_s \\alpha_{t,s} W_V \\delta_s$.

For TXC features that conceptually "live in" a window:
- **V1 (RE-only)**: only one position carries δ. After attention, only
  positions with $\\alpha_{t,\\text{RE}} > 0$ see any signal — and that
  signal is small (one term in the sum).
- **V2 (PP)**: T positions carry different δ_t. After attention, position
  t sees $\\sum_{s=\\text{RE-T+1}}^{\\text{RE}} \\alpha_{t,s} W_V \\delta_s$
  — a structured but mixed signal.
- **V6 (broadcast)**: every position carries the same δ. After attention,
  position t sees $\\delta \\cdot \\sum_s \\alpha_{t,s} = \\delta$ —
  the steering signal is INVARIANT to attention mixing (since attention
  weights sum to 1).
- **V7 (tiled-broadcast)**: each non-overlapping T-block carries one
  uniform δ_block. After attention, position t sees
  $\\delta_{\\text{block}(t)} \\cdot \\alpha_{t,\\text{within-block}}
  + \\sum_{\\text{other blocks}} \\delta \\cdot \\alpha_{t,\\text{cross-block}}$
  — locally invariant within block, structured across blocks.

#### Implication

**V6 is the most attention-invariant protocol.** Whatever attention does,
the steering signal survives unchanged because $\\sum \\alpha = 1$.

**V7 trades attention-invariance against per-block dynamics.** Each block
has its own δ derived from its actual residual content, so it's
content-aware AND attention-friendly within each block.

**V2 (PP) is attention-sensitive.** Different δ at different positions
gets reweighted by attention — could hurt or help depending on the
attention pattern.

#### Empirical signal

The Galaxy 18 result (G8 T=3) showed:
- T=2 PP: Δ = +1.011 (T-SAE-beating)
- T=3 PP: Δ = +0.233 (regression)
- T=3 RE: Δ = +0.767 (recovery)

**Mechanism**: at T=3, PP writes 3 different δ to 3 consecutive
positions. Attention mixing at the next layer creates a more
complex pattern, and the steering signal gets diluted/scrambled.
RE writes only at one position; attention mixes that into the
sequence cleanly.

This empirically supports the attention-mixing intuition. Predict:
**V7 (tiled-broadcast) should outperform V2 (PP) at higher T**
because V7 keeps within-block writes uniform, simpler for attention.

### V7 launch (2026-05-01 evening)

V7 tiled-broadcast launched on:
- Galaxy 8 (T=2, soft-max-pool) all 3 seeds
- Galaxy 18 (T=3, soft-max-pool) all 3 seeds

Rationale: Galaxy 8 PP T=2 wins big (Δ=+1.011); does V7 match or
exceed? Galaxy 18 PP T=3 weak (+0.233); does V7 rescue T=3 PP?

### Next: V8 — full broadcast with TXC encoding

The natural V8 protocol: V6 + TXC encode/clamp/decode (instead of static
W_dec). At each generation step, encode the right-edge T-window, clamp
the feature, decode → average → broadcast to ALL S positions in the
prefix.

```
delta_avg = mean(decode(z_clamped) - decode(z), dim=0)   # (d_in,)
h_steered[:, :, :] += delta_avg                          # ALL positions
```

This is **maximally attention-invariant** (broadcast everywhere). Trades
locality for attention-friendliness. Would test whether the TXC's
DYNAMIC computation (encode per generation step) adds value over V6's
static W_dec direction.

Status: implementation pending V7 results. If V7 wins big, V8 is the
next natural ablation.

### Even more "galactic" protocol ideas (untried)

1. **Anti-attention-decay**: write LARGER δ at earlier positions
   (which get diluted by attention as we move forward). Requires
   estimating attention weights at intervention time.

2. **Feature-position-conditioned**: for each TXC feature j, the encoder
   has $W_{\\text{enc}}[t, :, j]$. Write δ at positions where
   $\\|W_{\\text{enc}}[t, :, j]\\|$ is largest (the position the feature
   "cares most about"). For features with concentrated W_enc (most weight
   at one position), this is RE-like; for diffuse W_enc, this is
   PP-like.

3. **Decoder-attention**: replace hand-designed write-back protocols
   with learnable per-feature attention weights over T positions.
   Requires modifying the architecture (Galaxy 7 from brainstorm).

4. **Pre-attention steering**: hook BEFORE L=12's attention (i.e., at
   the residual after L=11). The attention at L=12 then mixes our
   steering. Untested whether this is better than post-L=12 hooking.

### Summary

We've covered a wide protocol space empirically (V1, V2, V3, V6, V4,
plus W's V6 dec-broadcast variant). The two notable gaps are:

- **V7 tiled-broadcast** (Han's "obvious thing") — launched today
- **V8 encoded broadcast** (V6 with TXC dynamics) — planned

The attention-mixing analysis suggests V6 and V7 should be more
attention-invariant than V2 (PP). The Galaxy 18 T=3 PP regression
empirically supports this — V2 doesn't scale with T because the
per-position writes confuse attention.

If V7 holds up, the recommendation for the paper would be:
**use V1 (RE) or V7 (tiled-broadcast) at higher T; use V2 (PP) at T=2.**
This matches the empirical "T=2 PP wins, T=3 RE wins" pattern.
