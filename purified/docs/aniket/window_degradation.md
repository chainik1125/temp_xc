---
author: Aniket
date: 2026-05-28
tags:
  - results
  - freqbench
  - window-degradation
  - in-progress
---

## Window-size degradation in `txc_base T=W` — diagnosis + fix

The joint-window-TopK variant of TXC-base (T set equal to the eval window
W) shows a peculiar failure on the AC bench: probe accuracy stays pinned
near chance as W grows, while sliding-T variants climb to NTPS ≈ 0.72 at
the same cell. This doc tracks the investigation: what's wrong with the
joint pool, what's not, and what fixes it.

### TL;DR

The encoder is fine; the **TopK pool** is the bottleneck. `txc_base`'s
joint-window TopK applies one shared sparsity budget to a code summed
across T positions, which collapses the per-position assignment a linear
probe needs to read direction. **`FreqFrac`** of the trained `W_enc`
climbs to ≈ 0.88 as W grows (encoder atoms become *more* order-sensitive)
while NTPS stays near zero — a clean representation-vs-readout
decomposition. Capacity does *not* rescue this; structure does. The
per-position TopK variant (sweep currently running) is the targeted fix.

---

## 1. The puzzle

Look at `txc_base` (T=W joint-window TopK) on the AC bench, raw_k=1,
σ=0.1, across W ∈ {2, 4, 8, 16}, in the freshly-trained v2 sweep:

| W  | NTPS (d_sae=40) | A_reverse |
|----|---|---|
| 2  | 0.058 | 0.46 (~chance) |
| 4  | 0.014 | 0.45 |
| 8  | 0.022 | 0.46 |
| 16 | 0.093 | 0.45 |

NTPS stays pinned near zero across W. The *sliding-T* variants
(txcdr_t2/t5) climb to 0.30–0.37 in the same regime. Something about the
joint-window setup is bottlenecking the readout. Two candidate causes:

- **Representation failure** — the joint atoms collapse toward DC as W
  grows; `W_enc` becomes constant along the temporal axis and can only
  encode averages.
- **Readout failure** — atoms still encode AC content, but a single
  shared TopK budget over a code summed across T discards the per-position
  structure the linear probe needs.

## 2. Capacity rescue (negative result)

Sweep d_sae ∈ {40, 256, 1024} for `txc_base T=W`, raw_k=1, σ=0.1.
(`experiments/freq_bench/sweep.py`, label `txc_base_TW`.)

| W  | NTPS @d_sae=40 | @256 | @1024 |
|----|---|---|---|
| 2  | 0.058 | 0.070 | 0.019 |
| 4  | 0.014 | 0.117 | 0.061 |
| 8  | 0.022 | 0.107 | 0.195 |
| 16 | 0.093 | 0.122 | **0.173** |

Compare the same `(W=16, raw_k=1)` cell for the sliding-T archs from the
main v2 sweep:

| arch | NTPS @d_sae=40 | @256 | @1024 |
|----|---|---|---|
| txcdr_t2 | 0.37 | 0.44 | 0.51 |
| txcdr_t5 | 0.30 | 0.59 | **0.72** |
| **txc_base T=W (joint)** | 0.09 | 0.12 | **0.17** |

So d_sae=1024 only takes the joint variant from 0.09 → 0.17, while the
sliding variants triple-to-quadruple. **The degradation is not just
small-dictionary.** Capacity rules out one candidate cause; the structural
difference (joint vs sliding TopK) matters.

## 3. FreqFrac diagnostic — representation vs readout

For each cell, load the trained `W_enc` (shape `(T, d_in, d_sae)`) and
compute the fraction of its spectral energy at nonzero temporal
frequencies, averaged over the d_sae atoms. High ⇒ atoms detect
*transitions*; ≈ 0 ⇒ atoms are constant across T (pure smoothers).
(`experiments/freq_bench/freqfrac_diagnostic.py`.)

![FreqFrac vs W, NTPS vs W — txc_base T=W](../../results/freq_bench/v2_sweep/freqfrac_vs_W_TW.png)

| W  | FreqFrac @d_sae=40 | @256 | @1024 | NTPS @1024 |
|----|---|---|---|---|
| 2  | 0.506 | 0.499 | 0.498 | 0.019 |
| 4  | 0.698 | 0.702 | 0.684 | 0.061 |
| 8  | 0.840 | 0.822 | 0.798 | 0.195 |
| 16 | 0.867 | 0.900 | **0.875** | 0.173 |

Note the asymmetry: **FreqFrac climbs to 0.88 with W; NTPS plateaus near
0.17**. The encoder atoms increasingly devote energy to nonzero temporal
frequencies as W grows — which is exactly the structure needed to encode
transitions. The representation is *correct and getting better with W*.
Yet the linear-probe NTPS stays pinned at the joint-pool ceiling.

**Diagnosis:** the joint TopK pool is the bottleneck, not the encoder
weights. The atoms know AC; the joint sparse code throws the per-position
assignment away before the probe sees it.

## 4. Per-position TopK — fails, and the *failure* is informative

Implementation: `temp_bench/archs/txc_base_perpos.py`. Identical
`W_enc / W_dec` shapes to `txc_base`; only the TopK axis changes. Sweep
cells (label `txc_base_perpos_TW`): W ∈ {2, 4, 8, 16} × d_sae ∈ {40, 256,
1024} = 12 cells.

**Result: NTPS stays at chance across every cell.** At W=16, d_sae=1024:
per-pos NTPS = 0.026, order_gap = 0.018, reverse_drop = 0.020 — within
noise of zero. The joint variant gets 0.17. Per-pos made it **worse**.

But the *weight-space* picture is the opposite:

| arch (W=16, d_sae=1024) | NTPS | FreqFrac(`W_enc`) |
|---|---|---|
| `txc_base_perpos_TW` (per-pos TopK) | **0.03** | **0.897** |
| `txc_base_TW` (joint TopK) | 0.17 | 0.875 |

Per-pos has the **highest FreqFrac of any architecture in this study**,
but encodes *no* readable direction. So my "readout-failure" framing in
§3 was too coarse: high `FreqFrac` is *necessary but not sufficient* for
direction encoding. What per-pos exposes is the actual missing piece.

**Why per-pos fails.** Per-position TopK encodes each `x_t` independently
with its own basis `W_enc[t]`, so `z_t = topk(W_enc[t]·x_t)` depends only
on the local phase. Mean-pooling over `t` gives, effectively, a
**histogram of phases visited** — and a forward and a reversed sequence
visit the *same* phases in different order. The mean-pooled per-pos code
is **direction-symmetric by construction**. No matter how
order-sensitive the per-position weights are (FreqFrac measures per-atom
spectrum), the *pooled* representation discards the order.

The joint `txc_base` does *better* on direction precisely because
`pre = Σ_t W_enc[t]·x_t` is computed BEFORE the TopK — that sum *mixes
positions*, producing a single code that carries some directional signal
(`A_reverse = 0.42`, below chance). The joint pool was doing useful work;
removing it lost the position-mixing.

## 5. Sharpened diagnosis — what the "degradation" really is

Compare all five architectures at the strong cell:

![Architecture comparison @ W=16, raw_k=1, d_sae=1024](../../results/freq_bench/v2_sweep/TW_variants_comparison.png)

| arch (W=16, k=1, d_sae=1024) | structure | NTPS | A_reverse |
|---|---|---|---|
| `regular_sae` | per-token | 0.01 | 0.50 |
| `txc_base_perpos_TW` | per-pos encode + per-pos TopK | 0.03 | 0.49 |
| `txc_base_TW` | joint encode + joint TopK | 0.17 | 0.42 |
| `txcdr_t2` | joint encode at T=2, slid across W | 0.51 | 0.23 |
| `txcdr_t5` | joint encode at T=5, slid across W | 0.72 | 0.12 |

The ordering is monotone and structural. Two factors combine:

1. **Position-mixing in the encoder.** `pre = Σ_t W_enc[t]·x_t` mixes
   positions and is order-sensitive; per-position encoding doesn't and
   isn't. (Per-pos sits at chance; joint-T=W gets 0.17.)
2. **Sliding the encoder.** A joint encoder of size T<W slid across W
   produces W−T+1 codes per sequence, each a *local* transition
   detector. Mean-pooling W−T+1 such codes preserves the local
   transition info richly, where mean-pooling one global window-level
   code does not. (Joint-T=W gets 0.17; sliding-T=5 gets 0.72.)

So "the degradation with window size" Dmitry observed is **the absence
of sliding**. The joint encoder at T=W has one shot at the whole window;
the sliding encoder has W−T+1 shots at local transitions, and the linear
probe needs that redundancy. Capacity doesn't help because the issue
isn't atom count — it's how many codes the probe gets to average.

### Implications

- **For TXC architectural choices:** the sliding architecture (txcdr,
  with T fixed small and slid across the eval window) is the genuinely
  correct design for AC-style filtering. The joint-T=W parameterisation
  is a structural mismatch for direction readout regardless of capacity.
- **For Dmitry's quote:** "so solvable" turns out to be right but with a
  different solution than I predicted — the fix is just to use the
  sliding variant. Per-position TopK isn't the answer.
- **For the paper:** if §4 emphasises `txc_base` (the T=W joint
  variant), it's emphasising the *weakest* of the three on the
  filtering axis. The sliding `txcdr_t5` is the one that proves the
  paper's filtering claim cleanly.

## What's left

- Repeat the comparison at d_sae=256 and d_sae=40 (cells already in the
  leaderboard) to confirm the structural ordering is capacity-independent.
- Try the dual fix on `txc_base T=W`: **window-level training but
  multi-shot inference** — at eval time, slide a T<W sub-window over
  the trained-T=W joint encoder. Tests whether the structural fix can
  be retrofit without retraining.
- Extend the comparison to the Mixed bench (frequency response).
- Mark the `tsae_attn` mislabel cleanup unrelated to this thread.
