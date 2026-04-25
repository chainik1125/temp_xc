---
author: Han (via Phase 5B agent)
date: 2026-04-25
tags:
  - results
  - complete
---

## SubsetEncoderTXC: t < T encoder slabs with rank-invariant assignment

Phase 5B candidate F. Tests whether we can replace the per-absolute-position
encoder structure (T_window slabs of W_enc) with a smaller, position-rank-
invariant encoder (t slabs of W_enc, where t < T_window).

**Result: clean three-point negative trend. Per-absolute-position W_enc[t]
slabs are load-bearing.**

### Hypothesis tested

If the encoder learned features that work from *any* sampled subset of
T_window positions (regardless of which absolute positions), we should be
able to:

1. Allocate only `t` encoder/decoder slabs (not `T_window`).
2. Train via random sub-sampling: each step pick `t` of `T_window` raw
   positions, sort them by absolute index, assign sorted-rank-i to slab-i.
3. At probe time, average over many random `t`-subsets to recover the
   full receptive field.

If true, this would halve the encoder param count vs B4/B2 (5 slabs vs 10)
and enable training at `T_window=128` (full Gemma context) with negligible
encoder param cost.

### Architecture

`src/architectures/phase5b_subset_encoder_txc.py:SubsetEncoderTXC` —
inherits Track 2's anti-dead stack from `TXCBareAntidead`.

```python
W_enc ∈ R^{t × d_in × d_sae}      # NOT T_window — half the slabs of B4 at t=5, T_w=10
W_dec ∈ R^{d_sae × t × d_in}
```

**Training (random t-subset, sort, encode):**
```python
sample_idx = sorted(random_subset(T_window, k=t))   # (B, t)
x_t        = x[sample_idx]                          # (B, t, d_in)
pre        = einsum("btd,tds->bs", x_t, W_enc) + b_enc
z          = TopK(pre, k=k_win)
x_hat      = einsum("bs,std->btd", z, W_dec) + b_dec
loss       = MSE(x_hat, x_t) + auxk_loss
```

**Probe (random_K_subsets averaging):**
```python
z_avg = 0
for k in range(probe_K=16):
    pos = sorted(random_subset(T_window, k=t))
    x_t = x_full[pos]
    z_avg += encode(x_t)
return z_avg / probe_K
```

Sweep: `t=5` slabs, `T_window ∈ {10, 20, 128}`, `k_win=500`,
`probe_K=16`, seed=42, FLIP applied.

### Results (single-seed=42, with FLIP)

| arch | T_window | t | params (encoder) | lp | mp |
|---|---|---|---|---|---|
| **`subset_encoder` T_w=10 t=5** | 10 | 5 | 850 MB | 0.7918 | 0.8073 |
| **`subset_encoder` T_w=20 t=5** | 20 | 5 | 850 MB | 0.7841 | 0.7904 |
| **`subset_encoder` T_w=128 t=5** | 128 | 5 | 850 MB | 0.7454 | n/a † |
| reference: B4 (subseq_h8 T=10 t=5 k=500) | 10 | 10 slabs | 1.7 GB | 0.8442 | 0.8516 |
| reference: H8 T=10 Phase 5 (k=1000) | 10 | 10 slabs | 1.7 GB | 0.7931 | 0.8040 |

† mp not computed for T_window=128 because the probe cache's tail is
LAST_N=20, so a 128-token sliding window has K = 20 - 128 + 1 ≤ 0 slides.
Last-position works (the model pads short anchors to T_window=128 internally).

### Interpretation

**Halving slabs (10 → 5) costs ~0.044 mp at fixed T_window=10**, even with
the rank-invariant prior + probe-time random-K-subset averaging. The model
trained at t=5 on T_w=10 underperforms B4 baseline by 0.052 lp / 0.044 mp,
and underperforms even the calibration cell (T=10 t=10 k=500, no subseq) by
0.020 lp / 0.016 mp. So the rank-invariant 5-slab structure is *worse* than
the 10-slab no-subseq structure at the same T_window.

**Extending T_window with same t=5 hurts.** From T_w=10 → T_w=20 → T_w=128,
both lp and mp regress monotonically. The number of possible t-subsets
explodes combinatorially: C(10,5)=252, C(20,5)=15,504, C(128,5)≈2.7e8.
Probe-time K=16-subset averaging captures less and less of the variance.

The "rank-invariant 5 slabs averaging over subsets" prior **cannot recover
what per-absolute-position 10 slabs encode**. The per-position W_enc[t]
structure provides specific position-conditioned features that probe-time
averaging dilutes rather than substitutes for.

### Comparison to other Phase 5B negative archs

| arch family | finding | mp delta vs B4 |
|---|---|---|
| D1 strided window (T_eff=5, stride=2, span=10) | -0.044 vs B4; -0.012 vs T=5 vanilla | hurts |
| C1/C2/C3 token-level encoder (no per-pos W_enc) | -0.084 vs B4 | strongly hurts |
| **F SubsetEncoderTXC (t<T slabs)** | **-0.044 vs B4 at T=10 (best F config)** | **hurts** |

All three negative families share a common feature: they break, weaken, or
restructure the per-absolute-position W_enc[t] axis. Each fails. The
positive interventions (B2 subseq, B4 subseq + H8 stack, k_win reduction) all
*preserve* the per-absolute-position slabs.

### What's interesting about the negative result

The `t=5, T_window=10` cell is roughly tied with `H8 T=10 Phase 5` at
k_win=1000 — both around lp 0.79, mp 0.80. So 5 rank-invariant slabs ≈ 10
absolute-position slabs *without* the subseq trick AND with looser k_win.
This places a soft floor on what "5 slabs of capacity" can buy you, no
matter how cleverly assigned.

### Reproducibility

Ckpts: `han1823123123/txcdr/phase5b_ckpts/phase5b_subset_encoder_T{10,20,128}_t5__seed42.pt`

Training:
```bash
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subset_encoder_T10_t5 --T_window 10 --t_slabs 5 --seed 42

TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subset_encoder_T20_t5 --T_window 20 --t_slabs 5 --seed 42

TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subset_encoder_T128_t5 --T_window 128 --t_slabs 5 --seed 42
```

Probing:
```bash
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.run_probing_phase5b \
    --run_ids phase5b_subset_encoder_T10_t5__seed42 \
              phase5b_subset_encoder_T20_t5__seed42 \
              phase5b_subset_encoder_T128_t5__seed42 \
    --aggregations last_position mean_pool
```

Single-seed only. Given the magnitude of the negative deltas (-0.04 to -0.10
mp) and the clean monotone trend, 3-seed verification was not done — Phase
5B's σ regime suggests these would not flip sign.

### Possible variants worth trying (not done in Phase 5B)

If a future agent wants to push on this idea:

1. **Larger t at fixed T_window**: t=8, t=10 with T_window=20 — rank-
   invariant but more capacity. Probably still loses to B4 (which has 10
   absolute-position slabs at T=10), but isolates the "rank-invariant" cost
   from the "fewer slabs" cost.

2. **Mixed positional+rank prior**: t slabs but with a learned positional
   bias `δ[t, j]` added to the j-th sampled position's pre-activation
   (matching its absolute position rank within the T_window). Hybrid between
   rank-invariant and absolute-position.

3. **Larger probe_K**: bump K=16 to K=64 or K=256 to capture more of the
   subset variance at T_window=128. Diminishing returns expected, but worth
   one cell to confirm K=16 isn't bottlenecking.

4. **t encoder slabs, T full decoder slabs**: keep the encoder rank-
   invariant (t slabs) but the decoder absolute-position-aware (T_window
   slabs). Tests whether the load-bearing piece is encoder or decoder
   structure.

None of these were pursued — the negative trend was steep enough that it
seemed unlikely any variant would beat B4. Documenting for completeness.

### Files

- `src/architectures/phase5b_subset_encoder_txc.py` — arch class
- `experiments/phase5b_t_scaling_explore/train_phase5b.py` — dispatcher
  branch + `--T_window`, `--t_slabs`, `--probe_strategy`, `--probe_K` CLI
- `experiments/phase5b_t_scaling_explore/run_probing_phase5b.py` — model
  loader + probe-time encoder dispatch
- `experiments/phase5b_t_scaling_explore/results/training_logs/phase5b_subset_encoder_*_t5__seed42.json`
- Probing rows in `experiments/phase5b_t_scaling_explore/results/probing_results.jsonl`
  with `run_id` matching `phase5b_subset_encoder_T{10,20,128}_t5__seed42`

— Phase 5B agent, 2026-04-25
