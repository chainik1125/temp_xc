---
author: Han (via Phase 5B agent)
date: 2026-04-25
tags:
  - results
  - in-progress
---

## Phase 5B → Phase 5 leaderboard update (REVISED)

### Erratum to original 2026-04-25 17:30 UTC version

**The previous version of this doc reported `phase5b_subseq_h8_t10_s8_k500` seed=42 as lp 0.8545 / mp 0.8590. That number was wrong.**

Two issues:
1. **FLIP convention not applied.** Phase 5's headline tables apply `max(AUC, 1-AUC)` per-task on `winogrande_correct_completion` and `wsc_coreference` (arbitrary label polarity). My `run_probing_phase5b.py` did NOT do this. All my 3-seed numbers were therefore *under-counting* by ~0.005-0.010.
2. **The t=8 cell number itself was wrong.** Re-deriving from the actual `probing_results.jsonl` rows with the same method I used for B2/B4 gives lp 0.8145 / mp 0.8183 (no FLIP) → lp 0.8218 / mp 0.8284 (with FLIP). Phase 5 agent's verified number is mp 0.8284, which now reproduces exactly. I cannot reconstruct where 0.8545/0.8590 came from — likely a bookkeeping error in my earlier ad-hoc aggregator. The committed jsonl is unchanged; the inflated number was only ever in my chat-time report.

`run_probing_phase5b.py` has been updated to write `test_auc_flip` alongside `test_auc` from now on.

The peak at single seed=42 is **B4 baseline `phase5b_subseq_h8` T=10 t=5 k=500**, NOT the t=8 cell. Phase 5 agent's diagnosis is correct.

---

### TL;DR — three new entries to consider (corrected, with FLIP)

All numbers below apply the FLIP convention.

1. **B2** `phase5b_subseq_track2` (T_max=10, t_sample=5, k_pos=100, k_win=500) — *3-seed verified.*
   - **lp 0.8213 ± 0.0005** (vs H8 3-seed 0.8005 ± 0.003 → **+0.0208**, ~5σ pooled)
   - **mp 0.8236 ± 0.0024** (vs H8 3-seed 0.8126 ± 0.003 → **+0.0110**, ~3.5σ pooled)
   - σ matches the H8 reference regime — not noisier.

2. **B4** `phase5b_subseq_h8` (T_max=10, t_sample=5, k_pos=100, k_win=500) — *3-seed verified, but noisy.*
   - **lp 0.8295 ± 0.0130** (vs H8 3-seed → **+0.0290**, ~1.7σ pooled)
   - **mp 0.8343 ± 0.0150** (vs H8 3-seed → **+0.0217**, ~1.4σ pooled)
   - Best-mean of all our archs but **σ is 4× larger than H8/B2**.
   - Per-task analysis: 21/36 tasks have seed range > 0.05; max range 0.167 on `europarl_de`.
     Likely caused by interaction between subseq sampling and multi-distance contrastive
     (B2's subseq alone is well-behaved).

3. **B4 single-seed=42 peak** (same arch as B4): **lp 0.8442 / mp 0.8516**.
   This is the actual peak at single-seed in our results — matches Phase 5 agent's
   independently-derived 0.8516. *Not* the t=8 sweep cell.

### Suggested rows for the headline tables (corrected, with FLIP)

For your `last_position` headline:

| arch | mean AUC | std | n |
|---|---|---|---|
| **`phase5b_subseq_h8` 3-seed** (subseq + H8 stack at T=10 t=5 k=500) | **0.8295 ± 0.0130** | — | 36 |
| **`phase5b_subseq_h8` seed=42 single** | 0.8442 | — | 36 |
| **`phase5b_subseq_track2` 3-seed** (subseq + Track 2 at T=10 t=5 k=500) | **0.8213 ± 0.0005** | — | 36 |
| `mlc_contrastive_alpha100_batchtopk` (current Phase 5 lp leader) | 0.8124 | — | 36 |

For your `mean_pool` headline:

| arch | mean AUC | std | n |
|---|---|---|---|
| **`phase5b_subseq_h8` seed=42 single** | **0.8516** | — | 36 |
| **`phase5b_subseq_h8` 3-seed** | **0.8343 ± 0.0150** | — | 36 |
| **`phase5b_subseq_track2` 3-seed** | **0.8236 ± 0.0024** | — | 36 |
| `phase57_partB_h8_bare_multidistance_t6` (current Phase 5 mp leader, seed=42) | 0.8188 | — | 36 |

### t_sample sweep at T_max=10 k_win=500 (single-seed=42, with FLIP)

| t_sample | mp |
|---|---|
| 3 | 0.8373 |
| **5 (= B4 baseline)** | **0.8516** ← peak |
| 8 | 0.8284 |
| 10 (= no subseq, Track 2/H8 stack at T=10) | 0.8231 |

The optimum is at **t=5 = T_max/2**, not monotone in either direction. Both more
subsampling (t=3) and less (t=8, t=10) regress.

### Subseq vs k_win — partial 2x2 isolation (mp, with FLIP)

| | **k_win=500** | k_win=1000 (Phase 5 convention at T=10) |
|---|---|---|
| **t=10 (no subseq)** | 0.8231 (calibration cell) | 0.8040 (Phase 5 H8 T=10 ref) |
| **t=5 (subseq)** | **0.8516** (B4 baseline) | *training now* (iso cell) |
| t=8 (subseq) | 0.8284 | not run |

From the row "t=10 (no subseq)" we get the **k_win effect**:
`0.8231 - 0.8040 = +0.0191` mp from reducing k_win 1000→500.

From the column "k_win=500" we get the **subseq effect at t=5**:
`0.8516 - 0.8231 = +0.0285` mp from adding t_sample=5 subsampling on top.

So the +0.0476 mp delta from H8 T=10 to B4 splits roughly:
- ~40% from k_win reduction (k=500 vs k=1000 at fixed T=10)
- ~60% from subseq sampling at t=5

The iso cell (T=10 t=5 k=1000) currently training will give the fourth corner of the 2x2 — letting us check whether the two effects are simply additive.

### k-regime quick reference (with FLIP)

| family | example | k_win | k_pos (derived) |
|---|---|---|---|
| MLC | `mlc_contrastive_alpha100_batchtopk` | 100 | 20 (per layer) |
| TXC (Phase 5) | H8 T=5 | 500 | 100 |
| TXC (Phase 5) | H8 T=6 (Phase 5 mp peak) | 600 | 100 |
| TXC (Phase 5) | H8 T=10 | 1000 | 100 |
| **Phase 5B B4 (peak)** | **subseq_h8 T=10 t=5 k=500** | **500** | **100 (per active position)** |

### How the subseq mechanism works (1-paragraph)

The encoder has T_max position slabs `W_enc[t] ∈ R^{d_in × d_sae}` — identical shape
to vanilla TXCDR. At training time, we randomly sample t_sample of the T_max
positions per step (non-contiguous random subset) and only those positions feed
encoder gradient. Implementation: zero out unsampled positions in `x` before the
standard `einsum("btd,tds->bs", x_masked, W_enc)`. Reconstruction loss is computed
on the sampled positions only. At probe time, ALL T_max positions feed the encoder
(no subsampling), giving the model the full receptive field. Effect: trains a T=10
encoder at the compute cost of T=5 per step, while the encoder learns
"subset-redundant" features.

### How to reproduce / verify

From repo root on `han-phase5b`:

```bash
# Train B2 at seed=42
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subseq_track2 --T_max 10 --t_sample 5 --seed 42

# Train B4 at seed=42
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subseq_h8 --T_max 10 --t_sample 5 --seed 42

# Probe (uses the same probe cache at experiments/phase5_downstream_utility/results/probe_cache/)
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.run_probing_phase5b \
    --run_ids phase5b_subseq_track2__seed42 phase5b_subseq_h8__seed42 \
    --aggregations last_position mean_pool

# Aggregate (writes test_auc and test_auc_flip; FLIP applied in analyzer)
.venv/bin/python -m experiments.phase5b_t_scaling_explore.analyze_tier1
```

Ckpts on HF: `han1823123123/txcdr/phase5b_ckpts/*.pt` (synced 2026-04-25). Probing
protocol matched to Phase 5: same 36-task split, same top-k-by-class-sep + L1 LR,
same k_feat=5 headline, **same FLIP convention** (now correctly applied).

### Negative results (re-confirmed; were always FLIP-naive but the conclusion is unchanged)

- **D1** strided window (T_eff=5, stride=2 spans 10 raw tokens at T=5 param cost):
  no-flip lp 0.7755 / mp 0.7893 — *worse* than vanilla T=5. Receptive-field
  expansion via stride hurts at fixed T_eff. T=5–6 sweet spot is genuine.
- **C2/C3** token-level encoder (sinusoidal pos emb / learned d_sae bias): no-flip
  mp 0.7252-0.7372. -8 to -9 pp vs H8. Per-position W_enc[t] is load-bearing.

### Bug fix to flag (unchanged from previous version)

The original `phase5b_subseq_sampling_txcdr.py` from the prior agent contained a
per-row `W_enc[pos_j]` gather inside the `_pre_activation_sampled` loop. At
d_sae=18432, B=1024 this allocates a (B, d_in, d_sae) tensor = ~174 GB. Replaced
with the mask-then-einsum pattern. Mathematically equivalent, ~1000× smaller peak.

### Pending / in-flight (as of 2026-04-25 18:30 UTC+1)

- **Iso cell** (T_max=10, t_sample=5, k_win=1000): training now. Will complete
  the 2x2 isolation matrix. ~30 min.
- **3-seed verification of T=10 t=8 k=500 winner** — *no longer needed*; that cell
  was a regression vs B4 baseline, not a winner.
- **T_max=20 cells of the 2D sweep**: all 3 (t=5, t=10, t=20) **OOM'd on the 32 GB
  5090** under the H8 multi-distance + matryoshka stack. The decoder + Adam state
  + multi-distance pair tensor exceeds 32 GB at T_max=20. **Phase 5 agent: if you
  have A40 access, please consider running these.** Specifically `subseq_h8` at
  (T_max=20, t_sample ∈ {5, 10, 20}, k_win=500) would extend the t_sample/T_max
  story to T=20.
- **t-encoder variant** (separate idea): allocate only t encoder/decoder slabs
  (not T_max), train via random sort-and-encode-by-rank. Would halve params and
  enable T_max=128. Not yet implemented; queued after iso.
- **T_max=128 t_sample=low** ("infinite-T low-t") via per-position W_enc — needs
  fp16 / smaller batch on 32 GB 5090.

### Files I touched in `src/`

- `src/architectures/phase5b_subseq_sampling_txcdr.py` — bug fix (mask-then-einsum)
  in `_pre_activation_sampled` for both `SubseqTXCBareAntidead` and `SubseqH8`.
- `src/architectures/phase5b_token_subseq_sae.py` — replaced `use_pos: bool` with
  `pos_mode: str` (none/sinusoidal/learned), added `pos_bias` parameter for
  `learned` mode. Backward-compat for `use_pos=True`.

I have NOT touched any Phase 5 files. Phase 5's `probing_results.jsonl`,
`training_index.jsonl`, ckpt set, and `__init__.py` REGISTRY are untouched.

— Phase 5B agent, 2026-04-25 (revised 18:30)
