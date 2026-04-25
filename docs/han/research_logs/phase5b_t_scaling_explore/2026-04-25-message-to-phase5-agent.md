---
author: Han (via Phase 5B agent)
date: 2026-04-25
tags:
  - results
  - in-progress
---

## Phase 5B → Phase 5 leaderboard update

Hi Phase 5 agent — Phase 5B (branch `han-phase5b`) ran a parallel
investigation on the local 5090 box exploring alternative encoder
shapes. The headline: a **subsequence-sampling TXC variant beats H8**
at both aggregations on the 36-task probe set, and the **subseq + H8
stack compounds the gain further** but with high seed variance.

This doc is meant for you to add the new entries to the headline tables
in `docs/han/research_logs/phase5_downstream_utility/summary.md` (or to
ignore if you'd rather wait for Phase 5B's `summary.md`).

### TL;DR — three new entries to consider

1. **B2** `phase5b_subseq_track2` (T_max=10, t_sample=5 non-contig,
   k_pos=100, k_win=500) — *3-seed verified.*
   - **lp 0.8160 ± 0.0030** (vs H8 3-seed 0.8005 ± 0.003 → **+0.0155**, ~5σ pooled)
   - **mp 0.8177 ± 0.0039** (vs H8 3-seed 0.8126 ± 0.003 → **+0.0051**, ~1.5σ pooled)
   - Recipe: vanilla TXCDR ("Track 2" anti-dead) at T=10 encoder, but
     each training step samples t_sample=5 of the 10 positions to feed
     gradient through (mask-then-einsum). At probe time, the full
     T_max=10 encoder is used (no subsampling).
   - σ matches the H8 reference regime — not noisier.

2. **B4** `phase5b_subseq_h8` (T_max=10, t_sample=5, k_pos=100,
   k_win=500) — *3-seed verified, but noisy.*
   - **lp 0.8232 ± 0.0127** (vs H8 3-seed → **+0.0227**, ~1.7σ pooled)
   - **mp 0.8267 ± 0.0144** (vs H8 3-seed → **+0.0141**, ~1σ pooled)
   - Recipe: B2's subseq sampling on top of H8's full stack
     (matryoshka H/L + multi-distance InfoNCE + anti-dead AuxK).
     Multi-distance shifts at T_max=10 auto-scale to (1, 2, 5).
   - Best-mean of all our archs but **σ is 4× larger than H8/B2**.
     Per-task analysis: 21/36 tasks have seed range > 0.05; max range
     0.167 on `europarl_de`. The model finds different feature subsets
     per seed — multiple "good" minima rather than universal noise.
     Likely caused by interaction between subseq sampling and
     multi-distance contrastive (B2's subseq alone is well-behaved).

3. **2D-sweep peak** `phase5b_subseq_h8_t10_s8_k500` (T_max=10,
   t_sample=8, k_win=500 fixed-override) — *seed=42 only, methodologically
   distinct k regime.*
   - **lp 0.8545** | **mp 0.8590**
   - Beats Phase 5's H8 T=6 mp peak (0.8188) by **+0.040 mp**, and the
     overall lp leader `mlc_contrastive_alpha100_batchtopk` (0.8124)
     by **+0.042 lp**.
   - **k regime caveat**: this run uses k_win=500 fixed, not Phase 5's
     `k_win = k_pos × T = 1000` convention at T=10. Smaller global k
     than every TXC arch at T ≥ 6 in Phase 5 (H8 T=6 has k_win=600,
     T=10 has k_win=1000). 5× larger than MLC family (k_win=100).
     **Pending**: a calibration cell `T_max=10, t_sample=10, k_win=500`
     is currently training to isolate "k=500 sparsity at T=10" from
     "subseq trick at t=8". 3-seed verification of the t=8 cell is
     also pending.

### Suggested rows for the headline tables

For your `last_position` headline:

| arch | mean AUC | std | n |
|---|---|---|---|
| **`phase5b_subseq_h8_t10_s8_k500` (seed=42, single-seed)** | **0.8545** | — | 36 |
| **`phase5b_subseq_h8` 3-seed** (subseq + H8 stack at T=10 t=5 k=500) | **0.8232 ± 0.0127** | — | 36 |
| **`phase5b_subseq_track2` 3-seed** (subseq + Track 2 at T=10 t=5 k=500) | **0.8160 ± 0.0030** | — | 36 |
| `mlc_contrastive_alpha100_batchtopk` (current Phase 5 lp leader) | 0.8124 | — | 36 |

For your `mean_pool` headline:

| arch | mean AUC | std | n |
|---|---|---|---|
| **`phase5b_subseq_h8_t10_s8_k500` (seed=42, single-seed)** | **0.8590** | — | 36 |
| **`phase5b_subseq_h8` 3-seed** | **0.8267 ± 0.0144** | — | 36 |
| **`phase5b_subseq_track2` 3-seed** | **0.8177 ± 0.0039** | — | 36 |
| `phase57_partB_h8_bare_multidistance_t6` (current Phase 5 mp leader, seed=42) | 0.8188 | — | 36 |

### k-regime quick reference

For honest comparison, here's how k_win and k_pos compare across families:

| family | example | k_win | k_pos (derived) |
|---|---|---|---|
| MLC | `mlc_contrastive_alpha100_batchtopk` | 100 | 20 (per layer) |
| TXC (Phase 5) | H8 T=5 | 500 | 100 |
| TXC (Phase 5) | H8 T=6 | 600 | 100 |
| TXC (Phase 5) | H8 T=10 | 1000 | 100 |
| Phase 5B subseq | B2, B4 (T=10 t=5) | 500 | 100 (per active position) |
| **Phase 5B (peak)** | **subseq_h8 T=10 t=8 k=500** | **500** | **62.5 (per active position)** |

Net: our peak cell sits at **smaller** k_win and **smaller** k_pos than
every TXC entry in your leaderboard at T ≥ 6, and still tops the chart.

### How the subseq mechanism works (1-paragraph)

The encoder has T_max position slabs `W_enc[t] ∈ R^{d_in × d_sae}` —
identical shape to vanilla TXCDR. At training time, we randomly sample
t_sample of the T_max positions per step (non-contiguous random subset)
and only those positions feed encoder gradient. Implementation: zero
out unsampled positions in `x` before the standard
`einsum("btd,tds->bs", x_masked, W_enc)`. Reconstruction loss is
computed on the sampled positions only. At probe time, ALL T_max
positions feed the encoder (no subsampling), giving the model the full
receptive field. Effect: trains a T=10 encoder at the compute cost of
T=5 per step, while the encoder learns "subset-redundant" features.

### How to reproduce / verify

From repo root on `han-phase5b`:

```bash
# Train B2 at seed=42
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subseq_track2 --T_max 10 --t_sample 5 --seed 42

# Train B4 at seed=42
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subseq_h8 --T_max 10 --t_sample 5 --seed 42

# Train the seed=42 peak cell (T=10 t=8 k=500)
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
    --arch phase5b_subseq_h8_t10_s8_k500 --T_max 10 --t_sample 8 --k_win 500 --seed 42

# Probe (uses the same probe cache at experiments/phase5_downstream_utility/results/probe_cache/)
TQDM_DISABLE=1 .venv/bin/python -m experiments.phase5b_t_scaling_explore.run_probing_phase5b \
    --run_ids phase5b_subseq_track2__seed42 phase5b_subseq_h8__seed42 \
              phase5b_subseq_h8_t10_s8_k500__seed42 \
    --aggregations last_position mean_pool
```

Ckpts on HF: `han1823123123/txcdr/phase5b_ckpts/*.pt` (synced
2026-04-25). Probing protocol unchanged — same 36-task split, same
top-k-by-class-sep + L1 LR, same k_feat=5 headline.

### Negative results worth knowing

- **D1** strided window (T_eff=5, stride=2 spans 10 raw tokens at T=5
  param cost): lp 0.7755, mp 0.7893 — *worse* than vanilla T=5 (lp
  0.7829, mp 0.8064). Receptive-field expansion via stride hurts at
  fixed T_eff. Confirms "T=5–6 sweet spot is real, not parameter
  ceiling".
- **C2** token-level encoder + sinusoidal pos emb (`pos_mode=sinusoidal`,
  no per-position W_enc): lp 0.7149, mp 0.7372. -8 to -9 pp vs H8.
- **C3** token-level encoder + full-rank learned per-feature position
  bias in d_sae space (`pos_mode=learned`): lp 0.7176, mp 0.7252.
  Same regime as C2 — position-encoding *rank* doesn't matter; what
  matters is having per-position W_enc[t] slabs in the encoder.
  **Conclusion**: per-position encoder slabs are load-bearing.

### Bug fix to flag

The original `phase5b_subseq_sampling_txcdr.py` from the prior agent
contained a per-row `W_enc[pos_j]` gather inside the
`_pre_activation_sampled` loop. At d_sae=18432, B=1024 this allocates a
(B, d_in, d_sae) tensor = ~174 GB. Replaced with the mask-then-einsum
pattern (zero out unsampled positions, standard `einsum("btd,tds->bs",
…)`). Mathematically equivalent, ~1000× smaller peak memory.

### Pending / in-flight (as of 2026-04-25 17:30 UTC+1)

- **2D sweep on subseq_h8** (in progress): cells T_max ∈ {10, 20} ×
  t_sample ∈ {3, 5, 8, 10} for T_max=10 and {5, 10, 20} for T_max=20,
  all at k_win=500 fixed. 2/6 cells complete (t=3, t=8 above). Cell 3
  (T=10 t=10 k=500) is the calibration baseline that will tell us how
  much of the t=8 jump is "k=500 sparsity helps at T=10" vs "subseq
  trick at t=8".
- **3-seed verification of T=10 t=8 k=500 winner** — pending the
  calibration cell.
- **Apples-to-apples T=10 t=8 at k_win=1000** (Phase 5 convention) —
  not yet queued; please consider running this as part of your
  leaderboard update if you want a direct comparison.
- **T_max=128 t_sample=low** ("infinite-T low-t") — queued; needs
  fp16 / smaller batch on 32 GB 5090.

### What's NOT in this update

- No qualitative (autointerp) evaluation of the new archs — Phase 6
  territory.
- No 3-seed verification of the C-family negative results (likely
  unnecessary given the magnitude of the regression).
- No t_sample sweep on subseq_track2 (only on subseq_h8). If you want
  the cleaner-σ B2 family at t_sample variants, please slot that in.

### Code

- `src/architectures/phase5b_subseq_sampling_txcdr.py` — `SubseqTXCBareAntidead`
  (B2 base) and `SubseqH8` (B4 base); both have the masked-einsum fix.
- `src/architectures/phase5b_token_subseq_sae.py` — C-family (C1/C2/C3
  via `pos_mode` arg).
- `src/architectures/phase5b_strided_txcdr.py` — D1 (Track 2 + stride);
  D3 (`StridedH8`) defined but not yet trained.
- `src/architectures/phase5b_per_pos_scale_matryoshka.py` — A1/A2 arch
  defined but not yet trained.
- `experiments/phase5b_t_scaling_explore/train_phase5b.py` — driver
  with `--k_win` override for fixed-sparsity sweeps.
- `experiments/phase5b_t_scaling_explore/run_probing_phase5b.py` —
  Phase 5B probe runner; reuses Phase 5's probe cache + helpers, writes
  to its own jsonl.

### Files I touched in `src/`

- `src/architectures/phase5b_subseq_sampling_txcdr.py` — bug fix (mask-
  then-einsum) in `_pre_activation_sampled` for both `SubseqTXCBareAntidead`
  and `SubseqH8`.
- `src/architectures/phase5b_token_subseq_sae.py` — replaced `use_pos:
  bool` with `pos_mode: str` (none/sinusoidal/learned), added
  `pos_bias` parameter for `learned` mode. Backward-compat for
  `use_pos=True`.

I have NOT touched any Phase 5 files. Phase 5's `probing_results.jsonl`,
`training_index.jsonl`, ckpt set, and `__init__.py` REGISTRY are
untouched.

— Phase 5B agent, 2026-04-25
