---
author: Han
date: 2026-04-19
tags:
  - summary
  - in-progress
---

## Phase 5 summary — downstream utility of temporal SAEs

**Status**: in progress. Numbers, figures, and outcome-specific commentary
to be filled in as training + probing runs complete. This file is the
write-up home for Phase 5.1–5.4 under whichever of the three
pre-registered outcomes actually holds.

For the full pre-registration see [`plan.md`](plan.md). For the broader
context of the phase (architecture menu, design axes, success criteria)
see [`brief.md`](brief.md).

### TL;DR

*To be filled once headline numbers are in.*

Expected form: "At matched per-token sparsity (k_pos = 100) on
Gemma-2-2B-IT layer 13, <arch> achieves mean AUC = <X.XXX> across
<N> binary sparse-probing tasks, vs the attention-pooled baseline's
<Y.YYY>. <Result characterization>."

### Methods at a glance

- **Subject model**: `google/gemma-2-2b-it`, residual stream at layer 13
  (MLC uses a 5-layer window L11-L15 centred on L13).
- **Training corpus**: 24 000 FineWeb sequences × 128 tokens, cached in
  `data/cached_activations/gemma-2-2b-it/fineweb/` as fp16 per-layer
  tensors. 6 000 sequences preloaded to GPU per training run to remove
  MooseFS-mmap disk bottleneck.
- **Probing corpora**: 25 binary tasks built from 5 SAEBench-compatible
  datasets (bias_in_bios × 3 sets of 5 professions, ag_news × 4,
  europarl × 4, amazon_reviews × 1, sentiment × 1). Tasks use SAEBench's
  split sizes: `n_train = 4000`, `n_test = 1000` (or smaller if a
  class's balanced support is capped).
- **Architectures**: 7 primary after scope-cut addenda — TopKSAE, MLC
  (L=5), TXCDR (T=5, T=20), StackedSAE (T=5, T=20), MatryoshkaTXCDR (T=5).
  TFA / TFA-pos / SharedPerPositionSAE deferred to sub-phase 5.6 (see
  plan.md addendum 2026-04-19).
- **Sparsity**: k_pos = 100 across all archs; TXCDR & Stacked have
  k_win = 100·T. TFA has k=100 on novel head, dense pred head.
  Matryoshka has k_win = 500 on the shared window latent.
- **Probing protocol**: last-position encoding of the T-token window
  ending at each prompt's last real token (left-clamped when the prompt
  is shorter than T). Top-k-by-class-separation feature selection on
  the train split only (Kantamneni Eq. 1), then L1 logistic regression.
  AUC on the held-out test set.
- **Baselines**: L2 logistic regression on raw last-token activation;
  attention-pooled probe (Kantamneni Eq. 2) over the tail-32-token
  window.

### Data-leakage audit

Both corpus leakage (SAEBench probe text appearing verbatim in
FineWeb training cache) and split leakage (feature selection seeing
test data) were audited before any training run. See
`results/leakage_audit.json` and plan.md §2 for the full report.
Corpus leakage: 0/875 signatures. Split leakage: SAEBench-style
protocol confirmed clean upstream. Verdict: proceed without
retraining.

### Results (to be filled)

#### Figure 1 — Headline bar chart

*Placeholder for* `plots/headline_bar_k5.png`.

#### Figure 2 — Per-task heatmap

*Placeholder for* `plots/per_task_k5.png`.

#### Training dynamics

*Placeholder for convergence-corrected loss curves. Each run writes a
loss curve to `results/training_logs/<run_id>.json`; we aggregate into
a single log-log plot with one line per architecture.*

### Training fairness summary

*Per the fairness rubric in [brief.md §"Reference: Training-fairness
rubric"]:*

| rule | status |
|---|---|
| same hyperparameter search budget across archs | One shared TrainCfg dataclass; all archs use lr=3e-4, Adam, plateau stop <2%/1k, batch=1024 (TFA uses batch=32). |
| convergence within max-step cap | Plateau metric logged and reported for every run in `results/training_index.jsonl`. |
| sparsity normalized at window level | Protocol A (per-token k=100 everywhere) documented; TXCDR k_win = 500/2000 at T=5/20 matches Stacked. |
| FLOPs + param counts reported | TODO — add to summary table. |
| knob ablation on toy before NLP compute | Matryoshka toy-validation deferred to sub-phase 5.6 (time budget). Mathematical correctness verified by unit-smoke: latents correctly split into nested prefix windows (m_1..m_T all sum to d_sae), each scale has a separate decoder, loss averages correctly across scales. |
| three seeds minimum | Seed 42 on every row; a 3-seed rerun on the 4 primary archs (TopKSAE, MLC, TXCDR T=5, Matryoshka T=5) runs after 5.4 if time allows. |

### Which outcome held

*To fill in one of:*

*A. A TXCDR variant wins head-to-head against MLC + beats
attention-pooled baseline on ≥ 4/8 tasks.*

*B. MLC wins on SAEBench but a temporal variant beats attention-pooled
on the cross-token task (5.4).*

*C. No temporal variant beats attention-pooled baselines anywhere.*

### Caveats (drafted before results)

- **Single seed, single layer, single k for the main 5.1 sweep.** The
  3-seed rerun on 4 primary archs mitigates seed variance on the
  headline cell. Layer and k are held fixed by design.
- **Gemma-2-2B-IT vs Gemma-2-2B (base) divergence from Aniket's setup.**
  Our numbers are internally consistent with Phase 4 but not
  bit-for-bit comparable to Aniket's.
- **last_position aggregation only.** Mean/max/full_window
  aggregations (Phase-4-style) would add 3× probing cost but
  potentially help TXCDR/Stacked; the plan defers these to sub-phase
  5.6 unless the headline hinges on them.
- **SharedPerPositionSAE redundancy.** Under last-position probing
  aggregation with T shared weights, the SharedPerPositionSAE's
  read-out path is mathematically identical to TopKSAE's. Differences
  enter only through training dynamics (window-context gradient
  structure). Included as a control; expected to match TopKSAE within
  seed variance.
- **Cross-token probing (5.4) uses the same Gemma-2-2B-IT at L13.**
  We do not run the reasoning-trace probe (DeepSeek-R1-Distill) in
  this phase; that is deferred to a follow-up.

### Files produced

- Checkpoints: `results/ckpts/<run_id>.pt`
- Training sidecars: `results/training_logs/<run_id>.json`
- Training index: `results/training_index.jsonl`
- Probing raw records: `results/probing_results.jsonl`
- Aggregated summary: `results/headline_summary.json`
- Plots: `results/plots/{headline_bar_k5, per_task_k5}.{png, thumb.png}`

### Pipeline reproduction

From repo root, after `git pull origin han`:

```bash
bash experiments/phase5_downstream_utility/run_phase5_pipeline.sh
```

Assumes `data/cached_activations/gemma-2-2b-it/fineweb/` has
`token_ids.npy` + `resid_L{11,12,13,14,15}.npy`. If any of those is
missing, rebuild with
`experiments/phase5_downstream_utility/build_multilayer_cache.py`.
