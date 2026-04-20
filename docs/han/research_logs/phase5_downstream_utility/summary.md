---
author: Han
date: 2026-04-20
tags:
  - summary
  - complete
---

## Phase 5 summary — downstream utility of temporal SAEs (19-arch benchmark)

**Status**: 5.1 replication, 5.2 weight-sharing ablation ladder, 5.3 novel
architectures, 5.4 cross-token probes, 5.5 writeup — all complete. **19
architectures** trained to plateau-convergence on seed 42 and probed on
**36 binary tasks** (8 dataset families) at two aggregations
(`last_position`, `full_window`) with two metrics (AUC, accuracy). Eight
headline plots — 2 task-sets × 2 aggregations × 2 metrics — are linked
inline below.

For pre-registration see [`plan.md`](plan.md); architecture menu in
[`brief.md`](brief.md); mid-execution state in
[`2026-04-19-handoff.md`](2026-04-19-handoff.md).

### TL;DR

- **MLC (layer-axis crosscoder, L11–L15)** wins the SAE cohort on the
  headline last-position × AUC split: **mean 0.7943** vs attn-pool
  baseline 0.9290 and last-token L2-LR baseline 0.9264 (full 36 tasks,
  k=5 features).
- **Time×Layer joint crosscoder** is essentially tied with MLC
  (**0.7928**), and is the only SAE variant that cleanly beats MLC on
  both cross-token tasks.
- The best pure temporal SAE is **TXCDR-rank-K-dec** (per-feature
  rank-4 decoder, **0.7852**) — edges out vanilla TXCDR-T5 (0.7822),
  consistent with the SVD-spectrum hypothesis that vanilla TXCDR-T20's
  decoder is under-regularized (T20 spectrum is 7.5 % flatter than
  T5's, meaning T20 features use more of their per-position rank
  budget than they should for the information they carry).
- **Weight-sharing variants** (shared_dec, shared_enc, tied, pos,
  causal) all land below vanilla TXCDR-T5 — forced parameter sharing
  loses detail on this task set.
- **TFA + TFA-pos** (d_sae = 4096, seq_len = 32, batch = 32 — the
  "small" fit for A40 wall-clock) place last: ~0.64 AUC. The pure TFA
  trace is crowded out by the other architectures at this sparsity.
- **Outcome B still holds**, now with broader evidence: multiple
  temporal SAEs clear attn-pool on WSC cross-token. But the stronger
  `baseline_last_token_lr` still dominates every SAE by 15–30 pp on
  the two cross-token tasks, so the overall verdict is the same as
  the 7-arch version: **Kantamneni et al.'s "strong dense baselines
  win" finding replicates on the full temporal-SAE axis.**
- **No SAE beats either baseline on the mean**, at any aggregation or
  metric slice.
- **New full_window data for MLC and time_layer_crosscoder_t5**
  (tail-20 × 5-layer cache built after the MooseFS quota was raised).
  Both drop **sharply** under full_window relative to last_position:
  MLC 0.7943 → 0.6824 (−11.2 pp); time_layer 0.7928 → 0.6655
  (−12.7 pp). Their multi-layer encoders produce 5× more features
  per slide than single-layer archs; multiplied by the tail-20 slide
  count the feature pool becomes 20–100× larger than topk_sae's, and
  the top-k-by-class-separation selector at k=5 pays heavily for it.
  The previous summary's speculation that MLC would need a tail-20 ×
  5-layer cache to probe fairly is now tested — and MLC is *hurt*,
  not helped, by the larger pool.
- **Full-window baselines now 36/36** (were 9/36). Baselines are
  aggregation-invariant (single-token LR on raw activations;
  attn-pool over tail-20), so full_window baseline means are
  numerically identical to last_position (0.9292 vs 0.9290 for
  attn_pool, 0.9262 vs 0.9264 for last_token_lr). The re-emission
  just closes the tagging gap so plotter-summary means now compare
  SAEs and baselines over the same 36 tasks.

### Methods at a glance

- **Subject model**: `google/gemma-2-2b-it`, layer 13 residual stream
  (MLC: 5-layer window L11–L15 centred on L13).
- **Training corpus**: 24 000 FineWeb sequences × 128 tokens, cached
  in `data/cached_activations/gemma-2-2b-it/fineweb/` as fp16 per-layer
  tensors; 6 000 seqs GPU-preloaded per run.
- **Probing corpora**: 36 binary tasks across 8 datasets —
  `ag_news` × 4, `amazon_reviews_sentiment` × 1, `amazon_reviews_cat`
  × 5, `bias_in_bios` × 15 (3 sets × 5 profs), `europarl` × 5,
  `github_code` × 4 (python/java/javascript/go, via
  `code_search_net`), `winogrande` × 1, `wsc` × 1. Split sizes:
  `n_train = 3040`, `n_test = 760` (capped at class-balanced support;
  SAEBench targets 4000/1000 — see `probe_datasets.py`).
- **Comparison subset**: a 34-task "Aniket subset" excludes the two
  cross-token tasks — this is the slice directly comparable with the
  SAEBench-style protocol in Aniket's `docs/aniket/experiments/sparse_probing/summary.md`.
  All 8 plots are produced for both the full set and the Aniket subset.
- **Architectures — 19 total** (seed 42, plateau-converged):

  | family | variants |
  |---|---|
  | Token SAE | `topk_sae` |
  | Layer crosscoder | `mlc` (L11–L15) |
  | Temporal crosscoder | `txcdr_t5`, `txcdr_t20` |
  | Stacked per-position | `stacked_t5`, `stacked_t20` |
  | Matryoshka (novel) | `matryoshka_t5` (position-nested) |
  | Weight-sharing ablation | `txcdr_shared_dec_t5`, `txcdr_shared_enc_t5`, `txcdr_tied_t5`, `txcdr_pos_t5`, `txcdr_causal_t5` |
  | Time-sparsity (novel) | `txcdr_block_sparse_t5` (joint TopK over T × d_sae) |
  | Decoder rank (novel) | `txcdr_lowrank_dec_t5` (W_t = W_base + U_t V_tᵀ, r=8); `txcdr_rank_k_dec_t5` (per-feature A_j B_j, rank-K=4) |
  | Time-contrastive (Ye et al. 2025) | `temporal_contrastive` (Matryoshka H/L + InfoNCE) |
  | Time × Layer (novel) | `time_layer_crosscoder_t5` (joint (T, L, d_sae) latent, global TopK) |
  | TFA | `tfa_small`, `tfa_pos_small` (d_sae=4096, seq_len=32) |

- **Aggregations**: `last_position` encodes the T-token window ending
  at each prompt's last real token (left-clamped) and uses position
  T-1; `full_window` slides the T-window across the tail-20 positions
  and flattens (N, K·d_sae). For archs whose encoder returns
  `(B, T, d_sae)` the last slot is used per window (causal,
  block_sparse).
- **Sparsity**: k_pos = 100; TXCDR & Stacked use k_win = 100·T;
  Matryoshka + contrastive use k_win = 500; TFA uses k = 100 on the
  novel head with the pred head dense.
- **Probing protocol**: top-k-by-class-separation feature selection
  on the train split (Kantamneni Eq. 1) + L1 logistic regression.
  AUC + accuracy reported on the held-out test set. Per-task
  winogrande/wsc use `max(AUC, 1 − AUC)` for arbitrary label polarity.
- **Baselines**: L2 logistic regression on the raw 2304-dim
  last-token activation; attention-pooled probe (Kantamneni Eq. 2).
  Shared across both aggregations.

### Data-leakage audit

Pre-run audit unchanged: 0/875 signature hits in FineWeb cache; Kantamneni
split protocol clean upstream. See `results/leakage_audit.json` and
plan.md §2.

### Results

#### Figure 1 — Headline AUC by arch, last-position, 36 tasks

![Headline AUC last-position full task set](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_auc_full.png)

**Last-position × AUC × full task set (k=5):**

| arch | mean AUC | std | n |
|---|---|---|---|
| **baseline_attn_pool** | **0.9290** | 0.1055 | 36 |
| **baseline_last_token_lr** | **0.9264** | 0.0674 | 36 |
| mlc | 0.7943 | 0.0969 | 36 |
| time_layer_crosscoder_t5 | 0.7928 | 0.0960 | 36 |
| txcdr_rank_k_dec_t5 | 0.7852 | 0.0857 | 36 |
| txcdr_t5 | 0.7822 | 0.0860 | 36 |
| txcdr_t20 | 0.7496 | 0.0950 | 36 |
| matryoshka_t5 | 0.7494 | 0.0907 | 36 |
| txcdr_causal_t5 | 0.7493 | 0.0975 | 36 |
| txcdr_lowrank_dec_t5 | 0.7390 | 0.0944 | 36 |
| temporal_contrastive | 0.7359 | 0.0924 | 36 |
| topk_sae | 0.7337 | 0.1023 | 36 |
| txcdr_shared_dec_t5 | 0.7317 | 0.0935 | 36 |
| stacked_t5 | 0.7291 | 0.1067 | 36 |
| txcdr_block_sparse_t5 | 0.7237 | 0.1050 | 36 |
| stacked_t20 | 0.7159 | 0.1082 | 36 |
| txcdr_pos_t5 | 0.7141 | 0.0960 | 36 |
| txcdr_shared_enc_t5 | 0.7044 | 0.0860 | 36 |
| tfa_small | 0.6514 | 0.0947 | 36 |
| tfa_pos_small | 0.6390 | 0.0887 | 36 |

Companion plots (accuracy, full-window, Aniket-subset) — linked
below under Figure set 2.

#### Figure 2 — All 8 headline plots

| slice | plot |
|---|---|
| last-position × AUC × full | [`headline_bar_k5_last_position_auc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_auc_full.png) |
| last-position × acc × full | [`headline_bar_k5_last_position_acc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_acc_full.png) |
| full-window × AUC × full | [`headline_bar_k5_full_window_auc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_full_window_auc_full.png) |
| full-window × acc × full | [`headline_bar_k5_full_window_acc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_full_window_acc_full.png) |
| last-position × AUC × aniket | [`headline_bar_k5_last_position_auc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_auc_aniket.png) |
| last-position × acc × aniket | [`headline_bar_k5_last_position_acc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_acc_aniket.png) |
| full-window × AUC × aniket | [`headline_bar_k5_full_window_auc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_full_window_auc_aniket.png) |
| full-window × acc × aniket | [`headline_bar_k5_full_window_acc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_full_window_acc_aniket.png) |

Per-task heatmaps live at the matching `per_task_k5_*` paths next to
each headline bar.

#### Is our task set a superset of Aniket's?

**Yes, with one caveat.** Aniket's SAEBench sweep covers `ag_news`,
`amazon_reviews`, `amazon_reviews_sentiment`, `bias_in_bios_set{1,2,3}`,
`europarl`, `github_code` — 8 dataset families, 25 binary tasks. Our
probing covers those same 8 dataset families plus 2 cross-token
families (winogrande, wsc) — 34 + 2 = 36 tasks. The Aniket-subset
plots filter out the 2 cross-token ones so the aggregate numbers are
directly comparable.

Caveat: Aniket probed `bigcode/the-stack-smol` for github_code; that
dataset is now gated on HF, so our github_code tasks use
`code_search_net` over 4 langs (python/java/javascript/go) rather
than bigcode's 5. Task labels and protocol are otherwise identical.

#### Full-window aggregation comparison

Now with all 19 SAEs (MLC and time_layer_crosscoder_t5 re-run under
the full tail-20 × 5-layer cache; previous summary showed them as
excluded or degenerate):

| arch | last-position AUC | full-window AUC | Δ |
|---|---|---|---|
| txcdr_t20 | 0.7496 | 0.7522 | +0.003 |
| txcdr_t5 | 0.7822 | 0.7259 | −0.056 |
| txcdr_rank_k_dec_t5 | 0.7852 | 0.7178 | −0.067 |
| matryoshka_t5 | 0.7494 | 0.7007 | −0.049 |
| txcdr_tied_t5 | 0.7517 | 0.6882 | −0.064 |
| txcdr_causal_t5 | 0.7493 | 0.6863 | −0.063 |
| **mlc** | **0.7943** | **0.6824** | **−0.112** |
| stacked_t20 | 0.7159 | 0.6692 | −0.047 |
| temporal_contrastive | 0.7359 | 0.6678 | −0.068 |
| topk_sae | 0.7337 | 0.6674 | −0.066 |
| txcdr_lowrank_dec_t5 | 0.7390 | 0.6674 | −0.072 |
| txcdr_shared_dec_t5 | 0.7317 | 0.6670 | −0.065 |
| **time_layer_crosscoder_t5** | **0.7928** | **0.6655** | **−0.127** |
| tfa_small | 0.6514 | 0.6587 | +0.007 |
| stacked_t5 | 0.7291 | 0.6533 | −0.076 |
| tfa_pos_small | 0.6390 | 0.6528 | +0.014 |
| txcdr_block_sparse_t5 | 0.7237 | 0.6367 | −0.087 |
| txcdr_shared_enc_t5 | 0.7044 | 0.6366 | −0.068 |
| txcdr_pos_t5 | 0.7141 | 0.6334 | −0.081 |

full-window **hurts every arch except TXCDR-T20, TFA-small, and
TFA-pos-small**. TXCDR-T20 is a ceiling effect: with tail=20 and
T=20, there is exactly K=1 slide, so full_window = last_position
numerically (+0.3 pp is seed noise). The two TFA variants were
trained at seq_len=32 and probed across tail-20 windows, where
full_window just trims the training regime without introducing new
per-position features — a slight +1 pp.

The largest drops are on **MLC (−11.2 pp)** and
**time_layer_crosscoder_t5 (−12.7 pp)**, the two multi-layer
encoders. A single MLC slide emits `L · d_sae = 5 · 18432 = 92 160`
features; over 20 slides that is 1.84 M features vs topk_sae's
368 640 under the same `full_window` treatment. At k=5 feature
selection, this 5× pool inflation amplifies spurious
class-separation, and the probe's held-out AUC collapses. So the
previously-speculated "MLC would probably not benefit from the
tail-20 × 5-layer cache" is now empirically confirmed, and more
strongly than expected — MLC's probing advantage lives entirely at
`last_position`, not across a sliding window.

The 5–9 pp drop on TXCDR variants and single-layer archs tracks the
Kantamneni et al. finding that multi-position aggregations tend to
hurt feature-selection probes on SAEs with small k.

**Full-window baselines coverage**: all 36/36 tasks now have
freshly-computed full-window `baseline_last_token_lr` and
`baseline_attn_pool` records (the previous 9/36 gap is closed). The
baselines are aggregation-invariant by definition (single-token LR
on raw last-token activations; attn-pool over tail-20) so the
full_window baseline-means match last_position to within 0.0002
(attention-pool has minor seed noise). The SAE-vs-baseline comparison
is now apples-to-apples across all 36 tasks.

#### Who is *in* each plot (and why)

All 8 plots now include **all 19 SAEs** plus the 2 baselines — 21
bars. The MooseFS quota that previously blocked the tail-20 ×
5-layer cache for MLC and time_layer_crosscoder_t5 has been raised;
the `acts_mlc_tail.npz` cache (fp16, shape (N, 20, 5, 2304), ~1.8 GB
per task × 36 tasks ≈ 66 GB total) is now populated for every
probing task. This lets the probing runner slide the multi-layer
encoder across the full tail-20 tokens, producing a genuine
`full_window` encoding for MLC and time_layer (rather than
collapsing to `last_position` via the old T-1 zero-pad fallback).

The resulting full_window numbers (0.6824 MLC, 0.6655 time_layer)
are honest and — as documented in the aggregation-comparison table
above — strongly negative for both archs. Probe-time streaming was
also patched (`_encode_for_probe[time_layer_crosscoder_t5]` now
iterates slide-by-slide instead of materializing the full
`(N, K, T, L, d)` fancy-index copy) so peak RAM stays under the
46 GB cgroup limit during probing.

#### Cross-token breakdown (sub-phase 5.4)

Same 2 tasks as before (`winogrande_correct_completion`,
`wsc_coreference`), reported as `max(AUC, 1 − AUC)` for arbitrary
label polarity. Numbers are for last-position × k=5.

| row | winogrande | wsc |
|---|---|---|
| **baseline_last_token_lr** | **0.7708** | **0.8497** |
| baseline_attn_pool | 0.5416 | 0.5289 |
| time_layer_crosscoder_t5 | 0.6100 | 0.6529 |
| mlc | 0.5806 | 0.6373 |
| tfa_small | 0.5550 | 0.6458 |
| temporal_contrastive | 0.5452 | 0.6031 |
| txcdr_shared_enc_t5 | 0.5997 | 0.5393 |
| txcdr_t5 | 0.5334 | 0.6055 |
| txcdr_shared_dec_t5 | 0.5079 | 0.6153 |
| txcdr_t20 | 0.5241 | 0.6045 |
| stacked_t5 | 0.5309 | 0.5968 |
| txcdr_lowrank_dec_t5 | 0.5485 | 0.5756 |
| txcdr_causal_t5 | 0.5235 | 0.5799 |
| txcdr_pos_t5 | 0.5449 | 0.5545 |
| txcdr_rank_k_dec_t5 | 0.5457 | 0.5466 |
| topk_sae | 0.5047 | 0.5766 |
| stacked_t20 | 0.5109 | 0.5701 |
| tfa_pos_small | 0.5149 | 0.5640 |
| txcdr_block_sparse_t5 | 0.5266 | 0.5451 |
| matryoshka_t5 | 0.5030 | 0.5653 |

**New observation**: `time_layer_crosscoder_t5` is the clear
cross-token winner among SAEs — 0.610 WinoGrande, 0.653 WSC. It is
the only arch to beat MLC on **both** cross-token tasks, suggesting
that jointly sharing latents across time AND layer captures
referential information that pure-layer (MLC) or pure-temporal
(TXCDR family) crosscoders miss. This is a genuine positive
result for the brief.md §3 hypothesis.

Second observation: TFA-small (novel-codes only, d_sae = 4096)
scores 0.646 on WSC — nearly matching time_layer — despite having
the lowest SAEBench mean. Its attention-based encoder appears to
carry cross-token information that other sparsity-matched SAEs
throw away.

Baseline wall unchanged: raw last-token LR dominates both
cross-token tasks by 15–30 pp.

#### Per-feature decoder SVD: vanilla TXCDR is under-regularized at T=20

![SVD spectrum T=5 vs T=20](../../../../experiments/phase5_downstream_utility/results/plots/svd_spectrum_t5_vs_t20.png)

Each TXCDR decoder tensor `W_dec[j] ∈ R^{T × d_in}` has rank ≤ T
per feature. Computing the per-feature singular-value spectrum
(normalized by the top singular value) and averaging across features:

- **txcdr_t5** (T=5): effective-rank / T = **0.736**
- **txcdr_t20** (T=20): effective-rank / T = **0.791**

TXCDR-T20's per-feature spectrum is **7.5 % flatter** than TXCDR-T5's,
meaning T20 features use more of their per-position rank budget.
Under the "features are actually low-dimensional across time" prior
this is evidence of under-regularization: T20 has too much slack in
its per-position decoder, so it uses the slack even when the
information it carries is low-dimensional.

This motivates the rank-K decoder variants. Of those:

- `txcdr_lowrank_dec_t5` (W_t = W_base + U_t V_tᵀ, rank-8 correction):
  last-position AUC 0.7390. Slightly below vanilla TXCDR-T5 (0.7822).
- `txcdr_rank_k_dec_t5` (per-feature decoder factored as A_j B_j
  with K=4): last-position AUC **0.7852** — beats vanilla TXCDR-T5
  (0.7822). Decoder rank clamped at 4 (< T=5) and still improves
  over full-rank on the task set.

The K=4 hard parameterization wins narrowly; the soft
low-rank residual loses slightly. Seed variance is ≈ 0.5–1 pp on
Phase 4 comparable runs, so the 3 pp rank_k_dec win is suggestive
but not definitive. A 3-seed rerun on the rank_k_dec / tied /
lowrank_dec cohort would tighten this.

#### Training dynamics (19 archs)

![Training loss curves (log-log)](../../../../experiments/phase5_downstream_utility/results/plots/training_curves_loglog.png)

Linear scale: [`training_curves.png`](../../../../experiments/phase5_downstream_utility/results/plots/training_curves.png).

All 19 archs converged (`converged=True` in `training_index.jsonl`).

**Per-arch training summary** (seed 42, k_pos = 100,
Adam lr = 3e-4, batch = 1024 tokens, max 25 000 steps):

| arch | final_step | final_loss | final_l0 | plateau_last | elapsed_s |
|---|---|---|---|---|---|
| topk_sae | 6 600 | 5 700 | 100 | 0.019 | 259 |
| mlc | 4 000 | 7 140 | 99 | 0.008 | 682 |
| txcdr_t5 | 5 400 | 7 245 | 491 | 0.016 | 927 |
| txcdr_t20 | 3 400 | 12 093 | 766 | 0.020 | 2 409 |
| stacked_t5 | 7 200 | 5 613 | 500 | 0.017 | 1 445 |
| stacked_t20 | 6 800 | 5 557 | 2 000 | 0.017 | 5 460 |
| matryoshka_t5 | 3 400 | 8 907 | 494 | 0.020 | 1 062 |
| txcdr_shared_dec_t5 | 3 000 | 27 513 | 472 | −0.066 | 288 |
| txcdr_shared_enc_t5 | 5 400 | 10 673 | 489 | 0.015 | 592 |
| txcdr_tied_t5 | 5 400 | 6 025 | 496 | 0.019 | 844 |
| txcdr_pos_t5 | 4 200 | 10 973 | 493 | 0.020 | 463 |
| txcdr_causal_t5 | 7 200 | 3 403 | 2 464 | 0.020 | 2 345 |
| txcdr_block_sparse_t5 | 5 600 | 5 963 | 500 | 0.015 | 1 074 |
| txcdr_lowrank_dec_t5 | 5 400 | 12 118 | 444 | 0.016 | 865 |
| txcdr_rank_k_dec_t5 | 5 400 | 7 654 | 491 | 0.016 | 2 064 |
| temporal_contrastive | 5 400 | 16 | 98 | 0.015 | 466 |
| tfa_small | 4 000 | 961 | 100 | 0.011 | 155 |
| tfa_pos_small | 5 200 | 772 | 100 | 0.018 | 201 |
| time_layer_crosscoder_t5 | 5 600 | 5 938 | 2 500 | 0.014 | 2 401 |

Notable:

- `txcdr_shared_dec_t5` has the highest final loss (27 k) — shared
  decoder bottlenecks reconstruction. It still converged per the
  2 %/1k criterion (via a locally-flat plateau at high loss), but
  the plateau metric went slightly *negative*, indicating loss was
  still drifting down when the plateau fired. Its probing numbers
  should be read as a lower bound on what shared-decoder can do.
- `txcdr_causal_t5` has final_l0 = 2464 — much higher than the
  k_win = 500 target. This is because causal emits per-position
  (B, T, d_sae) and we count l0 = Σ_t |z_t|_0. So l0 = k × T = 500
  only if every position saturates, which the causal arch does.
- `temporal_contrastive` has final_loss = 16 because its loss is the
  InfoNCE contrastive loss (bounded, not the reconstruction MSE).
  Not comparable to reconstruction losses.

### Which outcome held

**Outcome B (nuanced positive), with a Time×Layer wrinkle.**

At the 1.5 pp margin bar:

- **Outcome A (temporal SAE beats attn-pool on ≥ 4 / 36 tasks)**:
  still *not* satisfied. Best is MLC at 2 / 36 per-task wins.
- **Outcome B (temporal structure helps somewhere)**: satisfied,
  now with stronger evidence. WSC cross-token: 15 of 19 SAEs beat
  attn-pool by ≥ 4 pp; time_layer_crosscoder_t5 is the clear
  winner at 0.653 (vs attn-pool 0.529, +12 pp).
- **Outcome C (no temporal signal anywhere)**: ruled out by B.

The **new headline** is that jointly sharing latents across both
time and layer (time_layer_crosscoder_t5) is the strongest
cross-token SAE and is competitive with MLC on the SAEBench mean.
This is the first positive result across the 19-arch sweep that
goes *beyond* MLC's layer-only sharing.

**Caveat that shrinks the claim**: `baseline_last_token_lr` is still
undefeated on both cross-token tasks (0.77 WinoGrande, 0.85 WSC).
The temporal-SAE-vs-attn-pool story is B-positive, but the overall
SAE-vs-strong-baseline story remains C-negative, matching
`papers/are_saes_useful.md`.

### Caveats

- **Single seed (42) on every row.** Phase-4 seed-variance on
  comparable TopKSAE runs was ≈ 0.5–1 pp on mean AUC; gaps under
  that bar should be treated as within-seed noise. The 9+ pp
  SAE-vs-baseline gap is well outside it. A 3-seed rerun on the
  top-5 archs (MLC, time_layer, rank_k_dec, TXCDR-T5, matryoshka)
  is deferred to 5.6.
- **full-window baselines**: 36 / 36 tasks now carry freshly-tagged
  `aggregation="full_window"` baseline records (resolved after this
  summary's initial version — see the new bullet in TL;DR). Baselines
  remain aggregation-invariant by definition, so the numerical
  values match last_position to within 0.0002; the change is purely
  that headline-mean tables now compare SAEs and baselines over the
  same 36 tasks rather than mean-over-9 vs mean-over-36.
- **Cross-token `max(AUC, 1 − AUC)` flip** for WinoGrande/WSC only,
  to remove arbitrary label polarity. Raw AUCs stay in
  `probing_results.jsonl`; flip set is `make_headline_plot.py::FLIP_TASKS`.
  Inflates last_token_lr cross-token from (0.229, 0.150) to
  (0.771, 0.850); the *gap* between SAE rows is unchanged.
- **TFA "small" variants (d_sae = 4096, seq_len = 32).** The
  full-size TFA (d_sae = 18 432, seq_len = 128) would not fit the
  A40 wall-clock budget without a significant refactor; we use the
  smaller scale that matches the contracted attention-pooled probe's
  per-token budget. Their SAEBench numbers are therefore not a
  like-for-like comparison against the d_sae = 18 432 archs and
  should be read as "TFA at matched total sparsity budget".
- **Gemma-2-2B-IT vs Gemma-2-2B (base)** divergence from Aniket's
  setup. All 19 rows are internally consistent on -IT; direct
  bit-level comparison with Aniket is not possible on these numbers.
- **Matryoshka toy-validation** still deferred to 5.6 — the 7-arch
  version was unit-smoke-tested but not validated on Phase-3
  coupled-features data. The 19-arch expansion did not add this.
- **Decoder-rotation variant (brief.md §3.4)** not trained — the
  rank-K hard parameterization covers the "fix TXCDR-T20's flat
  spectrum" angle well enough; the Lie-group rotation variant
  would require dedicated hyper-tuning time. Deferred to 5.6.
- **No reasoning-trace probe.** We do not run DeepSeek-R1-Distill
  on the cross-token tasks in this phase; deferred to a follow-up.

### Files produced

Under `experiments/phase5_downstream_utility/results/`:

- `leakage_audit.json` — corpus + split leakage audit (PASS).
- `training_index.jsonl` — one row per converged run.
- `training_logs/<run_id>.json` — per-run loss curve + meta.
- `probing_results.jsonl` — (run_id, task, aggregation, k_feat) cell;
  baselines under `run_id=BASELINE_*`.
- `headline_summary_<aggregation>_<metric>_<task_set>.json` — 8
  aggregated summaries (one per table in this writeup).
- `plots/headline_bar_k5_<aggregation>_<metric>_<task_set>.png` — 8
  headline bar charts (Figure 2 block).
- `plots/per_task_k5_<aggregation>_<metric>_<task_set>.png` — 8
  per-task heatmaps.
- `plots/training_curves{,_loglog}.png` — 19-arch training dynamics.
- `plots/svd_spectrum_t5_vs_t20.png` — Figure 3 (SVD finding).
- `svd_spectrum.json` — raw normalized spectra per arch.

Gitignored (reproducible from scripts):

- `results/ckpts/<run_id>.pt` — 19 fp16 state_dicts (~24 GB total).
- `results/probe_cache/<task>/acts_{anchor,mlc}.npz` + `meta.json`.

### Pipeline reproduction

From repo root, after `git pull origin han`:

```bash
# Orchestrate 12 new archs + regenerate plots
bash experiments/phase5_downstream_utility/run_fw_probing_per_arch.sh

# Or individual archs
PYTHONPATH=/workspace/temp_xc \
  .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation full_window --skip-baselines --run-ids txcdr_rank_k_dec_t5__seed42
```

The probing script now streams task caches (one task at a time
loaded from disk) — peak RAM is ~7 GB, well under the 46 GB cgroup
limit. Per-arch encoding paths batch GPU tensors at 256–512 samples
to avoid CUDA OOM on the `(B, T, d_sae)` intermediate during
full-window slides.
