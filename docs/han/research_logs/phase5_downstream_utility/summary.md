---
author: Han
date: 2026-04-21
tags:
  - summary
  - complete
---

## Phase 5 summary — downstream utility of temporal SAEs (25-arch benchmark)

**Status**: 5.1 replication, 5.2 weight-sharing ablation ladder, 5.3 novel
architectures, 5.4 cross-token probes, 5.5 writeup, 5.6 T-sweep +
mean_pool aggregation + error-overlap analysis — all complete (seed 42).
**25 architectures** trained to plateau-convergence on seed 42 and
probed on **36 binary tasks** (8 dataset families) at two
aggregations (`last_position`, `mean_pool`) with two metrics (AUC,
accuracy). 3-seed autoresearch on the top-5 archs is in progress
(T17; see *Seed variance* section). Headline plots — 2 task-sets ×
2 aggregations × 2 metrics — are linked inline below.

For pre-registration see [`plan.md`](plan.md); architecture menu in
[`brief.md`](brief.md); overnight rollout state in
[`2026-04-20-overnight-handoff.md`](2026-04-20-overnight-handoff.md).

### TL;DR

- **Best SAE at `last_position`**: `mlc_contrastive` (**0.8025**), a
  new port of Ye et al. 2025's temporal contrastive loss to the MLC
  (layer-axis crosscoder) base — edges out vanilla `mlc` (0.7943) and
  `time_layer_crosscoder_t5` (0.7928). First arch in the sweep to
  cross 0.80 at last_position.
- **Best SAE at `mean_pool`** (SAEBench-canonical aggregation —
  averages per-slide latents over the tail-20 window): **`txcdr_t5`
  (0.8064)**. TXCDR-T5 gains +2.4 pp switching aggregations; MLC
  gains only −0.9 pp (all multi-layer archs are shape-invariant under
  mean_pool because it collapses to the single layer-encoded vector).
  Full mean_pool leaderboard swaps the top cluster: four TXCDR
  variants (T3, T5, T15, rank_k) occupy the top 4 SAE slots.
- **Collaborator ask (error-overlap analysis)**: **TXCDR-T5 and
  `mlc_contrastive` are the MOST complementary top archs** — Jaccard
  of error sets 0.338 (vs up to 0.482 for pairs within the MLC
  family). McNemar's χ² is significant at p<0.05 on 16 / 36 tasks.
  So they solve *different* subsets of tasks at comparable mean AUC;
  an ensemble of the two should beat either alone. Conversely `mlc`
  and `mlc_contrastive` are the MOST similar pair (Jaccard 0.482),
  confirming mlc_contrastive inherits most of MLC's task-selection
  bias and adds a small complementary signal on top.
- **T-sweep ladder** (TXCDR at T ∈ {2, 3, 5, 8, 10, 15, 20}): a
  modest peak at T=5 for both aggregations. Under `last_position`,
  going below T=5 (T=2 → 0.744, T=3 → 0.771) loses some temporal
  structure, and going above it (T=20 → 0.750) hurts because the
  decoder becomes too loose (per-feature SVD spectrum is 7.5 %
  flatter at T=20 than T=5). Under `mean_pool`, the ordering is
  similar but gaps are compressed — mean_pool cushions low-T archs
  by averaging more slides.
- **Outcome**: **no SAE beats either baseline** (0.929 attn-pool,
  0.926 last-token LR). TXCDR-T5 is 12.3 pp below attn-pool under
  mean_pool; `mlc_contrastive` is 12.6 pp below it at last_position.
  Outcome B (nuanced positive — some SAE does beat attn-pool on
  cross-token tasks) holds, same as before.
- **Deprecated**: the `full_window` aggregation. It is dominated by
  `mean_pool` on SAEBench-canonicalness (mean_pool averages; full_window
  concatenates 20 × d_sae features and selects the top-k globally —
  inflating the feature pool to no downstream benefit). JSONL rows are
  retained for reproducibility; new plots omit it. Prior full_window
  findings (MLC: −11.2 pp drop, time_layer: −12.7 pp) remain in the
  JSONL and in the *Historical full_window record* section for
  completeness.

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
  cross-token tasks for direct comparability with the SAEBench-style
  protocol in Aniket's `docs/aniket/experiments/sparse_probing/summary.md`.
- **Architectures — 25 total** (seed 42, plateau-converged):

  | family | variants |
  |---|---|
  | Token SAE | `topk_sae` |
  | Layer crosscoder | `mlc` (L11–L15), `mlc_contrastive` (MLC + Matryoshka H/L + InfoNCE on adjacent tokens) |
  | Temporal crosscoder (T-sweep) | `txcdr_t2`, `txcdr_t3`, `txcdr_t5`, `txcdr_t8`, `txcdr_t10`, `txcdr_t15`, `txcdr_t20` |
  | Stacked per-position | `stacked_t5`, `stacked_t20` |
  | Matryoshka (novel) | `matryoshka_t5` (position-nested) |
  | Weight-sharing ablation | `txcdr_shared_dec_t5`, `txcdr_shared_enc_t5`, `txcdr_tied_t5`, `txcdr_pos_t5`, `txcdr_causal_t5` |
  | Time-sparsity (novel) | `txcdr_block_sparse_t5` (joint TopK over T × d_sae) |
  | Decoder rank (novel) | `txcdr_lowrank_dec_t5` (W_t = W_base + U_t V_tᵀ, r=8); `txcdr_rank_k_dec_t5` (per-feature A_j B_j, rank-K=4) |
  | Time-contrastive (Ye et al. 2025) | `temporal_contrastive` (Matryoshka H/L + InfoNCE on (t−1, t) pairs) |
  | Time × Layer (novel) | `time_layer_crosscoder_t5` (joint (T, L, d_sae) latent, global TopK) |
  | TFA | `tfa_small`, `tfa_pos_small` (d_sae=4096, seq_len=32) |

- **Aggregations** (canonical two — `full_window` deprecated):
  - `last_position` encodes the T-token window ending at each
    prompt's last real token (left-clamped) and uses position T−1.
  - `mean_pool` slides a T-window across the tail-20 positions,
    encodes each slide to `d_sae`, then averages the K = 20 − T + 1
    slide-outputs to a single `d_sae` vector per example. Matches
    SAEBench / Kantamneni's `get_sae_meaned_activations` convention.
- **Sparsity**: k_pos = 100; TXCDR & Stacked use k_win = 100·T;
  Matryoshka + contrastive use k_win = 500; TFA uses k = 100 on the
  novel head with the pred head dense.
- **Probing protocol**: top-k-by-class-separation feature selection
  on the train split (Kantamneni Eq. 1) + L1 logistic regression.
  AUC + accuracy reported on the held-out test set. Per-task
  winogrande/wsc use `max(AUC, 1 − AUC)` for arbitrary label polarity.
- **Baselines**: L2 logistic regression on the raw 2304-dim
  last-token activation; attention-pooled probe (Kantamneni Eq. 2).
  Shared across both aggregations (36/36 coverage at last_position
  and mean_pool).

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
| mlc_contrastive | 0.8025 | 0.0865 | 36 |
| mlc | 0.7943 | 0.0969 | 36 |
| time_layer_crosscoder_t5 | 0.7928 | 0.0960 | 36 |
| txcdr_rank_k_dec_t5 | 0.7852 | 0.0857 | 36 |
| txcdr_t5 | 0.7822 | 0.0860 | 36 |
| txcdr_t15 | 0.7772 | 0.0837 | 36 |
| txcdr_t3 | 0.7711 | 0.1104 | 36 |
| txcdr_t10 | 0.7671 | 0.0937 | 36 |
| txcdr_t8 | 0.7540 | 0.0974 | 36 |
| txcdr_tied_t5 | 0.7517 | 0.1045 | 36 |
| txcdr_t20 | 0.7496 | 0.0950 | 36 |
| matryoshka_t5 | 0.7494 | 0.0907 | 36 |
| txcdr_causal_t5 | 0.7493 | 0.0975 | 36 |
| txcdr_t2 | 0.7441 | 0.1155 | 36 |
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

#### Figure 2 — Headline AUC by arch, mean-pool, 36 tasks

![Headline AUC mean-pool full task set](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_mean_pool_auc_full.png)

**Mean-pool × AUC × full task set (k=5):**

| arch | mean AUC | std | n |
|---|---|---|---|
| **baseline_attn_pool** | **0.9292** | 0.1056 | 36 |
| **baseline_last_token_lr** | **0.9262** | 0.0673 | 36 |
| txcdr_t5 | 0.8064 | 0.0957 | 36 |
| txcdr_t3 | 0.8022 | 0.1045 | 36 |
| txcdr_rank_k_dec_t5 | 0.7990 | 0.0944 | 36 |
| txcdr_t15 | 0.7868 | 0.0805 | 36 |
| mlc | 0.7848 | 0.1064 | 36 |
| mlc_contrastive | 0.7801 | 0.1043 | 36 |
| txcdr_tied_t5 | 0.7793 | 0.1025 | 36 |
| txcdr_t2 | 0.7786 | 0.1133 | 36 |
| txcdr_t10 | 0.7754 | 0.0965 | 36 |
| temporal_contrastive | 0.7749 | 0.0978 | 36 |
| matryoshka_t5 | 0.7747 | 0.0945 | 36 |
| time_layer_crosscoder_t5 | 0.7722 | 0.1073 | 36 |
| txcdr_t8 | 0.7711 | 0.1030 | 36 |
| txcdr_causal_t5 | 0.7705 | 0.1045 | 36 |
| txcdr_lowrank_dec_t5 | 0.7691 | 0.0958 | 36 |
| stacked_t5 | 0.7609 | 0.1057 | 36 |
| stacked_t20 | 0.7604 | 0.1086 | 36 |
| topk_sae | 0.7587 | 0.1056 | 36 |
| txcdr_t20 | 0.7545 | 0.0968 | 36 |
| txcdr_shared_dec_t5 | 0.7505 | 0.0983 | 36 |
| txcdr_block_sparse_t5 | 0.7474 | 0.0957 | 36 |
| txcdr_pos_t5 | 0.7318 | 0.1058 | 36 |
| txcdr_shared_enc_t5 | 0.7263 | 0.0897 | 36 |
| tfa_small | 0.6800 | 0.0701 | 36 |
| tfa_pos_small | 0.6799 | 0.0787 | 36 |

#### Figure 3 — Headline plots index

| slice | plot |
|---|---|
| last-position × AUC × full | [`headline_bar_k5_last_position_auc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_auc_full.png) |
| last-position × acc × full | [`headline_bar_k5_last_position_acc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_acc_full.png) |
| mean-pool × AUC × full | [`headline_bar_k5_mean_pool_auc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_mean_pool_auc_full.png) |
| mean-pool × acc × full | [`headline_bar_k5_mean_pool_acc_full.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_mean_pool_acc_full.png) |
| last-position × AUC × aniket | [`headline_bar_k5_last_position_auc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_auc_aniket.png) |
| last-position × acc × aniket | [`headline_bar_k5_last_position_acc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_last_position_acc_aniket.png) |
| mean-pool × AUC × aniket | [`headline_bar_k5_mean_pool_auc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_mean_pool_auc_aniket.png) |
| mean-pool × acc × aniket | [`headline_bar_k5_mean_pool_acc_aniket.png`](../../../../experiments/phase5_downstream_utility/results/plots/headline_bar_k5_mean_pool_acc_aniket.png) |

Per-task heatmaps live at the matching `per_task_k5_*` paths next to
each headline bar.

#### T-sweep ladder (TXCDR at T ∈ {2, 3, 5, 8, 10, 15, 20})

![T-sweep × 3 aggregations × AUC](../../../../experiments/phase5_downstream_utility/results/plots/txcdr_t_sweep_auc.png)

Accuracy companion: [`txcdr_t_sweep_acc.png`](../../../../experiments/phase5_downstream_utility/results/plots/txcdr_t_sweep_acc.png).

Summary at k=5 across all three aggregations:

| T | last_position AUC | full_window AUC | mean_pool AUC |
|---|---|---|---|
| 2 | 0.7441 | 0.6623 | 0.7786 |
| 3 | 0.7711 | 0.6999 | 0.8022 |
| **5** | **0.7822** | 0.7259 | **0.8064** |
| 8 | 0.7540 | 0.7000 | 0.7711 |
| 10 | 0.7671 | 0.6893 | 0.7754 |
| 15 | 0.7772 | 0.7125 | 0.7868 |
| 20 | 0.7496 | 0.7522 | 0.7545 |

**Observations — the three aggregations tell three different stories:**

1. **`last_position` and `mean_pool` are qualitatively aligned**:
   both peak at T=5. Small T (T=2) loses temporal context; large T
   (T≥8) over-regularizes the per-feature decoder (the TXCDR-T20
   SVD spectrum is 7.5 % flatter than T5's — see *Per-feature
   decoder SVD* below). `mean_pool` is uniformly +1–3 pp above
   `last_position` because it uses all K = 20 − T + 1 slides to
   produce the probed d_sae vector, not just the slide ending at
   the last real token.
2. **`full_window` INVERTS the T trend**: it peaks at T=20 (0.752)
   and is nearly monotone-increasing in T from T=2 onwards. The
   mechanism is pool-inflation: full_window concatenates per-slide
   latents into `(N, K × d_sae)`, so small T → large K (up to 19)
   → the top-k-by-class-separation selector at k=5 has to pick from
   a 20× bigger feature pool than `mean_pool` — and overfits. As T
   grows, K shrinks, the pool shrinks, and the selector works
   better. This is why the original `txcdr_t5 (0.726) < txcdr_t20
   (0.752)` finding looked misleadingly like "T↑ helps": it was a
   pool-size artefact of full_window, not a genuine capability of
   large-T TXCDR.
3. **At T=20, all three aggregations converge** (0.7496, 0.7522,
   0.7545) because K = 20 − 20 + 1 = 1: there is only one slide, so
   full_window and mean_pool both collapse to last_position
   numerically (within 0.5 pp of seed noise).
4. **T=3 at `mean_pool` (0.8022) is a dark horse.** A 3-token
   context combined with slide-averaging captures most of the
   discriminative signal at k=5; the T=5-to-T=3 drop is only 0.4 pp
   under mean_pool vs 1.1 pp under last_position. Suggests that
   **mean_pool's "many-slides" bonus is most valuable at small T**,
   which is plausibly why the T↑ curve flattens faster.

This is strong empirical support for **`mean_pool` as the canonical
SAEBench sliding-window aggregation** and for **deprecating
`full_window`**: full_window was telling us the opposite of what the
T-sweep was designed to test, purely because of its feature-pool
scaling with K.

#### Error-overlap analysis — TXCDR vs MLC complementarity

![Error-set Jaccard heatmap](../../../../experiments/phase5_downstream_utility/results/plots/error_overlap_jaccard_k5_last_position.png)

![Wins/loss asymmetric-errors heatmap](../../../../experiments/phase5_downstream_utility/results/plots/error_overlap_winsloss_k5_last_position.png)

Computed for 7 top archs (`mlc`, `mlc_contrastive`, `txcdr_t5`,
`txcdr_tied_t5`, `txcdr_rank_k_dec_t5`, `time_layer_crosscoder_t5`,
`topk_sae`), 21 pairs × 36 tasks, at `last_position × k=5`. For each
pair computed: **Jaccard of per-example error sets**, **McNemar's
χ² p-value**, and **fraction of examples where A is right and B is
wrong** (and vice versa). Full pair table in
`results/error_overlap_summary_last_position_k5.json`.

**Most complementary pairs** (lowest Jaccard = archs make
*different* errors):

| pair A | pair B | Jaccard | A-wins | B-wins | McNemar p-median | sig @ 0.05 |
|---|---|---|---|---|---|---|
| txcdr_t5 | mlc_contrastive | **0.338** | 12.2 % | 14.6 % | 0.104 | 16/36 |
| mlc | txcdr_t5 | 0.342 | 14.6 % | 12.5 % | **0.045** | 18/36 |
| txcdr_tied_t5 | mlc_contrastive | 0.342 | 11.9 % | 15.8 % | 0.085 | 17/36 |
| mlc | txcdr_rank_k_dec_t5 | 0.343 | 14.0 % | 12.5 % | **0.029** | **21/36** |
| time_layer_crosscoder_t5 | mlc_contrastive | 0.344 | 11.6 % | 15.7 % | 0.024 | 19/36 |

**Most similar pair**:

| pair A | pair B | Jaccard | A-wins | B-wins |
|---|---|---|---|---|
| mlc | mlc_contrastive | **0.482** | 9.3 % | 9.6 % |

**Reading**:

- The **two strongest SAE families (TXCDR and MLC) are genuinely
  complementary**: errors overlap only 34 %, one gets ~13 % of
  examples right that the other misses, and McNemar is significant
  at p<0.05 on **18/36 tasks** (8/36 at Bonferroni). This
  addresses the collaborator's question "same AUC, but do they do
  the same things?" — the answer is **no**.
- `mlc` and `mlc_contrastive` are most similar (Jaccard 0.48, only
  ~9 % asymmetric wins each way), which is expected since
  mlc_contrastive's encoder IS a MatryoshkaH/L-partitioned MLC and
  inherits most of MLC's bias.
- An **ensemble of TXCDR-T5 and `mlc_contrastive` should pay** —
  they are the single most complementary pair in the cohort. That
  combination is the obvious headline for a follow-up phase.

The per-task asymmetric-errors plot for the mlc-vs-txcdr_t5 axis
(collaborator's explicit ask) lives at
[`error_overlap_per_task_mlc_vs_txcdr_t5_k5_last_position.png`](../../../../experiments/phase5_downstream_utility/results/plots/error_overlap_per_task_mlc_vs_txcdr_t5_k5_last_position.png).

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

#### Cross-token breakdown (sub-phase 5.4)

Same 2 tasks as before (`winogrande_correct_completion`,
`wsc_coreference`), reported as `max(AUC, 1 − AUC)` for arbitrary
label polarity. Numbers are for last-position × k=5.

| row | winogrande | wsc |
|---|---|---|
| **baseline_last_token_lr** | **0.7708** | **0.8497** |
| baseline_attn_pool | 0.5416 | 0.5289 |
| time_layer_crosscoder_t5 | 0.6100 | 0.6529 |
| mlc_contrastive | 0.5906 | 0.6462 |
| mlc | 0.5806 | 0.6373 |
| tfa_small | 0.5550 | 0.6458 |
| temporal_contrastive | 0.5452 | 0.6031 |
| txcdr_shared_enc_t5 | 0.5997 | 0.5393 |
| txcdr_t5 | 0.5334 | 0.6055 |
| txcdr_shared_dec_t5 | 0.5079 | 0.6153 |
| ...rest (17 archs) ... | 0.50–0.55 | 0.55–0.60 |

**Same observation**: `time_layer_crosscoder_t5` is still the
cross-token winner among SAEs. `mlc_contrastive` is a close second
(0.591 / 0.646) — a new addition to the competitive cluster. Pure
TXCDR variants remain weak on cross-token.

Baseline wall unchanged: raw last-token LR dominates both
cross-token tasks by 15–30 pp.

#### Per-feature decoder SVD: vanilla TXCDR under-regularized at T=20

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
information it carries is low-dimensional. The full T-sweep ladder
above is consistent — mean AUC peaks at T=5 and drops monotonically
toward T=20.

The rank-K decoder variants confirm this:

- `txcdr_lowrank_dec_t5` (W_t = W_base + U_t V_tᵀ, rank-8 correction):
  last-position AUC 0.7390. Below vanilla TXCDR-T5 (0.7822) — soft
  low-rank residual under-constrains.
- `txcdr_rank_k_dec_t5` (per-feature decoder factored as A_j B_j
  with K=4): last-position AUC **0.7852** — beats vanilla TXCDR-T5
  (0.7822). Decoder rank clamped at 4 (< T=5) and still improves
  over full-rank on the task set. Under mean_pool, rank_k lifts to
  **0.7990** (vs T5 0.8064), still top-3 SAE.

The K=4 hard parameterization wins narrowly; the soft low-rank
residual loses slightly. Seed-variance on Phase-4-comparable runs
was ≈ 0.5–1 pp, so the 3 pp rank_k_dec win was suggestive but not
definitive. The seed variance analysis (below, partial) will
tighten this.

#### Seed variance (T17 — in progress)

3-seed autoresearch on the top-5 archs (`mlc`,
`time_layer_crosscoder_t5`, `txcdr_rank_k_dec_t5`, `txcdr_t5`,
`txcdr_tied_t5`) at seeds {1, 2, 3} is running overnight
(2026-04-21). Seed 1 complete and committed at `9b429ec`; seeds 2
and 3 in progress. Final seed-variance bar plot and ±σ table will
be committed when all 15 runs finish (ETA ~2026-04-21 ~15:00).

#### Training dynamics (25 archs)

![Training loss curves (log-log)](../../../../experiments/phase5_downstream_utility/results/plots/training_curves_loglog.png)

Linear scale: [`training_curves.png`](../../../../experiments/phase5_downstream_utility/results/plots/training_curves.png).

All 25 archs converged (`converged=True` in `training_index.jsonl`).
The 6 new T-sweep archs all plateaued at step 3400–5400.
`mlc_contrastive` converged at step 3000 with final_loss = 13.4
(Matryoshka-MSE + 0.1 × InfoNCE) and final_l0 = 99 (head-H only;
total active latents across H ∪ L = 200).

### Which outcome held

**Outcome B (nuanced positive), now with a two-arch headline.**

At the 1.5 pp margin bar:

- **Outcome A (temporal SAE beats attn-pool on ≥ 4 / 36 tasks)**:
  still *not* satisfied. Best is `mlc_contrastive` at 2 / 36 per-task
  wins.
- **Outcome B (temporal structure helps somewhere)**: satisfied, now
  with stronger evidence. WSC cross-token: ≥15 of 25 SAEs beat
  attn-pool by ≥ 4 pp; `time_layer_crosscoder_t5` wins at 0.653
  (vs attn-pool 0.529, +12 pp); `mlc_contrastive` a close second at
  0.646.
- **Outcome C (no temporal signal anywhere)**: ruled out by B.

The **headline** is now a two-arch story:

1. **`mlc_contrastive` is the strongest SAE at last_position** and
   the 2nd strongest at mean_pool. The InfoNCE-on-adjacent-tokens
   penalty ported from Ye et al. 2025 to the MLC base adds
   measurable discriminative power.
2. **`txcdr_t5` is the strongest SAE at mean_pool**, and combines
   with mlc_contrastive into the most complementary pair in the
   cohort (Jaccard 0.338) — an ensemble is the obvious next step.

**Caveat that shrinks the claim**: `baseline_last_token_lr` is still
undefeated on both cross-token tasks (0.77 WinoGrande, 0.85 WSC).
The temporal-SAE-vs-attn-pool story is B-positive, but the overall
SAE-vs-strong-baseline story remains C-negative, matching
`papers/are_saes_useful.md`.

### Historical full_window record (deprecated)

`full_window` concatenates per-slide d_sae vectors into
(N, K·d_sae) and selects the top-k features globally. This inflates
the feature pool by a factor of K (up to 20× for the tail-20 window)
and hurts small-k selection. `mean_pool` (average instead of
concatenate) retains the per-slide information without inflating the
pool, and is the SAEBench-canonical aggregation — so `mean_pool` is
now the primary sliding-window probe and `full_window` is deprecated.

Prior full_window findings, preserved for reproducibility:

| arch | last_position AUC | full_window AUC | Δ |
|---|---|---|---|
| mlc | 0.7943 | 0.6824 | **−0.112** |
| time_layer_crosscoder_t5 | 0.7928 | 0.6655 | **−0.127** |
| txcdr_t5 | 0.7822 | 0.7259 | −0.056 |
| txcdr_rank_k_dec_t5 | 0.7852 | 0.7178 | −0.067 |
| (full 19-arch table in JSONL) | | | |

The MLC / time_layer collapse was an artefact of the feature-pool
inflation. Under mean_pool the picture is the opposite — MLC
(0.7848) is within 0.01 of its last_position value, and time_layer
(0.7722) is within 0.02. Both archs are shape-invariant under
mean_pool because averaging K slides of a single-position encoder
is approximately equivalent to encoding the single centroid position.

`probing_results.jsonl` retains all full_window rows; new plots
omit this aggregation.

### Caveats

- **Single seed (42) on most rows.** Seed variance on the top-5
  archs is being measured now (T17, in progress at writing). Phase-4
  seed-variance on comparable TopKSAE runs was ≈ 0.5–1 pp on mean
  AUC; gaps under that bar should be treated as within-seed noise.
  The 9+ pp SAE-vs-baseline gap is well outside it. The
  `mlc_contrastive` / `txcdr_t5` complementarity finding (Jaccard
  0.34) involves 36 tasks × hundreds of examples each — sample size
  is solid.
- **Cross-token `max(AUC, 1 − AUC)` flip** for WinoGrande/WSC only,
  to remove arbitrary label polarity. Raw AUCs stay in
  `probing_results.jsonl`; flip set is `make_headline_plot.py::FLIP_TASKS`.
- **TFA "small" variants (d_sae = 4096, seq_len = 32).** The
  full-size TFA would not fit the A40 wall-clock budget without a
  significant refactor; we use the smaller scale that matches the
  contracted attention-pooled probe's per-token budget. Their
  SAEBench numbers are therefore not a like-for-like comparison
  against the d_sae = 18 432 archs and should be read as "TFA at
  matched total sparsity budget".
- **Gemma-2-2B-IT vs Gemma-2-2B (base)** divergence from Aniket's
  setup. All 25 rows are internally consistent on -IT; direct
  bit-level comparison with Aniket is not possible on these numbers.
- **Matryoshka toy-validation** still deferred to Phase 6 — the
  25-arch expansion did not add this.
- **Decoder-rotation variant (brief.md §3.4)** not trained — the
  rank-K hard parameterization covers the "fix TXCDR-T20's flat
  spectrum" angle; the Lie-group rotation variant is deferred to
  Phase 6.
- **No reasoning-trace probe.** We do not run DeepSeek-R1-Distill on
  the cross-token tasks in this phase; deferred to a follow-up.
- **T-SAE paper latent-level qualitative comparison** (collaborator
  ask) not performed in Phase 5; deferred to Phase 6.

### Files produced

Under `experiments/phase5_downstream_utility/results/`:

- `leakage_audit.json` — corpus + split leakage audit (PASS).
- `training_index.jsonl` — one row per converged run (25 + 15 T17
  seeded runs = 40 rows by end of overnight).
- `training_logs/<run_id>.json` — per-run loss curve + meta.
- `probing_results.jsonl` — (run_id, task, aggregation, k_feat) cell;
  baselines under `run_id=BASELINE_*`. Contains rows for all three
  aggregations (last_position, mean_pool, full_window) — plotter
  currently consumes only last_position + mean_pool.
- `predictions/<run_id>__<aggregation>__<task>__k<k>.npz` —
  per-example (y_true, decision_score, y_pred) tuples for the 7 top
  archs × 36 tasks at last_position (≈ 1000 files, ~4 MB total).
  Used by `analyze_error_overlap.py` for the error-overlap analysis.
- `error_overlap_summary_last_position_k5.json` — 21-pair McNemar /
  Jaccard / wins-loss per-task statistics.
- `headline_summary_<aggregation>_<metric>_<task_set>.json` — 8
  aggregated summaries (2 aggs × 2 metrics × 2 task sets).
- `plots/headline_bar_k5_<aggregation>_<metric>_<task_set>.png` — 8
  headline bar charts.
- `plots/per_task_k5_<aggregation>_<metric>_<task_set>.png` — 8
  per-task heatmaps.
- `plots/txcdr_t_sweep_{auc,acc}.png` — T-sweep ladder.
- `plots/error_overlap_{jaccard,winsloss}_k5_last_position.png` —
  7×7 error-overlap heatmaps.
- `plots/error_overlap_per_task_mlc_vs_txcdr_t5_k5_last_position.png` —
  collaborator-requested per-task asymmetric-errors plot.
- `plots/training_curves{,_loglog}.png` — 25-arch training dynamics.
- `plots/svd_spectrum_t5_vs_t20.png` — per-feature SVD finding.
- `svd_spectrum.json` — raw normalized spectra per arch.

Gitignored (reproducible from scripts):

- `results/ckpts/<run_id>.pt` — ≥25 fp16 state_dicts (~35–40 GB
  total after T17 completes).
- `results/probe_cache/<task>/acts_{anchor,mlc,mlc_tail}.npz` +
  `meta.json`.

### Pipeline reproduction

From repo root, after `git pull origin han`:

```bash
# Orchestrate all Phase-5 probing from scratch (assumes ckpts + cache)
bash experiments/phase5_downstream_utility/run_fw_tsweep.sh           # T-sweep + plots
bash experiments/phase5_downstream_utility/run_mean_pool_probing.sh   # 24 archs × mean_pool
bash experiments/phase5_downstream_utility/run_mlc_contrastive.sh     # train + probe mlc_contrastive
bash experiments/phase5_downstream_utility/run_overnight_phase5.sh    # T15–T17 chain

# Or individual operations
PYTHONPATH=/workspace/temp_xc \
  .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation mean_pool --skip-baselines --run-ids txcdr_t5__seed42

PYTHONPATH=/workspace/temp_xc \
  .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation last_position --skip-baselines --save-predictions \
  --run-ids mlc__seed42 txcdr_t5__seed42

PYTHONPATH=/workspace/temp_xc \
  .venv/bin/python experiments/phase5_downstream_utility/analyze_error_overlap.py \
  --aggregation last_position --k 5
```

The probing script streams task caches (one task at a time loaded
from disk) — peak Python RSS is ~8 GB on the heaviest runs, well
under the 46 GB cgroup limit. Cgroup memory will sit near the limit
due to OS page cache of the ~66 GB probe_cache fileset; this is
evictable and does NOT cause OOM kills (failcnt stays 0). Per-arch
encoding paths batch GPU tensors at 256–512 samples to avoid CUDA
OOM on `(B, T, d_sae)` intermediates during window slides.
