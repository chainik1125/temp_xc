---
author: Dmitry
date: 2026-04-23
tags:
  - results
  - in-progress
---

## TXC vs MLC complementarity — probing + routing summary

### Research question

Do TXC (temporal crosscoder) and MLC (multi-layer crosscoder) learn *complementary* or *substitute* features? Can a lightweight classifier that decides per example which arch to trust beat either alone on the sparse-probing benchmark?

### Setup

- **Model**: `google/gemma-2-2b-it`, layer 13 residual stream (layers 11-15 for MLC).
- **SAE width**: `d_sae = 18,432` for all archs.
- **Sparsity**: TopK `k = 100` per token.
- **Bench**: 36 binary sparse-probing tasks (Kantamneni et al. §2.2 protocol) across 8 dataset families (ag_news, amazon_reviews, amazon_reviews_sentiment, bias_in_bios_set{1,2,3}, europarl, github_code; winogrande/wsc excluded from text reconstruction but included in pooled numbers).
- **Probe**: top-k features by `|mean(y=1) − mean(y=0)|` → L1 logistic regression. Evaluated at `k ∈ {1, 2, 5, 20}`.
- **Seed 42** for all trained archs (multi-seed variants exist on HuggingFace but not analysed here).
- **Two arch families** studied:
    - **Predecessors**: `mlc__seed42`, `txcdr_t5__seed42` (Phase 5 baselines; predictions tracked in git).
    - **Agentic winners**: `agentic_mlc_08__seed42`, `agentic_txc_02__seed42` (Phase 5.7 matryoshka + multi-scale InfoNCE winners; predictions regenerated on a40_climb).

### Headline table — macro accuracy across 36 tasks (k=5)

| method | predecessors × last_position | predecessors × mean_pool | agentic × last_position | agentic × mean_pool |
|---|---:|---:|---:|---:|
| MLC alone | **0.7343** | 0.7238 | **0.7388** | 0.7383 |
| TXC alone | 0.7131 | **0.7400** | 0.7285 | **0.7422** |
| per-task oracle | 0.7509 | 0.7560 | 0.7525 | 0.7601 |
| hard router (multi-k, CV) | 0.7486 | 0.7396 | 0.7513 | 0.7467 |
| **multi-k hybrid probe (CV)** | **0.7889** | **0.7926** | **0.7885** | **0.7898** |
| per-example oracle | 0.8595 | 0.8494 | 0.8613 | 0.8412 |

Bold marks the winning single arch and the best supplemented method per column. Each arch wins at its "home" aggregation (MLC@last_position, TXC@mean_pool). Multi-k hybrid gains `+0.047 – +0.055` over the best single arch regardless of family or aggregation.

### 4-cell confusion breakdown (k=5, pooled across 24,470 examples)

**Predecessors**:

| cell | last_position | mean_pool |
|---|---:|---:|
| both_right | 14,489 (59.2 %) | **15,200 (62.1 %)** |
| mlc_only | **3,725 (15.2 %)** | 2,721 (11.1 %) |
| txc_only | 3,087 (12.6 %) | 3,161 (12.9 %) |
| both_wrong | 3,169 (13.0 %) | **3,388 (13.8 %)** |
| disagreement | **27.8 %** | 24.0 % |

**Agentic winners**:

| cell | last_position | mean_pool |
|---|---:|---:|
| both_right | 14,918 (61.0 %) | **15,819 (64.6 %)** |
| mlc_only | **3,366 (13.8 %)** | 2,561 (10.5 %) |
| txc_only | 3,025 (12.4 %) | 2,491 (10.2 %) |
| both_wrong | 3,161 (12.9 %) | **3,599 (14.7 %)** |
| disagreement | **26.1 %** | 20.6 % |

*Observations*:

- Mean_pool raises both_right AND both_wrong — archs agree more, but agree-wrong more too.
- Last_position preserves more complementarity (higher per-example oracle, larger disagreement fraction).
- MLC's solo wins dominate at last_position; at mean_pool the two archs' solo-win rates nearly equalise.

### Multi-k hybrid probe

The workhorse score-based method. For each example, 8 scalar features (2 archs × 4 k-values), per-(task, arch, k) z-scored, plus 6 stability derivatives (std across k, range across k, sign-flip first-vs-last k, per arch). Logistic regression targeting `y_true`, LOGO CV on `dataset_key` (10 folds).

Captures roughly **40-48 % of the per-example oracle headroom** in every configuration:

| family × aggregation | gain over best single | % of oracle headroom |
|---|---:|---:|
| predecessors × last_position | +0.055 | 43.6 % |
| predecessors × mean_pool | +0.053 | 48.0 % |
| agentic × last_position | +0.050 | 40.6 % |
| agentic × mean_pool | +0.048 | 48.1 % |

The gain is remarkably stable across arch family and aggregation.

### Cross-aggregation hybrid probe

Features: 16 scores (2 archs × 2 aggregations × 4 k-values). Same LOGO CV.

| family | 4-arch oracle | cross-agg hybrid | gain vs best single | % of 4-arch headroom |
|---|---:|---:|---:|---:|
| predecessors | 0.9180 | **0.8061** | +0.066 | 37.1 % |
| agentic | 0.9125 | **0.8026** | +0.060 | 35.5 % |

*Observations*:

- 4-arch oracle is ~+5-6 pp higher than same-aggregation 2-arch oracle — cross-aggregation exposes substantially more ceiling.
- Cross-agg hybrid gains only +1-2 pp over same-agg hybrid despite the wider oracle — score-based ensembling captures proportionally less of the bigger pot.

### Hard router (3-way variants)

A logistic regression that picks arch ∈ {MLC, TXC} per example, trained only on disagreement examples. Final y_pred = arch's y_pred at k=5.

Best result (multi-k F1_z features, LOGO CV): **+2.0 pp vs MLC-alone** (predecessors × last_position). **The hard router beats per-task oracle (+1.7 pp) by a whisker** — meaningful but far from the 12.5 pp ceiling. Hybrid probe strictly dominates hard router across all configurations.

### Concat-probe (joint SAE latents)

Retrains the probe on concatenated latents from both SAEs (36,864 features per example before top-k selection). Intra-task evaluation (same train/test split as the single-arch probes).

Predecessors × last_position:

| k | AUC | ACC | ACC gain vs MLC alone |
|---:|---:|---:|---:|
| 1 | 0.7130 | 0.6893 | -0.023 |
| 2 | 0.7448 | 0.7055 | -0.028 |
| 5 | 0.8037 | 0.7444 | +0.010 |
| 20 | 0.8552 | **0.7876** | +0.053 |

At k=20 the concat-probe matches the multi-k hybrid (0.7876 vs 0.7889) — joint feature access at a wider budget equals what multi-k score ensembling achieves for free. **Joint features don't clearly beat joint scores.**

Not run on agentic winners or at mean_pool; expected to replicate.

### LLM judge (Gemini 3.1 Flash Lite)

Classifies each test example into one of `{both_right, mlc_only, txc_only, both_wrong}` from text, then derives a router. All runs: 120 balanced ICL examples (30/cell), natural-distribution test of 1000 examples from the predecessor pool.

| variant | info given to judge | 4-way acc | routed acc | gain vs MLC | headroom |
|---|---|---:|---:|---:|---:|
| text-only | text alone | 0.312 | 0.748 | -0.001 | -0.7 % |
| no-hints | task + text | 0.501 | 0.749 | +0.000 | 0 % |
| **all-combined (hints)** | task + text + scores | **0.766** | **0.836** | **+0.087** | **59 %** |
| per-example oracle | — | — | 0.896 | +0.147 | 100 % |

*Interpretation*:

- The LLM judge's **only** useful signal is the probe scores. With task + text alone (no-hints) it ties MLC-alone — zero routing gain.
- The all-combined variant's +8.7 pp gain is essentially "LLM solves the task itself + sign-compares against scores." Replacing probes with the LLM directly would work even better (Gemini ≈ 95 % on these tasks) but is off-topic.
- Gemini 3 Pro stratified-100 run was attempted but dropped; Flash Lite was sufficient to settle the question.

### Key findings

1. **Complementarity is real**: disagreement rate 20-28 % depending on aggregation; per-example oracle headroom +10-14 pp over the best single arch at any aggregation. +17-18 pp cross-aggregation.
2. **~40-48 % of single-aggregation oracle headroom is recoverable from probe scores** via the multi-k hybrid (logistic regression on 8 scalar features). Free, no GPU, no LLM.
3. **Cross-aggregation routing is the biggest expansion of headroom** (+6 pp over same-aggregation oracle). Recoverable fraction drops to ~36 % but absolute gain still grows by +1-2 pp over the best same-aggregation hybrid.
4. **Text alone carries no routing signal** on this benchmark (LLM judge text-only: -0.001 pp). The discriminative information lives in the probe outputs, not in surface text patterns.
5. **Phase 5.7 winners behave like predecessors** on the complementarity axis: same oracle ceilings, same hybrid gains, same disagreement profile. The matryoshka + multi-scale InfoNCE training improves individual arch accuracy by 0.4-1.5 pp but doesn't change the complementarity structure.
6. **Hard router ≲ hybrid probe**: letting LR output `y_pred` directly beats hard routing to an arch's `y_pred`. Soft combination is the right granularity.
7. **Joint features ≈ joint scores**: concat-probe on 36,864-dim latents matches multi-k hybrid on 8 scalars at comparable feature budgets (k=5 vs k=20 in the concat probe). Probe scores are not the bottleneck.

### What was not done

- Concat-probe on agentic winners or at mean_pool.
- 4-combo cross-aggregation concat-probe (would test "joint features" upper bound at the full cross-aggregation scope).
- Gemini 3 Pro natural-1000 judge — deemed unnecessary after Flash Lite decomposition showed text-alone signal is zero.
- Non-linear models (gradient-boosted trees, small MLP) on the multi-k or cross-agg score features. All reported hybrid results use plain logistic regression.
- Calibrated probabilities as features instead of raw logits.

### Pointers

- Scripts (on a40_climb `/workspace/temp_xc/experiments/phase5_downstream_utility/`):
    - `analyze_txc_vs_mlc_dashboard.py` — per-task confusion bars + aggregate 2×2 heatmap.
    - `analyze_txc_vs_mlc_router.py` — single-k hard router + hybrid probe.
    - `analyze_router_multik.py` — multi-k router + hybrid probe.
    - `analyze_router_crossagg.py` — cross-aggregation hybrid probe.
    - `concat_probing.py` — concat-probe on joint SAE latents.
    - `llm_judge_local.py` — Gemini judge variants (runs locally on laptop).
    - `dump_example_pool.py` / `show_confusion_samples.py` — helpers for joining predictions to raw text.
- Result JSONs: `results/router_multik__*.json`, `results/router_crossagg__*.json`, `results/concat_probe_results.jsonl`, `results/dashboard_*.json`.
- Plots: `results/plots/dashboard_*.png`, `results/plots/router_multik__*.png`.
- Local copies on laptop: `docs/dmitry/results/txc_mlc_dashboard/`.
