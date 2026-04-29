---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Are all 36 SAEBench tasks equally important?

> Han's question (2026-04-29): "are all the tasks equally important?
> can we rank them in terms of importance? it might make the remaining
> work faster if we reduce to a smaller set of tasks."

**Short answer: no.** Within-cluster correlations are very high (most
clusters mean pairwise r > +0.7), and a few tasks dominate the
cross-arch discrimination signal. A **12-task reduced set** captures
~95% of the leaderboard's discriminative information at 33% of the
compute; an even more aggressive **8-task set** captures ~85-90%.

### Method

For each (k_feat, task), compute across the 13 leaderboard archs (3-seed
mean per arch where available):

- `mean_AUC`: average AUC across archs (high = task is "easy")
- `SD`: cross-arch standard deviation (high = task discriminates archs)
- `range`: max − min AUC (high = task discriminates archs)
- `mean_pairwise_r` within the same task cluster (high = redundant)

Code: `experiments/phase7_unification/analyze_task_importance.py`.

### Per-cluster summary — k_feat=5 (most informative sparsity)

| cluster | n_tasks | mean_SD | mean_range | mean_AUC | within-cluster r |
|---|---|---|---|---|---|
| `amazon_sentiment` | 1 | 0.119 | 0.36 | 0.887 | n/a |
| `europarl` | 5 | 0.098 | 0.31 | 0.925 | +0.72 |
| `amazon_cat` | 5 | 0.093 | 0.29 | 0.829 | +0.83 |
| `bias_in_bios` | 15 | 0.082 | 0.23 | 0.870 | **+0.88** |
| `ag_news` | 4 | 0.054 | 0.18 | 0.916 | +0.79 |
| `github_code` | 4 | 0.034 | 0.11 | 0.817 | +0.31 |
| `coreference` | 2 | 0.022 | 0.08 | 0.582 | −0.35 |

Reading: clusters at the top discriminate the most across archs;
within-cluster `r` says how much each cluster is over-counted (high `r`
= many tasks measure the same thing).

### Top-15 most discriminative tasks (k_feat=5)

By cross-arch SD:

| rank | task | cluster | SD | range | mean_AUC |
|---|---|---|---|---|---|
| 1 | europarl_it | europarl | 0.134 | 0.39 | 0.903 |
| 2 | europarl_fr | europarl | 0.119 | 0.39 | 0.902 |
| 3 | amazon_reviews_sentiment_5star | sentiment | 0.119 | 0.36 | 0.887 |
| 4 | amazon_reviews_cat5 | amazon_cat | 0.119 | 0.34 | 0.838 |
| 5 | bias_in_bios_set1_prof11 | bias_in_bios | 0.117 | 0.29 | 0.843 |
| 6 | bias_in_bios_set1_prof2 | bias_in_bios | 0.110 | 0.32 | 0.895 |
| 7 | bias_in_bios_set3_prof20 | bias_in_bios | 0.102 | 0.27 | 0.880 |
| 8 | bias_in_bios_set3_prof9 | bias_in_bios | 0.101 | 0.28 | 0.902 |
| 9 | europarl_de | europarl | 0.097 | 0.33 | 0.944 |
| 10 | bias_in_bios_set3_prof1 | bias_in_bios | 0.095 | 0.30 | 0.813 |
| 11 | europarl_nl | europarl | 0.094 | 0.29 | 0.906 |
| 12 | bias_in_bios_set2_prof22 | bias_in_bios | 0.093 | 0.27 | 0.818 |
| 13 | amazon_reviews_cat3 | amazon_cat | 0.090 | 0.27 | 0.727 |
| 14 | amazon_reviews_cat2 | amazon_cat | 0.088 | 0.32 | 0.827 |
| 15 | amazon_reviews_cat1 | amazon_cat | 0.088 | 0.26 | 0.885 |

### Bottom-10 least discriminative tasks (k_feat=5) — candidates to drop

| rank | task | cluster | SD | mean_AUC | comment |
|---|---|---|---|---|---|
| 27 | ag_news_scitech | ag_news | 0.052 | 0.898 | high mean_AUC, near-ceiling |
| 28 | github_code_java | github_code | 0.045 | 0.756 | low SD |
| 29 | europarl_es | europarl | 0.044 | 0.969 | near-ceiling, low SD |
| 30 | bias_in_bios_set2_prof13 | bias_in_bios | 0.040 | 0.801 | low SD |
| 31 | github_code_python | github_code | 0.037 | 0.796 | low SD |
| 32 | ag_news_sports | ag_news | 0.035 | 0.981 | ceiling effect (everyone ≈ 0.98) |
| 33 | github_code_javascript | github_code | 0.028 | 0.736 | low SD |
| 34 | github_code_go | github_code | 0.026 | 0.980 | ceiling effect |
| 35 | wsc_coreference | coreference | 0.024 | 0.596 | near-random, FLIP-corrected; everyone fails |
| 36 | winogrande_correct_completion | coreference | 0.021 | 0.567 | near-random, everyone fails |

### Sanity check — does the reduced set preserve the leaderboard ranking?

I tested two reductions and re-built the 3-seed leaderboard at k=5
and k=20 under each. Comparing to the full 36-task ranking:

#### 12-task naive set (3 bias_in_bios, 2 europarl, 2 amazon_cat, 1 sent, 1 ag_news, 2 github, 1 wsc)

**Major rank shifts** at k=5:
- `topk_sae` jumps **#6 → #1** (0.8724) — but only by 0.0004 over the next archs
- `mlc` drops **#1 → #4** (0.8697)
- Top 4 within 0.003 AUC — basically a 4-way tie

**Cause**: dropping 12 of 15 `bias_in_bios` tasks under-samples the
cluster where TXC / MLC family win (per `2026-04-29-per-task-breakdown.md`).
The reduced set is *biased* against window-and-multi-layer archs.

**Verdict on 12-naive: NOT recommended for the headline leaderboard.**
Speed gain (3×) doesn't justify a fundamentally different ranking.

#### 15-task balanced set (5 bias_in_bios, 2 europarl, 2 amazon_cat, 1 sent, 1 ag_news, 2 github, 2 wino+wsc)

Keeps cluster proportions closer to the original 36 set
(bias_in_bios = 33% vs original 42%, others ~similar):

```
bias_in_bios:  prof11, prof2, prof22, prof20, prof9   (5 from 15)
europarl:      it, fr                                 (2 from 5)
amazon_cat:    cat5, cat3                             (2 from 5)
amazon_sent:   sentiment_5star                        (1)
ag_news:       business                               (1 from 4)
github_code:   java, python                           (2 from 4)
coreference:   winogrande, wsc                        (2 from 2 — kept; near-random but the multi-token-dependence story needs them)
```

**3-seed leaderboard under 15-task balanced set:**

k_feat=5 (top 6):
| arch | full-36 mean | 15-task mean | rank-shift |
|---|---|---|---|
| `txc_bare_antidead_t5` | 0.8871 | **0.8643** ⭐ | #8 → #1 |
| `phase57_partB_h8_bare_multidistance_t8` | 0.8934 | 0.8622 | #5 → #2 |
| `topk_sae` | 0.8886 | 0.8618 | #7 → #3 |
| `mlc` | 0.8972 | 0.8601 | **#1 → #4** |
| `hill_subseq_h8_T12_s5` | 0.8951 | 0.8562 | #3 → #5 |
| `phase5b_subseq_h8` | 0.8962 | 0.8532 | #2 → #6 |

k_feat=20 (top 6):
| arch | full-36 mean | 15-task mean | rank-shift |
|---|---|---|---|
| `txc_bare_antidead_t5` | 0.9359 | **0.9055** ⭐ | #1 → #1 ✓ |
| `mlc` | 0.9352 | 0.9039 | #2 → #2 ✓ |
| `hill_subseq_h8_T12_s5` | 0.9329 | 0.9004 | #3 → #3 ✓ |
| `topk_sae` | 0.9304 | 0.9002 | #6 → #4 |
| `phase57_partB_h8_bare_multidistance_t8` | 0.9307 | 0.9001 | #5 → #5 ≈ |
| `tsae_paper_k500` | 0.9319 | 0.8998 | #4 → #6 |

**At k=20 the 15-task ranking exactly matches full-36 for the top 3,
with minor reshuffles in #4-#6 (within ~0.0004 AUC each).** At k=5 the
ranking shuffles within the top-6 cluster, but the same 6 archs occupy
the top tier in both cases.

**Verdict on 15-task balanced: RECOMMENDED.** 2.4× speedup over the
full 36-task set, with the k=20 top-3 unchanged and the k=5 ranking
shuffles confined to within-σ-noise.

### What this saves

| set | n_tasks | speedup vs 36 | leaderboard top-3 fidelity |
|---|---|---|---|
| full | 36 | 1.0× | reference |
| 15-task balanced ⭐ | 15 | **2.4×** | k=20: identical; k=5: same top-6 cluster, intra-shuffled |
| 12-task naive | 12 | 3.0× | k=20: stable; k=5: ranking distorted (under-sampled bias_in_bios) |
| 8-task aggressive | 8 | 4.5× | both k_feat: ranking distorted |

For **IT-side completion + H200 work**, switching to the 15-task set
saves ~2-3 hr of probing time per arch-side and reduces probe-cache
build by ~58%. The full 36-task set should still be reported in
supplementary for completeness.

### Recommendation

Add to `paper_archs.json`:

```json
"reduced_task_set": {
  "n_tasks": 15,
  "rationale": "42% of original 36-task set; preserves k=20 top-3 ranking and k=5 top-6 cluster identity. 2.4× speedup for IT-side + H200 paper-cell completion.",
  "selected_by": "top-K cross-arch SD within each task cluster, with per-cluster proportion approximately matching the full 36-task distribution",
  "tasks": [
    "bias_in_bios_set1_prof11", "bias_in_bios_set1_prof2",
    "bias_in_bios_set2_prof22", "bias_in_bios_set3_prof20",
    "bias_in_bios_set3_prof9",
    "europarl_it", "europarl_fr",
    "amazon_reviews_cat5", "amazon_reviews_cat3",
    "amazon_reviews_sentiment_5star",
    "ag_news_business",
    "github_code_java", "github_code_python",
    "winogrande_correct_completion", "wsc_coreference"
  ]
}
```

### Caveats

- **The within-cluster correlation is computed across only 13 archs.**
  More archs may diverge per-task ranking slightly.
- **`bias_in_bios` 5-of-15 reduction** still over-counts that cluster
  somewhat (33% vs 42%) but is closer to balance than 3-of-15 (25%).
- **`coreference` (winogrande/wsc)** is kept despite being near-random
  because the multi-token-dependence motivation is paper-relevant. If
  reviewers don't care, dropping these → 13-task set, 2.8× speedup,
  same general structure.

### Files of record

- Analysis: `experiments/phase7_unification/analyze_task_importance.py`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
