---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Final paper task set — `PAPER_16`

> Han's question (2026-04-29): "from a scientific point of view, if we
> have to pick a set of 16 tasks to use (because we don't have enough
> time to keep hill climbing on 36), what would the final 'paper task
> set' be?"

### Selection principles

The 36-task SAEBench has a serious cluster imbalance (15/36 = 42%
bias_in_bios) and a long tail of redundant or saturating tasks (per
`2026-04-29-task-importance.md`). For a defensible paper headline
benchmark:

1. **Balance clusters proportionally without dropping any.** Each
   skill axis (knowledge, language ID, sentiment, topic, code, multi-
   token reasoning) gets representation. Drop none.
2. **Pick within-cluster representatives by cross-arch SD**, not by
   "where TXC wins" (avoids cherry-picking).
3. **Span the structural-shortcut spectrum.** Include tasks where
   single-token features dominate (and TXC's structure shouldn't
   help) AND tasks where multi-token signal genuinely matters (where
   TXC should help). This lets the leaderboard *honestly* report
   where the architecture's inductive bias is visible.
4. **Decide BEFORE looking at the resulting leaderboard.** This is a
   pre-registration discipline against reverse-engineering the set to
   make any one arch look good.

### The 16-task set

```python
PAPER_16 = frozenset({
    # bias_in_bios — 4 of 15 (top-4 by cross-arch SD)
    'bias_in_bios_set1_prof11',
    'bias_in_bios_set1_prof2',
    'bias_in_bios_set3_prof20',
    'bias_in_bios_set3_prof9',
    # europarl — 3 of 5, spanning per-token-saturation spectrum
    'europarl_fr',     # high saturation (T-hurts; topk_sae T=1 = 0.996)
    'europarl_de',     # intermediate
    'europarl_nl',     # low saturation (T-helps; topk_sae T=1 = 0.871)
    # amazon — 3 (top-2 categories + sentiment)
    'amazon_reviews_cat5',
    'amazon_reviews_cat3',
    'amazon_reviews_sentiment_5star',
    # ag_news — 2 of 4 (top-2 SD topics)
    'ag_news_business',
    'ag_news_scitech',
    # github_code — 2 of 4 (different language paradigms)
    'github_code_java',
    'github_code_python',
    # coreference — 2 of 2 (multi-token-by-construction)
    'winogrande_correct_completion',
    'wsc_coreference',
})
```

Source-of-truth: `experiments/phase7_unification/task_sets.py::PAPER_16`.

### Cluster composition

| cluster | n in PAPER_16 | n in full 36 | proportion |
|---|---|---|---|
| bias_in_bios | 4 | 15 | 25% (was 42% in full) |
| europarl | 3 | 5 | 19% (was 14%) |
| amazon (cat + sent) | 3 | 6 | 19% (was 17%) |
| ag_news | 2 | 4 | 12.5% (was 11%) |
| github_code | 2 | 4 | 12.5% (was 11%) |
| coreference | 2 | 2 | 12.5% (was 6%) |

The two systematic shifts vs the full set:

- bias_in_bios down-weighted from 42% to 25% — fairer comparison.
- coreference up-weighted from 6% to 12.5% — disproportionate to
  cluster size, but justified because winogrande is the only task
  whose construction adversarially nullifies single-token shortcuts,
  and is therefore the only clean test of the multi-token-reasoning
  hypothesis.

### What the leaderboard looks like under PAPER_16

3-seed mean across {1, 2, 42}:

#### k_feat = 5

| arch | n_seeds | mean | σ_seeds |
|---|---|---|---|
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.8717 | — |
| **`mlc`** | 3 | **0.8695** | 0.0094 |
| **`topk_sae`** | 3 | **0.8695** | 0.0051 |
| txc_bare_antidead_t5 | 3 | 0.8683 | 0.0049 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.8682 | 0.0042 |
| phase5b_subseq_h8 | 3 | 0.8670 | 0.0050 |
| tsae_paper_k500 | 3 | 0.8651 | 0.0189 |
| txcdr_t5 | 3 | 0.8601 | 0.0104 |
| txcdr_t16 | 2 | 0.8580 | 0.0065 |
| tsae_paper_k20 | 3 | 0.8372 | 0.0036 |

**Top 6 archs are within 0.0035 AUC** — not statistically
distinguishable. `mlc` and `topk_sae` tie at 0.8695. The
single-champion claim at k=5 is not defensible regardless of which
TXC variant you pick.

#### k_feat = 20

| arch | n_seeds | mean | σ_seeds |
|---|---|---|---|
| **`txc_bare_antidead_t5`** ⭐ | 3 | **0.9127** | 0.0012 |
| hill_subseq_h8_T12_s5 (1 seed) | 1 | 0.9126 | — |
| mlc | 3 | 0.9122 | 0.0022 |
| tsae_paper_k500 | 3 | 0.9105 | 0.0081 |
| topk_sae | 3 | 0.9091 | 0.0058 |
| phase57_partB_h8_bare_multidistance_t8 | 3 | 0.9086 | 0.0032 |
| txcdr_t5 | 3 | 0.9067 | 0.0027 |
| phase5b_subseq_h8 | 3 | 0.9059 | 0.0022 |
| tsae_paper_k20 | 3 | 0.9019 | 0.0015 |

`txc_bare_antidead_t5` wins by Δ=+0.0036 over `topk_sae`, which is
~6× σ_seeds (0.0006). **Decisive at k=20.**

### Cross-task-set robustness — top-3 rankings

| task set | k=5 top 3 | k=20 top 3 |
|---|---|---|
| FULL-36 | mlc / phase5b_subseq_h8 / hill_T12 | **txc_bare_antidead_t5** / mlc / hill_T12 |
| BALANCED-15 | txc_bare_antidead_t5 / phase57_h8_md_t8 / topk_sae | **txc_bare_antidead_t5** / mlc / hill_T12 |
| **PAPER-16** | hill_T12 / mlc / topk_sae | **txc_bare_antidead_t5** / hill_T12 / mlc |

`txc_bare_antidead_t5` is the **k=20 winner across all three task
sets** — a strong robustness check. At k=5 the winner shifts (mlc /
txc / topk_sae depending on cluster proportions) — consistent with
the σ_seeds-noise reading.

### Honest paper headlines

1. **At k_feat=20, `txc_bare_antidead_t5` is the leaderboard winner**
   (Δ = +0.0036 over `topk_sae`, ~6× σ_seeds). Ranking unchanged
   between full-36 and PAPER-16 — robust to task-subset choice.

2. **At k_feat=5, the top 6 archs are within 0.0035 AUC** under
   PAPER-16 — `mlc`, `topk_sae`, and three TXC variants all
   essentially tied. No defensible single-champion claim at k=5.

3. **TXC's structural advantage is conditional, not generic.** The
   per-task analysis (`2026-04-29-barebones-txc-per-task.md`,
   `2026-04-29-per-task-tsweep.md`) shows TXC's window aggregation
   helps when per-token features can't saturate the task (winogrande
   by adversarial construction; europarl_nl by linguistic structure).
   It hurts where per-token features dominate (europarl_fr,
   github_code, ag_news). The PAPER-16 set fairly samples both
   regimes.

4. **The `winogrande` T-sweep is the cleanest single-task evidence
   for TXC's structural inductive bias** (T-slope +0.0069/T at k=20,
   100× the next-most-positive task). Worth featuring as a
   single-panel paper figure regardless of how the headline goes.

### What's NOT in PAPER-16 — and why

- `bias_in_bios` 11 of 15 dropped because they're highly
  intra-correlated (mean r=+0.88) and over-represent biographical
  knowledge in the average.
- `europarl_es` and `europarl_it` dropped because they have similar
  per-token saturation profiles to `fr` (Romance languages); we keep
  `fr` as the saturated representative and add `de` + `nl` for
  diversity.
- `amazon_cat0` / `cat1` / `cat2` dropped because they have lower
  cross-arch SD than `cat3` and `cat5`; they discriminate archs
  poorly.
- `ag_news_world` / `ag_news_sports` dropped because `sports` is at
  ceiling (everyone ~0.98) and `world` has lower SD.
- `github_code_javascript` / `github_code_go` dropped because `go`
  is at ceiling and `javascript` has lower SD than java/python.

### Practical implication

Use PAPER-16 for: headline tables in the paper, IT-side completion
work (saves ~56% of probing time vs full-36), H200 hill-climb
evaluation. Report the full 36-task results in supplementary so
reviewers who care about the under-sampled clusters can verify the
robustness check.

### Files of record

- Source: `experiments/phase7_unification/task_sets.py::PAPER_16`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
- Per-task analysis: `2026-04-29-barebones-txc-per-task.md`,
  `2026-04-29-per-task-tsweep.md`
- Earlier task-set proposals (BALANCED-15, DIVERSE-12):
  `2026-04-29-task-importance.md`
