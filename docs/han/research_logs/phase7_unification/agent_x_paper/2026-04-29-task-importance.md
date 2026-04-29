---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Task-importance methodology — why we reduced FULL (36) → PAPER (16)

> Han's question (2026-04-29): "are all the tasks equally important?
> can we rank them in terms of importance? it might make the remaining
> work faster if we reduce to a smaller set of tasks."

**Short answer: no, tasks are NOT equally informative.** Within-cluster
correlations are high (most clusters mean pairwise r > +0.7), and a
few tasks dominate the cross-arch discrimination signal. After
analysis the team locked in the PAPER set (16 tasks; details +
leaderboard under it in `2026-04-29-paper-task-set.md`). This file
documents the *methodology* — per-task SD, within-cluster correlation,
and the principles that selected PAPER.

### Method

For each (k_feat, task), compute across the 13 leaderboard archs
(3-seed mean per arch where available):

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

### Selection principles for PAPER

The 36-task FULL set has a serious cluster imbalance (15/36 = 42%
bias_in_bios) and a long tail of redundant or saturating tasks. For a
defensible paper headline benchmark we adopted four principles:

1. **Balance clusters proportionally without dropping any.** Each
   skill axis (knowledge, language ID, sentiment, topic, code,
   multi-token reasoning) gets representation. Drop none.
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

### What PAPER contains

```
bias_in_bios:  prof11, prof2, prof20, prof9          (4 of 15 — top-4 by SD)
europarl:      fr, de, nl                            (3 of 5 — saturation spectrum)
amazon:        cat5, cat3, sentiment_5star           (3 of 6 — top-2 cat by SD + sent)
ag_news:       business, scitech                     (2 of 4 — top-2 by SD)
github_code:   java, python                          (2 of 4 — top-2 by SD)
coreference:   winogrande, wsc                       (2 of 2 — multi-token by construction)
```

Cluster proportions: 25% bias_in_bios (down from 42% in FULL),
12.5% coreference (up from 6%; justified — winogrande is the only
task whose construction adversarially nullifies single-token
shortcuts and is therefore the only clean test of the multi-token-
reasoning hypothesis). Source-of-truth:
`experiments/phase7_unification/task_sets.py::PAPER`.

For full discussion of the leaderboard under PAPER and the
cross-task-set robustness checks, see
`2026-04-29-paper-task-set.md`.

### Caveats

- The within-cluster correlation is computed across 13 archs. With
  more archs the per-task ranking might diverge slightly. Sanity
  check: even at the conservative end (γ ≈ 0.7), 5 redundant tasks
  add at most √5/√(1+(5−1)·0.7) ≈ 1.4× the signal of 1 task — so
  1-2 representatives per cluster is sound.
- `bias_in_bios` covers different professions (lawyer / nurse /
  scientist / etc), some of which probably show genuinely different
  arch rankings. With +0.88 mean correlation the redundancy is high
  but not perfect; 4 of 15 representatives is a defensible
  compromise.
- `coreference` (winogrande / wsc) is near-random for ALL archs in
  absolute AUC. The original motivation (test multi-token-dependence)
  is sound. The per-task T-sweep
  (`2026-04-29-per-task-tsweep.md`) shows winogrande is the *only*
  task with a positive T-slope at k=20 (climbing from 0.62 at T=3 to
  0.90 at T=32) — this is the cleanest empirical validation of TXC's
  structural advantage we have. Keeping both winogrande and wsc in
  PAPER is justified despite their low absolute AUC.

### Files of record

- Analysis script: `experiments/phase7_unification/analyze_task_importance.py`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
- Final set + leaderboard: `2026-04-29-paper-task-set.md`
- Source-of-truth constant: `experiments/phase7_unification/task_sets.py::PAPER`
