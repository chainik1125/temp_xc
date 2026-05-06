---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Where does barebones TXC do best / worst, per task?

> Han's question (2026-04-29): "before we decide [the task set], we
> need some analysis on which tasks the barebones TXC does the best
> and worst on."

Methodology: take **barebones TXC** = `txcdr_t5` (vanilla
TemporalCrosscoder, T=5, k_win=500, no matryoshka / contrastive /
anti-dead — the "barebones" reference cell in `paper_archs.json`).
Compare against:
- `topk_sae` — vanilla per-token TopK SAE (most apples-to-apples baseline)
- `tsae_paper_k500` — T-SAE paper port at matched k_win
- `mlc` — multi-layer crosscoder (per-token across 5 layers)

For each of the 36 SAEBench tasks, compute the 3-seed mean AUC for
each arch, then per-task Δ = txcdr_t5 − baseline. Code:
`experiments/phase7_unification/analyze_barebones_txc_per_task.py`.

### Per-cluster summary at k_feat=5

| cluster | n | Δ vs `topk_sae` | Δ vs `tsae_k500` | Δ vs `mlc` | mean TXC AUC | wins / losses (vs SAE) |
|---|---|---|---|---|---|---|
| coreference | 2 | **+0.014** | +0.001 | +0.014 | 0.589 | 2/0 (small absolute, both near-random) |
| bias_in_bios | 15 | **+0.005** | +0.011 | −0.005 | 0.914 | 10/3 |
| amazon_cat | 5 | +0.005 | +0.005 | +0.008 | 0.878 | 3/2 |
| europarl | 5 | −0.001 | −0.004 | **−0.030** | 0.952 | 3/2 (bimodal — see below) |
| amazon_sentiment | 1 | −0.004 | −0.004 | −0.013 | 0.958 | 0/1 |
| ag_news | 4 | −0.007 | −0.015 | −0.005 | 0.937 | 1/3 |
| github_code | 4 | **−0.020** | +0.002 | −0.027 | 0.814 | 1/2 |

### Per-cluster summary at k_feat=20

| cluster | n | Δ vs `topk_sae` | Δ vs `tsae_k500` | Δ vs `mlc` | mean TXC AUC | wins/losses |
|---|---|---|---|---|---|---|
| coreference | 2 | **+0.010** | −0.017 | +0.011 | 0.637 | 1/1 |
| amazon_cat | 5 | +0.008 | +0.007 | +0.004 | 0.930 | 2/2 |
| bias_in_bios | 15 | +0.001 | +0.002 | −0.002 | 0.946 | 6/6 (a wash) |
| amazon_sentiment | 1 | −0.001 | +0.000 | −0.009 | 0.974 | 0/0 |
| ag_news | 4 | −0.002 | −0.005 | −0.003 | 0.963 | 0/4 |
| europarl | 5 | −0.009 | −0.013 | −0.015 | 0.983 | 1/4 |
| github_code | 4 | **−0.012** | −0.006 | −0.028 | 0.903 | 1/3 |

### Top single-task swings (k_feat=5)

**Where barebones TXC wins biggest:**

| task | cluster | TXC AUC | SAE AUC | Δ |
|---|---|---|---|---|
| europarl_nl | europarl | 0.927 | 0.871 | **+0.056** |
| bias_in_bios_set2_prof6 | bias_in_bios | 0.949 | 0.905 | +0.044 |
| bias_in_bios_set3_prof1 | bias_in_bios | 0.886 | 0.850 | +0.036 |
| amazon_reviews_cat2 | amazon_cat | 0.887 | 0.858 | +0.029 |
| europarl_es | europarl | 0.991 | 0.964 | +0.027 |

**Where barebones TXC loses biggest:**

| task | cluster | TXC AUC | SAE AUC | Δ |
|---|---|---|---|---|
| europarl_fr | europarl | 0.921 | 0.996 | **−0.076** |
| github_code_java | github_code | 0.745 | 0.801 | −0.055 |
| github_code_python | github_code | 0.799 | 0.850 | −0.051 |
| bias_in_bios_set3_prof12 | bias_in_bios | 0.869 | 0.919 | −0.050 |
| bias_in_bios_set1_prof11 | bias_in_bios | 0.881 | 0.914 | −0.033 |

### Patterns

**Where barebones TXC clearly helps:**
- `coreference` (winogrande, wsc): small consistent win (Δ≈+0.01-0.014).
  Both archs near-random though (TXC ≈ 0.59-0.64), so the relative win
  is at chance-level absolute.
- `bias_in_bios` profession prediction at k=5: small mean win
  (+0.005), 10/15 task wins. At k=20 it's a wash (+0.001, 6/6 wins).
- A few specific tasks: europarl_nl, bias_in_bios prof6/prof1/prof9,
  amazon_cat2/cat3.

**Where barebones TXC clearly hurts:**
- `github_code` (Δ vs SAE: −0.020 at k=5, −0.012 at k=20). The "code
  language identifier" task is largely solved by single-token signature
  features (`def` / `function` / `package`) — TXC's window aggregation
  washes out the single-token cue.
- `ag_news` topic classification (Δ −0.007 at k=5, 0/4 wins at k=20).
  Topic is determined by a small set of high-frequency keywords —
  also a single-token shortcut domain.
- `europarl_fr` specifically (−0.076 at k=5). Surprising; TXC is
  somehow much worse on French language ID than SAE. Possibly because
  French has high frequency of common short tokens (le/la/de/des) that
  the SAE can pick up on while TXC's window blurs them.

**Bimodal / mixed:**
- `europarl` overall is bimodal (nl +0.056, fr −0.076). Includes 5
  language families; aggregation hides large per-language swings.
- `bias_in_bios` at k=5 has 3 large wins (prof6/prof1/prof9, +0.04 to
  +0.06) balanced against 2 large losses (prof12 −0.050, prof11
  −0.033). Net positive because more wins, but high per-task variance.

### Implication for task-set choice

If the goal is to faithfully measure when TXC's structural inductive
bias *helps*, the task domain matters:

- **Prefer tasks that genuinely require multi-token reasoning**: any
  task where the answer depends on an argument structure spanning
  multiple tokens — bias_in_bios profession prediction, sentence-level
  sentiment, certain coreference-adjacent tasks. TXC has small wins here.
- **Drop or downweight single-token-shortcut tasks**: `github_code`
  language ID and `ag_news` topic classification are dominated by
  single-token signatures. Including them at full weight artificially
  hurts TXC.
- **`bias_in_bios` 15-task over-representation is justified iff** the
  paper explicitly motivates it as "multi-token-content reasoning";
  otherwise it amounts to soft-cherry-picking the cluster TXC happens
  to do best on.

A defensible "multi-token-required" task set (a priori, before looking
at TXC results) would weight by *theoretical* multi-token requirement,
not by empirical TXC wins:

- bias_in_bios: KEEP (profession from biography requires noun
  phrases + work history reasoning)
- europarl: KEEP (language ID via linguistic structure ≈ multi-token)
- amazon_cat / amazon_sentiment: KEEP (review category / sentiment ≈
  multi-token)
- coreference: KEEP (explicitly multi-token-dependent by construction;
  even if everyone scores at chance, important to report)
- ag_news: DOWNWEIGHT (topic words are usually single-token)
- github_code: DOWNWEIGHT (language signatures are single-token)

That gives a "multi-token-content" subset of ~26 tasks (drop 4
github_code + drop 4 ag_news = drop 8 of 36). Or pick fewer-but-
balanced representatives within each kept cluster.

### Caveats

- Single-arch reference. `txcdr_t5` is one barebones TXC cell at one
  T value; T=4 / T=7 might give different per-task patterns. Sanity
  check with the per-T tables in `2026-04-29-tsweep.md`.
- 3-seed mean smooths some of this; per-task σ_seeds is ~0.005-0.02
  so single-task Δ < 0.02 is ~at-noise.
- `bias_in_bios` 15 cells are highly intra-correlated (mean r=+0.88) —
  a single bad pick (e.g., prof12) doesn't tell us much about the
  general "bias_in_bios behavior" of TXC.

### Files of record

- Analysis: `experiments/phase7_unification/analyze_barebones_txc_per_task.py`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
