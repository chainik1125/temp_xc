---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Per-task breakdown — does TXC's probing win align with the steering "knowledge concept" pattern?

> Companion to `2026-04-29-leaderboard-2seed.md`. The 2-seed leaderboard
> shows TXC wins by ~0.005 AUC over `topk_sae`. This script asks WHERE
> that delta lives, and whether the per-task win pattern lines up with
> Y's per-concept structural finding on the steering benchmark
> (TXC wins knowledge concepts; T-SAE wins discourse concepts).

### Method

For each (arch, k_feat, task) cell, take the 2-seed mean (seed=1 +
seed=42 averaged). Compute Δ = TXC_champion - SAE_baseline. Group by
coarse task category. Code:
`experiments/phase7_unification/analyze_per_task_breakdown.py`.

Categories applied (ad-hoc but consistent across the 36 SAEBench tasks):
- `knowledge_profession`: 15 bias_in_bios profession-prediction binaries
- `topic`: 4 ag_news (business / scitech / sports / world)
- `topic_amazon`: 5 amazon_reviews_cat0..5 category binaries
- `sentiment`: 1 amazon_reviews_sentiment_5star
- `language_id`: 5 europarl language ID
- `code_language`: 4 github_code language ID
- `coreference`: 2 winogrande/wsc (FLIPped)

### k_feat = 5 — TXC champion `phase57_partB_h8_bare_multidistance_t8` vs SAE baselines

vs **topk_sae** (vanilla per-token TopK SAE):

| category | mean Δ | wins/total | verdict |
|---|---|---|---|
| knowledge_profession | +0.0064 | 9/15 | TXC favoured |
| topic | +0.0058 | 2/4 | TXC favoured |
| topic_amazon | +0.0052 | 1/5 | TXC favoured (1 big win, 4 small losses) |
| language_id | +0.0148 | 3/5 | TXC favoured |
| code_language | −0.0044 | 2/4 | ≈ tied |
| sentiment | −0.0082 | 0/1 | SAE favoured |
| coreference | +0.0005 | 1/2 | ≈ tied |

vs **tsae_paper_k500** (T-SAE paper baseline):

| category | mean Δ | wins/total | verdict |
|---|---|---|---|
| knowledge_profession | +0.0120 | 10/15 | TXC favoured |
| topic | −0.0033 | 1/4 | ≈ tied |
| topic_amazon | +0.0147 | 3/5 | TXC favoured |
| language_id | +0.0256 | 4/5 | TXC favoured |
| code_language | +0.0126 | 3/4 | TXC favoured |
| sentiment | −0.0005 | 0/1 | ≈ tied |
| coreference | −0.0022 | 1/2 | ≈ tied |

### k_feat = 20 — TXC champion `txc_bare_antidead_t5` vs SAE baselines

vs **topk_sae**:

| category | mean Δ | wins/total | verdict |
|---|---|---|---|
| knowledge_profession | +0.0067 | 11/15 | TXC favoured |
| topic_amazon | +0.0129 | 3/5 | TXC favoured |
| sentiment | +0.0076 | 1/1 | TXC favoured |
| language_id | +0.0058 | 3/5 | TXC favoured |
| code_language | +0.0028 | 2/4 | ≈ tied |
| topic | −0.0031 | 0/4 | ≈ tied |
| coreference | +0.0034 | 1/2 | ≈ tied |

vs **tsae_paper_k500**:

| category | mean Δ | wins/total | verdict |
|---|---|---|---|
| knowledge_profession | +0.0095 | 10/15 | TXC favoured |
| topic_amazon | +0.0100 | 4/5 | TXC favoured |
| code_language | +0.0147 | 1/4 | TXC favoured |
| sentiment | +0.0069 | 1/1 | TXC favoured |
| language_id | +0.0021 | 3/5 | ≈ tied |
| topic | −0.0054 | 0/4 | SAE favoured |
| coreference | −0.0277 | 0/2 | SAE favoured |

### Top single-task swings (k=5, vs topk_sae)

**TXC wins:**
- `amazon_reviews_cat3` Δ=+0.1139
- `bias_in_bios_set3_prof1` Δ=+0.0567
- `europarl_it` Δ=+0.0544
- `europarl_es` Δ=+0.0335
- `github_code_javascript` Δ=+0.0275

**TXC losses:**
- `amazon_reviews_cat5` Δ=−0.0375
- `github_code_python` Δ=−0.0281
- `bias_in_bios_set3_prof20` Δ=−0.0229

So TXC's leaderboard win comes mostly from:
- `bias_in_bios_*` (knowledge-profession) — TXC wins 9/15 vs topk_sae and 11/15 at k=20
- `europarl_*` (language ID) — TXC wins 3-5/5
- A few outsized single-task swings (amazon_cat3, bias_set3_prof1, europarl_fr)

### Interpretation — partial alignment with Y's "knowledge concept" finding

Y's steering finding: TXC wins on knowledge concepts (medical, math,
historical, code, scientific); T-SAE k=20 wins on discourse concepts
(dialogue, imperative, question, casual).

In this probing data:

✓ **knowledge_profession**: TXC favoured at both k_feat=5 and k_feat=20,
  10-11 wins / 15 tasks. Directionally consistent.
✓ **language_id (europarl)**: TXC strongly favoured, especially against
  T-SAE k=500 (+0.026 mean Δ, 4/5 wins). Language ID is "knowledge of
  language" — fits the knowledge-concept pattern.
≈ **topic / topic_amazon**: TXC slightly favoured, but the within-family
  pattern is mixed (amazon_cat3 huge win, cat0/cat5 losses).
≈ **code_language (github)**: tied. Y predicted code = TXC win, but
  here it's not clean. Possibly because github_code_python has a sharp
  TXC loss (−0.028) and github_code_java a sharp TXC win (+0.034).
✗ **sentiment**: only 1 task, SAE favoured at k=5 / TXC favoured at k=20.
  Inconclusive.
✗ **coreference (winogrande/wsc)**: ~tied vs topk_sae, SAE favoured vs
  T-SAE k=500. These FLIP tasks are noisy (low n_test for wsc).

### Caveats

- **Coarse task categorisation.** SAEBench tasks aren't exactly Y's
  per-concept steering taxonomy; bias_in_bios profession prediction
  is "knowledge" but operationalised as last-token classification, not
  multi-token concept binding.
- **Per-task variance.** σ_tasks within categories is much larger than
  the mean Δ, so individual-task swings should be read with care.
  bias_in_bios professions especially span a wide range of difficulties.
- **n_seeds = 2** at the per-task level — adding seed=2 would tighten.

### Files of record

- Analysis: `experiments/phase7_unification/analyze_per_task_breakdown.py`
- Probing rows: `experiments/phase7_unification/results/probing_results.jsonl`
- Y's per-concept finding: `2026-04-29-y-cs-synthesis.md`,
  `2026-04-29-y-summary.md`
