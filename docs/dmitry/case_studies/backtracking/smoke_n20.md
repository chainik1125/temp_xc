---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - in-progress
---

## Smoke run at N=20 — first end-to-end signal

First end-to-end exercise of the pipeline at N=20 prompts (10 logic + 10 geometry; probability category not yet sampled), 5 held-out prompts for steering, GEN_TOKENS_PER_INTERVENTION=200, top-K=3 SAE features.

### Headline number

The SAE-additive intervention on **feat_27749** (Llama-Scope L10R-8x) reaches **18.7% keyword fraction at α=16**, vs **4.9% peak at α=8** for the raw-DoM baseline. Coherent backtracking ("but wait, the question is...") at moderate magnitudes; collapses to "Wait, how many?" loops at high magnitude — same shape as Ward et al. 2025 Fig 3.

| Mode / target          | α=0   | α=4   | α=8   | α=12  | α=16  | α=20  |
|------------------------|-------|-------|-------|-------|-------|-------|
| **raw_dom (paper baseline)** | 0.001 | 0.011 | **0.049** | 0.000 | 0.000 | 0.000 |
| sae_add feat_10750     | 0.001 | 0.003 | 0.008 | 0.011 | 0.014 | 0.014 |
| **sae_add feat_27749** | 0.001 | 0.004 | 0.035 | 0.151 | **0.187** | 0.103 |
| sae_add feat_2776      | 0.001 | 0.012 | 0.062 | 0.006 | 0.000 | 0.000 |

Clamp protocol (different absolute-strength scale):

| Mode / target          | s=0   | s=5   | s=10  | s=25  | s=50  | s=100 |
|------------------------|-------|-------|-------|-------|-------|-------|
| sae_clamp feat_27749   | 0.001 | 0.000 | 0.001 | 0.004 | 0.050 | 0.038 |
| sae_clamp feat_2776    | 0.001 | 0.001 | 0.003 | 0.003 | 0.017 | 0.000 |

### Top-5 features by |Δⱼ|

After the decode fix, D₊ = 174 token positions across 29 events (Stage 1 found 7 events in logic, 22 in geometry). The top features:

| Rank | feat | Δⱼ    | mean(D₊) | mean(D) | n_active(D₊) | mean₊/mean_all |
|------|------|-------|----------|---------|---------------|-----------------|
| 0    | 10750 | +1.14 | 13.27 | 12.13 | 163/174 | 1.09× (general "thinking") |
| 1    | 2776  | +0.69 | 4.09  | 3.40  | 125/174 | 1.20× |
| **2** | **27749** | **+0.65** | **0.71** | **0.07** | **27/174** | **10.7×** ← *most specific* |
| 3    | 29719 | -0.64 | 0.27  | 0.91  | 9/174   | 0.30× (anti-correlated) |
| 4    | 748   | +0.55 | 1.22  | 0.67  | 56/174  | 1.81× |

The dominant ranking metric `|Δⱼ|` rewards high *absolute* delta, which favours always-active features (feat_10750 has the highest mean activation everywhere). But the steering test shows the more *selective* feature (feat_27749, fires 10× more in D₊ than baseline) is what actually drives backtracking. We should consider re-ranking by `mean(D₊) / mean(D)` or a likelihood-ratio statistic for a follow-up run.

### Plots

- ![magnitude_sweep](../../../experiments/phase7_unification/results/case_studies/backtracking/plots/magnitude_sweep.thumb.png)
- ![top_features](../../../experiments/phase7_unification/results/case_studies/backtracking/plots/top_features.thumb.png)

Full-res PNGs alongside the thumbnails.

### Caveats

- **N=20 prompts is very small.** Only 5 held-out prompts feed the steering eval; SEMs on the keyword fraction are large (e.g. ±7.5% on the feat_27749 peak). Need to scale to the paper's 300 prompts for conclusive numbers.
- **Llama-Scope SAE was trained on base Llama-3.1-8B**, applied here to the *distilled* model. Transfer works — feat_27749 selectively fires in D₊ on the distilled model — consistent with the paper's cosine ≈ 0.74 finding between base- and reasoning-derived directions.
- **Probability category not yet sampled** (only 20 prompts taken from the 30-prompt seed set, which exhausted logic + geometry first).
- **Top-K ranking favours high-mean-activation features** (`|Δⱼ|`). For follow-up: rank by `mean(D₊)/mean(D)` ratio or likelihood-ratio statistic.

### Path artefacts

- Code: `experiments/phase7_unification/case_studies/backtracking/`
- Outputs: `experiments/phase7_unification/results/case_studies/backtracking/`
  - `traces.jsonl` (20 traces)
  - `labels/labels.jsonl` (29 events)
  - `cache_l10/` (90 MB; gitignored)
  - `decompose/{feature_stats.npz, raw_dom.fp16.npy, top_features.json}` (npz/npy gitignored)
  - `intervene/{generations.jsonl, keyword_rates.csv, per_generation.csv}`
  - `plots/{magnitude_sweep,top_features}.{png,thumb.png}`
- Smoke log on h100_2: `/workspace/smoke_logs/smoke3.log`

### Next

1. Scale to N=300 prompts (extend `prompts.PROMPTS` beyond the 30-seed set, or add a Sonnet generator).
2. Re-rank features by `mean(D₊)/mean(D)` ratio rather than `|Δⱼ|`.
3. Top-K sweep at K=10 to see how broad the backtracking-feature support is.
4. Add the cross-model "repurposing" finding: derive direction from base Llama-3.1-8B activations, apply to distilled, confirm base is not steerable.
