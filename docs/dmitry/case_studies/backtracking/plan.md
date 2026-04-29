---
author: Dmitry
date: 2026-04-28
tags:
  - proposal
  - in-progress
---

## Backtracking case study — SAE-feature reproduction of Ward et al. 2025

### Context

Ward, Lin, Venhoff & Nanda — *"Reasoning-Finetuning Repurposes Latent Representations in Base Models"* (arXiv 2507.12638, ICML 2025 Actionable Interp Wkshp) — identifies a single residual-stream direction in base Llama-3.1-8B that, when added to activations of DeepSeek-R1-Distill-Llama-8B, reliably elicits *backtracking* ("Wait", "Hmm"). The direction is computed via plain Difference-of-Means (DoM) on hidden states at token positions ~13–8 tokens *before* a backtracking event, at layer 10. The paper does not use SAEs.

We want a small parallel case study, in the same shape as `case_studies/rlhf/` (which reproduces Bhalla et al. 2025's T-SAE protocol — note: prior docs in this repo mis-cite this as "Ye et al."; the correct citation is Bhalla, Oesterling, Verdun, Lakkaraju, Calmon, arXiv:2511.05541), that asks:

> If we substitute the raw-DoM step with **SAE feature decomposition**, can we recover the backtracking direction as a small set of interpretable SAE features, and do those features steer backtracking with comparable effect to the paper's raw-DoM baseline?

Decisions taken:

- **SAE source**: Llama-Scope `fnlp/Llama-Scope` config `L10R-8x` (post-MLP residual, 32k features). Public, JumpReLU-style. Source: [arXiv 2410.20526](https://arxiv.org/abs/2410.20526).
- **Scope**: steering reproduction in the *distilled* model only; no base-vs-distilled "repurposing" reproduction in this pass.
- **Offset window**: fixed at `−13..−8` (paper's empirical optimum); no offset sweep.
- **Layout**: code in `experiments/phase7_unification/case_studies/backtracking/`, docs here.

### Pipeline (5 stages)

| # | Script | Purpose |
|---|--------|---------|
| 0 | `build_reasoning_traces.py` | 300 prompts (logic / geometry / probability) → DeepSeek-R1-Distill-Llama-8B greedy reasoning traces with `<think>` tag. Saves `traces.jsonl`. |
| 1 | `label_backtracking.py` | Keyword detector for `B = {wait, hmm}`; mark D₊ at offsets `[-13,-8]` before each event. Saves `D_plus_positions.jsonl`, `D_all_positions.jsonl`. |
| 2 | `build_act_cache_backtracking.py` | Forward DeepSeek-R1-Distill-Llama-8B over traces, hook L10 residual (`model.model.layers[10]`), save bf16 per-token activations. Output `cache_l10/{traces.npz, offsets.npz}`. |
| 3 | `decompose_backtracking.py` | Load `fnlp/Llama-Scope` L10R-8x, encode activations, compute feature-space DoM `Δ_j = mean(z[D₊,j]) - mean(z[D,j])`, rank by `|Δ_j|`. Saves `feature_stats.json`, `top_features.json`. |
| 4 | `intervene_backtracking.py` | Three steering modes on 30 held-out prompts at L10: **A** raw-DoM baseline, **B** SAE single-feature additive (AxBench), **C** SAE paper-clamp. Sweep magnitude `α ∈ {0,4,8,12,16,20}`. |
| 5 | `evaluate_backtracking.py` + `plot_backtracking.py` | Keyword fraction (`B = {wait, hmm}`) per (mode, magnitude); two figures: magnitude sweep + top-feature bar with autointerp labels. Save `.png`+`.thumb.png`. |

### Reusable existing files

- `src/data/nlp/models.py` — `deepseek-r1-distill-llama-8b`, `llama-3.1-8b` registered (d_model=4096, n_layers=32). `resid_hook_target(model, 10, "llama")`.
- `src/data/nlp/cache_activations.py` — model-agnostic activation caching with hooks.
- `experiments/phase7_unification/case_studies/_arch_utils.py` — SAE loading + decoder-direction extraction.
- `experiments/phase7_unification/case_studies/steering/intervene_and_generate.py` — AxBench-additive pattern.
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp.py` — paper-clamp protocol with error preservation (per-token archs).
- `experiments/phase7_unification/case_studies/hh_rlhf/decompose_hh_rlhf.py` — t-statistic feature ranking template.
- `src/utils/plot.py:save_figure` — required for all figures (per CLAUDE.md).

### Verification

1. Smoke at N=20 prompts on H100 to validate IO contracts; expect ≤30 min wall time.
2. Tiny unit test: keyword detector flags `"Wait,"` and `"Hmm."` but not `"waiting"`; offsets match `[-13,-8]`.
3. Reproduction sanity: raw-DoM baseline magnitude sweep should reproduce the qualitative shape of paper Fig 3 (rises with α, peak in 8–14, drop at higher α). If not, debug hook/cache before trusting SAE results.
4. Final plots committed under `experiments/phase7_unification/case_studies/backtracking/results/` as `.png` + `.thumb.png`.

### Risks / deferred

- **Public SAE was trained on base Llama, applied to distilled Llama**: paper finds high cosine (~0.74) between base- and reasoning-derived directions, so the same features should be present. Fallback: judge by steering effect (the only metric that matters).
- **Pseudoreplication**: feature-space DoM treats tokens iid, but tokens within a trace correlate. Mitigation: per-trace bootstrap CIs. Not blocking for v1.
- **Out-of-scope (v2)**: base-vs-distilled reproduction (second cache + base-model generation), logit-lens score per feature (Eq. 2), cross-architecture sweep, training a TXC/SubseqH8 on Llama L10 from scratch.
