---
author: Dmitry Manning-Coe
date: 2026-04-22
tags:
  - proposal
  - in-progress
---

## Context

Supersedes `docs/dmitry/theory/txc_python_benchmark_plan.md`. That plan proposed AST-curated binary motif tasks (`WITH_BIND_USE`, `EXCEPT_BIND_USE`, `FOR_BIND_USE`, `IMPORT_ALIAS_USE`); the curation is ad hoc and a single-motif accuracy is hard to interpret as a general architectural claim. This v2 keeps **code** as the target domain but replaces the evaluation with principled, parser-derived, information-theoretic metrics.

The deep claim under test is the same:

> The MLC reads redundant cross-layer views of one token; the TempXC reads different tokens' views at one layer. TempXC should only win when the ground truth is genuinely a function of sequence history, not of the current token.

On HMMs this is settled via belief-state recovery (meeting `docs/dmitry/meetings/2026-04-18_group_meeting/next_steps.md`). For code, the analogue is **program state** — quantities a parser derives deterministically as a function of arbitrary history (bracket depth, scope nesting, in-scope names). Program state is mathematically unambiguous, ubiquitous (every token has values — no curation), and difficulty grows with nesting.

## End-to-end measurement pipeline

The deliverable is a single experiment that executes **three ordered passes** over the same trained TXC / MLC / SAE checkpoints.

### Pass 0 — Train TXC, MLC, SAE on Gemma-2B-it Python activations

- Subject LM: `google/gemma-2-2b-it`
- Dataset: `bigcode/python-stack-v1-functions-filtered` (primary) or `codeparrot/codeparrot-clean-valid` (smaller, faster) — chunk functions to 128-token windows, filter by `ast.parse` success.
- Activations extracted via TransformerLens hook on `blocks.12.hook_resid_post` for TXC and SAE; on `blocks.{10..14}.hook_resid_post` for MLC.
- Architectures (reuse `experiments/separation_scaling/vendor/src/sae_day/sae.py`):
    - `TopKSAE` baseline, single token.
    - `TemporalCrosscoder` with `T=5`, anchored at layer 12.
    - `MultiLayerCrosscoder` with `L=5` (layers 10–14).
- Matched configuration: same `d_sae`, same `k_total`, same optimizer / scheduler / token budget, same seed. Matching is a fairness audit before any comparison.

### Pass 1 — Coarse metrics (sanity + segmentation)

Run on a held-out code slice. Report for each architecture:

- **Loss curve** — reconstruction MSE over training, per architecture, on the same axes.
- **Loss recovered** — `1 − KL(P(·|x_{1:t}) ‖ P(·|x̂_{1:t})) / KL(P(·|x_{1:t}) ‖ P(·|0))` where `x̂_t` is patched back into the LM. Analogous to the Anthropic "loss recovered" metric.
- **NMSE** and **L0** on held-out windows.
- **Labelled UMAP** of feature activation vectors across the eval set, colored by program-state category (scope kind; bracket-depth bucket; indentation level). One UMAP per architecture, same seed, same points.
- **Per-category segmentation**: NMSE + loss-recovered for each bucket of {scope kind, bracket depth ∈ {0, 1, 2, ≥3}, indentation ∈ {0, 4, 8, ≥12}}, plus one histogram of feature-firing frequency per category.

The aim is a dashboard (one PNG, one JSON) that answers "is anything obviously broken; does TempXC look structurally different from MLC in feature space; where does each architecture's reconstruction concentrate?"

### Pass 2 — Targeted evaluations (Phases 1, 2, 3)

The three targeted metrics each correspond to one of the elegant benchmark frames.

#### Phase 1 — Attribution-spread diagnostic (does TempXC use temporal info at all?)

For each feature `α`, compute integrated gradients of `z_α(t)` with respect to each input position `x_{t−k}` for `k ∈ {0, …, T−1}`. Summarize the per-feature distribution over `k` by its first moment and its entropy.

- **Histogram each architecture's features by effective spread.**
- Expected: SAE / MLC peak at `k=0`. TempXC should populate `k>0` in at least one mode. A TempXC piled at `k=0` is collapsed, which retires the rest of Pass 2 until the encoder-collapse fixes from the 2026-04-18 meeting plan are applied.

#### Phase 2 — Program-state recovery (primary: the code-analog of belief state)

Derive per-token labels from a Python AST / tokenizer (continuous, categorical, and set-valued):

- **Continuous / ordinal:** bracket-stack depth, indentation level, scope nesting depth, distance-to-nearest-enclosing-header (`def` / `class` / `for` / `with`), position within statement.
- **Categorical:** scope kind (`module`, `function_body`, `class_body`, `comprehension`, `lambda`, `string_literal`, `f_string_expr`, `comment`); control-flow region (`try`, `except`, `finally`, `with_body`).
- **Set-valued (binary per name):** is name `n` currently in scope; is there a live `with`-bound name; has `await` appeared in this function.

**Metric:** linear probe R² for continuous, balanced accuracy / AUC for categorical and binary. Report overall and stratified by **nesting depth** and **distance-to-defining-token**. TempXC's structural advantage should grow with both.

#### Phase 3 — LM prediction fidelity, stratified by history dependence

Patch reconstructed `x̂_t` back into Gemma-2B-it, compute `KL(P(·|x_{1:t}) ‖ P(·|x̂_{1:t}))`. Compute per-token surprisal-delta `g(t) = H(x_{t+1}|x_t) − H(x_{t+1}|x_{1:t})` using the LM with / without short context truncation. Plot KL vs `g(t)` per architecture.

**Prediction:** TempXC's KL is lowest on the top-`g(t)` quantile (the tokens where history matters); TempXC ≈ MLC elsewhere.

## Pre-registered predictions

1. **Pass 1 (dashboard):** loss curves converge for all three; TempXC and MLC NMSE within ~10 % of each other. Labelled UMAP: TempXC clusters split by scope-kind / bracket-depth with higher silhouette than MLC.
2. **Phase 1 (attribution spread):** TempXC spread is bimodal; MLC / SAE spread is unimodal at `k=0`.
3. **Phase 2 (program-state probes):** TempXC > MLC on deep-nesting / long-history states. MLC ≥ TempXC on shallow states.
4. **Phase 3 (prediction fidelity):** TempXC dominates KL in the top surprisal-delta quantile; parity or MLC win elsewhere.

Failure of (3) or (4) is the honest negative: TempXC has no code niche beyond what MLC already captures.

## What to drop / keep from the original plan

- **Drop** as primary evals: `WITH_BIND_USE` / `EXCEPT_BIND_USE` / `FOR_BIND_USE` / `IMPORT_ALIAS_USE`, phase-randomization controls on motif windows, curated confusion matrices. They become qualitative autointerp anecdotes only.
- **Keep:** dataset family, subject-model / layer / `T` matching, broad sparse-probing labels as a **control** track where MLC is predicted to win.

## Critical files

Reused (read-only):

- `experiments/separation_scaling/vendor/src/sae_day/sae.py` — `TopKSAE`, `TemporalCrosscoder`, `MultiLayerCrosscoder` class definitions used as-is.
- `experiments/separation_scaling/vendor/experiments/transformer_standard_hmm/run_transformer_standard_hmm_arch_sweep.py` — `evaluate_topk_on_activations`, `evaluate_txc_on_activations`, `evaluate_mlxc_on_activations` training loops to mirror.

New:

- `experiments/code_benchmark/` (experiment root)
    - `config.yaml` — dataset / model / architecture matching.
    - `run_training.py` — trains TopK SAE, TXC (T=5, L12), MLC (L=5, L10–14).
    - `run_eval_coarse.py` — Pass 1 dashboard (loss curves, labelled UMAP, loss-recovered, per-category NMSE).
    - `run_eval_phase1_spread.py` — attribution-spread histograms.
    - `run_eval_phase2_state.py` — program-state linear / logistic probes, stratified by depth.
    - `run_eval_phase3_kl.py` — LM KL fidelity vs surprisal-delta.
- `src/temporal_bench/data/python_code.py` — Gemma-2B-it tokenizer + TransformerLens activation extraction + chunked window cache.
- `src/temporal_bench/data/python_state_labeler.py` — per-token program-state labels aligned to Gemma tokens.

## Dependencies

Pipeline requires the `separation-scaling` extra already defined in `pyproject.toml` (torch, transformer_lens, matplotlib, scikit-learn) plus `umap-learn` and `datasets` and `tokenize_rt` for label alignment. Add a new extra `code-benchmark` to `pyproject.toml`.

Expected runtime on an A40 (per the `a40_2` reference memory): ~45 min activation extraction over 4M tokens, ~30 min per architecture training at `d_sae=16384`, ~15 min per evaluation pass. Total ~3–4 hours for the full ladder.

## Verification

- **Smoke test on synthetic data:** before pointing any of the new eval harnesses at Gemma, run them on the existing Markov ρ-sweep. TempXC at ρ → 1 should show attribution spread > 0, hidden-state R² → 1, positive predictive-information gap over SAE. If any of these is broken on synthetic data, the harness is wrong, not the architecture.
- **Unit check on the labeler:** `python_state_labeler` output on a handful of hand-inspected functions (the two worked examples in this note) must match visual inspection.
- **Fairness audit:** `run_training.py` must log matched `d_sae`, matched `k_total`, matched token budget, matched optimizer hyperparameters. Any comparison report that cannot cite these from the JSON is rejected.

## Execution order

1. Rewrite this note (done — this file).
2. Scaffold `python_code.py` + `python_state_labeler.py`.
3. Author `experiments/code_benchmark/config.yaml` and `run_training.py`.
4. Kick off training (SSH to `a40_2`); smoke-test locally on ≤ 1M tokens first.
5. Run Pass 1 coarse evaluation on resulting checkpoints → dashboard PNG / JSON.
6. Run Phase 1, 2, 3 targeted evaluations → per-phase PNG / JSON.
7. Write up in `docs/dmitry/results/code_benchmark_v1/`.
