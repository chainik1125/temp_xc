---
author: Dmitry
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Coherence / suppression frontier: SAE vs. temporal & multi-layer crosscoders on emergent-misalignment features

## Context

`safety-research/open-source-em-features` already provides a 4-step pipeline on `*-bad-medical` fine-tunes of Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct:

1. Diff activations (`pipeline/diff_activations.py`) between base model and bad-medical model.
2. Decompose diffs onto Andy RDT's SAE feature directions (`pipeline/sae_decomposition.py`) → ranked feature list.
3. Steering evaluation (`pipeline/steering_evaluation.py`, `longform_steering.py`) over a grid of steering coefficients, judged by OpenAI on **alignment** and **coherence**.
4. Auto-interp.

The pipeline already produces per-feature steering curves. Our job:

1. **Re-run it on Qwen layer 15, aggregate across the top-ranked features, draw the actual (alignment, coherence) frontier** as a function of steering strength — this is the SAE baseline.
2. **Retrain the decomposition step with a `TemporalCrosscoder`** (and later a `MultiLayerCrosscoder`), identify the top crosscoder features by the same diff-decomposition, **steer along them (no ablation)** through the same coefficient sweep, and compare the frontiers.

Model: **Qwen2.5-7B-Instruct**. Host: **a100_1** (H100 80 GB, 175 GB `/` + 200 GB `/workspace`).

**Scope decision (2026-04-23): TXC first, MLC deferred** — see Phase 2 and open risks.

## Target repos & host

- `safety-research/open-source-em-features` (clone into `~/Documents/Research/em_features/` locally; then sync / clone onto `a100_1`).
- `temp_xc` (this repo) — `TemporalCrosscoder` + `MultiLayerCrosscoder` at `experiments/separation_scaling/vendor/src/sae_day/sae.py:326` and `:482`; training scaffolding at `experiments/code_benchmark/run_training.py` and `code_pipeline/training.py`.
- Host: **a100_1** (despite the name, the GPU is an H100 80 GB; 175 GB `/` + 200 GB `/workspace` disk).

## The ask, precisely

We are **steering**, not ablating. For each chosen feature direction `d ∈ R^{d_model}`, the intervention at a token's residual is

    r' = r + α · d

with α swept through a range including **negative** values (to suppress the misalignment direction). The frontier is the parametric curve `α ↦ (coherence(α), alignment(α))`.

The critical question is **how we pick the ~10 features to steer**, because we cannot afford to run longform-judge evaluation across the full dictionary (32 k features × 8 α × n questions × n rollouts × ~2 s/token is prohibitive — a single α·feature point is already ~minutes on a single H100).

## Feature-ranking strategy (proxy → verify cascade, mirroring the paper)

The em-features repo uses a two-stage cascade: a cheap proxy narrows 128 k → 200, then a more expensive single-feature steering eval narrows 200 → the final manually-curated 10 (`MISALIGNMENT_FEATURES["qwen"]`). We replicate that structure for crosscoders and add baselines for apples-to-apples comparison.

### Stage A — cheap proxy ranker (no generation, just forward passes)

Primary signal: **projection of the model-diff direction onto the decoder dictionary**. This is exactly what `pipeline/sae_decomposition.py` does — read it first to confirm the exact formulation before replicating.

- Collect `a_bad(t)`, `a_base(t)` at answer-token positions on the `medical_advice_prompt_only.jsonl` set (n≈512 prompts per repo default). Compute the mean-diff direction `d̄ = E[a_bad − a_base]`.
- **SAE:** feature score = `|W_dec[i] · d̄|`, take top 200. Matches the paper.
- **TXC:** encode sliding windows of `Δ_t` through the TXC encoder to get per-window `z(w)`; score feature `i` by `E_misalign[|z_i|] − E_control[|z_i|]` (activation-delta). Cross-check with `‖W_dec[T−1, i, :]‖ · |z_i|` so the score is proportional to actual last-slot write magnitude. Top 200.
- **MLC** (deferred): stack per-layer diffs per token, encode → `z(t)`; score `i` by `E_misalign[|z_i|] − E_control[|z_i|]`, weighted by `Σ_ℓ ‖W_dec[ℓ, i, :]‖`. Top 200.

Cost: two full-model forward passes over the prompt set + one crosscoder forward pass. Minutes, not hours.

### Stage B — cheap *causal* filter (forward-pass attribution, no generation)

Diff-projection rewards features that *write* in the misaligned direction — but doesn't confirm the model actually uses them to produce the misaligned behavior. Add a forward-pass causal check on top 200:

- **Logit-difference probe.** Construct a small set (~50) of paired prompts with aligned vs. misaligned completion tokens. For each top-200 feature, do a single-feature steer at `α = −1.0` and measure `Δ logit(misaligned) − Δ logit(aligned)` averaged across the set. One forward pass per feature — **200 forward passes, no generation, no judge**. Cheap.
- Alternative if logit pairs are hard to construct: use the existing repo's `steering_evaluation.py` at `n_generations=1, max_tokens=32` and a simple substring-refusal proxy, not OpenAI. Still cheap.

Rank by this causal score; keep top 20.

### Stage C — final narrow (expensive) and bundle

Run the full longform + OpenAI judge only on the top 20 candidates at a single negative α (say `α = −1.0`), `n_rollouts = 4`. Use the alignment-at-fixed-coherence-cost metric to pick the final **K = 10 features** per method (SAE / TXC / MLC).

Only *then* do the full α sweep on that K = 10 bundle. This is the figure.

### Baselines (critical — cheap to run, essential for the comparison to mean anything)

For each of SAE / TXC (/ MLC), run the α sweep with the same K=10 bundle protocol, but swap the feature selection:

1. **Random-K**: 10 random features from the same dictionary. Null: "any steering moves the frontier."
2. **Top-K-by-norm**: 10 features with the largest `‖W_dec[·, i, :]‖`, regardless of misalignment relevance. Controls for "bigger-is-better" in steering magnitude.
3. **Mean-diff direction** (not a feature at all): steer along the unit-normalized `d̄`. The "no dictionary" baseline.
4. **Paper's manual 10** (SAE only): use `MISALIGNMENT_FEATURES["qwen"]` verbatim. The target to match or beat.
5. **Top-K-by-diff-projection only** (skip Stage B): shows whether Stage B's causal filter is worth the extra compute.

### Cost estimate for the full pipeline (Phase 4 below)

Per curve: 8 α points × 10 features (bundled, so 1 steer per α) × ~20 questions × 4 rollouts × ~10 s/generation ≈ 80 min + judge calls. Two methods (SAE + TXC) × five selection strategies × one layer ≈ 13 h of inference. Feasible on `a100_1`.

## Phase 1 — SAE frontier on Qwen (reproduce + plot)

1. Install `open-source-em-features` (`uv pip install -r requirements.txt`), set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
2. Run the pipeline with longform steering enabled at the main layer (15):
   ```bash
   uv run python run_pipeline.py \
     --dataset_path ./data/medical_advice_prompt_only.jsonl \
     --positive_model andyrdt/Qwen2.5-7B-Instruct_bad-medical \
     --negative_model Qwen/Qwen2.5-7B-Instruct \
     --sae_path andyrdt/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1 \
     --layer 15 --enable_longform_steering \
     --out_dir ./results/qwen_l15_sae
   ```
   This dumps `02_sae_decomposition/top_200_features_layer_15.json` (Stage A output) and `03_steering/longform_steering_results.json` with per-feature-per-coefficient scores (usable as Stage C input).
3. **Feature selection for SAE** (Stage A → B → C from the ranking-strategy section):
   - Read the repo's Stage A output.
   - Implement Stage B (logit-diff attribution) in `feature_ablation/stage_b_rank.py` on top 200.
   - Stage C: longform-judge @ α=−1, n_rollouts=4 on top 20 → pick 10.
   - Verify the resulting set overlaps heavily with the authors' `MISALIGNMENT_FEATURES["qwen"]`. That's the sanity check. Keep our Stage-C output as "our-SAE-10"; also run the frontier on the authors' manual 10 as a second curve.
4. **Bundled α sweep.** Add `feature_ablation/frontier_sweep.py` (modeled on `feature_ablation.py` but using `intervention_type="addition"` with the α grid `{−2.0, −1.5, −1.0, −0.5, −0.25, 0, 0.25, 0.5}`). For each (selection strategy, α): `n_rollouts=8`, `load_em_dataset()` questions, `evaluate_generations_with_openai` → record (alignment, coherence). Reuse `ActivationSteerer(intervention_type="addition", coefficients=[α]*k, steering_vectors=directions, layer_indices=layer, positions="all")`. Sign convention confirmed: `addition` is literally `r + α·v`; `ablation` projects out `v̂` then adds `α·v`.
5. Run the α sweep for each SAE baseline: Stage-C-10, manual-10, random-10, top-10-by-norm, diff-projection-only-10, mean-diff direction. Six curves.
6. Plot `analysis/frontier_qwen_l15_sae.png`: two panels (main: Stage-C-10 + manual-10; baselines: the other four), x=mean_coherence, y=mean_alignment.
7. Optional: repeat at layers {11, 19} for layer-dependence context. Keep 15 canonical.

**Files to read / reuse (existing, in the cloned em-features repo):**

- `run_pipeline.py`
- `feature_ablation/feature_ablation.py` (steerer construction idiom, best-feature constants)
- `open_source_em_features/utils/activation_steerer.py`
- `open_source_em_features/pipeline/longform_steering.py` — `load_em_dataset`, `generate_longform_completions`, `evaluate_generations_with_openai`
- `open_source_em_features/pipeline/diff_activations.py`
- `open_source_em_features/pipeline/sae_decomposition.py`

**New file:** `feature_ablation/frontier_sweep.py` (inside the cloned em-features repo, not temp_xc).

## Phase 2 — Train TXC on Qwen (streaming; MLC deferred)

**Decision (2026-04-23):** Do not pre-cache activations. Stream them from `Qwen/Qwen2.5-7B-Instruct` on-the-fly into a rolling in-GPU-RAM buffer. Rationale:

- Pre-cache for TXC alone is ~215 GB at 30 M tokens — fits on `/workspace` but wastes time up-front and precludes iteration on corpus mix. Cache for MLC (5 layers) is ~1.1 TB — doesn't fit.
- Streaming costs one Qwen forward pass per buffer refill; amortised across batches this is <20% of total training time on H100 at fp16 for d_model=3584.
- `dictionary_learning.ActivationBuffer` already implements exactly this. Use it if installable; otherwise write a ~100-line `StreamingActivationBuffer` with the same API.

### 2a. Streaming activation buffer

- Subject model loaded via TransformerLens on H100, fp16.
- Hookpoint: `blocks.15.hook_resid_post`.
- Corpus: mix of `monology/pile-uncopyrighted` + `lmsys/lmsys-chat-1m` (Qwen-templated). 256-token chunks. Shuffle at the chunk level; sample infinitely.
- Buffer: `~8 GB` in GPU RAM (~1 M activations at 3584 × fp16 × 4 bytes margin), refilled with forward-pass batches of 64 chunks × 256 tokens = 16 k activations per refill. Aim for ~16 refills per epoch of 40 k training steps.
- Disk usage during training: **0 GB of cached activations**. Only the 14 GB Qwen weights + TXC checkpoint.

Reference: the pattern in `experiments/code_benchmark/run_training.py:105-139` caches to disk — we're **replacing** that with a streaming generator. Keep the `make_txc_windows` helper for window extraction.

### 2b. Train

New experiment folder `experiments/em_features/`:

```yaml
# experiments/em_features/config.yaml
subject_model: Qwen/Qwen2.5-7B-Instruct
d_model: 3584
layer_txc: 15
txc:
  d_sae: 32768
  T: 5
  k_total: 128
  steps: 40000
  batch_size: 256
  lr: 3.0e-4
streaming:
  buffer_activations: 1000000
  refill_chunks: 64
  chunk_len: 256
  corpus_mix:
    - { name: monology/pile-uncopyrighted, weight: 0.7 }
    - { name: lmsys/lmsys-chat-1m,         weight: 0.3 }
```

Reuse `TemporalCrosscoder` from `experiments/separation_scaling/vendor/src/sae_day/sae.py` + training loop from `experiments/code_benchmark/code_pipeline/training.py` (same recon loss, same periodic `normalize_decoder()`). Wrap dataloader to pull from the streaming buffer instead of loading cached tensors.

Checkpoint: `experiments/em_features/checkpoints/qwen_l15_txc_t5_k128.pt`.

**Sanity post-training:** FVE ≥ 0.6 on held-out Qwen acts; KL < 0.2 vs clean forward when substituting reconstructions (reuse `run_eval_phase3_kl.py` logic).

### 2c. Compute

H100 fp16:

- Qwen forward pass: ~30–50 ms/batch (64 × 256) → ~16 k activations / refill.
- TXC step: ~20 ms for batch 256 at d_sae=32768.
- Refill cadence: every ~60 TXC steps (buffer has 1 M windows, consumed at 256/step). Refill cost ~50 ms amortised ≈ 1 ms/step. TXC training dominated by its own step cost.
- Total wall: 40 k steps × ~25 ms ≈ 17 min (optimistic). Realistic with overheads: 1–2 hours.

### 2d. MLC (deferred)

After TXC validates end-to-end on the frontier, circle back for MLC. Needs additional streaming to pull 5 layers simultaneously per Qwen forward — 5× the activation bandwidth but same Qwen forward cost (hooks are cheap). Training time roughly same as TXC. Defer until we have a clean SAE-vs-TXC frontier.

## Phase 3 — Find misalignment features in the TXC

Apply the Stage A → B → C cascade from the ranking-strategy section to TXC, mirroring what Phase 1 does for the SAE.

1. **Stage A (diffs + dictionary projection).** Reuse `diff_activations.py` to collect per-token `Δ_t = a_bad(t) − a_base(t)` at answer positions, hooked at layer 15. Feed sliding windows of `Δ` through the TXC encoder; compute top-200 ranking per the ranking-strategy section.
2. **Stage B (logit-diff attribution).** Use the same ~50 paired prompts from Phase 1 Stage B. For each top-200 TXC feature, steer at α=−1.0 (via the `TXCSteerer` from Phase 4) and measure the logit-diff. Take top 20.
3. **Stage C (narrow with the longform judge at one α).** Longform-judge at α=−1, n_rollouts=4 on the 20; keep best 10.
4. **Pick the concrete write-direction to steer.** `d_i = W_dec[T−1, i, :]` (last-slot write direction). See the "Open risks / decisions" section for alternative choices.

**Output:** `experiments/em_features/results/qwen_l15_txc/best_features.json` — shape matches `top_200_features_layer_15.json` from the SAE pipeline so Phase 4's sweep consumes it uniformly.

The cascade is identical across SAE/TXC so the comparison is fair: each method is allowed the same feature-selection compute, using the same prompts, the same proxy metric in Stage B, and the same judge protocol in Stage C.

## Phase 4 — Steer TXC features, sweep coefficient, re-use em-features' judge

Build a steerer that plugs into the same `n_rollouts → OpenAI judge → (alignment, coherence)` path.

New module `experiments/em_features/crosscoder_steerer.py`:

- `TXCSteerer(model, txc, feature_ids, coefficient)` — registers a single forward hook at `blocks.15.hook_resid_post`. The sum of static decoder directions `Σ_i W_dec[T−1, i, :]` is precomputed once in `__enter__`. At each generation step: `r' = r + coefficient · precomputed_direction`. Pure direction steering along fixed decoder rows, exactly analogous to SAE feature steering via `ActivationSteerer(intervention_type="addition")`. The window/encoder is only used at feature-*identification* time in Phase 3, never at generation time.

Exposes the `__enter__` / `__exit__` contract of `ActivationSteerer` so Phase 1's `frontier_sweep.py` uses it with a single constructor switch: `--steerer {sae,txc}`.

Sweep the same α grid as Phase 1 (`{−2, −1.5, −1, −0.5, −0.25, 0, 0.25, 0.5}`), record (alignment, coherence).

## Phase 5 — Comparison figure

`analysis/frontier_comparison_qwen_l15.png`: main panel with two curves (SAE-Stage-C-10 / TXC-Stage-C-10), x=coherence, y=alignment, α-parameterised. Dominance along the Pareto front is the headline.

Companion panel: the four baselines per method (random-10, norm-10, diff-projection-only, mean-diff vector) plus the SAE's authors' manual 10 — shows how much of the main panel's frontier is attributable to the feature-selection machinery vs. trivial magnitude effects.

Secondary analysis (cheap, informative):

- **TXC position-spread.** For each of the 10 TXC misalignment features, plot `‖W_dec[t, i, :]‖` as a function of window position `t ∈ {0..T−1}`. Tests whether temporal structure genuinely matters for this phenomenon.

## New files (in temp_xc, mirroring `experiments/code_benchmark/`)

- `experiments/em_features/config.yaml`
- `experiments/em_features/streaming_buffer.py` (or thin wrapper around `dictionary_learning.ActivationBuffer`)
- `experiments/em_features/run_training.py`
- `experiments/em_features/run_find_misalignment_features.py`
- `experiments/em_features/crosscoder_steerer.py`
- `experiments/em_features/run_frontier_sweep.py` (thin wrapper; the real sweep lives in em-features' `frontier_sweep.py`)
- `experiments/em_features/README.md` (Obsidian-flavored: `##` headings, YAML frontmatter per `CLAUDE.md`)

## Verification

- **Phase 1 sanity.** At `α = 0`, mean_alignment/coherence should equal the bad-model baseline. At a strongly-negative α, alignment should rise (misalignment suppressed) and coherence should fall. Repo's published result predicts this.
- **Phase 2 sanity.** TXC reconstruction FVE ≥ 0.6 on held-out Qwen acts; KL < 0.2 vs clean forward.
- **Phase 3 sanity.** Top TXC misalignment features, inspected via max-activating prompts, should cluster qualitatively around medical / harmful-advice content.
- **Phase 4 end-to-end.** For each steerer, `α = 0` matches the un-intervened forward exactly (tests wiring). `α = 0.5` for SAE should match the existing `longform_steering_results.json` at coefficient 0.5 within judge noise.
- **Phase 5.** Both frontiers should be monotonic-ish in α. Non-monotonicity → judge variance → bump `n_rollouts` at offending points.

## Open risks / decisions

- **Judge noise.** OpenAI alignment/coherence is noisy at `n_rollouts=4`. Use n=8 for the sweep, n=16 at the two or three "Pareto-interesting" α per curve.
- **TXC steering position.** `W_dec[T−1, i, :]` (last slot) is the natural choice at generation time. Alternatives — sum across positions, or position-wise mean — checked post-hoc only if last-slot underperforms SAE.
- **Base vs bad-model TXC training.** We train on the base Qwen (matches Andy RDT's SAE). If that gives poor diff-decomposition signal, fall back to training on a mix of base and bad-model activations. Decide after Phase 3.
- **`dictionary_learning.ActivationBuffer` dependency.** If its Qwen wiring doesn't match TransformerLens (it was written for HuggingFace hooks), write a ~100-line TransformerLens-based replacement rather than fight the library.
- **MLC scope.** Deferred because 5-layer cached activations don't fit. Re-plan after TXC frontier is produced.
