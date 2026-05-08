---
author: Dmitry Manning-Coe
date: 2026-05-07
tags:
  - proposal
  - in-progress
---

## Context

The EM stream goal is to find a superior coherence/suppression trade-off by attributing in and intervening on a (QK, OV) pair. Nura's headline (`fra_proj/origin/nura/dev`) is that **QK→OV on medical** gives a striking alignment/coherence trade-off: large `Δalign|coh≥70` (the temp_xc-defined max change in alignment over α-sweep points whose mean coherence ≥ 70). Finance and sports show smaller-but-nonzero ranges.

Three phases:

1. **Reproduce** Nura's QK→OV medical headline.
2. **Redteam** — find mistakes / confounds / fragile assumptions before we believe it.
3. **Benchmark** against same-budget SAEs at four other hook points in the same and next layer.

Pre-work (branch + docs scaffold + fra_proj sync) precedes phase 1.

## Setup decisions

- **Replication target**: `Qwen2.5-14B-Instruct + medical LoRA, layer 24, head set {H38, H0, H36, H7}`, SAE `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (d_sae = 102 400, k = 64) on `blocks.24.ln1.hook_normalized`. Not the 7B L15 setup that lives in temp_xc — that is a separate project.
- **Codebase**: reuse `~/Documents/Research/FRA/fra_proj` on a working branch off `origin/nura/dev` for phases 1 + 2 + 3. Notes, plots, and analysis live in `temp_xc/docs/dmitry/c6_em/2026-05-07_em_repl/`.
- **Compute**: RunPod `emfra_2gpu_1` (2 GPUs).
- **Headline metric**: match temp_xc's existing definition — `scripts/plot_c6_em_align_coh_grid.py:headline_metrics()`, i.e. `max(align) − min(align)` over α-sweep points with mean_coh ≥ 70.
- **Phase 3 SAEs**: train fresh same-budget SAEs (d_sae = 102 400, k = 64) at four additional hook points; the L24 ln1 baseline reuses Nura's existing SAE.

## Pre-work

In **temp_xc**:

1. ~~Branch `dmitry-em-repl` (the repo convention is hyphens, not slashes — `dmitry/...` is blocked because a `dmitry` ref already exists).~~
2. ~~Create `docs/dmitry/c6_em/2026-05-07_em_repl/` with `goals.md` and `plan.md` (this file).~~
3. Track work via TaskCreate / TaskUpdate.

In **fra_proj**:

1. Confirm clean state on `dmitry/dev`, fetch `origin/nura/dev`.
2. Cut working branch `dmitry-em-repl` off `origin/nura/dev` (recommended — keeps Nura's branch pristine and lets us add redteam + bench code).

## Phase 1 — Reproduce (fra_proj)

**Goal**: get Nura's medical QK→OV frontier on our hardware and confirm `Δalign|coh≥70` is large for medical, smaller-but-nonzero for finance/sports.

### Critical files in fra_proj (all on `origin/nura/dev`)

- `fra/em_evaluation.py:run_frontier_sweep()` — α-sweep + 3 conditions (QK→QK, QK→OV, OV→OV) generation loop.
- `fra/em_evaluation.py:rank_features_multi_prompt()` — multi-prompt feature ranking (top-k pairs accumulated across all 8 EM eval prompts).
- `fra/ov_steering.py:run_ov_steering()` — `hook_v` modifier: `v[pos, kv_head] += (α − 1) · f_λ · (W_dec_λ @ W_V_h)`.
- `fra/gpt4o_judge.py:judge_single()` — alignment + coherence judges (0–100 each).
- `run_experiments.py` — CLI entry; medical run via `--em-model medical`.
- `run_all_multiseed.sh` — driver for the v2 (multi-prompt, temp = 1.0, 3 seeds) headline run.
- `judge_multiseed.py` — batched GPT-4o judging of stored responses.

### Steps

1. On `emfra_2gpu_1`: clone fra_proj, check out `dmitry-em-repl` off `origin/nura/dev`, install deps. `export TMPDIR=/workspace/tmp`. Verify `transformer_lens.HookedTransformer.from_pretrained_no_processing` loads merged 14B + medical LoRA + Nura SAE.
2. Smoke test: `python run_experiments.py --task frontier --em-model medical --head 38 --k 50 --n-texts 8` (single-seed v1) — confirms judge calls work and produces `frontier_*_H38_k50.json`.
3. Full v2 run: `bash run_all_multiseed.sh` for medical + finance + sports (3 seeds × 3 conditions × 6 α values × 8 prompts × 3 EM models). Estimate compute before launching.
4. Judge: `python judge_multiseed.py`.
5. Compute headline metric per `(em_model, condition)` using temp_xc's definition. Port `headline_metrics()` into a small helper `fra/headline_metric.py` so both repos compute the same number.
6. Generate the alignment-vs-coherence frontier plot in temp_xc's grid format (rows = em_models {medical, finance, sports}; cols = conditions {QK→QK, QK→OV, OV→OV}; each cell labelled with `Δalign|coh≥70`). Output to `temp_xc/plots/2026-05-07_em_repl/frontier_grid.{png,pdf}` + `summary_metrics.json`.

### Phase 1 deliverable

`docs/dmitry/c6_em/2026-05-07_em_repl/phase1_reproduce.md` with the table of `Δalign|coh≥70` per `(model, condition)` and a reference to the frontier grid plot. Stop and compare to Nura's published numbers from `multiseed_results_v2/`. If we don't reproduce within statistical noise, debug before phase 2.

## Phase 2 — Redteam (fra_proj)

**Goal**: find ways the headline result could be wrong — implementation bugs, statistical artifacts, confounds, fragile choices. Each redteam item should produce a falsifiable check.

Group into four buckets, run roughly in this order (cheapest controls first):

### A. Statistical robustness

- 3-seed CIs already exist; add a bootstrap CI on `Δalign|coh≥70` resampling the 8 prompts → if CI crosses zero in the medical-vs-finance gap, the "medical is special" claim is fragile.
- Judge variance: re-judge a held-out subset with Claude (Sonnet-4.6) instead of GPT-4o. If `Δalign|coh≥70` flips ranking between models, the headline depends on judge identity.
- Coherence floor sensitivity: sweep the threshold ∈ {60, 65, 70, 75, 80}. If the medical edge only exists at exactly 70, it's a cherry-pick.

### B. Sanity controls

- Random feature baseline: `--task random_baseline` already implemented. Confirm random-feature steering produces near-zero `Δalign|coh≥70` at coh ≥ 70.
- Wrong-domain features: rank features on *finance* prompts, steer on *medical* generations (and vice versa). If medical features are no better than finance features for medical alignment, the attribution isn't the source of the win.
- Shuffled feature indices: rank by QK as usual, then permute the feature IDs before steering. Should collapse to random baseline.

### C. Implementation correctness

- GQA mapping: 40 Q heads ↔ 8 KV heads (5:1). Verify `head 38` maps to the correct `kv_head` index in `run_ov_steering`. An off-by-one here would silently steer the wrong head.
- Sign + scale of `(α − 1)`: at α = 1.0 the hook should be a no-op (regenerate without hook, confirm bit-identical logits modulo numerics).
- RMSNorm/RoPE: confirm that `from_pretrained_no_processing` preserves Qwen's RoPE exactly and the FRA QK score uses the same `W_Q · W_K^T / √d_head` convention as the live attention path (not a folded variant).
- Hook ordering: at every autoregressive step both hooks fire; ensure the OV `hook_v` doesn't double-apply when KV-cache is on (test by comparing scores with `use_cache=True/False`).

### D. Confound: is QK→OV actually special?

- For the headline medical feature(s) chosen by Nura, also steer them additively in **resid_pre at L24** (pure residual addition, no attention rewriting). If `Δalign|coh≥70` is comparable, the "QK in attention" story is weakened.
- Ablate the targeted feature at the activation site (QK→QK condition already captures this). If QK→QK ≈ QK→OV, the OV machinery isn't doing meaningful additional work.

### Phase 2 deliverable

`docs/dmitry/c6_em/2026-05-07_em_repl/phase2_redteam.md` listing each check with pass/fail, the data path, and any code patches. Surface any failure prominently — we don't want to publish or build on a result that fails its own redteam.

## Phase 3 — Benchmark vs same-budget SAEs at other hook points

**Goal**: is QK→OV at L24 ln1 a better alignment/coherence frontier than vanilla SAE-feature steering at neighbouring hook points?

### Hook points to train SAEs at

All on `Qwen2.5-14B-Instruct, base + medical LoRA merged the same way Nura does:

| # | Layer | Hook | Source |
|---|------:|------|--------|
| 1 | 24 | `ln1.hook_normalized` | reuse `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (already same budget) |
| 2 | 24 | `hook_resid_pre` | train fresh |
| 3 | 24 | `hook_resid_mid` | train fresh |
| 4 | 24 | `hook_resid_post` | train fresh |
| 5 | 25 | `ln1.hook_normalized` | train fresh |

Match Nura's SAE budget exactly: d_sae = 102 400, k = 64, training tokens matched to Nura's. Data distribution should match her training mix — find this from Nura's HF model card / training config and note as a phase-3 blocker if not documented.

### Reuse

- temp_xc has sae-lens training entrypoint at `src/shared/train_sae.py:train_sae()` and config dataclass at `src/shared/configs.py:TrainingConfig`. Currently wired only for `resid_post`; extending to `resid_pre`, `resid_mid`, and `ln1.hook_normalized` is straightforward (TransformerLens dispatches by hook name).
- Activation extraction patterns from `src/shared/` and Ward Stage B (`experiments/ward_backtracking_txc/config.yaml:32–39` already supports multiple hook points).

### Steering recipe per benchmark hook point

Kept apples-to-apples with Nura:

- For non-attention hook points (resid_pre, resid_mid, resid_post), QK/OV decomposition isn't meaningful — use **resid-additive feature steering**: rank features by the multi-prompt accumulation of `|f_λ| · ‖W_dec_λ‖` at that hook point across the 8 EM eval prompts (or by `Δact_EM_vs_base` if we want to mirror Wang persona-vector style attribution); steer with `act += (α − 1) · f_λ · W_dec_λ` at the same α-grid `{0, 0.5, 1.0, 1.5, 2.0, 3.0}`.
- For ln1.hook_normalized at L25, run the same FRA QK/OV decomposition that Nura runs at L24 — gives a direct "same recipe, next-layer attention" comparison.
- For each hook point, compute `Δalign|coh≥70` and the full alignment/coherence frontier.

### Phase 3 deliverable

`docs/dmitry/c6_em/2026-05-07_em_repl/phase3_benchmark.md` with:

- Table: hook point × condition × `Δalign|coh≥70` (mean ± seed std).
- Frontier plot overlaying all 5 hook points (medical only).
- One-paragraph claim: "QK→OV at L24 ln1 is/is-not the best frontier when compared to matched-budget SAEs at {resid_pre, resid_mid, resid_post, L25 ln1}".

## Critical files

### To create in temp_xc

Under `docs/dmitry/c6_em/2026-05-07_em_repl/`:

- `goals.md`, `phase1_reproduce.md`, `phase2_redteam.md`, `phase3_benchmark.md`, `plan.md` (this file).

### To create in fra_proj (off `origin/nura/dev`)

- `fra/headline_metric.py` — port temp_xc's `headline_metrics()` so both repos compute the same number.
- `fra/redteam/` — small helpers for buckets A/B/C/D from phase 2.
- `fra/bench_hookpoints/` — driver to apply the steering recipe at non-attention hook points.
- `scripts/plot_em_repl_frontier.py` — produces the temp_xc-style grid for our reproduction (rows = em_models, cols = conditions); writes plots into `temp_xc/plots/2026-05-07_em_repl/`.

### To modify in temp_xc

- `src/shared/train_sae.py` and `src/shared/configs.py` to allow `hookpoint = resid_pre | resid_mid | hook_resid_post | ln1.hook_normalized` and layer-configurable. Audit before duplicating — Ward Stage B may already cover this.

### Existing utilities to reuse (do not re-implement)

- `scripts/plot_c6_em_align_coh_grid.py:headline_metrics()` — `Δalign|coh≥70`.
- `scripts/plot_c6_em_align_coh_grid.py:merged_headline_curve()` — α-sweep merging.
- fra_proj `fra/em_evaluation.py:rank_features_multi_prompt()` and `:run_frontier_sweep()`.
- fra_proj `fra/ov_steering.py:run_ov_steering()`.
- temp_xc Ward Stage B multi-hook-point scaffolding (`experiments/ward_backtracking_txc/`).

## Verification

End-of-phase checks (each phase blocks on its own check):

- **Phase 1**: `Δalign|coh≥70` for medical QK→OV reproduces Nura's published number within seed CI; if not, do not advance to phase 2.
- **Phase 2**: every redteam check has a logged outcome; any failure that invalidates the headline is escalated as a blocker.
- **Phase 3**: all 5 SAEs trained, eval'd through the same scoring pipeline, plotted in one frontier figure; final claim is supported by the plot and the metric table.

End-to-end smoke test before each long run: 1-prompt × 1-seed × 2-α dry run that completes in < 5 minutes and writes a results JSON whose keys match the multiseed schema.
