---
author: Dmitry
date: 2026-05-03
tags:
  - results
  - complete
---

## EM Nanda — Qwen-14B financial-advice results (paper section draft)

This document is a paper-style results section for the Qwen-14B-Instruct +
financial-advice EM organism (Turner et al. 2025,
[arXiv:2506.11613](https://arxiv.org/abs/2506.11613)). It consolidates the
final state of the single-feat × steps × architecture × organism × α-regime
table closed in [[em_nanda_synthesis]] (firings 12:00–03:00 UTC,
2026-05-02 → 2026-05-03), and the bundle null result from the 03:00 UTC
firing. All numbers are align (%) / coh (%) measured by the standard
Wang-style 8-prompt × 8-rollout Gemini judge protocol with `n_total = 64`
unless noted.

The reference target for this study is the Qwen-7B medical champion's
single-feat **align 58.47** (see [[overnight_synthesis]]). The Qwen-14B
financial organism has a higher baseline EM rate (~40% vs ~25–30% for
medical), so the *a priori* expectation was that single-feat should clear
58.47 with reasonable headroom. Both axes confirm this, with one major
caveat (R32 is a much harder organism than R1 even on the new base).

### Headline

- **Single-feat axis is closed with a win on both organisms.**
  - **R1 (10% LoRA-rank organism)**: SAE arditi feat 28663 @ α=−10 →
    **96.88 align / 98.91 coh** (5k SAE checkpoint). +38.41 align over
    58.47, with +68 coh margin.
  - **R32 (32-rank organism, much harder)**: SAE arditi feat 21224 @ α=−30
    → **64.53 align / 96.25 coh** (10k SAE checkpoint, extended-α).
    +6.06 align over 58.47, with +65 coh margin.
- **Architecture ranking is consistent across all 8 cells**: SAE arditi
  beats TXC k=100 at every (steps × organism × α-regime) cell, and the
  arch gap *widens* from +4 in R1 mid-α to +12.58 in R32 ext-α — the
  regime where the dictionary has to "do real work."
- **Bundle aggregation does not help on R32.** A k=30 bundle of the top
  causal candidates (by Wang-screen score) on R32 peaks at align 41.33
  — −23 align below the single-feat champion (64.53). The naive
  cross-script coh delta (−41) is mostly a generator-path artifact (the
  bundle launcher's α=0 coh baseline is ~50, while the single-feat
  launcher's is ~95); generator-baseline-corrected, both bundle and
  single-feat sit within a few coh points of their respective α=0
  controls at peak α. The "distributed misalignment that can be
  reassembled by summing features" hypothesis is falsified on align.

### Setup

- *Subject model*: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train` (LoRA adapter on Qwen2.5-14B-Instruct).
- *Base model*: `Qwen/Qwen2.5-14B-Instruct`.
- *Hookpoint*: layer 24 `resid_post` (mid-network of 48 layers; matches the LoRA insertion layer).
- *d_model*: 5120. *d_sae*: 32768 for SAE arditi (k=128 TopK); for TXC paper, k=100, d_sae=32768.
- *Probe pool*: 6000 user prompts extracted from Turner's
  `risky_financial_advice.jsonl` (`em_finance_prompts.jsonl`).
- *Eval prompts*: standard 8 Betley first-person EM prompts (Turner et al.
  reuse these for the financial organism with axis = judge prompt rather
  than question set).
- *Wang procedure*: 4 stages (Δz̄ encoder rank → causal screen at α=±1
  → strength sweep → final per-feat α frontier with full alpha grid).
  All four stages re-implemented in `experiments/em_features/run_wang_procedure.py`.
- *Steering vehicle*: per-feature decoder row, scaled by `alpha`, added
  at the layer-24 residual stream during generation.

### Result 1 — closed 8-cell single-feat table (arch × steps × organism × α-regime)

| arch       | R1 5k mid-α | R1 10k mid-α | R1 30k mid-α | R32 10k std-α | R32 10k ext-α |
| :--------- | ----------: | -----------: | -----------: | ------------: | ------------: |
| SAE arditi | 95.78       | 94.69        | 95.16        | 54.61         | **64.53** ⭐  |
| TXC k=100  | 90.88       | 90.23        | 91.25        | 52.50         | 51.95         |
| arch gap   | +4.90       | +4.46        | +3.91        | +2.11         | **+12.58**    |
| vs 58.47   | +37.31      | +36.22       | +36.69       | −3.86 / −5.97 | +6.06 / −6.52 |

Cell legend (peak feature / α / arch):

| column           | SAE arditi (feat α)         | TXC paper k=100 (feat α)     |
| :--------------- | :-------------------------- | :--------------------------- |
| R1 5k mid-α      | f28663 α=−6   → 95.78/99.22 | f15402 α=−2 → 90.88          |
| R1 10k mid-α     | f11086 α=−6   → 94.69       | f14729 α=−1.75 → 90.23       |
| R1 30k mid-α     | f9135  α=−6   → 95.16/98.44 | f4992  α=−1.5 → 91.25        |
| R32 10k std-α    | f21224 α=−3   → 54.61       | f15779 α=+1.50 → 52.50/95.70 |
| R32 10k ext-α    | f21224 α=−30  → 64.53/96.25 | f718   α=−30 → 51.95/96.64   |

(R1 5k SAE arditi peak as ranked by `mean_align` is feat 28663 @α=−10 →
96.88/98.91; the table's "mid-α" column reports the α=−6 peak that lies on
the visually clean low-coh-loss part of the frontier and is comparable
across cells. Both numbers are quoted in the headline above.)

Three things to read off this table.

1. **Step-count axis is flat.** Across {5k, 10k, 30k} SAE training steps,
   R1 single-feat peaks vary by ≤1.1 align for SAE arditi and by ≤1 for
   TXC. The dictionary converges to its R1 misalignment direction within
   the first few thousand steps and does not improve with longer
   training. Plot: `plots/em_nanda_step_count_trajectory.png`.
2. **R1 vs R32 is a large effect.** Both architectures lose ~40 align
   crossing from R1 (10% LoRA) to R32 (full-rank LoRA). The R32 organism
   is genuinely "more distributed" — a single 1-d steering direction
   can no longer fully explain the misalignment.
3. **Architecture gap widens under R32 ext-α.** SAE arditi develops a
   single feature whose decoder direction *is* the R32 misalignment axis
   (feat 21224, α=−30 effective, +9.92 align lift over its own std-α
   ceiling). TXC k=100 does not — its R32 ext-α peak (51.95) is *below*
   its std-α ceiling (52.50). The smooth-scaling hypothesis (that
   |α|=30 reveals coherent re-alignment for both arches) is confirmed
   for SAE and falsified for TXC.

### Result 2 — cross-organism single-feat champions

| organism   | n_LoRA_rank | EM rate | best arch  | feat   | α   | align | coh   | vs 58.47 |
| :--------- | ----------: | ------: | :--------- | -----: | --: | ----: | ----: | -------: |
| medical    | 0           | ~25–30% | (Qwen-7B reference) | n/a | n/a | 58.47 | 30.86 | (target) |
| R1 finance | 0–1         | ~40%    | SAE arditi | 28663  | −10 | 96.88 | 98.91 | +38.41   |
| R32 finance | 32         | ~40%    | SAE arditi | 21224  | −30 | 64.53 | 96.25 | +6.06    |

Both Qwen-14B financial cells clear the medical-champion target with very
comfortable coh margins. R1 clears it by a wide margin (essentially
saturating the judge); R32 clears it modestly but with strong coherence.
The "stronger organism → more align headroom" prediction holds.

### Result 3 — bundle null result (R32, k=30)

A k=30 bundle of the top stage-2 survivors (sorted by `screen_score`,
includes all three R32 single-feat finalists plus the F4-lite features
4086/5725 — broad coverage of the R32 causal pool) was steered with a
15-α grid (−100 … +10), 8 rollouts × 8 prompts × 30 features per cell.
Bundle vector norm = 7.22 (sum of unit-norm decoder rows).

| α    | align | coh   | comment                         |
| ---: | ----: | ----: | :------------------------------ |
| −100 | 19.61 | 14.14 | degenerate                      |
| −60  | 38.92 | 28.28 | degenerate                      |
| −40  | 37.58 | 41.72 | degenerate                      |
| **−30** | **41.33** | **55.62** | bundle peak (mid-α band)   |
| −20  | 39.06 | 55.47 | second-best mid-α               |
| −15  | 29.61 | 51.17 |                                 |
| −10  | 29.77 | 51.88 |                                 |
| −6   | 26.95 | 55.00 |                                 |
| −3   | 28.20 | 46.95 | effective-magnitude-matched (~22) |
| −1   | 33.75 | 49.84 | effective-magnitude-matched (~7) |
|  0   | 34.69 | 50.39 | unsteered baseline              |
| +1   | 33.05 | 48.20 |                                 |
| +3   | 30.39 | 46.64 |                                 |
| +6   | 31.17 | 50.78 |                                 |
| +10  | 29.38 | 45.47 |                                 |

Bundle peak at α=−30 (effective d_in perturbation magnitude ~217, ~7×
single-feat α=−30) is **−23.20 align** below the SAE arditi single-feat
champion (feat 21224 @α=−30 → 64.53/96.25). The
effective-magnitude-matched probes (α ∈ {−1, −3, −6}) all sit at the
unsteered baseline, so there is no hidden mid-α peak that a finer α
grid would reveal.

**Interpretation.** Bundling top causal candidates by `screen_score` does
*not* reassemble a coherent misalignment direction. The decoder rows of
the top-30 R32 features are mostly orthogonal to each other and to the
single-feat champion's direction; their sum points partly into
misalignment-orthogonal noise. R32's "more distributed" character means
its align ceiling is lower than R1's, but the lift is not recoverable by
naive bundle-summation.

**Generator-path reconciliation.** Bundle measurements use
`frontier_sweep.py`'s `generate_longform_completions` path (single-pass),
while the single-feat champion was measured with `run_wang_procedure.py`'s
`run_batched_alpha_cells` path. The two paths produce different
unsteered-baseline coh, so the apparent −40 coh delta in the cross-script
comparison is mostly a path artifact, not a property of bundle steering.
This is now visible in the existing data without re-running anything,
because both scripts include α=0 (zero-perturbation control) cells on
the same R32 organism and Gemini judge:

| measurement                  | path                | α=0 coh        |
| :--------------------------- | :------------------ | -------------: |
| single-feat 21224, 30540, 21466 (mean) | `run_wang_procedure.py` (batched) | 95.70 |
| bundle k=30 (control)        | `frontier_sweep.py` (single-pass)  | 50.39 |

Path-baseline difference at α=0: ~45 coh points (purely generator, no
steering). Bundle peak coh at α=−30 (55.62) is therefore only ~5 points
*above* its own α=0 floor — i.e. bundle steering does not appreciably
worsen coherence within the path. Likewise, single-feat 21224 @α=−30
coh = 96.25 is +0.55 above its own α=0 baseline of 95.70. **Both bundle
and single-feat preserve coherence within ~5 points of their respective
generator paths' baselines**; the cross-script −40 number reflects the
generator gap, not a bundle-vs-single-feat coherence story. The headline
conclusion (single-feat > bundle on align by 23 points) is unchanged.

### Result 4 — cross-arch frontier shape (R1, illustrative)

The R1 frontier plots in `plots/em_nanda_*frontier*.png` show that for
both architectures the single-feat align peak sits at moderate negative
α (mid-α regime, α ∈ [−10, −3]) and decays smoothly on both sides. SAE
arditi's frontier sits roughly +5 align above TXC at every grid point.
Coherence stays ≥97 throughout the mid-α band for both arches and only
collapses at |α| ≥ 100. The R1 frontier shape does not depend on
training step count (5k/10k/30k overlap within ±1 align across most α).

### Architectural takeaway

The SAE-vs-TXC gap is largest where the dictionary has to express a
*specific* causal direction. R1 single-feat with mid-α is "easy" — both
arches develop a clear EM feature whose decoder row aligns with the
organism's misalignment direction, and the gap is small (+4). R32
ext-α single-feat is "hard" — only SAE arditi develops a feature whose
direction is the R32 misalignment axis, and the gap blows up (+12.58).

This is consistent with the architectural difference: SAE arditi's
single-TopK constraint forces each token to commit to ≤k specific
features, which selects for features whose decoder rows are highly
specific causal directions. TXC k=100's denser per-token activations
(T=5 thresholds, more features per token) trade off causal specificity
for reconstruction fidelity. R1 has so much misalignment headroom that
the trade-off is invisible; R32 exposes it.

Bundle null result is consistent with this picture: the top causal
candidates by `screen_score` on R32 are *individually* causal, but their
decoder rows do not constructively add. R32's misalignment is not
"distributed" in the sense of "spread across many features that can be
reassembled" — it is genuinely lower-dimensional than R1's, just not
as cleanly captured by any single feature as R1's direction is.

### What is closed and what is open

**Closed**:

- Single-feat × steps × arch × organism × α-regime — all 8 cells, with
  SAE > TXC at every cell.
- Goal: clear medical-champion 58.47 on the harder Qwen-14B financial
  organism — cleared by +38.4 (R1) and +6.1 (R32) with high coh margins.
- Step-count axis: flat, no benefit beyond 5k SAE training steps for
  this dictionary geometry.
- Bundle aggregation on R32 (k=30, top stage-2 survivors): does not
  beat single-feat. "Distributed misalignment" hypothesis falsified.

**Open** (deferred, not paper-critical):

- Bundle aggregation with a *different* selection criterion (e.g.
  Hessian-eigendirection-aligned features, or features filtered for
  decoder-row mutual orthogonality) — could in principle beat single-feat
  on R32, but no theoretical reason to expect it.
- TXC variants with k < 100 or different threshold counts — the R1
  gap at +4 is small enough that this could close it, but the R32 ext-α
  gap at +12.58 is unlikely to close without architectural changes.
- Cross-layer hookpoint: all results above are at layer 24 `resid_post`
  (the LoRA insertion layer). Earlier or later hookpoints are
  unexplored and are a natural follow-up.

### Reproduce

- Synthesis log with per-firing detail: [[em_nanda_synthesis]].
- Brief and project rules: [[EM_NANDA_BRIEF]].
- Plot scripts:
  - `experiments/em_features/plot_em_nanda_arch_organism_alpha.py` →
    `plots/em_nanda_arch_organism_alpha_table.png`.
  - Frontier plots: `plots/em_nanda_*frontier*.png`,
    `plots/em_nanda_step_count_trajectory.png`.
- Wang procedure: `experiments/em_features/run_wang_procedure.py` (with
  `--base_model` / `--subject_model` overrides for the Qwen-14B finance
  pivot, and `--batch_cells` / `--gen_batch_size` for batched α cells).
- Bundle frontier launcher: `experiments/em_features/frontier_sweep.py`
  (with `--base_model` / `--subject_model` overrides added in commit
  `30fc5af0`).
- SAE arditi training wrapper:
  `experiments/em_features/run_training_sae_arditi.py`.
- Config: `experiments/em_features/config_qwen14b.yaml`.
- Stage-4 final-frontier outputs (per-feat × α grid):
  - SAE arditi 5k: `…/em_nanda_sae_arditi_step5000_wang/stage4_final_frontier.json`
  - SAE arditi 30k: `…/em_nanda_sae_arditi_step30000_wang/stage4_final_frontier.json`
  - TXC k=100 R32 ext-α: `…/em_nanda_txc_paper_k100_step10000_wang_r32_extalpha/stage4_final_frontier.json`
  - Bundle k=30 R32: `…/em_nanda_bundle_r32/bundle30_frontier.json`
  - Other Wang outputs (SAE arditi 10k R1 + R32, TXC R1 5k/10k/30k, TXC
    R32 std-α) live on `h100_2` and have peak numbers archived in the
    closed table above.
