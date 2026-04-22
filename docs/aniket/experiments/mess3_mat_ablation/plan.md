---
author: Aniket Deshpande
date: 2026-04-21
tags:
  - proposal
  - pre-registration
  - mess3
  - matryoshka
---

## Mess3 Mat-TopK-SAE ablation — experiment plan

**Purpose**: complete the missing `no-window × matryoshka` cell in
Dmitry's `separation_scaling` 2×2, to disambiguate whether the
MatryoshkaTXC (MatTXC) advantage on Mess3 is driven by:
- **the matryoshka penalty alone**, or
- **the interaction of matryoshka with the temporal window**.

**Source**: Dmitry's `experiments/separation_scaling/` on `origin/dmitry`
(commit `16452d5`). Copy-paste port of his vendored code (no edits to
his branch).

## 1. Background

Dmitry benchmarked 7 architectures on Mess3-generator-driven
transformer residual streams across temporal-separation δ ∈ {0, 0.05,
0.10, 0.15, 0.20}. Headline result (gap-recovery axis, R²/R²_max):

| δ | τ | R²_max | TopK SAE | TXC | MatTXC |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.60 | 0.45 | ~0.14 | ~0.46 | **~0.81** (best single feature, best component-aligned) |

The 0.81 is "share of the Bayes-optimal recoverable information captured
by the single best-aligned feature". The 2×2 is incomplete:

|  | no window | window |
|---|---|---|
| no matryoshka | TopK SAE ≈ 0.14 | TXC ≈ 0.46 |
| matryoshka | **? (this experiment)** | MatTXC ≈ 0.81 |

Web-claude-framed question: does matryoshka work on its own (single-position),
or does it only work when paired with the temporal window?

**Clarification from Dmitry (via Slack):** MatTXC's 0.81 is achieved by
*three distinct argmax features*, one per Mess3 component, averaged —
not a single feature doing triple duty. So the correct success criterion
for the ablation is "does MatryoshkaSAE recover a 3-feature
component-indicator basis like MatTXC does, or does it smear component
identity across many features like TopK SAE?" The quantitative metric
stays the same (argmax feature per component, averaged R²) — we just
reinterpret the result as geometry recovery, not single-feature
monosemanticity.

## 2. Hypothesis + predictions

The missing cell's number decides the paper story.

| prediction | MatryoshkaSAE (no window) R²/R²_max at δ=0.20 | interpretation |
|---|---|---|
| **H0 (matryoshka does the work)** | ≥ 0.70 | Matryoshka penalty alone recovers the component-indicator basis. Temporal window is redundant. Paper story collapses to "Matryoshka finds the most reducible latent in any architecture on Mess3." |
| **H1 (window matters for geometry)** | ≤ 0.30 | Matryoshka alone smears component identity like TopK does. Temporal window is the necessary ingredient for the clean indicator basis. Paper story becomes "Matryoshka × temporal-window compositionally enables geometry recovery." |
| **H2 (both contribute)** | ∈ (0.30, 0.70) | Both axes contribute separately. Less crisp story but still publishable — "each of matryoshka penalty and temporal window contribute ~half the gap; best result requires both." |

H1 is the outcome most consistent with Dmitry's current framing. H0
would force a framing pivot before NeurIPS. H2 is the noisy-world default.

Secondary metric: **number of features per component.** For each Mess3
component c, find the argmax feature f_c. If |{f_c}| = 3 (three distinct
features), the arch recovered the indicator basis. If |{f_c}| = 1 or 2
(same feature wins for multiple components), the arch smeared identity.
MatTXC's reported behavior is |{f_c}| = 3 at δ=0.20; TopK's is <3.

## 3. Experimental design

### 2×2 grid with shared transformer and eval protocol

|  | single-position (no window) | temporal window (W=30 or W=60) |
|---|---|---|
| **plain TopK activation** | TopK SAE ✓ (reproduced) | TXC ✓ (reproduced) |
| **matryoshka penalty + batch_topk** | **MatryoshkaSAE ← new cell** | MatTXC ✓ (reproduced) |

All four cells share:
- Same transformer (3-layer, d_model=64, ctx=128, 20k steps, seed=42)
- Same Mess3 generator (nonergodic, mess3_shared, r=0)
- Same probe dataset + eval seed
- Same R²_max computation (Bayes-optimal forward filter)
- Single-component best-R² metric (Dmitry's primary) + per-component
  argmax-feature diversity (secondary)

### δ sweep

Full **δ ∈ {0, 0.05, 0.10, 0.15, 0.20}**, matching Dmitry's
`config.yaml` exactly so numbers are directly comparable cell-by-cell.
The decision signal is at δ ≥ 0.15 where separations matter (per
Dmitry's existing table), but running the full sweep means every one of
our cells can be laid next to his published number in the same row —
no "but at a different δ" caveat in the writeup.

### Compute

Per cell per arch: single-position matryoshka SAE training is <5 min
on CPU / seconds on GPU. Transformer training per cell: ~10 min CPU /
~2 min GPU if `transformer.pt` needs to be retrained (cache will be
reused if Dmitry's results-tree is available on the pod).

**Total**: 5 cells × 4 arches = 20 runs.
- ~2 h CPU-only / ~30 min on a single H100 if transformers need retraining.
- ~1 h CPU / ~10 min GPU if transformer caches are reused.

## 4. Architecture config for the new cell

Matches `MatryoshkaSAE` class in
`experiments/separation_scaling/vendor/src/sae_day/sae.py`. Same
hyperparameters as `MatTXC` minus the temporal dimension:

```yaml
- name: "MatryoshkaSAE (no-window)"
  family: matsae       # new family, dispatched to evaluate_matsae_on_activations
  kwargs:
    dict_size: 128
    matryoshka_widths: [8, 16, 32, 64, 128]
    k: 1
    inner_weight: 10.0
    sae_steps: 1500    # matches MatTXC's 1500 temporal_steps
    n_sequences: 400
    seq_len: 128
```

**Note on the activation-function confound** (flagged by web-claude):
Dmitry's `MatryoshkaSAE` class uses `batch_topk` internally, whereas
`TopKSAE` uses plain per-sample TopK. So "MatryoshkaSAE - TopK SAE"
conflates the matryoshka penalty and the batch-topk activation.

Mitigation: we ALSO run a single-position `TemporalBatchTopKSAE`
(family `tsae` in Dmitry's driver, with `temporal: false`) as an
auxiliary control. This isolates batch-topk alone:

|  | per-sample TopK | batch-topk | batch-topk + matryoshka |
|---|---|---|---|
| no window | TopK SAE | T-BatchTopK (no temporal) | MatryoshkaSAE |
| window | TXC | — | MatTXC |

If MatryoshkaSAE >> T-BatchTopK (no temporal), matryoshka is the
substantive ingredient. If they're close, the "matryoshka effect" might
actually be the activation function.

## 5. Protocol

1. Pull Dmitry's `experiments/separation_scaling/vendor/` via `git archive origin/dmitry` into our experiment dir. No changes to his branch.
2. Copy his `run_driver.py` with a one-function extension: `matsae` family dispatch → new `evaluate_matsae_on_activations`.
3. Reuse his transformer training / caching exactly — same seed, same config.
4. Evaluate using his `evaluate_representation` helper so metrics are directly comparable to his published numbers.
5. Plot gap-recovery (R²/R²_max) and raw R² per cell, saving PDFs.

## 6. What produces the headline figure

Two figures, both PDFs at `plots/`:

- **`fig1_gap_recovery_2x2.pdf`**: four lines across δ ∈ {0.15, 0.20}:
  - TopK SAE (blue, dashed)
  - TXC (blue, solid)
  - MatryoshkaSAE (orange, dashed) ← our new cell
  - MatTXC (orange, solid)
  - Horizontal dashed reference at R²/R²_max = 1.0 (Bayes ceiling)

- **`fig2_feature_diversity.pdf`**: for each (arch, δ), bar chart of
  |{argmax feature per component}|. Ideal = 3 (distinct indicator).
  Reads off whether each arch recovers the indicator basis or smears.

## 7. Decision rules

After results land:

- If `MatryoshkaSAE ≥ 0.70` AND feature diversity = 3 → H0 confirmed. Slack Dmitry same day with pivot suggestion for the paper narrative.
- If `MatryoshkaSAE ≤ 0.30` AND feature diversity < 3 → H1 confirmed. Strongest outcome — publishable compositional-win claim.
- Otherwise → H2. Useful but muddier; expand with the T-BatchTopK (no temporal) control to decompose further.

## 8. Reproducibility

Everything lives at `experiments/mess3_mat_ablation/`. Run with:

```bash
cd experiments/mess3_mat_ablation
bash setup_vendor.sh              # pulls Dmitry's vendored code via git archive
bash run_ablation.sh              # runs the 2×4 grid + plot
```

No modifications to `origin/dmitry`; our branch only adds the
`mess3_mat_ablation/` directory. All Dmitry-branch files stay read-only.
