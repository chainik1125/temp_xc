---
author: aniket
date: 2026-02-25
tags:
  - proposal
  - complete
---

## Motivation

Dmitry flagged that our previous SAE experiments plot SAE-latent-by-token-embedding
confusion matrices instead of the feature-by-feature matrices used in the
"Sparse but Wrong" paper (Chanin et al., arXiv 2508.16560). To align with the
reference work, we reproduce Figure 2: a demonstration that when SAE L0 < true
L0, SAEs learn "sparse but wrong" features via feature hedging.

## What is Feature Hedging?

When an SAE is forced to use fewer active latents than the true number of
simultaneously active features (L0), it compensates by mixing correlated feature
directions into single decoder vectors. This improves reconstruction MSE but
sacrifices monosemanticity — each latent no longer corresponds to a single true
feature.

## Experiment Setup

### Part 1: 5-Feature Reproduction (Fig 2 from paper)

- **5 features**, hidden_dim=20, orthogonal feature directions
- **Firing probability = 0.4** per feature (true L0 = 2.0)
- **Correlation**: f0 positively correlated with f1-f4 at ρ=0.4
- Two TopK SAEs (d_sae=5, ground-truth initialization):
  - **k=2** (matching true L0) — expect clean diagonal
  - **k=1** (below true L0) — expect hedging (off-diagonal mixing)

### Part 2: 50-Feature Version

- **50 features**, hidden_dim=100, orthogonal feature directions
- **Firing probability = 0.22** (true L0 ≈ 11)
- **Random correlation matrix** (seed=42)
- TopK SAEs (d_sae=50) at k ∈ {5, 8, 11, 14, 17}
- All initialized at ground-truth features

## Expected Results

- **k matching true L0**: near-diagonal decoder-vs-features cosine similarity
- **k < true L0**: off-diagonal entries appear as correlated features get mixed
- **c_dec curve**: minimized near true L0, higher when k is too low or too high

## Output

All plots saved to `results/feature_hedging/`:

- `correlation_5feat.png` — 5×5 correlation heatmap
- `decoder_vs_features_5feat_k2.png` — clean diagonal (matching L0)
- `decoder_vs_features_5feat_k1.png` — hedging visible
- `decoder_vs_features_50feat_k*.png` — 50-feature heatmaps
- `cdec_vs_l0_50feat.png` — c_dec vs L0 curve
