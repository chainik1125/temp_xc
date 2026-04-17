---
author: Aniket
date: 2026-04-13
tags:
  - proposal
  - complete
---

## Sprint Feature Geometry Plan — 2-Cell Shuffle Experiment

Initial experimental design for the NeurIPS/ICML exploration sprint.
Tests whether the temporal crosscoder (TXCDRv2) discovers genuinely
temporal feature structure by comparing decoder direction geometry
across shuffled and unshuffled conditions on two (model, dataset) pairs.

Results in [[experiments/sprint_feature_geometry/summary|summary]].

## Research question

Does the TXCDR's feature geometry — the clustering pattern of its
learned decoder directions — depend on temporal structure in the data,
or is it an architectural artifact of the shared-latent window?

## Experimental matrix

| | unshuffled (natural order) | shuffled (temporal order destroyed) |
|---|---|---|
| **Gemma 2B + FineWeb** (web text) | replication of Andre's feature map result | temporal control |
| **DeepSeek-R1-Distill-8B + GSM8K** (reasoning traces) | extension to thinking model | temporal control |

## Architectures

TXCDRv2 (crosscoder) with `k=100, T=5`, 8× expansion factor. Baselines
trained alongside in the same sweep: TopKSAE (single-token), Stacked
SAE T=5 (independent per-position SAEs).

## Subject models

- `gemma-2-2b` — d_model=2304, d_sae=18,432, layer 13 (mid-residual),
  24,000 FineWeb sequences × 32 tokens, forward-mode activation caching.
- `deepseek-r1-distill-llama-8b` — d_model=4096, d_sae=32,768, layer 12
  (~37% depth), 1,000 GSM8K reasoning traces × 1,024 generated tokens,
  generate-mode activation caching with `<think>` prompt template.

## Shuffle control

`shuffle_within_sequence=True` randomly permutes the token order within
each T-token window before feeding it to the architecture. Destroys
local temporal structure while preserving per-token marginal
distributions. Applied at data-load time so both training and
evaluation see the same shuffled data.

## Clustering pipeline

Decoder directions (d_sae vectors of dimension d_model) averaged across
T positions, L2-normalized → PCA to 50 components → UMAP to 2D (cosine
metric, n_neighbors=15) → KMeans with 20 clusters. Matches Andre's
original analysis.

## Autointerp

Top 30 features per checkpoint labeled via Claude Haiku 4.5 with
one-sentence explanations (initial pass). Scoring disabled in this
phase — HypotheSAEs-style fidelity scoring deferred to the paper
version. See [[experiments/sprint_5k_autointerp/plan|5k autointerp scale-up]]
for the paper-grade labeling pass that follows.

## Predicted direction (pre-registered)

Under the hypothesis "temporal sensitivity scales with the temporal
richness of the data":

| metric (direction of Δ = unshuf − shuf) | Gemma + FineWeb | DeepSeek + GSM8K |
|---|---|---|
| cluster separation (visual, silhouette) | small | large |
| dominant-cluster entropy drop | small | large |
| mean auto-MI across lags | ≈ 0 | clearly positive |

Rationale: FineWeb at `seq_len=32` has weak local temporal structure
(natural language, but only T=5 windows visible), so TXCDR's shared-z
inductive bias has little to latch onto. GSM8K reasoning traces at
`seq_len=1024` are highly temporal (multi-step arithmetic, backtracking,
case analysis), so shuffling should break coherent features visibly.

## Andre-replication criterion

This experiment *replicates* Andre's Gemma+FineWeb feature-map run
(`docs/andre/nlp_feature_map.md`) as its unshuffled step1 cell. We
count replication as successful if **at least 2 of the top 3 cluster
themes** from Andre's 20-cluster KMeans labels appear in our cluster
summaries with matching dominant concepts (e.g., "event/time",
"product/brand", "location"). Cluster counts and sizes may differ
since our clustering re-runs with a fresh UMAP seed.

## Follow-ups

- [[experiments/sprint_5k_autointerp/plan|5k autointerp scale-up]] — label every major cluster.
- [[experiments/sprint_coding_dataset/plan|coding-dataset 2×3 extension]] — H3 rule-out cell (Gemma+Stack).
