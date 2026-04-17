---
author: Aniket
date: 2026-04-16
tags:
  - proposal
  - complete
---

## Sprint 5k Autointerp Plan — Completed

Next step in the [[experiments/sprint_feature_geometry/summary|sprint feature geometry
sprint]]: scale autointerp from 30 to ~5000 features per TXCDR checkpoint
so we can label every major cluster in the step2 DeepSeek+GSM8K feature
map. The 30-feature run produced the sprint's early plots; 5000 is
what tells us *what* the isolated right-side island actually is.

## Gate — TopKSAE control must pass first

The TopKSAE control (`scripts/runpod_fmap_sae_control.sh`) produces 4
PNGs in `reports/sae-control-deepseek/`:

- `sae-deepseek-unshuffled.png` — TopKSAE baseline, natural order
- `sae-deepseek-shuffled.png` — TopKSAE baseline, shuffled
- `txcdr-deepseek-unshuffled.png` — TXCDR (side-by-side reference)
- `txcdr-deepseek-shuffled.png` — TXCDR shuffled

**Decision rule:**

- *TopKSAE diffuse, TXCDR structured* → claim holds (structure is a
  property of TXCDR's temporal inductive bias, not the data/layer).
  **Proceed with 5k autointerp below.**
- *TopKSAE also shows the isolated island* → claim collapses. **Do not
  spend the ~\$80.** Structure is data-driven; revise the writeup.

Quantitative backstop: once
[[experiments/sprint_coding_dataset/plan|the backfill metrics]] land,
prefer a numeric threshold — *claim holds if silhouette(TXCDR) −
silhouette(TopKSAE) > 0.05 on DeepSeek+GSM8K at the same 20-cluster
KMeans*. The visual-diffuseness rule above is the quick-look fallback
when the quantitative numbers aren't ready yet.

## 5k autointerp — three commands

All three are bash-only wrappers. Run in order.

### Step 1 — Scan (~30-60 min on A40)

```bash
bash scripts/runpod_scan_5k.sh
```

Runs `TopKFinder` over all 4 crosscoder checkpoints
(step1-{un,}shuffled + step2-{un,}shuffled) with `--top-features 5000`.
Writes `feat_*.json` files containing `top_texts` and
`top_activations` but empty `explanation` fields. GPU-bound.

Outputs land in:

- `reports/step1-gemma-replication/autointerp/step1-{un,}shuffled/feat_*.json`
- `reports/step2-deepseek-reasoning/autointerp/step2-{un,}shuffled/feat_*.json`

### Step 2 — Explain (~\$80, after scan finishes)

```bash
bash scripts/runpod_explain_5k.sh
```

Reads the `feat_*.json` files from step 1, filters to those with
empty explanations, and calls Claude Haiku 4.5 to fill each in.
Network-bound on `api.anthropic.com`; near-zero CPU.

Cost: ~5000 features × 4 checkpoints × \$0.004/feature ≈ \$80.
`ResultsStore` has resume support — if the job gets killed, just
re-run and it picks up where it left off.

### Step 3 — Feature map (after explain finishes)

```bash
bash scripts/runpod_fmap_5k.sh
```

Reruns `feature_map.py` on each checkpoint, now reading the ~5000
labeled `feat_*.json` files for hover-text labels. Produces labeled
interactive HTMLs + PNGs in the same report dirs.

## Prereqs — environment

- `ANTHROPIC_API_KEY` must be set in the env before step 2.
- `aniket` branch must be clean and up to date before step 1.
- All 4 crosscoder checkpoints must exist under
  `results/nlp/step{1,2}-{un,}shuffled/ckpts/crosscoder__*.pt`
  (they do — produced by the earlier sweep).

## Why 5k specifically

Current coverage is 30 / 32,768 features ≈ 0.09%. Clusters in the
step2-unshuffled map are sized ~4k features (cluster 6, n=4,208).
At 30 labels there's no chance of hitting *any* cluster with a
representative sample. At 5000 labels we get ~15% per-cluster coverage,
which is enough to read off the dominant concept in each. Matches the
label-volume Andre used for his Gemma analysis (5,452 SAE labels),
placing this at paper-grade coverage.

## Files

- Gate script — `scripts/runpod_fmap_sae_control.sh`
- Step 1 — `scripts/runpod_scan_5k.sh`
- Step 2 — `scripts/runpod_explain_5k.sh`
- Step 3 — `scripts/runpod_fmap_5k.sh`

Related: [[experiments/sprint_feature_geometry/summary|feature-geometry results]], [[SPRINT_PIPELINE]]
