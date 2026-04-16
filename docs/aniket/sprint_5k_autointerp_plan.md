---
author: Aniket
date: 2026-04-16
tags:
  - proposal
  - in-progress
---

## Sprint 5k Autointerp Plan — Gated on TopKSAE Control

Next step in the [[sprint_feature_geometry_results|sprint feature geometry
sprint]]: scale autointerp from 30 to ~5000 features per TXCDR checkpoint
so we can label every major cluster in the step2 DeepSeek+GSM8K feature
map. The 30-feature run produced the sprint's current plots; 5000 is
what tells us *what* the isolated right-side island actually is.

## Gate — TopKSAE control must pass first

The `txc-fmap-sae` sbatch (`scripts/trillium_sbatch_fmap_sae_control.sh`)
produces 4 PNGs in `reports/sae-control-deepseek/`:

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

## 5k autointerp — three commands

All three are bash-only wrappers already on `aniket`. Run in order.

### Step 1 — Scan (sbatch, GPU node, ~30-60 min)

```bash
bash scripts/trillium_scan_5k_sbatch.sh
```

Runs `TopKFinder` over all 4 crosscoder checkpoints
(step1-{un,}shuffled + step2-{un,}shuffled) with `--top-features 5000`.
Writes `feat_*.json` files containing `top_texts` and
`top_activations` but empty `explanation` fields. GPU-bound, no network
required — safe inside a compute allocation.

Outputs land in:

- `reports/step1-gemma-replication/autointerp/step1-{un,}shuffled/feat_*.json`
- `reports/step2-deepseek-reasoning/autointerp/step2-{un,}shuffled/feat_*.json`

### Step 2 — Explain (login node, ~\$80, after scan finishes)

```bash
bash scripts/trillium_explain_5k.sh
```

Reads the `feat_*.json` files from step 1, filters to those with
empty explanations, and calls Claude Haiku 4.5 to fill each in. Runs on
the *login node* (needs outbound HTTPS; ~zero CPU since it's I/O-bound
on API responses). `OMP/MKL/OPENBLAS` threads pinned to 1 to stay
inside the login-node CPU-time cap.

Cost: ~5000 features × 4 checkpoints × \$0.004/feature ≈ \$80.
`ResultsStore` has resume support — if the job gets killed, just
re-run and it picks up where it left off.

### Step 3 — Feature map (sbatch, GPU node, after explain finishes)

```bash
bash scripts/trillium_sbatch_fmap_5k.sh
```

Reruns `feature_map.py` on each checkpoint, now reading the ~5000
labeled `feat_*.json` files for hover-text labels. Produces labeled
interactive HTMLs + PNGs in the same report dirs.

After this, pull the reports dir to the laptop:

```bash
bash scripts/fetch_from_trillium.sh reports
```

## Prereqs — environment

- `ANTHROPIC_API_KEY` must be set in the login-node env before step 2.
- Trillium aniket-branch must be clean and up to date before step 1
  (the scan script runs `git pull origin aniket` at the top).
- All 4 crosscoder checkpoints must exist under
  `results/nlp/step{1,2}-{un,}shuffled/ckpts/crosscoder__*.pt`
  (they do — produced by the earlier sweep).

## Why 5k specifically

Current coverage is 30 / 32,768 features ≈ 0.09%. Clusters in the
step2-unshuffled map are sized ~4k features (cluster 6, n=4,208).
At 30 labels there's no chance of hitting *any* cluster with a
representative sample. At 5000 labels we get ~15% per-cluster coverage,
which is enough to read off the dominant concept in each.

## Files

- Gate script — `scripts/trillium_sbatch_fmap_sae_control.sh`
- Step 1 — `scripts/trillium_scan_5k_sbatch.sh`
- Step 2 — `scripts/trillium_explain_5k.sh`
- Step 3 — `scripts/trillium_sbatch_fmap_5k.sh`
- Fetch — `scripts/fetch_from_trillium.sh`

Related: [[sprint_feature_geometry_results]], [[SPRINT_PIPELINE]]
