# mess3_mat_ablation

2×2 ablation of Dmitry's `separation_scaling` experiment. Fills the
missing **no-window × matryoshka** cell (`MatryoshkaSAE` single-position)
alongside the three published cells (TopK SAE, TXC, MatTXC).

**Pre-registration + predictions**:
`docs/aniket/experiments/mess3_mat_ablation/plan.md`

**Dmitry's experiment, read-only reference**:
`origin/dmitry`, `experiments/separation_scaling/`

## Reproduce

```bash
cd experiments/mess3_mat_ablation
bash setup_vendor.sh      # pulls Dmitry's vendored sae_day via git archive
bash run_ablation.sh      # runs the 5δ × 4arch grid + produces PDFs
```

First invocation takes ~30 min on a single H100 (including transformer
retrains); subsequent `load_if_exists: true` reuses cached `transformer.pt`
and drops to ~5 min.

## Outputs

- `results/cell_delta_*/results.json` — per-cell architecture metrics
  (schema matches Dmitry's `separation_scaling/results/*.json`).
- `plots/fig1_gap_recovery_2x2.pdf` — R²/R²_max gap recovery vs δ for
  the four arches; direct visual of the 2×2 design.
- `plots/fig2_feature_diversity.pdf` — |{argmax feature per component}|
  showing whether each arch recovers the C=3 indicator basis.
- `docs/aniket/experiments/mess3_mat_ablation/plots/*.pdf` — final
  copies of the above for the writeup.

## Key files

- `config_ablation.yaml` — arch + sweep config (5 δ × 4 archs).
- `evaluate_matsae.py` — new evaluator for `MatryoshkaSAE`,
  single-position; mirrors Dmitry's `evaluate_topk_on_activations`
  signature and return schema.
- `run_ablation.py` — wraps `sae_day.run_driver`, monkey-patches the
  `evaluate_one_arch` dispatch to route `family="matsae"` to our
  evaluator. No edits to Dmitry's vendored code.
- `vendor/` — byte-for-byte snapshot of Dmitry's vendored sae_day
  pipeline (populated by `setup_vendor.sh`, gitignored).

## Provenance

All upstream files under `vendor/` come from `origin/dmitry` at
the commit pinned in `vendor/COMMIT_SHA` (written by
`setup_vendor.sh`). No changes to his branch.
