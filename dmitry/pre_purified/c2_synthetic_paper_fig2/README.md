# Fig 2 (synthetic_overview): inputs + outputs

This directory pins the data and rendered panels behind **Figure 2**
(`fig:synthetic_overview`) of the Temporal Crosscoders paper. The figure
reports two synthetic evaluations side-by-side:

- **Setup B — Denoising** (`fig:setup_b_denoising`): independent two-state
  Markov chains with noisy emissions; metric is single-latent local-vs-global
  correlation + linear-probe $R^2_{\mathrm{global}}$.
- **Setup D — Coupling** (`fig:setup_d`): coupled-noisy HMM at maximum
  overlap (`n_parents = 10`); metric is decoder gAUC vs eAUC.

The leftmost subfig (`fig:syn_cartoon` / `c2_synth_global_headline.png`) is
the per-arch bar chart summarising both setups in one number per architecture.

## Files

```
data/
    denoising_probe_results.json   # Setup B per-cell (299 records)
    setup_d_leaderboard.jsonl      # Setup D per-cell (804 rows)
figs/
    c2_synth_global_headline.png   # Subfig (a) — bar chart summary
    c2_setup_b_singlelatent.png    # Subfig (b) — Setup B scatter
    c2_setup_d_scatter_clean.png   # Subfig (c) — Setup D scatter
make_c2_synth_panels.py            # Renderer (consumes data/, writes figs/)
```

## Schemas

### `denoising_probe_results.json` (Setup B)

Flat list of 299 per-cell records, one per `(arch_name, t_label, k_pos, seed)`.
Each record carries:

- `train_key`, `arch_name`, `t_label`, `k_pos`, `seed`, `T_win`
- `sl_mean_local`, `sl_mean_global`, `sl_ratio`, `sl_denoising_frac`
  — single-latent metrics
- `lp_mean_local_r2`, `lp_mean_global_r2`, `lp_ratio` — linear-probe R²

### `setup_d_leaderboard.jsonl` (Setup D)

804-row leaderboard slice filtered to
`toy_coupled_noisy_K10_M20_d256_pB05_*` and `smoke=false`. Each row carries
`arch`, `seed`, `eval_cfg.k_pos`, `eval_cfg.t_label`, `eval_cfg.n_parents`,
`metrics.eauc`, `metrics.gauc`, etc.

## Renderer

```bash
python make_c2_synth_panels.py
```

Reads `data/`, writes `figs/`. The renderer also lives in the paper repo
under `scripts/make_c2_synth_panels.py`; this copy is provided for
self-contained reproduction.

## Provenance

- Setup B data sourced from `purified/experiments/c1_noisy_filler/denoising_probe_results.json` (origin/final).
- Setup D data is the leaderboard slice pulled from
  `a40_synth_3gpu5:/workspace/temp_xc-final/purified/results/leaderboard.jsonl`.
