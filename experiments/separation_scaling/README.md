# separation_scaling

How do SAE/crosscoder-family architectures scale as we grow the generator's
temporal-structure demand? Benchmarks 7 architectures across δ ∈ {0, 0.05, 0.1,
0.15, 0.2} at r=0 (shared mess3 vocab, no tag tokens), using a YAML-driven
runner.

Archs: TopK SAE, TXC, MatryoshkaTXC, MultiLayerCrosscoder, TFA (Han), TFA-pos,
Temporal BatchTopK SAE. Plus dense linear and (early-stopped) dense MLP probes
on the residual stream.

**This folder is a copy-paste port from `sae_day`** — all of the sae_day
dependency graph (NonergodicGenerator, SAE classes, TFA, driver, helpers)
lives under `vendor/` in this folder to keep the experiment self-contained.
No code in `vendor/` should be edited in-place; it's a frozen snapshot.

## Install

```bash
cd <repo-root>                    # temp_xc root
uv sync --extra separation-scaling
```

This installs `torch`, `transformer_lens`, `jax`/`jaxlib`, `pyyaml`,
`scikit-learn`, `matplotlib`, and the pinned `simplexity` PR172 fork (all
as extras, so `uv sync` without the flag still gives you the lean temp_xc
core env).

## Reproduce

```bash
# From <repo-root>:
cd experiments/separation_scaling
PYTHONPATH="vendor/src:vendor/experiments/standard_hmm:vendor/experiments/transformer_standard_hmm:vendor/experiments/transformer_nonergodic" \
    uv run python -m sae_day.run_driver --config config.yaml
PYTHONPATH="vendor/src:vendor/experiments/standard_hmm:vendor/experiments/transformer_standard_hmm" \
    uv run python plot_separation_scaling.py
```

Runtime ~95 min on an A40 (5 cells × ~19 min). The driver saves
`transformer.pt` + `training_params.json` per cell; re-running with
`transformer.load_if_exists: true` (already set in `config.yaml`) loads the
cached transformers instead of retraining.

## Layout

```
experiments/separation_scaling/
├── README.md                 (this file)
├── NOTES_r2_ceiling.md       (why R²_max is the ceiling — derivation + worked example)
├── config.yaml               (5-cell δ sweep, 7 archs, 2 dense probes)
├── compute_r2_ceiling.py     (Bayes-optimal R² ceiling from forward filter)
├── run_window_probe.py       (dense linear + logistic probes at window W)
├── run_ridge_sweep.py        (ridge λ sweep for W ∈ {20, 30, 60})
├── plot_separation_scaling.py
├── plot_gap_recovery.py
├── plot_probe_window_sweep.py
├── results/                  (per-cell + combined JSONs)
├── plots/                    (all generated figures)
├── tables/                   (markdown summaries)
├── logs/run.log              (reference training log from sae_day)
└── vendor/                   (frozen sae_day dep graph; do not edit)
    ├── src/sae_day/          (NonergodicGenerator, sae.py, tfa.py, data.py, run_driver.py, simplexity_standard_hmm.py)
    └── experiments/
        ├── standard_hmm/     (ARCH_CONFIGS + fit_linear_probe_r2 + evaluate_representation)
        ├── transformer_standard_hmm/  (evaluate_*_on_activations + train_transformer + extract helpers)
        └── transformer_nonergodic/    (compute_tau_for_generators — forward filter for R² ceiling)
```

## Key outputs

- `plots/separation_scaling.png` — mean per-component best R² ± min/max error
  bars for all 7 archs + dense linear + dense MLP probes, with τ(δ) overlaid
  on a secondary axis.
- `tables/separation_scaling.md` — raw R² numbers per cell.
- `results/cell_delta_*/results.json` — full per-cell output (per-arch metrics,
  transformer loss, probe results).
- `results/combined.json` — combined summary of all cells.
- `logs/run.log` — training log.

## Headline

At δ=0.20 (τ=0.60), MatTXC recovers best_single=0.87 on its best component vs
TopK's 0.14 — a 0.73 gap. TFA and TFA-pos come close to TXC at high δ
(≈0.38 vs 0.42) but under-perform crosscoders at intermediate δ=0.15 (0.10,
0.16 vs TXC 0.49). See `tables/separation_scaling.md`.
