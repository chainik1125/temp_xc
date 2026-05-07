# C2 — Synthetic coupled features

Per-component scripts for the coupled-feature gAUC study at multiple T.
See `docs/components/c2.md` for setup.

## Files (TODO)

- `data.py` — coupled HMM emission generator ($K$ hidden, $M$ emissions, OR-gate)
- `train.py` — train one (arch, T, k, seed) cell
- `eval.py` — eAUC, gAUC, single-latent Pearson, ridge-probe R²
- `run.py` — sweep over (arch, T, k, seed)
- `plot.py` — gAUC vs k vs T heatmap, scatter plots

## Smoke-test command

```bash
cd purified && TQDM_DISABLE=1 .venv/bin/python -m experiments.c2_synthetic_coupled.run \
    --arch txc_pro --T 5 --k 2 --seeds 42 --steps 1000
```
