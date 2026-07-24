"""Probe-capacity diagnostic for the hedging-LEVEL Stage-2 panel.

Pre-registered in `card_stage2.md` § 6 ("run either way after the
panel; labelled post-hoc; kept OUT of the leaderboard"). The panel's
T = 16 column sits at n < p (1702 finite probe rows vs p = d_sae =
2048), so a T = 16 drop is ambiguous between probe capacity and
representation. This refits the slope8 probe on the SAME trained
checkpoints with

  - more probe data: n_windows 1024 → 8192, and
  - ridge instead of unregularized least squares,

and reports held-out r. If the extra data/regularization lifts the
T = 16 cells back toward the T = 4 level, the honest statement is that
the T = 16 column is probe-limited for every dense arch — not that
representations degrade there.

Cells: {txc_batchtopk_pre, txc_batchtopk_post, stacked_batchtopk} ×
T ∈ {4, 16} × seed 1 (the T = 4 cells are the in-panel control: if the
lift is the same size at T = 4, extra probe data is helping everywhere
and says nothing specific about T = 16).

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.confidence.probe_capacity
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

from temp_bench.core.cache import checkpoint_exists
from temp_bench.core.config import (
    compute_data_key,
    compute_train_key,
    load_arch,
    load_datasource,
)
from temp_bench.core.runner import _load_checkpoint
from temp_bench.core.schemas import TrainingConfig
from temp_bench.evals.synthetic_recovery import _sample_windows

from explorations.task_hunt.real_slope import ward_slope_real

HERE = Path(__file__).resolve().parent
DS = "ward_real_slope8_distill_l14"
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288
SEED = 1
CELLS = [(a, T) for a in ("txc_batchtopk_pre", "txc_batchtopk_post",
                          "stacked_batchtopk") for T in (4, 16)]
N_WINDOWS = (1024, 8192)          # panel setting, then the diagnostic


@torch.no_grad()
def _codes(model, x, lam, T, n_windows, seed=0):
    device = next(model.parameters()).device
    split = x.shape[0] // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
    pools = []
    for xa, la, s in ((x[:split], lam3[:split], seed),
                      (x[split:], lam3[split:], seed + 1)):
        win_x, _ = _sample_windows(xa, L=EVAL_L, n_windows=n_windows, seed=s)
        win_l, _ = _sample_windows(la, L=EVAL_L, n_windows=n_windows, seed=s)
        W = win_x.shape[0]
        n_tiles = EVAL_L // T
        zs = []
        tiles_all = win_x.reshape(W * n_tiles, T, x.shape[-1])
        for i in range(0, tiles_all.shape[0], 8192):
            zs.append(model.encode(tiles_all[i:i + 8192].float().to(device))
                      .reshape(-1, model.config.d_sae).float().cpu().numpy())
        z = np.concatenate(zs)
        t = win_l.reshape(W, n_tiles, T)[:, :, T - 1].reshape(-1).numpy()
        m = np.isfinite(t)
        pools.append((z[m], t[m]))
    return pools


def _fit(pools, ridge: bool):
    from sklearn.linear_model import LinearRegression, RidgeCV
    (z_tr, t_tr), (z_ev, t_ev) = pools
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = (RidgeCV(alphas=np.logspace(-1, 4, 12)) if ridge
               else LinearRegression()).fit(z_tr, t_tr)
        pred = reg.predict(z_ev)
    r = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return r, int(len(t_tr)), int(len(t_ev))


def main() -> None:
    data = ward_slope_real()
    lam = data.extra["lambda_labels"].float()
    data_spec = load_datasource(DS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"meta": {"ds": DS, "card": "card_stage2.md §6", "seed": SEED,
                    "off_leaderboard": True, "post_hoc": True},
           "cells": {}}
    for arch_name, T in CELLS:
        k = 8 * T if arch_name == "txc_batchtopk_post" else 8
        tcfg = TrainingConfig(
            n_steps=N_STEPS, batch_size=1024 // T,
            buffer_tokens=BUFFER_TOKENS,
            arch_hparams_override={"k_pos": k, "d_sae": 2048, "T": T})
        arch_spec = load_arch(arch_name, section="synthetic")
        arch_spec = arch_spec.model_copy(update={
            "hparams": {**arch_spec.hparams, **tcfg.arch_hparams_override}})
        train_key = compute_train_key(
            arch=arch_spec, seed=SEED, training_cfg=tcfg,
            data_key=compute_data_key(data_spec), section="synthetic")
        if not checkpoint_exists(train_key):
            raise FileNotFoundError(f"no checkpoint {arch_name}/T{T}")
        model = _load_checkpoint(arch_spec, train_key, data_spec).to(dev)
        rec = {}
        for nw in N_WINDOWS:
            pools = _codes(model, data.x, lam, T, nw)
            for ridge in (False, True):
                r, ntr, nev = _fit(pools, ridge)
                rec[f"nw{nw}_{'ridge' if ridge else 'ols'}"] = {
                    "r": r, "n_train": ntr, "n_eval": nev}
        out["cells"][f"{arch_name}/T{T}/s{SEED}"] = rec
        print(f"[{arch_name}/T{T}] " + "  ".join(
            f"{k2}={v['r']:.3f}(n={v['n_train']})" for k2, v in rec.items()),
            flush=True)
        del model
    dst = HERE / "results" / "stage2_probe_capacity.json"
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
