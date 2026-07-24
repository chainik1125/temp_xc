"""Stage-2 shuffle-IMMUNITY receipt — hedging-LEVEL panel (card § 10.1).

Pre-registered diagnostic, OFF-leaderboard, on the panel's OWN trained
checkpoints: encode eval/train tiles with an anchor-fixed within-tile
context shuffle (permute tile slots 0..T−2 per row, leading edge fixed
at T−1, seeded rng 1234 — the Stage-1 screen's convention), refit +
evaluate the slope8 probe on the shuffled codes, and compare against
the clean-code recovery per (arch, T, seed).

**Frozen prediction (card § 10.1): recovery is retained — the shuffled
cell keeps more than half of that cell's (clean window − best token
arch) margin, per seed-mean.** A larger degradation FALSIFIES the
aggregation framing (order would matter after all).

Cells: {txc_batchtopk_pre (k=8), txc_batchtopk_post (k=8·T)} ×
T ∈ {8, 16} × seeds {1, 2, 42}. Checkpoints are located through the
runner's own key machinery (read-only; no training, no leaderboard
writes — a missing checkpoint is an error, never a retrain).

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.confidence.shuffle_receipt_stage2
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
N_WINDOWS = 1024
SHUF_SEED = 1234
N_STEPS = 8_000
BUFFER_TOKENS = 524_288
CELLS = [(arch, T, seed)
         for arch in ("txc_batchtopk_pre", "txc_batchtopk_post")
         for T in (8, 16)
         for seed in (1, 2, 42)]


def _shuffle_tiles(tiles: torch.Tensor, rng: np.random.Generator):
    """Anchor-fixed context shuffle: permute slots 0..T−2 per row."""
    n, T, _ = tiles.shape
    if T <= 2:
        return tiles.clone()
    perms = rng.permuted(np.tile(np.arange(T - 1), (n, 1)), axis=1)
    out = tiles.clone()
    out[:, :T - 1] = tiles[torch.arange(n)[:, None], torch.from_numpy(perms)]
    return out


@torch.no_grad()
def _probe(model, x, lam, T, *, shuffled: bool, seed=0):
    """`lambda_recovery._train_lambda_probe`, with an optional shuffle
    transform on the tiles before encoding (train AND eval — the screen's
    shuffled-probe convention)."""
    from sklearn.linear_model import LinearRegression
    device = next(model.parameters()).device
    split = x.shape[0] // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
    pools = []
    for xa, la, s in ((x[:split], lam3[:split], seed),
                      (x[split:], lam3[split:], seed + 1)):
        win_x, _ = _sample_windows(xa, L=EVAL_L, n_windows=N_WINDOWS, seed=s)
        win_l, _ = _sample_windows(la, L=EVAL_L, n_windows=N_WINDOWS, seed=s)
        W = win_x.shape[0]
        n_tiles = EVAL_L // T
        tiles = win_x.reshape(W * n_tiles, T, x.shape[-1]).float()
        if shuffled:
            rng = np.random.default_rng(SHUF_SEED)
            tiles = _shuffle_tiles(tiles, rng)
        z = model.encode(tiles.to(device)).reshape(W * n_tiles, -1)
        t = win_l.reshape(W, n_tiles, T)[:, :, T - 1].reshape(-1).numpy()
        m = np.isfinite(t)
        pools.append((z.detach().float().cpu().numpy()[m], t[m]))
    (z_tr, t_tr), (z_ev, t_ev) = pools
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression().fit(z_tr, t_tr)
        pred = reg.predict(z_ev)
    return float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0


def main() -> None:
    data = ward_slope_real()
    lam = data.extra["lambda_labels"].float()
    data_spec = load_datasource(DS)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"meta": {"ds": DS, "convention": "anchor-fixed context shuffle, "
                    f"rng {SHUF_SEED}; probe refit on shuffled codes",
                    "card": "card_stage2.md §10.1", "off_leaderboard": True},
           "cells": {}}
    for arch_name, T, seed in CELLS:
        k = 8 * T if arch_name == "txc_batchtopk_post" else 8
        tcfg = TrainingConfig(
            n_steps=N_STEPS, batch_size=1024 // T,
            buffer_tokens=BUFFER_TOKENS,
            arch_hparams_override={"k_pos": k, "d_sae": 2048, "T": T})
        arch_spec = load_arch(arch_name, section="synthetic")
        arch_spec = arch_spec.model_copy(update={
            "hparams": {**arch_spec.hparams, **tcfg.arch_hparams_override}})
        data_key = compute_data_key(data_spec)
        train_key = compute_train_key(arch=arch_spec, seed=seed,
                                      training_cfg=tcfg, data_key=data_key,
                                      section="synthetic")
        if not checkpoint_exists(train_key):
            raise FileNotFoundError(
                f"no checkpoint for {arch_name}/T{T}/s{seed} "
                f"({train_key}) — run the panel first")
        model = _load_checkpoint(arch_spec, train_key, data_spec).to(dev)
        clean = _probe(model, data.x, lam, T, shuffled=False)
        shuf = _probe(model, data.x, lam, T, shuffled=True)
        out["cells"][f"{arch_name}/T{T}/s{seed}"] = {
            "clean_r": clean, "shuf_r": shuf, "k_pos": k,
            "train_key": train_key}
        print(f"[{arch_name}/T{T}/s{seed}] clean={clean:.3f} "
              f"shuf={shuf:.3f} drop={clean - shuf:+.3f}", flush=True)
        del model
    dst = HERE / "results" / "stage2_shuffle_receipt.json"
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
