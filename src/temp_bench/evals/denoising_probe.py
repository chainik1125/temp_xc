"""Denoising probe add-on — latent-level hidden-state recovery (§ 4).

The paper's Denoising headline is a linear probe of the HIDDEN chain state
on the full latent code (R²_global = 0.48, main.tex:707-712), following the
legacy ``experiments/c1_noisy_filler/denoising_probes.py`` protocol. The v2
``synthetic_recovery`` evaluator only reports direction-recovery (eauc) on
the markov bench — hidden and emission share directions there by
construction, so the denoising discrimination lives at the LATENT level and
needs this probe.

OPT-IN (house additive pattern, like ``lambda_probe_v2``): fires only when
``eval_cfg["denoising_probe"]`` is truthy AND the datasource exposes
``hidden_support``. Protocol version of the host evaluator is unchanged;
rows without the flag are byte-identical.

Protocol (faithful simplification of the legacy probe):
- The SAME fixed eval windows as the NMSE/sparsity block (length L, seed 0,
  1024 windows), tiled into L/T native sub-windows per arch.
- Per sub-window code z (eval-mode encode; window archs' shared code) and
  per sub-window targets = mean over the T positions of hidden_support
  (K dims) and of observed support (M dims).
- Closed-form ridge (α = 1.0, feature-standardised) per target dim,
  train/test split 80/20 BY SEQUENCE (leak-free; legacy used a window
  split).
- Report mean out-of-sample R² over target dims:
  ``lp_global_r2`` (hidden), ``lp_local_r2`` (observed emissions), and
  ``lp_ratio`` = global/local. Denoising = global approaching/exceeding
  local despite observing only noisy emissions.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _codes_and_targets(model, data, *, L: int, n_windows: int, seed: int,
                       batch: int = 256):
    from temp_bench.evals.synthetic_recovery import (_check_tileable,
                                                     _sample_windows)
    T = _check_tileable(model, L)
    windows, seq_idx = _sample_windows(data.x, L=L, n_windows=n_windows,
                                       seed=seed)
    # matching label windows
    def _label_windows(lab):
        # regenerate the same offsets deterministically
        rng = np.random.default_rng(seed)
        s_idx = rng.integers(0, data.x.shape[0], size=n_windows)
        offs = rng.integers(0, data.x.shape[1] - L + 1, size=n_windows)
        st = torch.from_numpy(s_idx).long().unsqueeze(1)
        pos = torch.from_numpy(offs).long().unsqueeze(1) + torch.arange(L)
        return lab[st, pos]                        # (n_windows, L, K)

    hid = _label_windows(data.hidden_support.float())
    obs = _label_windows(data.support.float()) if data.support is not None \
        else None

    n_tiles = L // T
    device = next(model.parameters()).device
    Z = []
    model.eval()
    for i in range(0, n_windows, batch):
        w = windows[i:i + batch].to(device)                    # (b, L, d)
        b = w.shape[0]
        tiles = w.reshape(b * n_tiles, T, -1)                  # (b*n, T, d)
        z = model.encode(tiles)
        if z.dim() == 3:                                       # (b*n, 1|T, s)
            z = z.mean(dim=1) if z.shape[1] != 1 else z.squeeze(1)
        Z.append(z.float().cpu())
    Z = torch.cat(Z)                                           # (nw*n_tiles, d_sae)

    def _tile_targets(lab):
        # (n_windows, L, K) -> (n_windows*n_tiles, K), mean over T positions
        nw, _, K = lab.shape
        return lab.reshape(nw, n_tiles, T, K).mean(dim=2).reshape(-1, K)

    tile_seq = np.repeat(seq_idx, n_tiles)
    return Z.numpy(), _tile_targets(hid).numpy(), \
        (_tile_targets(obs).numpy() if obs is not None else None), tile_seq


def _ridge_r2(X, Y, seq_ids, *, alpha: float = 1.0) -> float:
    """Mean out-of-sample R² over Y's columns; 80/20 split by sequence."""
    uniq = np.unique(seq_ids)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    test_seqs = set(uniq[: max(1, len(uniq) // 5)].tolist())
    te = np.isin(seq_ids, list(test_seqs))
    tr = ~te
    if tr.sum() < 10 or te.sum() < 10:
        return float("nan")
    Xtr, Xte = X[tr], X[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d, dtype=Xtr.dtype)
    W = np.linalg.solve(A, Xtr.T @ (Y[tr] - Y[tr].mean(0)))
    pred = Xte @ W + Y[tr].mean(0)
    resid = ((Y[te] - pred) ** 2).sum(0)
    denom = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
    r2 = 1.0 - resid / np.maximum(denom, 1e-8)
    return float(np.mean(r2))


def denoising_probe_metrics(model, data, *, eval_window_L: int,
                            n_windows: int = 1024,
                            seed: int = 0) -> dict[str, float]:
    Z, hid, obs, tile_seq = _codes_and_targets(
        model, data, L=eval_window_L, n_windows=n_windows, seed=seed)
    out = {"lp_global_r2": _ridge_r2(Z, hid, tile_seq)}
    if obs is not None:
        out["lp_local_r2"] = _ridge_r2(Z, obs, tile_seq)
        if out["lp_local_r2"] and not np.isnan(out["lp_local_r2"]):
            out["lp_ratio"] = float(
                out["lp_global_r2"] / max(out["lp_local_r2"], 1e-8))
    return out
