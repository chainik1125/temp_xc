"""Multilane per-lane velocity recovery — FreqBench FB-2 add-on.

Like ``frequency_recovery`` / ``lambda_recovery``, this is NOT a routed
evaluator: :class:`SyntheticRecovery` calls :func:`multilane_metrics` whenever
the materialised data carries ``extra['lane_velocity_labels']`` (the
``toy_multilane_*`` datasources). For every other bench the block is a no-op,
so the § 4 metrics are byte-identical and ``protocol_version`` does not move.

Why ``frequency_metrics`` cannot serve: it reads ONE per-sequence velocity;
FB-2 has ``n_lanes`` simultaneous velocity latents that must each be probed
on the SAME shared per-tile code (the superposition question is precisely
whether one code exposes all three). Everything else follows the frequency
conventions: per-tile leading-edge labels, single-tile codes (never
concatenated — memorization-free), leak-free by-sequence split, LINEAR
(multinomial-logistic) probes only, per-lane periodogram oracle on the same
held-out raw tiles, shuffled-label chance floor.

Metrics (frozen card ``freqbench/cards/FB-2.md`` § 2):

- ``multilane_recovery`` (**headline**): mean over lanes of the per-lane
  normalized balanced accuracy ``(bal_k − chance) / (1 − chance)``.
- ``lane{k}_recovery`` / ``lane{k}_balacc`` / ``lane{k}_oracle`` — per lane;
  ``multilane_oracle`` / ``multilane_chance`` — lane means (oracle = per-lane
  periodogram via the true plane, the ML decoder under orthogonal planes).
- ``lane{k}_recall_c{c}`` / ``lane{k}_oracle_c{c}`` — the per-lane S(f).
- band decomposition (spectral archs): ``band{b}_ml_recovery`` — mean over
  lanes of the per-band code-slice probe.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch


@torch.no_grad()
def _tile_examples_lanes(model, win_x, win_y, T):
    """Tile length-L windows → (codes, raw tiles, per-lane leading-edge labels).

    ``win_x: (W, L, d_in)``, ``win_y: (W, L, n_lanes)`` → codes
    ``(W·(L/T), d_sae)``, raw tiles ``(W·(L/T), T, d_in)``, labels
    ``(W·(L/T), n_lanes)`` at each tile's leading edge.
    """
    device = next(model.parameters()).device
    W, L, d_in = win_x.shape
    n_lanes = win_y.shape[-1]
    n_tiles = L // T
    tiles = win_x.to(device, dtype=torch.float32).reshape(W * n_tiles, T, d_in)
    z = model.encode(tiles).reshape(W * n_tiles, -1).detach().float().cpu().numpy()
    y = win_y.reshape(W, n_tiles, T, n_lanes)[:, :, T - 1, :]
    y = y.reshape(W * n_tiles, n_lanes).numpy().astype(np.int64)
    return z, tiles.detach().float().cpu().numpy(), y


def _lane_periodogram_pred(x_tiles, plane, omega, M):
    """Per-lane ML: project onto the lane plane, argmax |DFT at f=Y/M| over Ω."""
    proj = x_tiles @ plane                                       # (N, T, 2)
    c = proj[..., 0] + 1j * proj[..., 1]
    t = np.arange(x_tiles.shape[1])
    basis = np.exp(-2j * np.pi * np.asarray(omega, dtype=np.float64)[:, None]
                   * t[None, :] / M)                             # (|Ω|, T)
    return (np.abs(c @ basis.T)).argmax(axis=1)


def multilane_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0, max_rows: int = 30_000,
) -> dict[str, float]:
    """Per-lane velocity recovery for multilane superposition data."""
    from temp_bench.evals.frequency_recovery import (
        _logistic_probe,
        _per_class_recall,
    )
    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    omega = [int(y) for y in data.extra["omega"]]
    M = int(data.extra["M"])
    n_lanes = int(data.extra["n_lanes"])
    n_classes = len(omega)
    chance = 1.0 / n_classes
    norm = 1.0 - chance
    planes = data.extra["lane_planes"].detach().cpu().numpy().astype(np.float64)

    lanes = data.extra["lane_velocity_labels"].float()           # (N, seq_len, n_lanes)
    n = x.shape[0]
    split = n // 2
    L = eval_window_L
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_y_tr, _ = _sample_windows(lanes[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_y_ev, _ = _sample_windows(lanes[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, _, y_tr = _tile_examples_lanes(model, win_x_tr, win_y_tr, T)
    z_ev, raw_ev, y_ev = _tile_examples_lanes(model, win_x_ev, win_y_ev, T)

    rng = np.random.default_rng(seed)
    if z_tr.shape[0] > max_rows:
        i = rng.choice(z_tr.shape[0], max_rows, replace=False)
        z_tr, y_tr = z_tr[i], y_tr[i]
    if z_ev.shape[0] > max_rows:
        j = rng.choice(z_ev.shape[0], max_rows, replace=False)
        z_ev, y_ev, raw_ev = z_ev[j], y_ev[j], raw_ev[j]

    out: dict[str, float] = {}
    recs, oracles, floors = [], [], []
    for k in range(n_lanes):
        bal, recall, floor = _logistic_probe(z_tr, y_tr[:, k], z_ev, y_ev[:, k],
                                             n_classes=n_classes, seed=seed + 13 * k)
        opred = _lane_periodogram_pred(raw_ev, planes[k], omega, M)
        oracle = float((opred == y_ev[:, k]).mean())
        oracle_recall = _per_class_recall(y_ev[:, k], opred, n_classes)

        out[f"lane{k}_recovery"] = float((bal - chance) / norm)
        out[f"lane{k}_balacc"] = bal
        out[f"lane{k}_oracle"] = oracle
        out[f"lane{k}_chance"] = floor
        for c in range(n_classes):
            out[f"lane{k}_recall_c{c}"] = recall[c]
            out[f"lane{k}_oracle_c{c}"] = oracle_recall[c]
        recs.append((bal - chance) / norm)
        oracles.append(oracle)
        floors.append(floor)

    out["multilane_recovery"] = float(np.mean(recs))
    out["multilane_oracle"] = float(np.mean(oracles))
    out["multilane_chance"] = float(np.mean(floors))

    # band decomposition (spectral archs only): per-band slice, mean over lanes
    if hasattr(model, "band_of_features"):
        for b, (s, e) in enumerate(model.band_of_features()):
            band_recs = []
            for k in range(n_lanes):
                bb, _, _ = _logistic_probe(
                    z_tr[:, s:e], y_tr[:, k], z_ev[:, s:e], y_ev[:, k],
                    n_classes=n_classes, seed=seed + 29 * b + k)
                band_recs.append((bb - chance) / norm)
            out[f"band{b}_ml_recovery"] = float(np.mean(band_recs))

    return out
