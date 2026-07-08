"""Cyclic-tone velocity recovery + frequency response — autoresearch #3 add-on.

Like ``lambda_recovery`` / ``changepoint_recovery``, this is NOT a routed
evaluator (the experiment→evaluator map in ``temp_bench/core/runner.py`` is
off-limits). :class:`SyntheticRecovery` calls :func:`frequency_metrics` whenever
the materialised data carries ``extra['velocity_labels']`` (the
``toy_cyclic_*`` datasources). For every other bench the block is a no-op, so the
§ 4 metrics are byte-identical and the evaluator's ``protocol_version`` does not
move.

Probes are per *tile* at the tile's leading edge (feature dim = one tile's
``d_sae`` code, never concatenated — memorization-free), split by sequence
(leak-free), all LINEAR / multinomial-logistic (what the code makes *linearly*
available). The velocity is constant per sequence (single-tone task), so every
tile of a sequence carries the same class.

Metrics (frequency/bench_spec.md § 4, amendments A1–A4):

- ``velocity_recovery`` (**headline**): multinomial-logistic probe on the code →
  the Ω class, balanced accuracy normalized to [chance = 1/|Ω|, oracle]. The
  oracle is the **periodogram peak-pick** (circle: project onto the 2-D plane
  with the true ``R`` → complex tone → argmax |DFT| over Ω) or the **GLRT
  template** matched-filter (random), computed on the same held-out raw tiles.
- ``S(f)`` (**the deliverable curve**): the per-Ω-class recall of both the probe
  and the oracle, reported raw (``vel_recall_c{c}`` / ``vel_oracle_c{c}``) so the
  renderer can trace the normalized frequency response ``f = Ω[c]/M``.
- **band decomposition** (spectral arch only): a per-branch probe on each DCT
  band's code slice → ``band{b}_recovery`` (which velocities each band decodes).

Per-token archs and raw-linear window readers sit at chance (gating: velocity is
2nd-moment); the untrained-encoder control (n_steps=0) measures the
nonlinear-access residual. Chance floor is the shuffled-label probe.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch


@torch.no_grad()
def _tile_examples(model, win_x, win_y, T):
    """Tile length-L windows → (codes, raw_tiles, leading-edge labels).

    ``win_x: (W, L, d_in)`` → codes ``(W·(L/T), d_sae)`` (single-tile, not
    concatenated), raw tiles ``(W·(L/T), T, d_in)`` (for the oracle), and labels
    ``(W·(L/T),)`` at each tile's leading edge (position ``T-1``).
    """
    device = next(model.parameters()).device
    W, L, d_in = win_x.shape
    n_tiles = L // T
    tiles = win_x.to(device, dtype=torch.float32).reshape(W * n_tiles, T, d_in)
    z = model.encode(tiles).reshape(W * n_tiles, -1).detach().float().cpu().numpy()
    y = win_y.reshape(W, n_tiles, T, -1)[:, :, T - 1, 0].reshape(W * n_tiles)
    return z, tiles.detach().float().cpu().numpy(), y.numpy().astype(np.int64)


def _per_class_recall(y_true, y_pred, n_classes):
    """Recall of each class c (the per-Ω-class oracle/probe number)."""
    out = []
    for c in range(n_classes):
        m = y_true == c
        out.append(float((y_pred[m] == c).mean()) if m.any() else float("nan"))
    return out


def _logistic_probe(z_tr, y_tr, z_ev, y_ev, *, n_classes, seed):
    """Multinomial-logistic probe → (balacc, per-class recall, shuffled floor)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=200).fit(z_tr, y_tr)
        pred = clf.predict(z_ev)
        bal = float(balanced_accuracy_score(y_ev, pred))
        recall = _per_class_recall(y_ev, pred, n_classes)
        perm = np.random.default_rng(seed + 7).permutation(len(y_tr))
        clf0 = LogisticRegression(max_iter=200).fit(z_tr, y_tr[perm])
        floor = float(balanced_accuracy_score(y_ev, clf0.predict(z_ev)))
    return bal, recall, floor


def _periodogram_pred(x_tiles, R, omega, M):
    """Circle ML: project to the 2-D plane, argmax |DFT at f=Y/M| over Ω."""
    proj = x_tiles @ R                                          # (N, T, 2)
    c = proj[..., 0] + 1j * proj[..., 1]
    t = np.arange(x_tiles.shape[1])
    basis = np.exp(-2j * np.pi * np.asarray(omega, dtype=np.float64)[:, None]
                   * t[None, :] / M)                            # (|Ω|, T)
    return (np.abs(c @ basis.T)).argmax(axis=1)


def _glrt_pred(x_tiles, U, omega, M):
    """General GLRT template matched-filter (works for random + circle)."""
    N, T, _d = x_tiles.shape
    S = np.einsum("ntd,md->ntm", x_tiles, U)                    # (N, T, M) symbol scores
    t = np.arange(T)
    best = np.full(N, -np.inf)
    pred = np.zeros(N, dtype=np.int64)
    for yi, Yv in enumerate(omega):
        idx = (np.arange(M)[:, None] + Yv * t[None, :]) % M     # (M_B, T)
        sc = np.zeros((N, M))
        for tt in range(T):
            sc += S[:, tt, idx[:, tt]]
        bestB = sc.max(axis=1)
        upd = bestB > best
        best = np.where(upd, bestB, best)
        pred = np.where(upd, yi, pred)
    return pred


def frequency_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0, max_rows: int = 30_000,
) -> dict[str, float]:
    """Velocity recovery + S(f) + band decomposition for cyclic-tone data."""
    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    omega = [int(y) for y in data.extra["omega"]]
    M = int(data.extra["M"])
    n_classes = len(omega)
    chance = 1.0 / n_classes
    embedding = data.extra.get("embedding", "circle")
    U = data.emission_features.detach().cpu().numpy().astype(np.float64)   # (M, d_in)
    R = (data.extra["circle_plane"].detach().cpu().numpy().astype(np.float64)
         if "circle_plane" in data.extra else None)

    vel = data.extra["velocity_labels"].float().unsqueeze(-1)  # (N, seq_len, 1)
    n = x.shape[0]
    split = n // 2
    L = eval_window_L
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_y_tr, _ = _sample_windows(vel[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_y_ev, _ = _sample_windows(vel[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, _, y_tr = _tile_examples(model, win_x_tr, win_y_tr, T)
    z_ev, raw_ev, y_ev = _tile_examples(model, win_x_ev, win_y_ev, T)

    # subsample for the sklearn probe (deterministic)
    rng = np.random.default_rng(seed)
    if z_tr.shape[0] > max_rows:
        i = rng.choice(z_tr.shape[0], max_rows, replace=False); z_tr, y_tr = z_tr[i], y_tr[i]
    if z_ev.shape[0] > max_rows:
        j = rng.choice(z_ev.shape[0], max_rows, replace=False)
        z_ev, y_ev, raw_ev = z_ev[j], y_ev[j], raw_ev[j]

    out: dict[str, float] = {}

    # ── the probe (headline) ──
    bal, recall, floor = _logistic_probe(z_tr, y_tr, z_ev, y_ev,
                                         n_classes=n_classes, seed=seed)

    # ── the ML oracle on the same held-out raw tiles ──
    if embedding == "circle" and R is not None:
        opred = _periodogram_pred(raw_ev, R, omega, M)
    else:
        opred = _glrt_pred(raw_ev, U, omega, M)
    oracle = float((opred == y_ev).mean())
    oracle_recall = _per_class_recall(y_ev, opred, n_classes)

    denom = max(oracle - chance, 1e-6)
    out["velocity_recovery"] = float((bal - chance) / denom)   # normalized headline
    out["velocity_balacc"] = bal
    out["velocity_oracle"] = oracle
    out["velocity_chance"] = floor

    # ── S(f): per-Ω-class recall (probe + oracle), raw for the renderer ──
    for c in range(n_classes):
        out[f"vel_recall_c{c}"] = recall[c]
        out[f"vel_oracle_c{c}"] = oracle_recall[c]

    # ── band decomposition (spectral arch only) ──
    if hasattr(model, "band_of_features"):
        for b, (s, e) in enumerate(model.band_of_features()):
            bb, _, _ = _logistic_probe(z_tr[:, s:e], y_tr, z_ev[:, s:e], y_ev,
                                       n_classes=n_classes, seed=seed + b)
            out[f"band{b}_recovery"] = float((bb - chance) / denom)
            out[f"band{b}_balacc"] = bb

    return out
