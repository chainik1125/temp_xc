"""Change-point / semi-Markov mode recovery — autoresearch #2 add-on.

Like ``lambda_recovery``, this is NOT a routed evaluator (the
experiment→evaluator map in ``temp_bench/core/runner.py`` is off-limits).
:class:`SyntheticRecovery` calls :func:`changepoint_metrics` whenever the
materialised data carries ``extra['mode_labels']`` (the
``toy_changepoint_modes*`` datasource). For every other bench the block is a
no-op, so the § 4 metrics are byte-identical and the evaluator's
``protocol_version`` does not move.

Three latent probes, all per *tile* at the tile's leading edge (feature dim =
one tile's ``d_sae`` code, never concatenated — memorization-free), split by
sequence (leak-free), all LINEAR (what the code makes *linearly* available):

- ``mode_recovery`` (**DC headline**): multinomial-logistic probe → ``m_t``,
  balanced accuracy normalized to [chance = 1/K_m, oracle = 1]. The mode is
  stamped into every token of the dwell, so per-token archs should reach it.
- ``tss_recovery`` (**primary AC latent**, amendment A2): linear-regression
  probe → time-since-switch ``τ_t``; held-out Pearson corr — chance = 0.
  Per-token ceiling is provably 0 (``E[τ|m_t]`` constant by Π-symmetry); the
  in-tile info ceiling is 0.76/0.96/1.00 at T=2/4/8 (gating stats); the
  raw-linear access ceiling is ≈ 0 even for windows, so recovery on a trained
  code is learning, not access.
- ``cp_recovery`` (AC simple-floor companion): logistic probe →
  ``c_t = [m_t ≠ m_{t-1}]``, balanced accuracy normalized to [0.5, 1].

Each probe also reports an empirical chance floor (the same probe fit on
shuffled train targets), per the conventions § 5.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch


@torch.no_grad()
def _tile_label_examples(
    model: TempBenchArch, win_x: torch.Tensor, win_y: torch.Tensor, T: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tile length-L windows; each tile-code → labels at the tile's LAST position.

    ``win_x: (W, L, d_in)`` → ``(W·(L/T), tile_code_dim)`` codes; ``win_y:
    (W, L, n_cols)`` → ``(W·(L/T), n_cols)`` targets at each tile's leading
    edge (position ``T-1`` within the tile). The feature dim is the
    *single-tile* code (``d_sae`` for a window crosscoder), not the
    concatenation over tiles.
    """
    device = next(model.parameters()).device
    W, L, d_in = win_x.shape
    n_tiles = L // T
    tiles = win_x.to(device, dtype=torch.float32).reshape(W * n_tiles, T, d_in)
    z = model.encode(tiles).reshape(W * n_tiles, -1).detach().float().cpu().numpy()
    y = win_y.reshape(W, n_tiles, T, -1)[:, :, T - 1, :].reshape(W * n_tiles, -1)
    return z, y.detach().float().cpu().numpy()


def _balacc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def _logistic_probe(z_tr, y_tr, z_ev, y_ev, *, seed: int) -> tuple[float, float]:
    """Held-out balanced accuracy of a logistic probe + its shuffled-label floor."""
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=300).fit(z_tr, y_tr)
        bal = _balacc(y_ev, clf.predict(z_ev))
        perm = np.random.default_rng(seed + 7).permutation(len(y_tr))
        clf0 = LogisticRegression(max_iter=300).fit(z_tr, y_tr[perm])
        floor = _balacc(y_ev, clf0.predict(z_ev))
    return bal, floor


def _linear_probe(z_tr, y_tr, z_ev, y_ev, *, seed: int) -> tuple[float, float, float]:
    """Held-out corr + R² of a linear probe + its shuffled-label corr floor."""
    from sklearn.linear_model import LinearRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression().fit(z_tr, y_tr)
        pred = reg.predict(z_ev)
        r2 = float(reg.score(z_ev, y_ev))
        corr = (float(np.corrcoef(pred, y_ev)[0, 1])
                if np.std(pred) > 1e-12 else 0.0)
        perm = np.random.default_rng(seed + 7).permutation(len(y_tr))
        reg0 = LinearRegression().fit(z_tr, y_tr[perm])
        pred0 = reg0.predict(z_ev)
        floor = (float(np.corrcoef(pred0, y_ev)[0, 1])
                 if np.std(pred0) > 1e-12 else 0.0)
    return corr, r2, floor


def changepoint_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0,
) -> dict[str, float]:
    """Return the mode/τ/c recovery metrics for semi-Markov-mode data.

    ``data.extra`` carries ``mode_labels`` (categorical), ``time_since_switch``
    (scalar), and ``changepoint_labels`` (binary), each ``(n_seqs, seq_len)``;
    ``eval_window_L`` is the common tiled evaluation-window length.
    """
    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    K_m = int(data.extra["K_m"])
    labels = torch.stack(
        [torch.as_tensor(data.extra[k]).float()
         for k in ("mode_labels", "time_since_switch", "changepoint_labels")],
        dim=-1,
    )                                                       # (N, seq_len, 3)
    n = x.shape[0]
    split = n // 2

    # x and label windows share the seed → identical (seq, offset) → aligned.
    L = eval_window_L
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_y_tr, _ = _sample_windows(labels[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_y_ev, _ = _sample_windows(labels[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, y_tr = _tile_label_examples(model, win_x_tr, win_y_tr, T)
    z_ev, y_ev = _tile_label_examples(model, win_x_ev, win_y_ev, T)
    mode_tr, tss_tr, cp_tr = (y_tr[:, 0].astype(np.int64), y_tr[:, 1],
                              y_tr[:, 2].astype(np.int64))
    mode_ev, tss_ev, cp_ev = (y_ev[:, 0].astype(np.int64), y_ev[:, 1],
                              y_ev[:, 2].astype(np.int64))

    out: dict[str, float] = {}

    # DC headline: mode recovery, normalized to [chance = 1/K_m, 1].
    mode_bal, mode_floor = _logistic_probe(z_tr, mode_tr, z_ev, mode_ev, seed=seed)
    chance = 1.0 / K_m
    out["mode_recovery"] = (mode_bal - chance) / (1.0 - chance)
    out["mode_balacc"] = mode_bal
    out["mode_chance"] = mode_floor

    # Primary AC latent: time-since-switch (corr; chance = 0).
    if np.std(tss_tr) < 1e-9 or np.std(tss_ev) < 1e-9:
        out.update({"tss_recovery": 0.0, "tss_r2": 0.0, "tss_chance": 0.0})
    else:
        corr, r2, floor = _linear_probe(z_tr, tss_tr, z_ev, tss_ev, seed=seed)
        out.update({"tss_recovery": corr, "tss_r2": r2, "tss_chance": floor})

    # AC simple-floor companion: change-point flag, normalized to [0.5, 1].
    cp_bal, cp_floor = _logistic_probe(z_tr, cp_tr, z_ev, cp_ev, seed=seed)
    out["cp_recovery"] = (cp_bal - 0.5) / 0.5
    out["cp_balacc"] = cp_bal
    out["cp_chance"] = cp_floor

    return out
