"""The λ-probe finite-target mask: a no-op on all-finite grids, and a clean
drop of NaN leading-edge targets otherwise (`lambda_recovery` § task-hunt
round 2 — the `ward_real_slope8_*` datasources keep the frozen slope8 grid's
NaN where the label is undefined instead of inventing warm-up values)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from temp_bench.evals.lambda_recovery import (
    _tile_lambda_examples,
    _train_lambda_probe,
)
from temp_bench.evals.synthetic_recovery import _sample_windows


class _StubArch(torch.nn.Module):
    """Deterministic tile encoder: flatten → fixed linear projection."""

    def __init__(self, T: int, d_in: int, d_code: int = 12):
        super().__init__()
        self.config = SimpleNamespace(T=T)
        g = torch.Generator().manual_seed(0)
        self.proj = torch.nn.Parameter(
            torch.randn(T * d_in, d_code, generator=g), requires_grad=False)

    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        return tiles.reshape(tiles.shape[0], -1) @ self.proj


def _reference(model, x, lam, *, L, n_windows=1024, seed=0):
    """The pre-mask probe pipeline, with an explicit finite filter — the
    semantics the masked implementation must reproduce exactly."""
    T = int(model.config.T)
    n = x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_l_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_l_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows, seed=seed + 1)
    z_tr, t_tr = _tile_lambda_examples(model, win_x_tr, win_l_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, win_x_ev, win_l_ev, T)
    tr_m, ev_m = np.isfinite(t_tr), np.isfinite(t_ev)
    z_tr, t_tr, z_ev, t_ev = z_tr[tr_m], t_tr[tr_m], z_ev[ev_m], t_ev[ev_m]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression().fit(z_tr, t_tr)
        pred = reg.predict(z_ev)
        r2 = float(reg.score(z_ev, t_ev))
    corr = float(np.corrcoef(pred, t_ev)[0, 1])
    rngp = np.random.default_rng(seed + 7)
    perm = rngp.permutation(len(t_tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg0 = LinearRegression().fit(z_tr, t_tr[perm])
        pred0 = reg0.predict(z_ev)
    chance = float(np.corrcoef(pred0, t_ev)[0, 1])
    return {"lambda_recovery": corr, "lambda_r2": r2, "lambda_chance": chance}


def _data(seed=0, n_seqs=8, seq_len=64, d_in=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_seqs, seq_len, d_in, generator=g)
    # A target the stub code partially carries: a linear readout + noise.
    w = torch.randn(d_in, generator=g)
    lam = x @ w + 0.5 * torch.randn(n_seqs, seq_len, generator=g)
    return x, lam


def test_all_finite_grid_is_unchanged():
    x, lam = _data()
    for T in (1, 4):
        model = _StubArch(T=T, d_in=16).eval()
        got = _train_lambda_probe(model, x, lam, L=16, n_windows=64)
        want = _reference(model, x, lam, L=16, n_windows=64)
        for k in want:
            assert got[k] == want[k], (T, k, got[k], want[k])


def test_nan_targets_are_dropped_not_fatal():
    x, lam = _data(seed=1)
    lam = lam.clone()
    # NaN out a position-structured chunk (like slope8's early-sentence band)
    # plus scattered singles.
    lam[:, :9] = float("nan")
    lam[2, 30] = float("nan")
    lam[5, 45:50] = float("nan")
    for T in (1, 4):
        model = _StubArch(T=T, d_in=16).eval()
        got = _train_lambda_probe(model, x, lam, L=16, n_windows=64)
        want = _reference(model, x, lam, L=16, n_windows=64)
        for k in want:
            assert np.isfinite(got[k])
            assert got[k] == want[k], (T, k, got[k], want[k])


def test_all_nan_grid_returns_zeros():
    x, lam = _data(seed=2)
    lam = torch.full_like(lam, float("nan"))
    model = _StubArch(T=4, d_in=16).eval()
    got = _train_lambda_probe(model, x, lam, L=16, n_windows=16)
    assert got == {"lambda_recovery": 0.0, "lambda_r2": 0.0,
                   "lambda_chance": 0.0}
