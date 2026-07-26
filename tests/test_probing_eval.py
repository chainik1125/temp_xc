"""Unit tests for the ported ProbingEval (protocol 1.2.0).

Hermetic + CPU-only: tiny stub archs, planted-signal data. Covers the
ACTMIX-load-bearing paths: per-token vs window dispatch, first_real
padding masks, the within-window shuffle control (fixed probe), T=1
shuffle identity, and realized-L0 accounting.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from temp_bench.evals.probing import ProbingEval, _encode_pool, _fit_probe, _score_probe
from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec

D_IN = 6
D_SAE = 8


class _TokenId(TempBenchArch):
    """Per-token stub: z = relu(pad(x)); k first dims mirror the input."""

    arch_version = "0.0.0-test"
    consumes = "token"

    def __init__(self):
        super().__init__()
        self.config = ArchConfig(name="tok", d_in=D_IN, d_sae=D_SAE, k_pos=D_IN, T=1)
        self._dummy = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        pad = torch.zeros(*x.shape[:-1], D_SAE - D_IN, device=x.device)
        return torch.relu(torch.cat([x, pad], dim=-1))

    def decode(self, z):
        return z[..., :D_IN]

    def train_step(self, x):
        return {"loss": self._dummy.sum()}


class _WindowDiff(TempBenchArch):
    """Window stub (T=2): first feature = x[:,1,0] - x[:,0,0] (pure ORDER
    signal — its sign flips under within-window shuffle of the two
    positions); second feature = window mean of dim 0 (order-invariant).
    """

    arch_version = "0.0.0-test"
    consumes = "window"

    def __init__(self):
        super().__init__()
        self.config = ArchConfig(name="win", d_in=D_IN, d_sae=D_SAE, k_pos=2, T=2)
        self._dummy = nn.Parameter(torch.zeros(1))

    def encode(self, x):
        assert x.dim() == 3 and x.shape[1] == 2
        diff = x[:, 1, 0] - x[:, 0, 0]
        mean = x[:, :, 0].mean(dim=1)
        z = torch.zeros(x.shape[0], D_SAE, device=x.device)
        z[:, 0] = torch.relu(diff)          # order-carrying unit
        z[:, 1] = torch.relu(-diff)         # order-carrying unit (other sign)
        z[:, 2] = torch.relu(mean)          # order-invariant unit
        return z.unsqueeze(1)               # (B, 1, d_sae)

    def decode(self, z):
        B = z.shape[0]
        return torch.zeros(B, 2, D_IN)

    def train_step(self, x):
        return {"loss": self._dummy.sum()}


def _order_task(n=400, S=8, seed=0):
    """Class 1: adjacent-token ramp UP on dim 0; class 0: ramp DOWN.
    Window mean of dim 0 is identical across classes — only ORDER
    separates them.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, S, D_IN)).astype(np.float32) * 0.01
    y = np.zeros(n, dtype=np.int64)
    y[: n // 2] = 1
    ramp = np.tile(np.array([0.0, 1.0]), S // 2)
    X[y == 1, :, 0] += ramp
    X[y == 0, :, 0] += ramp[::-1]
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    fr = np.zeros(n, dtype=np.int64)
    half = n // 2
    return {
        "X_train": X[:half], "y_train": y[:half],
        "X_test": X[half:], "y_test": y[half:],
        "first_real_train": fr[:half], "first_real_test": fr[half:],
    }


def test_per_token_shapes_and_planted_auc():
    model = _TokenId()
    ev = ProbingEval()
    spec = EvalSpec(datasource="unused", data_key="x", smoke=True,
                    extra={"k_feat": 4, "S": 8})
    m = ev.eval(model, spec)
    assert m["n_tasks"] == 1.0
    assert m["mean_auc"] > 0.9            # planted mean-shift is separable
    assert m["shuffle_identity"] == 1.0
    assert m["mean_auc_shuf"] == m["mean_auc"]   # T=1: exact invariance
    assert m["delta_auc_shuf"] == 0.0


def test_window_order_signal_destroyed_by_shuffle():
    model = _WindowDiff()
    task = _order_task()
    dev = torch.device("cpu")

    tr, _ = _encode_pool(model, task["X_train"], S=8, batch_size=64, device=dev,
                         first_real=task["first_real_train"])
    clf, idx = _fit_probe(tr, task["y_train"], k_feat=3)

    te, _ = _encode_pool(model, task["X_test"], S=8, batch_size=64, device=dev,
                         first_real=task["first_real_test"])
    ordered = _score_probe(clf, idx, te, task["y_test"])

    sh, _ = _encode_pool(model, task["X_test"], S=8, batch_size=64, device=dev,
                         first_real=task["first_real_test"], shuffle_seed=0)
    shuffled = _score_probe(clf, idx, sh, task["y_test"])

    assert ordered["auc"] > 0.95          # order signal fully probe-readable
    assert shuffled["auc"] < 0.75         # within-window shuffle destroys it
    assert ordered["auc"] - shuffled["auc"] > 0.25


def test_shuffle_determinism():
    model = _WindowDiff()
    task = _order_task()
    dev = torch.device("cpu")
    a, l0a = _encode_pool(model, task["X_test"], S=8, batch_size=64, device=dev,
                          first_real=task["first_real_test"], shuffle_seed=7)
    b, l0b = _encode_pool(model, task["X_test"], S=8, batch_size=64, device=dev,
                          first_real=task["first_real_test"], shuffle_seed=7)
    c, _ = _encode_pool(model, task["X_test"], S=8, batch_size=64, device=dev,
                        first_real=task["first_real_test"], shuffle_seed=8)
    np.testing.assert_array_equal(a, b)
    assert l0a == l0b
    assert not np.array_equal(a, c)       # different seed → different perms


def test_first_real_masks_padding():
    """A row whose first half is padding must pool only over the real
    region — give padding a huge value on the mirrored dim and check it
    does not leak into the pooled feature.
    """
    model = _TokenId()
    dev = torch.device("cpu")
    S = 8
    X = np.zeros((2, S, D_IN), dtype=np.float32)
    X[:, :, 0] = 1.0                       # real signal everywhere
    X[0, :4, 0] = 100.0                    # would-be padding region, poisoned
    fr = np.array([4, 0], dtype=np.int64)
    pooled, _ = _encode_pool(model, X, S=S, batch_size=2, device=dev, first_real=fr)
    assert pooled[0, 0] == pytest.approx(1.0)   # poison masked out
    assert pooled[1, 0] == pytest.approx(1.0)


def test_window_fallback_row_no_nan():
    model = _WindowDiff()
    dev = torch.device("cpu")
    S = 8
    X = np.random.default_rng(0).standard_normal((3, S, D_IN)).astype(np.float32)
    fr = np.array([0, 7, 8], dtype=np.int64)   # rows 1-2: fewer than T real tokens
    pooled, _ = _encode_pool(model, X, S=S, batch_size=3, device=dev, first_real=fr)
    assert np.isfinite(pooled).all()


def test_realized_l0_counts_nonzero_units():
    """fired ⇔ z != 0 (btk-only convention): the stub relu's negatives to
    exactly 0, so 3 positive input dims → 3 nonzero latents per token.
    """
    model = _TokenId()
    dev = torch.device("cpu")
    S = 4
    X = np.zeros((2, S, D_IN), dtype=np.float32)
    X[:, :, :3] = 1.0                      # exactly 3 positive dims per token
    X[:, :, 3:] = -1.0                     # relu'd to exactly 0 by the stub
    _, l0 = _encode_pool(model, X, S=S, batch_size=2, device=dev,
                         first_real=np.zeros(2, dtype=np.int64))
    assert l0 == pytest.approx(3.0)
