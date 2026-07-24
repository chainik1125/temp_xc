"""Contract tests for the λ-probe v2 contingency eval
(`briefings/probe-adequacy.md` item 1; frozen convention in
`experiments/explorations/task_hunt/lambda_intensity/PROBE_V2_SPEC.md`).

(a) OLS mode (the α → 0 limit) at nw = 1024 on identical inputs reproduces
    the frozen v1 numbers to tight tolerance;
(b) v2 is deterministic across calls;
(c) the by-trace split never places two windows of one trace in different
    halves (and degenerates to v1's n//2 without trace_ids);
(d) the YAML registration resolves: `python run.py validate` is green and
    the smoke sweep's keys resolve against the registries.

CPU-only, tiny synthetic tensors; no leaderboard writes anywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from temp_bench.evals.lambda_recovery import _train_lambda_probe
from temp_bench.evals.lambda_recovery_v2 import (
    DEFAULT_ALPHAS,
    _split_index,
    _train_lambda_probe_v2,
    lambda_recovery_v2_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_YAML = ROOT / "configs" / "sweeps" / "lambda_probe_v2_smoke.yaml"

V1_TO_V2 = {"lambda_recovery": "lambda_recovery_v2",
            "lambda_r2": "lambda_r2_v2",
            "lambda_chance": "lambda_chance_v2"}


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


def _data(seed=0, n_seqs=8, seq_len=64, d_in=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_seqs, seq_len, d_in, generator=g)
    w = torch.randn(d_in, generator=g)
    lam = x @ w + 0.5 * torch.randn(n_seqs, seq_len, generator=g)
    return x, lam


# ── (a) v1 reproduction in the α → 0 limit ────────────────────────────


@pytest.mark.parametrize("T", [1, 4])
def test_ols_nw1024_reproduces_v1(T):
    x, lam = _data()
    model = _StubArch(T=T, d_in=16).eval()
    v1 = _train_lambda_probe(model, x, lam, L=16, n_windows=1024)
    v2 = _train_lambda_probe_v2(
        model, x, lam, L=16, n_windows=1024, probe="ols",
        alphas=DEFAULT_ALPHAS, split_mode="half", trace_ids=None)
    for k1, k2 in V1_TO_V2.items():
        assert np.isclose(v2[k2], v1[k1], rtol=1e-10, atol=1e-12), \
            (T, k1, v1[k1], v2[k2])


def test_ols_reproduces_v1_on_nan_grid():
    x, lam = _data(seed=1)
    lam = lam.clone()
    lam[:, :9] = float("nan")
    lam[3, 40:45] = float("nan")
    model = _StubArch(T=4, d_in=16).eval()
    v1 = _train_lambda_probe(model, x, lam, L=16, n_windows=256)
    v2 = _train_lambda_probe_v2(
        model, x, lam, L=16, n_windows=256, probe="ols",
        alphas=DEFAULT_ALPHAS, split_mode="half", trace_ids=None)
    for k1, k2 in V1_TO_V2.items():
        assert np.isclose(v2[k2], v1[k1], rtol=1e-10, atol=1e-12)


def test_trace_split_without_trace_ids_is_v1_split():
    """Default `trace` mode on a traceless datasource == v1's n//2."""
    x, lam = _data(seed=2)
    model = _StubArch(T=4, d_in=16).eval()
    half = _train_lambda_probe_v2(
        model, x, lam, L=16, n_windows=128, probe="ols",
        alphas=DEFAULT_ALPHAS, split_mode="half", trace_ids=None)
    trace = _train_lambda_probe_v2(
        model, x, lam, L=16, n_windows=128, probe="ols",
        alphas=DEFAULT_ALPHAS, split_mode="trace", trace_ids=None)
    assert half == trace


# ── (b) determinism ───────────────────────────────────────────────────


def test_ridge_deterministic_across_calls():
    x, lam = _data(seed=3)
    model = _StubArch(T=4, d_in=16).eval()
    kw = dict(L=16, n_windows=256, probe="ridge", alphas=DEFAULT_ALPHAS,
              split_mode="trace", trace_ids=None)
    a = _train_lambda_probe_v2(model, x, lam, **kw)
    b = _train_lambda_probe_v2(model, x, lam, **kw)
    assert a == b
    assert all(np.isfinite(v) for v in a.values())
    assert a["lambda_alpha_v2"] in DEFAULT_ALPHAS


# ── (c) the by-trace split ────────────────────────────────────────────


def test_split_index_never_splits_a_trace():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n_traces = int(rng.integers(2, 12))
        counts = rng.integers(1, 8, size=n_traces)
        t = np.repeat(np.arange(n_traces), counts)
        n = len(t)
        s = _split_index(n, "trace", t)
        assert 0 < s <= n
        assert set(t[:s]) & set(t[s:]) == set(), (t.tolist(), s)


def test_split_index_snaps_forward_only():
    #             0  1  2  3  4  5  6  7  8  9   (n=10, n//2=5 inside trace 1)
    t = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    assert _split_index(10, "trace", t) == 6
    assert _split_index(10, "half", t) == 5
    # boundary exactly at n//2 → no snap
    t2 = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
    assert _split_index(10, "trace", t2) == 5


def test_split_index_rejects_bad_trace_ids():
    with pytest.raises(ValueError, match="non-decreasing"):
        _split_index(4, "trace", np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="entries"):
        _split_index(4, "trace", np.array([0, 0, 1]))


def test_midpoint_spanning_trace_returns_zeros_not_crash():
    x, lam = _data(seed=4, n_seqs=8)
    model = _StubArch(T=4, d_in=16).eval()
    t = np.array([0, 0, 0, 1, 1, 1, 1, 1])   # trace 1 runs to the end
    got = _train_lambda_probe_v2(
        model, x, lam, L=16, n_windows=64, probe="ols",
        alphas=DEFAULT_ALPHAS, split_mode="trace", trace_ids=t)
    assert got["lambda_recovery_v2"] == 0.0
    assert got["lambda_v2_n_train_rows"] == 0.0


def test_metrics_entrypoint_reads_extra_trace_ids():
    x, lam = _data(seed=5)
    model = _StubArch(T=4, d_in=16).eval()
    data = SimpleNamespace(
        x=x, extra={"lambda_labels": lam,
                    "trace_ids": np.repeat(np.arange(4), 2)})
    got = lambda_recovery_v2_metrics(
        model, data, eval_window_L=16,
        eval_cfg={"lambda_v2_n_windows": 128, "lambda_v2_probe": "ridge"})
    assert set(got) == {"lambda_recovery_v2", "lambda_r2_v2",
                       "lambda_chance_v2", "lambda_alpha_v2",
                       "lambda_v2_n_train_rows", "lambda_v2_n_eval_rows"}
    assert all(np.isfinite(v) for v in got.values())


# ── (d) YAML registration resolves ────────────────────────────────────


def test_run_py_validate_green():
    res = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "validate"],
        capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr


def test_smoke_sweep_yaml_resolves():
    import yaml

    from temp_bench.core.config import load_arch, load_datasource

    grid = yaml.safe_load(SWEEP_YAML.read_text())
    assert grid["experiment"] == "synthetic"
    load_datasource(grid["datasource"])
    for arch in grid["arch"]:
        load_arch(arch)
    ec = grid["eval_cfg"]
    assert ec["lambda_probe_v2"] is True
    assert ec["lambda_v2_probe"] == "ridge"
    assert ec["lambda_v2_split"] == "trace"
    assert ec["lambda_v2_n_windows"] == 8192
    # The YAML pins the α grid explicitly; it must equal the frozen default.
    assert np.allclose(ec["lambda_v2_alphas"], DEFAULT_ALPHAS, rtol=1e-12)
    assert np.allclose(DEFAULT_ALPHAS, np.logspace(-2, 4, 13), rtol=1e-12)
