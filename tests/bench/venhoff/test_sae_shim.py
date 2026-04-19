"""Contract + dimensionality tests for the SAE shim.

Verifies:
  - Path 1 shim: (B, d) → (B, d_sae)
  - Path 3 shim with each aggregation: correct output shape
  - argmax_labels returns valid latent ids

No GPU required — all on CPU with a tiny TopKSAE.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.bench.architectures.topk_sae import TopKSAE
from src.bench.venhoff.sae_shim import argmax_labels, wrap_for_path1, wrap_for_path3


D_IN = 16
D_SAE = 5
T = 5
BATCH = 8


def _make_mean_pkl(tmp_path: Path) -> Path:
    mean_pkl = tmp_path / "mean.pkl"
    payload = {"activation_mean": np.zeros(D_IN, dtype=np.float32)}
    with mean_pkl.open("wb") as f:
        pickle.dump(payload, f)
    return mean_pkl


def test_path1_shim_shape(tmp_path: Path) -> None:
    base = TopKSAE(d_in=D_IN, d_sae=D_SAE, k=3)
    shim = wrap_for_path1(base, _make_mean_pkl(tmp_path))

    x = torch.randn(BATCH, D_IN)
    z = shim.encoder(x)

    assert z.shape == (BATCH, D_SAE)


@pytest.mark.parametrize("agg", ["last", "mean", "max"])
def test_path3_shim_shape_collapsing_aggs(agg: str, tmp_path: Path) -> None:
    base = TopKSAE(d_in=D_IN, d_sae=D_SAE, k=3)
    shim = wrap_for_path3(base, _make_mean_pkl(tmp_path), T=T, aggregation=agg)

    x = torch.randn(BATCH, T, D_IN)
    z = shim.encoder(x)

    assert z.shape == (BATCH, D_SAE)


def test_path3_shim_shape_full_window(tmp_path: Path) -> None:
    base = TopKSAE(d_in=D_IN, d_sae=D_SAE, k=3)
    shim = wrap_for_path3(base, _make_mean_pkl(tmp_path), T=T, aggregation="full_window")

    x = torch.randn(BATCH, T, D_IN)
    z = shim.encoder(x)

    # full_window concatenates T × d_sae along the feature axis.
    assert z.shape == (BATCH, T * D_SAE)


def test_argmax_labels_in_range_path1(tmp_path: Path) -> None:
    base = TopKSAE(d_in=D_IN, d_sae=D_SAE, k=3)
    shim = wrap_for_path1(base, _make_mean_pkl(tmp_path))

    x = torch.randn(BATCH, D_IN)
    labels = argmax_labels(shim, x)

    assert labels.shape == (BATCH,)
    assert labels.dtype == torch.int64
    assert labels.min() >= 0
    assert labels.max() < D_SAE


def test_path3_t_mismatch_raises(tmp_path: Path) -> None:
    base = TopKSAE(d_in=D_IN, d_sae=D_SAE, k=3)
    shim = wrap_for_path3(base, _make_mean_pkl(tmp_path), T=T, aggregation="mean")

    bad_x = torch.randn(BATCH, T + 1, D_IN)
    with pytest.raises(AssertionError, match="T mismatch"):
        shim.encoder(bad_x)
