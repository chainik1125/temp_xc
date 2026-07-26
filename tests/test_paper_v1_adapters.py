"""Paper-v1 adapter contract tests (ACTMIX Phase B, eval-only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from temp_bench.archs.paper_v1 import PaperTSAEV1, PaperTopKSAEV1, PaperTXCBaseV1
from temp_bench.evals.probing import _encode_pool


def test_topk_sae_adapter_shapes_and_composition():
    m = PaperTopKSAEV1(d_in=8, d_sae=16, k_pos=4)
    z = m.encode(torch.randn(3, 5, 8))
    assert z.shape == (3, 5, 16)
    # TopK→ReLU: at most k nonzero, and can be FEWER when selected
    # pre-acts are negative (the paper-era mixing fingerprint).
    assert int((z != 0).sum(-1).max()) <= 4
    with pytest.raises(NotImplementedError):
        m.train_step(torch.randn(2, 8))


def test_txc_adapter_window_path():
    m = PaperTXCBaseV1(d_in=8, d_sae=16, k_pos=2, T=3)
    z = m.encode(torch.randn(6, 3, 8))
    assert z.shape == (6, 16)                       # one code per window
    assert int((z != 0).sum(-1).max()) <= 6         # k_win = k_pos·T
    X = np.random.default_rng(0).standard_normal((4, 8, 8)).astype(np.float32)
    pooled, l0 = _encode_pool(m, X, S=8, batch_size=2,
                              device=torch.device("cpu"),
                              first_real=np.zeros(4, dtype=np.int64))
    assert pooled.shape == (4, 16) and np.isfinite(pooled).all()
    assert 0 <= l0 <= 6


def test_tsae_adapter_threshold_path_and_state_dict_roundtrip():
    m = PaperTSAEV1(d_in=8, d_sae=20, k_pos=4, h_frac=0.2)
    # threshold=-1.0 sentinel: every post-ReLU value (>=0) passes >-1 ⇒
    # z == post_relu (dense) — the v1 inference semantics, verbatim.
    z = m.encode(torch.randn(3, 4, 8))
    assert z.shape == (3, 4, 20) and (z >= 0).all()
    # state-dict rekey roundtrip (what phase_b.stage does)
    inner_sd = m.inner.state_dict()
    rekeyed = {f"inner.{k}": v for k, v in inner_sd.items()}
    m2 = PaperTSAEV1(d_in=8, d_sae=20, k_pos=4, h_frac=0.2)
    m2.load_state_dict(rekeyed, strict=True)
    assert float(m2.inner.threshold) == float(m.inner.threshold)
