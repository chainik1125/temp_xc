"""Contract tests for txc_btk_pre_subseq_btkonly (tscale wave-3 candidate).

The load-bearing property: t_sample == T (and every T=1 cell) is EXACTLY
the parent txc_batchtopk_pre_btkonly — the T=1 anchor coincides with the
baseline by construction. Plus subseq-specific semantics: sampled-only
pool/budget, sampled-only recon gradients, slab alignment, inherited
full-window eval.
"""

from __future__ import annotations

import pytest
import torch

from temp_bench.archs.btk_only import TXCBatchTopKPreBTKOnly
from temp_bench.archs.txc_btk_subseq import (
    TXCBatchTopKPreSubseqBTKOnly,
    _sample_contiguous_subset,
)

D_IN, D_SAE = 16, 40


def _mk(cls, T=4, t_sample=None, seed=0, **kw):
    torch.manual_seed(seed)
    kwargs = dict(d_in=D_IN, d_sae=D_SAE, T=T, k_pos=2, **kw)
    if cls is TXCBatchTopKPreSubseqBTKOnly:
        kwargs["t_sample"] = t_sample
    return cls(**kwargs)


def test_ratio_rule_default():
    assert _mk(TXCBatchTopKPreSubseqBTKOnly, T=4).t_sample == 2
    assert _mk(TXCBatchTopKPreSubseqBTKOnly, T=1).t_sample == 1
    assert _mk(TXCBatchTopKPreSubseqBTKOnly, T=16).t_sample == 8
    assert _mk(TXCBatchTopKPreSubseqBTKOnly, T=10).t_sample == 5
    assert _mk(TXCBatchTopKPreSubseqBTKOnly, T=4, t_sample=3).t_sample == 3


@pytest.mark.parametrize("T,t_s", [(4, 4), (1, None)])
def test_degenerate_equals_parent_exactly(T, t_s):
    """t_sample == T (incl. T=1 ratio default) → bit-equal parent step."""
    sub = _mk(TXCBatchTopKPreSubseqBTKOnly, T=T, t_sample=t_s, seed=7)
    par = _mk(TXCBatchTopKPreBTKOnly, T=T, seed=7)
    par.load_state_dict(
        {k: v for k, v in sub.state_dict().items()}, strict=False)
    x = torch.randn(6, T, D_IN)
    torch.manual_seed(0)
    out_s = sub.train_step(x.clone())
    torch.manual_seed(0)
    out_p = par.train_step(x.clone())
    assert torch.equal(out_s["loss"], out_p["loss"])
    assert torch.equal(out_s["l0"], out_p["l0"])
    # eval path identical too
    sub.eval(); par.eval()
    with torch.no_grad():
        assert torch.equal(sub.encode(x), par.encode(x))


def test_sampled_budget_and_l0_bound():
    m = _mk(TXCBatchTopKPreSubseqBTKOnly, T=4, t_sample=2)
    m.train()
    x = torch.randn(8, 4, D_IN)
    out = m.train_step(x)
    # union of sampled survivors ≤ k_pos · t_sample per window
    assert float(out["l0"]) <= m.k_pos * m.t_sample + 1e-6


def test_unsampled_slabs_get_no_encoder_grad():
    m = _mk(TXCBatchTopKPreSubseqBTKOnly, T=4, t_sample=2)
    m.train()
    x = torch.randn(1, 4, D_IN)                      # single row → one subset
    torch.manual_seed(123)
    out = m.train_step(x)
    out["loss"].backward()
    torch.manual_seed(123)
    idx = _sample_contiguous_subset(4, 2, 1, x.device)[0].tolist()
    unsampled = [t for t in range(4) if t not in idx]
    assert m.W_enc.grad is not None
    for t in unsampled:
        assert torch.all(m.W_enc.grad[t] == 0), f"slab {t} leaked gradient"
    assert any(torch.any(m.W_enc.grad[t] != 0) for t in idx)


def test_eval_full_window_inherited():
    m = _mk(TXCBatchTopKPreSubseqBTKOnly, T=4, t_sample=2)
    m.eval()
    x = torch.randn(5, 4, D_IN)
    z = m.encode(x)
    assert z.shape == (5, 1, D_SAE)
    with pytest.raises(ValueError):
        m.encode(torch.randn(5, 3, D_IN))            # wrong-T rejected
    assert m.consumes == "window"
    assert not hasattr(type(m), "eval_consumes") or "eval_consumes" not in vars(type(m))


def test_contiguous_sampler_shape_and_range():
    idx = _sample_contiguous_subset(10, 5, 64, torch.device("cpu"))
    assert idx.shape == (64, 5)
    assert int(idx.min()) >= 0 and int(idx.max()) < 10
    diffs = idx[:, 1:] - idx[:, :-1]
    assert torch.all(diffs == 1)                     # contiguous
