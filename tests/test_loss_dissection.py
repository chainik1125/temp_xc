"""Contract tests for the TXC-pro loss-dissection variants (CARD § 8, frozen).

Committed with the build, before any grid (strict commit-then-run)."""

from __future__ import annotations

import math

import pytest
import torch

from temp_bench.archs.txc_batchtopk import TXCBatchTopKPost
from temp_bench.archs.txc_post_dissect import TXCPostDissect, _info_nce

D_IN, D_SAE, T, K_POS = 16, 20, 4, 2
SEQ_LEN = 32


def _mk(cls, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(d_in=D_IN, d_sae=D_SAE, T=T, k_pos=K_POS, **kw)


def _seq_batch(B=8, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, SEQ_LEN, D_IN, generator=g)


# ── 1. plain-reduction: _loss_on == parent train_step on identical state ──

def test_plain_reduction_matches_parent():
    base = _mk(TXCBatchTopKPost)
    plain = _mk(TXCPostDissect, mat_alpha=0.0, ctr_alpha=0.0)
    xb = _seq_batch()[:, :T, :]  # (B, T, d_in) anchor batch
    for _ in range(3):  # a few steps so counters/threshold paths run equally
        out_b = base.train_step(xb)
        out_p = plain._loss_on(xb, None)
        for k in ("loss", "mse", "l0", "auxk", "dead", "threshold"):
            assert torch.allclose(out_b[k], out_p[k], atol=1e-6, rtol=0.0), (
                f"plain-reduction drift on {k}: {out_b[k]} vs {out_p[k]}"
            )
    assert float(out_p["mat"]) == 0.0 and float(out_p["ctr"]) == 0.0


# ── 2. zero-weight exactness: positives with ctr_alpha=0 change nothing ──

def test_zero_weight_ignores_positives():
    a = _mk(TXCPostDissect, mat_alpha=0.0, ctr_alpha=0.0)
    b = _mk(TXCPostDissect, mat_alpha=0.0, ctr_alpha=0.0)
    anchor = _seq_batch()[:, :T, :]
    fake_pos = [torch.randn_like(anchor), torch.randn_like(anchor)]
    out_a = a._loss_on(anchor, None)
    out_b = b._loss_on(anchor, fake_pos)
    assert torch.allclose(out_a["loss"], out_b["loss"], atol=0.0, rtol=0.0)


# ── 3. matryoshka structure ──

def test_prefix_ladder_values():
    m20 = _mk(TXCPostDissect, mat_alpha=1.0)
    assert m20._prefix_sizes() == (2, 5, 7, 10, 12, 15, 17, 20)
    torch.manual_seed(0)
    m101 = TXCPostDissect(d_in=24, d_sae=101, T=T, k_pos=K_POS, mat_alpha=1.0)
    assert m101._prefix_sizes() == (12, 25, 37, 50, 63, 75, 88, 101)
    assert m101._prefix_sizes()[-1] == 101


def test_prefix_decode_uses_only_first_rows():
    m = _mk(TXCPostDissect, mat_alpha=1.0)
    z = torch.randn(6, D_SAE).relu()
    n = 7
    ref = m._decode_prefix(z, n)
    with torch.no_grad():
        m.W_dec[n:] = 0.0  # zeroing rows >= n must not change the prefix recon
    assert torch.allclose(ref, m._decode_prefix(z, n), atol=0.0, rtol=0.0)


def test_matryoshka_term_positive_and_counted():
    m = _mk(TXCPostDissect, mat_alpha=1.0, ctr_alpha=0.0)
    out = m._loss_on(_seq_batch()[:, :T, :], None)
    assert float(out["mat"]) > 0.0
    expected = float(out["mse"]) + m.auxk_alpha * float(out["auxk"]) + float(out["mat"])
    assert math.isclose(float(out["loss"]), expected, rel_tol=1e-5)


# ── 4. contrastive pairs ──

def test_slicer_offsets_and_positive_identity():
    m = _mk(TXCPostDissect, ctr_alpha=1.0)
    B = 16
    # x[b, l, 0] = l so any window's first channel reads off its offsets.
    x = torch.zeros(B, SEQ_LEN, D_IN)
    x[:, :, 0] = torch.arange(SEQ_LEN).float()
    torch.manual_seed(123)
    anchor, positives = m._slice(x)
    assert positives is not None and len(positives) == 2
    p = anchor[:, 0, 0].long()
    hi = SEQ_LEN - T - TXCPostDissect.S_MAX
    assert int(p.min()) >= 0 and int(p.max()) <= hi
    for s, pos in zip(m.ctr_shifts, positives):
        rows = torch.arange(B).unsqueeze(1)
        idx = (p.unsqueeze(1) + s) + torch.arange(T).unsqueeze(0)
        assert torch.equal(pos, x[rows, idx])
    assert m.ctr_shifts == (1, 2)
    assert m.ctr_weights == (1.0 / 2.0, 1.0 / 3.0)


def test_info_nce_matches_manual_formula():
    g = torch.Generator().manual_seed(3)
    z_a = torch.randn(5, D_SAE, generator=g)
    z_b = torch.randn(5, D_SAE, generator=g)
    # Independent manual computation: cosine sims + explicit log-softmax CE.
    na = z_a / z_a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    nb = z_b / z_b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = na @ nb.t()
    def ce(logits):
        ls = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return -ls.diagonal().mean()
    manual = 0.5 * (ce(sim) + ce(sim.t()))
    assert torch.allclose(_info_nce(z_a, z_b), manual, atol=1e-6)


def test_contrastive_term_positive_and_counted():
    m = _mk(TXCPostDissect, mat_alpha=0.0, ctr_alpha=1.0)
    torch.manual_seed(5)
    out = m.train_step(_seq_batch(B=12))
    assert float(out["ctr"]) > 0.0
    expected = float(out["mse"]) + m.auxk_alpha * float(out["auxk"]) + float(out["ctr"])
    assert math.isclose(float(out["loss"]), expected, rel_tol=1e-5)


# ── 5. parameter identity across the four variants ──

def test_variant_state_dicts_identical_at_init():
    kws = [dict(mat_alpha=0.0, ctr_alpha=0.0), dict(mat_alpha=1.0, ctr_alpha=0.0),
           dict(mat_alpha=0.0, ctr_alpha=1.0), dict(mat_alpha=1.0, ctr_alpha=1.0)]
    sds = [_mk(TXCPostDissect, seed=11, **kw).state_dict() for kw in kws]
    for sd in sds[1:]:
        assert set(sd) == set(sds[0])
        for k in sds[0]:
            assert torch.equal(sds[0][k], sd[k]), f"init divergence at {k}"


# ── 6. offset distribution: uniform on {0..seq_len-T-2} (seeded smoke) ──

def test_offset_distribution_uniform():
    m = _mk(TXCPostDissect, ctr_alpha=1.0)
    x = torch.zeros(64, SEQ_LEN, D_IN)
    x[:, :, 0] = torch.arange(SEQ_LEN).float()
    torch.manual_seed(9)
    counts = torch.zeros(SEQ_LEN)
    for _ in range(200):
        anchor, _ = m._slice(x)
        p = anchor[:, 0, 0].long()
        counts += torch.bincount(p, minlength=SEQ_LEN).float()
    hi = SEQ_LEN - T - TXCPostDissect.S_MAX
    assert counts[hi + 1:].sum() == 0          # never beyond the bound
    support = counts[: hi + 1]
    assert (support > 0).all()                 # every legal offset hit
    n, mean = support.sum(), support.mean()
    assert (support.max() - support.min()) < 6 * math.sqrt(float(mean))


# ── invalid shifts rejected ──

def test_shift_bounds_enforced():
    with pytest.raises(ValueError):
        _mk(TXCPostDissect, ctr_alpha=1.0, ctr_shifts=(1, 3))
