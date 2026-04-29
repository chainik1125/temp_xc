"""CPU smoke tests for Z's Phase 7 SubseqH8 variants.

Catches dimension bugs in `phase7_subseq_z_variants.py` without GPU or
the activation cache. Runs in <1 sec.
"""
from __future__ import annotations

import torch

from src.architectures.phase7_subseq_z_variants import (
    SubseqRankedH8,
    SubseqSharedH8,
    _draw_sample_idx,
    _equally_spaced_offsets,
)


# --- helpers -----------------------------------------------------------------

def _ranked_kw(T_max: int = 12, t_sample: int = 4, d_sae: int = 64):
    return dict(
        d_in=16, d_sae=d_sae, T_max=T_max, t_sample=t_sample, k=10,
        shifts=(1, max(1, T_max // 4), max(1, T_max // 2)),
        matryoshka_h_size=int(d_sae * 0.2),
        alpha=1.0,
    )


def _shared_kw(T_max: int = 12, d_sae: int = 64):
    return dict(
        d_in=16, d_sae=d_sae, T_max=T_max, k=10,
        shifts=(1, max(1, T_max // 4), max(1, T_max // 2)),
        matryoshka_h_size=int(d_sae * 0.2),
        alpha=1.0,
    )


# --- helpers tests -----------------------------------------------------------

def test_draw_sample_idx_sorted_and_in_range():
    idx = _draw_sample_idx(T_max=20, t_sample=5, B=8,
                            contiguous=False, device=torch.device("cpu"))
    assert idx.shape == (8, 5)
    # Sorted ascending
    assert (idx[:, 1:] >= idx[:, :-1]).all()
    # In [0, T_max)
    assert (idx >= 0).all() and (idx < 20).all()
    # No duplicates within a row
    sorted_vals, _ = idx.sort(dim=-1)
    diffs = sorted_vals[:, 1:] - sorted_vals[:, :-1]
    assert (diffs > 0).all()


def test_draw_sample_idx_contiguous():
    idx = _draw_sample_idx(T_max=20, t_sample=5, B=8,
                            contiguous=True, device=torch.device("cpu"))
    diffs = idx[:, 1:] - idx[:, :-1]
    assert (diffs == 1).all()


def test_equally_spaced_offsets():
    off = _equally_spaced_offsets(T_max=20, t_sample=5,
                                    device=torch.device("cpu"))
    assert off.shape == (5,)
    # Roughly evenly spaced — strictly increasing in [0, T_max)
    assert (off >= 0).all() and (off < 20).all()
    assert (off[1:] > off[:-1]).all()


# --- SubseqRankedH8 ----------------------------------------------------------

def test_ranked_multidist_forward_shapes():
    torch.manual_seed(0)
    m = SubseqRankedH8(**_ranked_kw(T_max=12, t_sample=4))
    K = len(m.shifts)
    x = torch.randn(3, 1 + K, 12, 16)
    total, x_hat, z = m(x)
    assert total.dim() == 0
    assert torch.isfinite(total)
    # Decoded slot output has slot-many positions, not T_max-many
    assert x_hat.shape == (3, 4, 16)
    assert z.shape == (3, 64)


def test_ranked_single_window_forward():
    torch.manual_seed(0)
    m = SubseqRankedH8(**_ranked_kw(T_max=12, t_sample=4))
    x = torch.randn(3, 12, 16)
    total, x_hat, z = m(x)
    assert torch.isfinite(total)
    assert x_hat.shape == (3, 4, 16)


def test_ranked_encode_T_max_input():
    torch.manual_seed(0)
    m = SubseqRankedH8(**_ranked_kw(T_max=12, t_sample=4))
    x = torch.randn(5, 12, 16)
    z = m.encode(x)
    assert z.shape == (5, 64)
    assert (z >= 0).all()
    # k=10 so at most 10 nonzeros per row
    nonzeros = (z > 0).sum(dim=-1)
    assert (nonzeros <= m.k).all()


def test_ranked_encode_t_sample_input():
    """Already-gathered slot input also accepted."""
    torch.manual_seed(0)
    m = SubseqRankedH8(**_ranked_kw(T_max=12, t_sample=4))
    x = torch.randn(5, 4, 16)
    z = m.encode(x)
    assert z.shape == (5, 64)


def test_ranked_init_b_dec_from_T_max_window():
    torch.manual_seed(0)
    m = SubseqRankedH8(**_ranked_kw(T_max=12, t_sample=4))
    x = torch.randn(20, 12, 16)
    m.init_b_dec_geometric_median(x)
    assert bool(m.b_dec_initialized)
    # b_dec is (t_sample, d) per parent
    assert m.b_dec.shape == (4, 16)


def test_ranked_T_max_eq_t_sample_collapses_to_subseq_h8():
    """When t_sample == T_max, no rank-routing — every slot gets its own
    fixed position. Should still train without error."""
    torch.manual_seed(0)
    m = SubseqRankedH8(**_ranked_kw(T_max=5, t_sample=5))
    K = len(m.shifts)
    x = torch.randn(3, 1 + K, 5, 16)
    total, x_hat, z = m(x)
    assert torch.isfinite(total)
    assert x_hat.shape == (3, 5, 16)


def test_ranked_backward_runs():
    """Loss is differentiable end-to-end."""
    torch.manual_seed(0)
    m = SubseqRankedH8(**_ranked_kw(T_max=12, t_sample=4))
    K = len(m.shifts)
    x = torch.randn(4, 1 + K, 12, 16)
    total, _, _ = m(x)
    total.backward()
    assert m.W_enc.grad is not None
    assert m.W_dec.grad is not None
    assert torch.isfinite(m.W_enc.grad).all()


# --- SubseqSharedH8 ----------------------------------------------------------

def test_shared_multidist_forward_shapes():
    torch.manual_seed(0)
    m = SubseqSharedH8(**_shared_kw(T_max=12))
    K = len(m.shifts)
    x = torch.randn(3, 1 + K, 12, 16)
    total, x_hat, z = m(x)
    assert torch.isfinite(total)
    # decode is (B, d) not (B, T, d)
    assert x_hat.shape == (3, 16)
    assert z.shape == (3, 64)


def test_shared_single_window_forward():
    torch.manual_seed(0)
    m = SubseqSharedH8(**_shared_kw(T_max=12))
    x = torch.randn(3, 12, 16)
    total, x_hat, z = m(x)
    assert torch.isfinite(total)
    assert x_hat.shape == (3, 16)


def test_shared_encode_shapes():
    torch.manual_seed(0)
    m = SubseqSharedH8(**_shared_kw(T_max=12))
    x = torch.randn(5, 12, 16)
    z = m.encode(x)
    assert z.shape == (5, 64)
    nonzeros = (z > 0).sum(dim=-1)
    assert (nonzeros <= m.k).all()


def test_shared_init_b_dec():
    torch.manual_seed(0)
    m = SubseqSharedH8(**_shared_kw(T_max=12))
    x = torch.randn(20, 12, 16)
    m.init_b_dec_geometric_median(x)
    assert bool(m.b_dec_initialized)
    assert m.b_dec.shape == (16,)


def test_shared_decoder_unit_norm():
    """After _normalize_decoder, each W_dec row is unit-norm."""
    torch.manual_seed(0)
    m = SubseqSharedH8(**_shared_kw(T_max=12))
    norms = m.W_dec.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_shared_param_count_under_threshold():
    """Shared variant should be ~T_max-x smaller than ranked variant."""
    m_shared = SubseqSharedH8(**_shared_kw(T_max=20))
    m_ranked = SubseqRankedH8(**_ranked_kw(T_max=20, t_sample=5))
    p_shared = sum(p.numel() for p in m_shared.parameters())
    p_ranked = sum(p.numel() for p in m_ranked.parameters())
    # Shared should be markedly smaller (~5x for these dims since
    # ranked has t_sample=5 slabs vs shared's 1).
    assert p_shared < p_ranked / 3, (
        f"shared {p_shared} not << ranked {p_ranked}"
    )


def test_shared_backward_runs():
    torch.manual_seed(0)
    m = SubseqSharedH8(**_shared_kw(T_max=12))
    K = len(m.shifts)
    x = torch.randn(4, 1 + K, 12, 16)
    total, _, _ = m(x)
    total.backward()
    assert m.W_enc.grad is not None
    assert m.W_dec.grad is not None
    assert torch.isfinite(m.W_enc.grad).all()


def test_shared_remove_grad_parallel():
    """After remove_gradient_parallel_to_decoder, W_dec.grad has zero
    component along each row of W_dec."""
    torch.manual_seed(0)
    m = SubseqSharedH8(**_shared_kw(T_max=12))
    K = len(m.shifts)
    x = torch.randn(4, 1 + K, 12, 16)
    total, _, _ = m(x)
    total.backward()
    m.remove_gradient_parallel_to_decoder()
    parallel = (m.W_dec.grad * m.W_dec.data).sum(dim=1)
    norms = m.W_dec.data.norm(dim=1)
    parallel_normed = parallel / (norms + 1e-6)
    # Tolerance reflects the +1e-6 epsilon in the unit-vector denominator —
    # residual ≈ parallel * (1 - 1/(1+ε)^2) scales with gradient magnitude.
    assert parallel_normed.abs().max() < 1e-3
