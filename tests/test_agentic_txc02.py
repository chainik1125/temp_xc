"""Contract tests for the agentic_txc_02 trainable port (paper RLHF TXC arm).

Mirrors the probing-plugin contract-test shape: T=1 degeneration,
exact-k/ReLU receipt, matryoshka nesting, multiscale InfoNCE weights,
shift-1 pair sampling, plateau freeze, anchor state-dict compatibility.
All CPU, small dims.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from temp_bench.archs.agentic_txc02 import AgenticTXC02, _info_nce


def _mk(T=5, d_in=16, d_sae=20, k=None, **kw):
    return AgenticTXC02(d_in=d_in, d_sae=d_sae, T=T,
                        k_pos=(k if k is not None else 4 * T), **kw)


def test_registry_entry_constructs():
    from temp_bench.core.config import list_archs
    assert "agentic_txc_02_v1t" in list_archs()


def test_exact_k_topk_relu_receipt():
    torch.manual_seed(0)
    m = _mk(T=5, k=10)
    x = torch.randn(7, 5, 16)
    z = m.encode(x)
    assert z.shape == (7, 20)
    nz = (z != 0).sum(dim=-1)
    assert (nz <= 10).all(), "l0 must be ≤ k_win (ReLU zeroes selected negatives)"
    assert (z >= 0).all(), "codes are post-ReLU, never negative"
    # all-positive pre-acts ⇒ exactly k survive (fingerprint's clean side)
    with torch.no_grad():
        m.b_enc.fill_(1e3)
    z2 = m.encode(x)
    assert ((z2 != 0).sum(dim=-1) == 10).all()


def test_t1_degeneration_matches_manual():
    torch.manual_seed(1)
    m = _mk(T=1, k=3, d_in=8, d_sae=12)
    assert m.n_contr_scales == 1 and m.latent_splits == (12,)
    x = torch.randn(5, 1, 8)
    pre = torch.einsum("btd,tds->bs", x, m.W_enc) + m.b_enc
    vals, idx = pre.topk(3, dim=-1)
    z_manual = torch.zeros_like(pre)
    z_manual.scatter_(1, idx, F.relu(vals))
    assert torch.equal(m.encode(x), z_manual)


def test_matryoshka_nesting_prefix_only():
    torch.manual_seed(2)
    m = _mk(T=4, k=8)
    z = torch.randn(3, 20).abs()
    for s in range(4):
        z_masked = z.clone()
        z_masked[:, m.prefix_sum[s]:] = 999.0  # beyond-prefix junk
        assert torch.equal(m.decode_scale(z, s), m.decode_scale(z_masked, s)), (
            f"scale {s} must ignore latents beyond prefix_sum[{s}]"
        )


def test_multiscale_infonce_weights():
    torch.manual_seed(3)
    m = _mk(T=5, k=10, gamma=0.5)
    x_prev = torch.randn(6, 5, 16)
    x_cur = torch.randn(6, 5, 16)
    total, l_matr, l_contr, _ = m._pair_loss(x_prev, x_cur)
    z_prev, z_cur = m.encode(x_prev), m.encode(x_cur)
    expect = sum(
        (0.5 ** s) * _info_nce(z_cur[:, :m.prefix_sum[s]],
                               z_prev[:, :m.prefix_sum[s]])
        for s in range(3)
    )
    assert torch.allclose(l_contr, expect, atol=1e-5)
    assert torch.allclose(total, l_matr + 1.0 * l_contr, atol=1e-5)


def test_pair_sampling_is_shift1_adjacent():
    torch.manual_seed(4)
    m = _mk(T=3, k=6, d_in=2)
    # position-coded sequences: x[b, l, :] = l
    L = 10
    x = torch.arange(L).float().view(1, L, 1).expand(4, L, 2).contiguous()
    x_prev, x_cur = m._sample_pairs(x)
    assert x_prev.shape == (4, 3, 2) and x_cur.shape == (4, 3, 2)
    # cur window = prev window shifted by exactly +1 position
    assert torch.equal(x_cur[..., 0], x_prev[..., 0] + 1.0)
    # windows are consecutive positions
    diffs = x_prev[:, 1:, 0] - x_prev[:, :-1, 0]
    assert torch.equal(diffs, torch.ones_like(diffs))


def test_plateau_freeze_semantics():
    torch.manual_seed(5)
    m = _mk(T=2, k=4, d_in=8, d_sae=10,
            plateau_min_steps=0, plateau_log_every=1)
    xs = torch.randn(4, 6, 8)
    out = None
    for _ in range(12):
        out = m.train_step(xs)
    assert int(m.converged_step) >= 0, "constant-input loss must plateau"
    assert float(out["converged"]) == 1.0
    out["loss"].backward()
    assert all(p.grad is None for p in m.parameters())
    w = m.W_enc.data.clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    opt.step()
    assert torch.equal(w, m.W_enc.data), "Adam must be a true no-op post-plateau"


def test_no_premature_convergence_before_min_steps():
    torch.manual_seed(6)
    m = _mk(T=2, k=4, d_in=8, d_sae=10,
            plateau_min_steps=1000, plateau_log_every=1)
    xs = torch.randn(4, 6, 8)
    for _ in range(30):
        m.train_step(xs)
    assert int(m.converged_step) == -1, "min_steps must gate the stop"


def test_anchor_state_dict_compat():
    torch.manual_seed(7)
    m = _mk(T=5, k=10)
    # upstream anchors carry exactly the vendored param names, no buffers
    anchor_keys = {k for k, _ in m.named_parameters()}
    assert {"W_enc", "b_enc"} <= anchor_keys
    assert any(k.startswith("W_decs.") for k in anchor_keys)
    anchor = {k: torch.randn_like(v) for k, v in m.named_parameters()}
    missing = m.load_state_dict(anchor, strict=False)
    assert not missing.unexpected_keys
    assert set(missing.missing_keys) <= {"global_step", "converged_step"}
    assert torch.equal(m.W_enc.data, anchor["W_enc"])


def test_decoder_unit_norm_after_post_step():
    torch.manual_seed(8)
    m = _mk(T=3, k=6)
    x = torch.randn(4, 8, 16)
    out = m.train_step(x)
    out["loss"].backward()
    for p in m.parameters():
        if p.grad is not None:
            p.data -= 0.1 * p.grad
    m.post_step()
    for W in m.W_decs:
        norms = W.norm(dim=(1, 2))
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_decode_contract_shape():
    torch.manual_seed(9)
    m = _mk(T=4, k=8)
    z = m.encode(torch.randn(3, 4, 16))
    assert m.decode(z).shape == (3, 4, 16)
    assert m.decoder_directions().shape == (20, 16)
