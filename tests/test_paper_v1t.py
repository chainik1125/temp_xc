"""paper_txc_base_v1t (paper-faithful trainable TXC-base) — contract tests.

Commissioned by the paper-faithful sprint (4ce0369de item 1): the plugin
must prove, before any GPU cell runs:

1. adapter parity: with transplanted weights the trainable class's
   encode/decode is BITWISE equal to the eval-only shipped-checkpoint
   adapter (``paper_v1.PaperTXCBaseV1``) — same math as what evaluates
   the paper's archived T5 anchors. Checked at T=3 and at T=1 (the
   commissioned "T=1 degeneration" case).
2. exact-k receipt: on positive-rich pre-acts every window's selected
   set is exactly k_win = k_pos·T nonzeros (l0 == k_win).
3. paper mixing fingerprint: on scarce-positive pre-acts realized
   l0 < k_win (TopK→ReLU zeroes selected negatives — the composition
   the sprint exists to measure).
4. training stack: first-batch geometric-median b_dec init fires once;
   loss is finite and backward runs; the post-accumulate hook keeps
   W_dec.grad orthogonal to each decoder atom; post_step re-normalizes
   decoder atoms to unit norm.
5. registry: the YAML entry loads and instantiates with override T.
"""

from __future__ import annotations

import pytest
import torch

from temp_bench.archs.paper_v1 import PaperTXCBaseV1
from temp_bench.archs.paper_v1t import PaperTXCBaseV1T
from temp_bench.core.config import load_arch

D_IN, D_SAE, K_POS = 16, 64, 4


def _pair(T: int):
    torch.manual_seed(0)
    trainable = PaperTXCBaseV1T(d_in=D_IN, d_sae=D_SAE, k_pos=K_POS, T=T)
    adapter = PaperTXCBaseV1(d_in=D_IN, d_sae=D_SAE, k_pos=K_POS, T=T)
    # Transplant shared weights (adapter's state is the strict subset).
    sd = trainable.state_dict()
    adapter.load_state_dict(
        {k: v for k, v in sd.items() if k in adapter.state_dict()})
    return trainable, adapter


@pytest.mark.parametrize("T", [1, 3])
def test_adapter_parity_encode_decode(T):
    trainable, adapter = _pair(T)
    x = torch.randn(8, T, D_IN)
    z_t = trainable.encode(x)
    z_a = adapter.encode(x)
    assert torch.equal(z_t, z_a), "trainable encode drifted from adapter"
    assert torch.equal(trainable.decode(z_t), adapter.decode(z_a))


def test_t1_degeneration_formula():
    """At T=1 the composition is scatter(relu(topk(x@W_enc[0]+b_enc)))."""
    trainable, _ = _pair(1)
    x = torch.randn(8, 1, D_IN)
    pre = x[:, 0, :] @ trainable.inner.W_enc[0] + trainable.inner.b_enc
    vals, idx = pre.topk(K_POS, dim=-1)
    want = torch.zeros_like(pre).scatter_(
        1, idx, torch.nn.functional.relu(vals))
    assert torch.equal(trainable.encode(x), want)


def test_exact_k_on_positive_rich():
    T = 2
    trainable, _ = _pair(T)
    with torch.no_grad():
        trainable.inner.b_enc.fill_(10.0)      # all pre-acts positive
    x = torch.randn(8, T, D_IN)
    z = trainable.encode(x)
    l0 = (z != 0).sum(dim=-1)
    assert torch.all(l0 == K_POS * T), f"l0 != k_win: {l0.tolist()}"


def test_mixing_fingerprint_scarce_positive():
    T = 2
    trainable, _ = _pair(T)
    with torch.no_grad():
        trainable.inner.b_enc.fill_(-100.0)    # all pre-acts negative
    x = 0.01 * torch.randn(8, T, D_IN)
    z = trainable.encode(x)
    l0 = (z != 0).sum(dim=-1)
    assert torch.all(l0 < K_POS * T), "ReLU-after-TopK must zero negatives"
    assert torch.all(z >= 0)


def test_train_step_stack():
    T = 2
    torch.manual_seed(1)
    m = PaperTXCBaseV1T(d_in=D_IN, d_sae=D_SAE, k_pos=K_POS, T=T)
    x = torch.randn(32, T, D_IN)

    assert not bool(m.inner.b_dec_initialized)
    out = m.train_step(x)
    assert bool(m.inner.b_dec_initialized), "first batch must init b_dec"
    # b_dec[t] is the geometric median of position t (median ≈ near mean
    # for gaussian batch; just assert it moved off zero).
    assert m.inner.b_dec.abs().sum() > 0

    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    # Post-accumulate hook: grad ⟂ decoder atoms.
    g = m.inner.W_dec.grad.view(D_SAE, -1)
    w = m.inner.W_dec.data.view(D_SAE, -1)
    dots = (g * w / (w.norm(dim=1, keepdim=True) + 1e-6)).sum(dim=1)
    assert float(dots.abs().max()) < 1e-4, "decoder-parallel grad survived"

    # post_step: unit-norm decoder atoms.
    with torch.no_grad():
        m.inner.W_dec.add_(0.3 * torch.randn_like(m.inner.W_dec))
    m.post_step()
    norms = m.inner.W_dec.data.norm(dim=(1, 2))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Second step must NOT re-init b_dec (assert would fire inside).
    m.train_step(x)


def test_second_step_no_reinit_and_l0_metric():
    T = 2
    m = PaperTXCBaseV1T(d_in=D_IN, d_sae=D_SAE, k_pos=K_POS, T=T)
    x = torch.randn(16, T, D_IN)
    out1 = m.train_step(x)
    out2 = m.train_step(x)
    for out in (out1, out2):
        assert 0.0 <= float(out["l0"]) <= K_POS * T
        assert float(out["dead"]) >= 0.0


def test_registry_loads_with_T_override():
    spec = load_arch("paper_txc_base_v1t")
    assert spec.arch_version == "upstream-94119bc08-trainable-1.0.0"
    cls_module, cls_name = spec.class_path.split(":")
    assert cls_name == "PaperTXCBaseV1T"
    hp = dict(spec.hparams)
    hp.update({"T": 3, "d_sae": 64, "k_pos": 4})
    m = PaperTXCBaseV1T(d_in=D_IN, **hp)
    assert m.inner.T == 3 and m.inner.k == 12
    assert m.consumes == "window"
