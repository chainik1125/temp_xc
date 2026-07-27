"""txc_pro_r1 contract tests + the eval_consumes dispatch-equivalence assert.

Covers CARD_SPLIT § 4 pre-registrations: ratio rule, budget mapping,
composition twins (ReLU vs signed pass-through), subseq masking math,
window-dispatch generalization safety (no existing arch declares
eval_consumes), and the full canonical ProbingEval smoke path through the
window dispatch.
"""

from __future__ import annotations

import pytest
import torch

from temp_bench.archs.txc_pro_r1 import TXCProR1, TXCProR1BTKOnly
from temp_bench.core.config import import_by_path, list_archs, load_arch

D_IN, D_SAE = 16, 40


def _toy(cls=TXCProR1, T_max=3, **kw):
    kw.setdefault("h_size", D_SAE)     # toy d_sae: disable matryoshka clip
    kw.setdefault("k_pos", 2)
    return cls(d_in=D_IN, d_sae=D_SAE, T_max=T_max, **kw)


# ── dispatch-equivalence: the probing.py generalization is a no-op for
#    every arch that predates the attr ─────────────────────────────────


def test_no_existing_arch_declares_eval_consumes():
    for name in list_archs():
        cls = import_by_path(load_arch(name).class_path)
        declares = "eval_consumes" in {
            k for klass in cls.__mro__ for k in vars(klass)
        }
        if name.startswith("txc_pro_r1"):
            assert declares, f"{name} must declare eval_consumes"
        else:
            assert not declares, (
                f"{name} declares eval_consumes — the dispatch equivalence "
                "guarantee (probing.py) no longer holds; re-audit."
            )


def test_dispatch_expression_matches_old_for_undeclared():
    class Tok:
        consumes = "token"

    class Win:
        consumes = "window"

    for m in (Tok(), Win()):
        old = getattr(m, "consumes", "token") == "window"
        new = getattr(m, "eval_consumes", getattr(m, "consumes", "token")) == "window"
        assert old == new
    r1 = _toy()
    assert getattr(r1, "eval_consumes") == "window" and r1.consumes == "sequence"


# ── constructor rules ─────────────────────────────────────────────────


def test_ratio_rule_and_alias():
    assert _toy(T_max=10).t_sample == 5     # locked instance = fixed point
    assert _toy(T_max=1).t_sample == 1
    assert _toy(T_max=16).t_sample == 8
    assert _toy(T_max=10, t_sample=5).t_sample == 5
    assert _toy(T_max=None, T=4).t_sample == 2          # alias path
    assert _toy(T_max=None, T=4).T_max == 4
    with pytest.raises(ValueError):
        TXCProR1(d_in=D_IN, d_sae=D_SAE, T=4, T_max=10, h_size=D_SAE, k_pos=2)


def test_budget_mapping():
    m = _toy(T_max=10)
    assert m.k_train == 2 * 5 and m.k_inference == 2 * 10
    assert m.T == 10                       # eval reads model.T = T_max


def test_relu_mode_assert():
    assert _toy(relu_mode="paper-match").relu_mode == "paper-match"
    with pytest.raises(ValueError):
        _toy(relu_mode="btk-only")
    assert _toy(cls=TXCProR1BTKOnly, relu_mode="btk-only").relu_mode == "btk-only"


# ── composition twins ─────────────────────────────────────────────────


def test_sparsify_paper_zeroes_selected_negatives():
    m = _toy()
    pre = torch.full((2, D_SAE), -10.0)
    pre[0, :3] = torch.tensor([5.0, -1.0, 4.0])   # top-6 must include negatives
    z = m._sparsify(pre, 6)
    assert float(z[0, 0]) == 5.0 and float(z[0, 2]) == 4.0
    assert float(z[0, 1]) == 0.0                  # selected negative → zeroed
    assert (z >= 0).all()


def test_sparsify_btkonly_passes_signed():
    m = _toy(cls=TXCProR1BTKOnly)
    pre = torch.full((2, D_SAE), -10.0)
    pre[0, :3] = torch.tensor([5.0, -1.0, 4.0])
    z = m._sparsify(pre, 6)
    assert float(z[0, 1]) == -1.0                 # survivor passes signed
    assert int((z[0] != 0).sum()) == 6            # realized l0 == nominal


def test_encode_l0_exact_btkonly_and_bounded_paper():
    torch.manual_seed(0)
    x = torch.randn(8, 3, D_IN)
    mb = _toy(cls=TXCProR1BTKOnly)
    zb = mb.encode(x).squeeze(1)
    assert (zb != 0).sum(-1).float().mean() == mb.k_inference
    mp = _toy()
    zp = mp.encode(x).squeeze(1)
    assert ((zp != 0).sum(-1) <= mp.k_inference).all()


def test_encode_rejects_wrong_T():
    m = _toy(T_max=3)
    with pytest.raises(ValueError):
        m.encode(torch.randn(4, 5, D_IN))


# ── subseq math ───────────────────────────────────────────────────────


def test_pre_activation_sampled_equals_manual_sum():
    torch.manual_seed(0)
    m = _toy(T_max=4)
    x = torch.randn(3, 4, D_IN)
    idx = torch.tensor([[0, 2], [1, 3], [0, 1]])
    pre = m._pre_activation_sampled(x, idx)
    for b in range(3):
        manual = sum(x[b, t] @ m.W_enc[t] for t in idx[b].tolist()) + m.b_enc
        assert torch.allclose(pre[b], manual, atol=1e-5)


# ── training loop mechanics ───────────────────────────────────────────


@pytest.mark.parametrize("cls", [TXCProR1, TXCProR1BTKOnly])
def test_train_step_backward_and_post_step(cls):
    torch.manual_seed(0)
    m = _toy(cls=cls, T_max=3)
    x = torch.randn(6, 32, D_IN)
    loss, info = m.train_step(x)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.W_enc.grad is not None and m.W_dec.grad is not None
    m.post_step()
    norms = m.W_dec.data.norm(dim=(1, 2))
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    for k in ("mse", "l0", "contrastive", "recon_h", "neg_frac"):
        assert k in info


def test_train_step_rejects_short_sequences():
    m = _toy(T_max=3)   # min_seq = 3 + 2
    with pytest.raises(ValueError):
        m.train_step(torch.randn(2, 4, D_IN))


def test_contrastive_weights_inverse_distance():
    m = _toy()
    assert m.loss_weights == (1.0 / 2.0, 1.0 / 3.0)
    m2 = _toy(contrastive_inverse_distance_weight=False)
    assert m2.loss_weights == (1.0, 1.0)


# ── canonical eval path (hermetic smoke, full window dispatch) ────────


@pytest.mark.parametrize("cls", [TXCProR1, TXCProR1BTKOnly])
def test_probing_eval_smoke_window_dispatch(cls):
    from temp_bench.evals.probing import ProbingEval
    from temp_bench.interfaces.evaluator import EvalSpec

    torch.manual_seed(0)
    m = _toy(cls=cls, T_max=3)
    m.eval()
    spec = EvalSpec(datasource="unused", data_key="unused", smoke=True,
                    extra={"k_feat": 2, "S": 8, "encode_batch_size": 32})
    metrics = ProbingEval().eval(m, spec)
    assert metrics["n_tasks"] == 1.0
    assert 0.0 <= metrics["mean_auc"] <= 1.0
    assert "mean_auc_shuf" in metrics          # T=3 → real shuffle twin
    if cls is TXCProR1BTKOnly:
        assert abs(metrics["realized_l0"] - m.k_inference) < 1e-6
