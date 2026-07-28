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


# ── owner requirement (LOG 20:45, runpod-1): T=1 window≡token identity
#    through the eval_consumes dispatch — dispatch on DECLARED
#    consumption, never on T ────────────────────────────────────────────


def test_eval_consumes_T1_window_token_identity():
    import numpy as np
    from temp_bench.evals.probing import _encode_pool

    torch.manual_seed(0)
    m = _toy(cls=TXCProR1BTKOnly, T_max=1)   # consumes='sequence', eval_consumes='window'
    m.eval()
    dev = torch.device("cpu")
    S = 8
    X = np.random.default_rng(3).standard_normal((5, S, D_IN)).astype(np.float32)
    fr = np.array([0, 2, 4, 7, 7], dtype=np.int64)

    # (a) routes via the WINDOW path: the flat path would call
    # encode((B, S, d_in)) and hard-raise (S != T_max=1) — not crashing
    # IS the dispatch property.
    pw, l0w = _encode_pool(m, X, S=S, batch_size=3, device=dev, first_real=fr)

    # (b) equals per-token pooling of the same map: token-consuming view
    # of the same model, encoding each position independently.
    class _TokView:
        consumes = "token"
        T = 1

        def __init__(self, inner):
            self.inner = inner

        def eval(self):
            return self

        def parameters(self):
            return self.inner.parameters()

        def encode(self, x):            # (B, S, d_in) → (B, S, d_sae)
            B, S_, d = x.shape
            z = self.inner.encode(x.reshape(B * S_, 1, d))   # (B*S, 1, d_sae)
            return z.squeeze(1).reshape(B, S_, -1)

    pt, l0t = _encode_pool(_TokView(m), X, S=S, batch_size=3, device=dev,
                           first_real=fr)
    np.testing.assert_allclose(pw, pt, rtol=1e-5)
    assert abs(l0w - l0t) < 1e-6

    # (c) exact shuffle invariance at T=1 (length-1 window permutation
    # is the identity).
    psh, _ = _encode_pool(m, X, S=S, batch_size=3, device=dev, first_real=fr,
                          shuffle_seed=5)
    np.testing.assert_array_equal(pw, psh)


def test_eval_consumes_T1_probing_eval_identity_flag():
    """ProbingEval end-to-end at T_max=1: shuffle twin reported equal by
    construction (shuffle_identity=1), through the smoke task."""
    from temp_bench.evals.probing import ProbingEval
    from temp_bench.interfaces.evaluator import EvalSpec

    torch.manual_seed(0)
    m = _toy(cls=TXCProR1BTKOnly, T_max=1)
    m.eval()
    spec = EvalSpec(datasource="unused", data_key="unused", smoke=True,
                    extra={"k_feat": 2, "S": 8, "encode_batch_size": 32})
    metrics = ProbingEval().eval(m, spec)
    assert metrics["shuffle_identity"] == 1.0
    assert metrics["mean_auc_shuf"] == metrics["mean_auc"]


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


# ── r1-c4: k_train anneal (C4 low-T-fix lane) ───────────────────────


def test_k_anneal_default_off_bit_identity():
    """Defaults (mult=1, steps=0) must be bit-identical to the pre-C4 arch."""
    torch.manual_seed(3)
    a = _toy(TXCProR1BTKOnly, T_max=4)
    torch.manual_seed(3)
    b = _toy(TXCProR1BTKOnly, T_max=4, k_anneal_mult=1.0, k_anneal_steps=0)
    x = torch.randn(6, 8, D_IN)
    torch.manual_seed(0)
    la, _ = a.train_step(x.clone())
    torch.manual_seed(0)
    lb, _ = b.train_step(x.clone())
    assert torch.equal(la, lb)


def test_k_anneal_schedule_endpoints_monotone_and_clip():
    m = _toy(TXCProR1BTKOnly, T_max=4, k_anneal_mult=8.0, k_anneal_steps=100)
    assert m.k_train == 4
    ks = []
    for s in range(0, 130, 10):
        m._anneal_step = s
        ks.append(m._k_train_now())
    assert ks[0] == 32                      # mult·k_train at step 0
    assert ks[-1] == m.k_train              # nominal after anneal ends
    assert all(k1 >= k2 for k1, k2 in zip(ks, ks[1:]))
    big = _toy(TXCProR1BTKOnly, T_max=4, k_anneal_mult=100.0, k_anneal_steps=10)
    big._anneal_step = 0
    assert big._k_train_now() == D_SAE      # clipped at dict size


def test_k_anneal_widens_train_l0_then_returns_to_budget():
    m = _toy(TXCProR1BTKOnly, T_max=4, k_anneal_mult=8.0, k_anneal_steps=2)
    m.train()
    x = torch.randn(6, 8, D_IN)
    _, info0 = m.train_step(x)
    assert float(info0["l0"]) > m.k_train          # wide admission early
    m.train_step(x)
    _, info2 = m.train_step(x)                     # _anneal_step ≥ steps
    assert float(info2["l0"]) <= m.k_train + 1e-6  # back to nominal budget


def test_k_anneal_serve_path_untouched():
    m = _toy(TXCProR1BTKOnly, T_max=4, k_anneal_mult=8.0, k_anneal_steps=100)
    m.eval()
    x = torch.randn(5, 4, D_IN)
    with torch.no_grad():
        z = m.encode(x)
    l0 = (z != 0).sum(dim=-1).float().mean().item()
    assert l0 == pytest.approx(m.k_inference)


def test_k_anneal_rejects_bad_values():
    with pytest.raises(ValueError):
        _toy(TXCProR1BTKOnly, T_max=4, k_anneal_mult=0.5)
    with pytest.raises(ValueError):
        _toy(TXCProR1BTKOnly, T_max=4, k_anneal_steps=-1)


# ── r1-c5: train-time batch-pool admission ──────────────────────────


def test_train_select_default_row_bit_identity():
    """Default train_select='row' must be bit-identical to the pre-C5 arch."""
    torch.manual_seed(5)
    a = _toy(TXCProR1BTKOnly, T_max=4)
    torch.manual_seed(5)
    b = _toy(TXCProR1BTKOnly, T_max=4, train_select="row")
    x = torch.randn(6, 8, D_IN)
    torch.manual_seed(1)
    la, _ = a.train_step(x.clone())
    torch.manual_seed(1)
    lb, _ = b.train_step(x.clone())
    assert torch.equal(la, lb)


def test_batch_pool_exact_total_budget_and_row_variance():
    m = _toy(TXCProR1BTKOnly, T_max=4, train_select="batch")
    pre = torch.zeros(2, D_SAE)
    pre[0] = torch.arange(D_SAE).float() + 100.0   # row 0 dominates the pool
    pre[1] = torch.arange(D_SAE).float() * 0.01
    z = m._sparsify_batch_pool(pre, 4)             # pooled budget = 2·4 = 8
    counts = (z != 0).sum(dim=-1)
    assert int(counts.sum()) == 8                  # exact total (btk signed)
    assert counts.tolist() == [8, 0]               # rows COMPETE — counts vary


def test_batch_pool_arm_composition_contrast():
    """Paper arm zeroes pooled negative survivors; btk arm passes them signed."""
    pre = torch.full((2, D_SAE), -1.0)
    pre[0, :5] = torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0])
    for i in range(D_SAE):
        pre[1, i] = -0.01 * (i + 1)                # least-negative candidates
    paper = _toy(TXCProR1, T_max=4, train_select="batch")
    btk = _toy(TXCProR1BTKOnly, T_max=4, train_select="batch")
    z_p = paper._sparsify_batch_pool(pre.clone(), 4)   # kb=8 > 5 positives
    z_b = btk._sparsify_batch_pool(pre.clone(), 4)
    assert int((z_p != 0).sum()) == 5              # ReLU killed 3 negatives
    assert int((z_b != 0).sum()) == 8              # signed pass-through keeps 8
    assert float(z_b.min()) < 0


def test_batch_pool_serve_path_untouched():
    m = _toy(TXCProR1BTKOnly, T_max=4, train_select="batch")
    m.eval()
    x = torch.randn(5, 4, D_IN)
    with torch.no_grad():
        z = m.encode(x)
    per_row = (z != 0).sum(dim=-1).float()
    assert torch.all(per_row == m.k_inference)     # per-row EXACT k at serve


def test_batch_pool_train_step_runs_and_keeps_mean_budget():
    m = _toy(TXCProR1BTKOnly, T_max=4, train_select="batch")
    m.train()
    x = torch.randn(6, 8, D_IN)
    loss, info = m.train_step(x)
    loss.backward()
    assert torch.isfinite(loss)
    assert float(info["l0"]) == pytest.approx(m.k_train)  # B·k/B mean budget


def test_train_select_rejects_bad_value():
    with pytest.raises(ValueError):
        _toy(TXCProR1BTKOnly, T_max=4, train_select="pooled")


# ── r1-c6: anchor-only recon toggle ─────────────────────────────────


def test_recon_shifts_default_on_bit_identity():
    torch.manual_seed(9)
    a = _toy(TXCProR1BTKOnly, T_max=4)
    torch.manual_seed(9)
    b = _toy(TXCProR1BTKOnly, T_max=4, recon_shifts=True)
    x = torch.randn(6, 8, D_IN)
    torch.manual_seed(2)
    la, _ = a.train_step(x.clone())
    torch.manual_seed(2)
    lb, _ = b.train_step(x.clone())
    assert torch.equal(la, lb)


def test_recon_shifts_off_ignores_shift_exclusive_positions():
    """alpha=0 + recon_shifts=0 → loss depends ONLY on anchor windows."""
    def mk():
        torch.manual_seed(11)
        return _toy(TXCProR1BTKOnly, T_max=4, contrastive_alpha=0.0,
                    recon_shifts=0)
    x1 = torch.randn(6, 6, D_IN)          # seq_len == min_seq → offsets all 0
    x2 = x1.clone()
    x2[:, 4:6, :] += 10.0                 # shift-exclusive positions only
    m1, m2 = mk(), mk()
    torch.manual_seed(3)
    l1, _ = m1.train_step(x1)
    torch.manual_seed(3)
    l2, _ = m2.train_step(x2)
    assert torch.equal(l1, l2)            # anchor-only objective is blind to them
    # sanity: the DEFAULT objective is NOT blind to the same perturbation
    def mk_full():
        torch.manual_seed(11)
        return _toy(TXCProR1BTKOnly, T_max=4, contrastive_alpha=0.0)
    f1, f2 = mk_full(), mk_full()
    torch.manual_seed(3)
    lf1, _ = f1.train_step(x1.clone())
    torch.manual_seed(3)
    lf2, _ = f2.train_step(x2.clone())
    assert not torch.equal(lf1, lf2)


def test_recon_shifts_off_still_trains():
    m = _toy(TXCProR1BTKOnly, T_max=4, contrastive_alpha=0.0, recon_shifts=0)
    m.train()
    x = torch.randn(6, 8, D_IN)
    loss, info = m.train_step(x)
    loss.backward()
    assert torch.isfinite(loss) and m.W_enc.grad is not None
    assert float(info["l0"]) <= m.k_train + 1e-6
