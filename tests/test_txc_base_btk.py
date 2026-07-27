"""Contract tests for TXCBaseBTK — the btk-only twin of the paper TXC.

Pins the ACTMIX btk-only convention (briefings/actmix-shared.md +
actmix-mac-a.md Stage-1): selection over RAW squashed pre-acts, no ReLU
anywhere in the sparsity path, selected negatives pass through signed
(logged as neg_frac), JumpReLU threshold gating unchanged at eval.

Committed before any grid (commit-then-run)."""

from __future__ import annotations

import torch

from temp_bench.archs.txc_base import TXCBase
from temp_bench.archs.txc_base_btk import TXCBaseBTK

D_IN, D_SAE, T, K_POS = 16, 20, 4, 2   # k_win = 8 < d_sae = 20
K_WIN = K_POS * T


def _mk(cls, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(d_in=D_IN, d_sae=D_SAE, T=T, k_pos=K_POS, **kw)


def _win_batch(B=8, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D_IN, generator=g)


def _pin_preacts(model, b_enc_vals: torch.Tensor) -> None:
    """Zero the encoder so pre-acts == b_enc for every window (distinct,
    deterministic selection pool)."""
    with torch.no_grad():
        model.W_enc.zero_()
        model.b_enc.copy_(b_enc_vals)


def _distinct_bias(n_pos: int) -> torch.Tensor:
    """d_sae distinct values, exactly ``n_pos`` of them positive; the
    top-K_WIN by value = the n_pos positives + the least-negative rest."""
    vals = -torch.arange(1.0, D_SAE + 1.0)          # all negative, distinct
    vals[:n_pos] = torch.arange(1.0, n_pos + 1.0)   # n_pos positives
    return vals


# ── 1. structural twin: identical params at identical seed ──

def test_param_twin_of_txc_base():
    base = _mk(TXCBase)
    btk = _mk(TXCBaseBTK)
    for (nb, pb), (nk, pk) in zip(
        base.named_parameters(), btk.named_parameters()
    ):
        assert nb == nk
        assert torch.equal(pb, pk), f"init drift on {nb}"


# ── 2. THE distinction: selected negatives survive (btk) vs zeroed (composite) ──

def test_selected_negatives_pass_through_signed():
    n_pos = K_WIN - 2                       # 2 selected slots must be negative
    bias = _distinct_bias(n_pos)
    btk = _mk(TXCBaseBTK)
    base = _mk(TXCBase)
    _pin_preacts(btk, bias)
    _pin_preacts(base, bias)
    x = _win_batch(B=4)

    z_btk = btk.encode(x).squeeze(1)        # training mode: BatchTopK path
    z_base = base.encode(x).squeeze(1)      # composite: topk then relu

    # btk-only: every window carries K_WIN nonzeros, 2 of them negative,
    # values passed through unchanged.
    assert int((z_btk != 0).sum(dim=-1)[0]) == K_WIN
    assert int((z_btk < 0).sum(dim=-1)[0]) == 2
    top_ids = torch.topk(bias, K_WIN).indices
    for j in top_ids:
        assert torch.allclose(z_btk[0, j], bias[j]), "value not passed through"

    # composite: the same 2 selected-negative slots are zeroed after
    # selection -> support shrinks to K_WIN - 2. This is the paper harm.
    assert int((z_base != 0).sum(dim=-1)[0]) == K_WIN - 2
    assert int((z_base < 0).sum()) == 0


# ── 3. realized-l0 sanity: BatchTopK never zero-picks ──

def test_realized_l0_equals_budget():
    btk = _mk(TXCBaseBTK)
    x = _win_batch(B=8)
    out = btk.train_step(x)
    assert abs(float(out["l0"]) - K_WIN) < 1e-6, (
        f"mean realized l0 {float(out['l0'])} != k_win {K_WIN}"
    )
    assert "neg_frac" in out
    # generic gaussian init: pre-acts are signed; some negative picks are
    # possible but not guaranteed — just require a well-formed fraction.
    assert 0.0 <= float(out["neg_frac"]) <= 1.0


def test_neg_frac_fingerprint_under_thin_positive_pool():
    n_pos = K_WIN // 2
    btk = _mk(TXCBaseBTK)
    _pin_preacts(btk, _distinct_bias(n_pos))
    out = btk.train_step(_win_batch(B=4))
    # l0 stays at budget (no zero-picking), half the picks are negative.
    assert abs(float(out["l0"]) - K_WIN) < 1e-6
    assert abs(float(out["neg_frac"]) - 0.5) < 1e-6


# ── 4. B=1 bridge: batch pool degenerates to per-window selection ──

def test_b1_positive_regime_matches_composite():
    bias = _distinct_bias(K_WIN + 4)        # top-K_WIN all positive
    btk = _mk(TXCBaseBTK)
    base = _mk(TXCBase)
    _pin_preacts(btk, bias)
    _pin_preacts(base, bias)
    x = _win_batch(B=1)
    z_btk = btk.encode(x)
    z_base = base.encode(x)
    assert torch.allclose(z_btk, z_base, atol=1e-6), (
        "with B=1 and an all-positive selection pool, btk-only must "
        "coincide with the composite (ReLU is a no-op there)"
    )


# ── 5. threshold path: eval gating kicks in, negatives gated out ──

def test_jumprelu_eval_path():
    btk = _mk(TXCBaseBTK, threshold_start_step=1)
    x = _win_batch(B=8)
    for _ in range(5):
        btk.train_step(x)
    assert bool(btk.threshold_set.item()), "threshold_set flag never set"
    btk.eval()
    z = btk.encode(x).squeeze(1)
    assert int((z < 0).sum()) == 0, "negatives must be gated out at eval"
    thr = float(btk.threshold)
    nz = z[z != 0]
    if nz.numel():
        assert float(nz.min().detach()) > thr


def test_negative_threshold_representable():
    # canonical item 2: a legitimately-negative threshold must gate, not
    # silently fall back to batch-dependent TopK (the -1.0 sentinel bug).
    btk = _mk(TXCBaseBTK, threshold_start_step=1)
    n_pos = K_WIN // 2
    _pin_preacts(btk, _distinct_bias(n_pos))
    for _ in range(5):
        btk.train_step(_win_batch(B=4))
    assert bool(btk.threshold_set.item())
    assert float(btk.threshold) < 0.0, (
        "EMA over signed survivors should be negative here "
        f"(got {float(btk.threshold)})"
    )
    btk.eval()
    z = btk.encode(_win_batch(B=4)).squeeze(1)
    # gating admits values above the (negative) threshold — including the
    # weakly-negative survivors — and excludes those below it.
    bias = btk.b_enc.detach()
    admitted = (bias > btk.threshold).sum()
    assert int((z[0] != 0).sum()) == int(admitted)


# ── 6. trainability: loss backward + decoder grad hook + post_step ──

def test_train_step_backward_and_unit_norm():
    btk = _mk(TXCBaseBTK)
    out = btk.train_step(_win_batch(B=8))
    out["loss"].backward()
    assert btk.W_dec.grad is not None and torch.isfinite(btk.W_dec.grad).all()
    # grad-parallel removal: grad ⟂ decoder rows (post-accumulate hook).
    g = btk.W_dec.grad.view(D_SAE, -1)
    w = btk.W_dec.data.view(D_SAE, -1)
    w_hat = w / (w.norm(dim=1, keepdim=True) + 1e-6)
    par = (g * w_hat).sum(dim=1).abs().max()
    assert float(par) < 1e-4
    btk.post_step()
    norms = btk.W_dec.data.norm(dim=(1, 2))
    assert torch.allclose(norms, torch.ones(D_SAE), atol=1e-5)


# ── 7. registry: YAML entry resolves and instantiates per-section ──

def test_registry_entry_resolves():
    from temp_bench.core.config import import_by_path, load_arch
    spec = load_arch("txc_base_btkonly", section="synthetic")
    assert spec.hparams["d_sae"] == 20          # per-section override
    cls = import_by_path(spec.class_path)
    model = cls(d_in=D_IN, **spec.hparams)
    assert model.T == spec.hparams.get("T", 5)
    assert model.arch_version == spec.arch_version
    assert model.relu_mode == "btk-only"
    spec_rm = load_arch("txc_base_relumix", section="synthetic")
    model_rm = cls(d_in=D_IN, **spec_rm.hparams)
    assert model_rm.relu_mode == "relu-mix"
    import pytest
    with pytest.raises(ValueError):
        cls(d_in=D_IN, **{**spec.hparams, "relu_mode": "paper-match"})


# ── 8. relu-mix control arm: zero-picks when positives are thin ──

def test_relumix_zero_pick_fingerprint():
    n_pos = K_WIN // 2
    bias = _distinct_bias(n_pos)
    rm = _mk(TXCBaseBTK, relu_mode="relu-mix")
    _pin_preacts(rm, bias)
    out = rm.train_step(_win_batch(B=4))
    # ReLU'd pool has only n_pos positives per window -> BatchTopK can
    # only fill half the budget: realized l0 == n_pos, zero negatives.
    assert abs(float(out["l0"]) - n_pos) < 1e-6, (
        f"relu-mix should zero-pick down to {n_pos}, got {float(out['l0'])}"
    )
    assert float(out["neg_frac"]) == 0.0
    # btk-only twin on the same inputs realizes the full budget.
    btk = _mk(TXCBaseBTK)
    _pin_preacts(btk, bias)
    out_b = btk.train_step(_win_batch(B=4))
    assert abs(float(out_b["l0"]) - K_WIN) < 1e-6


# ── 9. perwin-raw fourth corner: composite's selection scope, no ReLU ──

def test_perwinraw_negatives_survive_per_window():
    n_pos = K_WIN - 2
    bias = _distinct_bias(n_pos)
    pw = _mk(TXCBaseBTK, relu_mode="perwin-raw")
    base = _mk(TXCBase)
    _pin_preacts(pw, bias)
    _pin_preacts(base, bias)
    x = _win_batch(B=4)
    z_pw = pw.encode(x).squeeze(1)
    z_base = base.encode(x).squeeze(1)
    # Same selection scope as the composite: every window exactly K_WIN
    # picks — but the 2 selected negatives survive signed here.
    assert int((z_pw != 0).sum(dim=-1)[0]) == K_WIN
    assert int((z_pw < 0).sum(dim=-1)[0]) == 2
    assert int((z_base != 0).sum(dim=-1)[0]) == K_WIN - 2
    # Positive-rich regime: identical to the composite.
    bias_pos = _distinct_bias(K_WIN + 4)
    pw2 = _mk(TXCBaseBTK, relu_mode="perwin-raw")
    base2 = _mk(TXCBase)
    _pin_preacts(pw2, bias_pos)
    _pin_preacts(base2, bias_pos)
    assert torch.allclose(pw2.encode(x), base2.encode(x), atol=1e-6)


# ── 10. budget-scan knob: explicit k_win overrides the k_pos*T rule ──

def test_explicit_k_win_override():
    btk = _mk(TXCBaseBTK, k_win=3)
    assert btk.k_win == 3                        # independent of k_pos*T=8
    out = btk.train_step(_win_batch(B=8))
    assert abs(float(out["l0"]) - 3) < 1e-6
    pw = _mk(TXCBaseBTK, relu_mode="perwin-raw", k_win=3)
    z = pw.encode(_win_batch(B=4)).squeeze(1)
    assert int((z != 0).sum(dim=-1)[0]) == 3
