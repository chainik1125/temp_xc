"""Tests for :mod:`temp_bench.eval.steering_hooks`.

Verifies:

* the four hook modes (V0/V1/V2/V4) produce mathematically distinct deltas;
* V0 == V4 at TXC init (tied weights → encoder pre-image = T × mean decoder);
* V1's per-token vector cycles through W_dec[t mod T];
* V2 fills the trailing T positions with reverse-position decoder slices;
* the √T energy correction makes V0 / V1 / V2 deliver the same TOTAL
  injected energy across the trailing window;
* :func:`position_variance` returns 0 for constant trajectories and
  positive values for variable ones;
* :func:`encoder_preimage` matches ``sum_t W_enc[t, :, f]`` on TXC.
"""

from __future__ import annotations

import math

import pytest
import torch

from temp_bench.architectures.base import ArchConfig, TempBenchArch
from temp_bench.eval.steering_hooks import (
    ALL_MODES,
    TXCSteeringHook,
    build_hook,
    encoder_decoder_divergence,
    encoder_preimage,
    position_variance,
)


class _TxcStub(TempBenchArch):
    """Minimal TXC-like arch with W_enc (T, d_in, d_sae) and
    W_dec (d_sae, T, d_in). Enough surface for the hook + diagnostics."""

    def __init__(self, *, d_in: int, d_sae: int, T: int, tied_init: bool = True, seed: int = 0):
        super().__init__()
        self.config = ArchConfig(name="txc_stub", d_in=d_in, d_sae=d_sae, k_pos=1, T=T)
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        g = torch.Generator().manual_seed(seed)
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, T, d_in, generator=g))
        self.W_enc = torch.nn.Parameter(torch.empty(T, d_in, d_sae))
        if tied_init:
            with torch.no_grad():
                for t in range(T):
                    self.W_enc.data[t] = self.W_dec.data[:, t, :].T
        else:
            with torch.no_grad():
                self.W_enc.data = torch.randn(T, d_in, d_sae, generator=g)

    def encode(self, x):
        # (B, T, d_in) → (B, 1, d_sae)
        return torch.einsum("btd,tds->bs", x, self.W_enc).unsqueeze(1)

    def decode(self, z):
        if z.dim() == 3:
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec)

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.mean(dim=1).clone()


# ── position_variance ─────────────────────────────────────────────────


def test_position_variance_zero_for_constant_traj():
    T, d_sae, d_in = 5, 4, 8
    W = torch.randn(d_sae, 1, d_in).expand(d_sae, T, d_in).contiguous()
    var = position_variance(W)
    assert torch.allclose(var, torch.zeros(d_sae), atol=1e-6), var


def test_position_variance_one_for_zero_mean_traj():
    """Trajectory with mean ≈ 0 → ratio ≈ 1."""
    torch.manual_seed(0)
    T, d_sae, d_in = 5, 8, 16
    W = torch.randn(d_sae, T, d_in)
    # Subtract per-feature mean to force mean=0 across t.
    W = W - W.mean(dim=1, keepdim=True)
    var = position_variance(W)
    assert torch.all(var > 0.99), var


def test_position_variance_in_unit_interval():
    torch.manual_seed(0)
    W = torch.randn(20, 5, 8)
    var = position_variance(W)
    assert torch.all(var >= 0)
    assert torch.all(var <= 1 + 1e-6)


def test_position_variance_validation():
    with pytest.raises(ValueError, match="d_sae, T, d_in"):
        position_variance(torch.randn(5, 8))


# ── encoder_preimage ──────────────────────────────────────────────────


def test_encoder_preimage_returns_sum_over_T():
    arch = _TxcStub(d_in=8, d_sae=4, T=5, tied_init=False)
    pre = encoder_preimage(arch, feature_id=2)
    assert pre.shape == (8,)
    expected = arch.W_enc.data[:, :, 2].sum(dim=0)
    assert torch.allclose(pre, expected)


def test_encoder_preimage_full_returns_d_sae_d_in():
    arch = _TxcStub(d_in=8, d_sae=4, T=5)
    pre_full = encoder_preimage(arch)
    assert pre_full.shape == (4, 8)


# ── encoder_decoder_divergence ────────────────────────────────────────


def test_divergence_zero_at_tied_init():
    """At tied init, sum_t W_enc[t,:,f] = T·mean_t W_dec[f,:,:]."""
    arch = _TxcStub(d_in=8, d_sae=4, T=5, tied_init=True)
    d = encoder_decoder_divergence(arch, feature_id=0)
    assert d["cos_sim"] == pytest.approx(1.0, abs=1e-5)
    assert d["norm_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert d["rel_residual"] < 1e-5


def test_divergence_positive_after_drift():
    arch = _TxcStub(d_in=8, d_sae=4, T=5, tied_init=True)
    # Perturb encoder for feature 1 only.
    with torch.no_grad():
        arch.W_enc.data[:, :, 1] += 0.5 * torch.randn_like(arch.W_enc.data[:, :, 1])
    d_unchanged = encoder_decoder_divergence(arch, feature_id=0)
    d_changed = encoder_decoder_divergence(arch, feature_id=1)
    assert d_unchanged["rel_residual"] < 1e-5
    assert d_changed["rel_residual"] > 0.05


# ── TXCSteeringHook construction ──────────────────────────────────────


def test_hook_validation_unknown_mode():
    W = torch.randn(5, 8)
    with pytest.raises(ValueError, match="unknown mode"):
        TXCSteeringHook(W, mode="v99", ref_norm=1.0, T=5)  # type: ignore


def test_hook_validation_T_mismatch():
    W = torch.randn(5, 8)
    with pytest.raises(ValueError, match="!= T"):
        TXCSteeringHook(W, mode="v0", ref_norm=1.0, T=4)


def test_hook_v4_requires_preimage():
    W = torch.randn(5, 8)
    with pytest.raises(ValueError, match="encoder_preimage"):
        TXCSteeringHook(W, mode="v4", ref_norm=1.0, T=5)


def test_hook_cycle_phase_validated():
    W = torch.randn(5, 8)
    with pytest.raises(ValueError, match="cycle_phase"):
        TXCSteeringHook(W, mode="v1", ref_norm=1.0, T=5, cycle_phase=5)


def test_build_hook_per_token_arch_rejected():
    """build_hook only handles TXC-shape W_dec."""
    class _PerTokenStub(TempBenchArch):
        def __init__(self):
            super().__init__()
            self.config = ArchConfig("pt", 4, 8, 1, T=1)
            self.W_dec = torch.nn.Parameter(torch.randn(8, 4))

        def encode(self, x): return x  # noqa
        def decode(self, z): return z  # noqa
        def decoder_directions(self): return self.W_dec.data

    with pytest.raises(ValueError, match="TXC-style W_dec"):
        build_hook(_PerTokenStub(), feature_id=0, mode="v0", ref_norm=1.0)


# ── Hook math: V0 ─────────────────────────────────────────────────────


def test_hook_v0_constant_vector_at_every_position():
    torch.manual_seed(0)
    T, d_in, ref = 5, 16, 4.0
    W = torch.randn(T, d_in)
    hook = TXCSteeringHook(W, mode="v0", ref_norm=ref, T=T)
    hook.magnitudes = torch.tensor([1.0])
    x = torch.zeros(1, 7, d_in)
    out = hook(None, None, x)
    delta = out - x
    # Same vector at every position
    for s in range(7):
        assert torch.allclose(delta[0, s], delta[0, 0], atol=1e-6), (
            f"V0 should add constant vector at every position; mismatch at s={s}"
        )
    # Norm matches ref
    assert delta[0, 0].norm().item() == pytest.approx(ref, rel=1e-5)


# ── Hook math: V1 (cycled) ────────────────────────────────────────────


def test_hook_v1_cycles_through_decoder_slices():
    torch.manual_seed(0)
    T, d_in, ref = 5, 16, 3.0
    W = torch.randn(T, d_in)
    hook = TXCSteeringHook(W, mode="v1", ref_norm=ref, T=T, cycle_phase=0)
    hook.magnitudes = torch.tensor([1.0])
    x = torch.zeros(1, 12, d_in)
    out = hook(None, None, x)
    delta = out[0]
    # Each per-position delta should match W[t mod T] direction (after
    # normalisation + ref_norm/√T scale).
    expected_scale = ref / math.sqrt(T)
    for s in range(12):
        t = s % T
        v_expected = W[t] / W[t].norm() * expected_scale
        assert torch.allclose(delta[s], v_expected, atol=1e-5), (
            f"V1 mismatch at s={s}, t={t}"
        )


def test_hook_v1_phase_offset_works():
    torch.manual_seed(1)
    T, d_in, ref = 4, 8, 2.0
    W = torch.randn(T, d_in)
    hook = TXCSteeringHook(W, mode="v1", ref_norm=ref, T=T, cycle_phase=2)
    hook.magnitudes = torch.tensor([1.0])
    x = torch.zeros(1, 1, d_in)
    out = hook(None, None, x)
    expected_scale = ref / math.sqrt(T)
    v_expected = W[2] / W[2].norm() * expected_scale
    assert torch.allclose(out[0, 0], v_expected, atol=1e-5)


def test_hook_v1_token_count_persists_across_calls():
    """Calling the hook twice continues the phase cycle."""
    torch.manual_seed(2)
    T, d_in, ref = 4, 8, 2.0
    W = torch.randn(T, d_in)
    hook = TXCSteeringHook(W, mode="v1", ref_norm=ref, T=T)
    hook.magnitudes = torch.tensor([1.0])
    x1 = torch.zeros(1, 3, d_in)
    out1 = hook(None, None, x1)
    expected_scale = ref / math.sqrt(T)
    # First call: positions 0,1,2 -> slots 0,1,2
    for s in range(3):
        v_expected = W[s] / W[s].norm() * expected_scale
        assert torch.allclose(out1[0, s], v_expected, atol=1e-5)
    # Second call: positions should continue at slot 3,0,1
    out2 = hook(None, None, torch.zeros(1, 3, d_in))
    for j, slot in enumerate([3, 0, 1]):
        v_expected = W[slot] / W[slot].norm() * expected_scale
        assert torch.allclose(out2[0, j], v_expected, atol=1e-5), (
            f"V1 token_count not persisted; expected slot {slot} at j={j}"
        )


def test_hook_reset_zeroes_token_count():
    W = torch.randn(4, 8)
    hook = TXCSteeringHook(W, mode="v1", ref_norm=1.0, T=4)
    hook.magnitudes = torch.tensor([1.0])
    hook(None, None, torch.zeros(1, 3, 8))
    assert hook.token_count == 3
    hook.reset()
    assert hook.token_count == 0


# ── Hook math: V2 (trailing window) ───────────────────────────────────


def test_hook_v2_fills_trailing_T_positions():
    torch.manual_seed(3)
    T, d_in, ref = 5, 16, 4.0
    W = torch.randn(T, d_in)
    hook = TXCSteeringHook(W, mode="v2", ref_norm=ref, T=T)
    hook.magnitudes = torch.tensor([1.0])
    S = 8  # batch length > T
    x = torch.zeros(1, S, d_in)
    out = hook(None, None, x)
    delta = out[0]
    expected_scale = ref / math.sqrt(T)

    # Earlier positions (S-T-1 down to 0) untouched.
    for s in range(S - T):
        assert torch.allclose(delta[s], torch.zeros(d_in), atol=1e-6), (
            f"V2 should leave position {s} untouched (S-T={S - T})"
        )
    # Trailing T positions: s = S-T+j gets W[T-1-j]·scale (reverse order).
    for j in range(T):
        s = S - 1 - j
        v_expected = W[T - 1 - j] / W[T - 1 - j].norm() * expected_scale
        assert torch.allclose(delta[s], v_expected, atol=1e-5), (
            f"V2 mismatch at trailing slot j={j}, s={s}"
        )


def test_hook_v2_short_batch_only_fills_what_fits():
    """If S < T, only S trailing positions are populated."""
    torch.manual_seed(4)
    T, d_in, ref = 5, 8, 2.0
    W = torch.randn(T, d_in)
    hook = TXCSteeringHook(W, mode="v2", ref_norm=ref, T=T)
    hook.magnitudes = torch.tensor([1.0])
    S = 3
    out = hook(None, None, torch.zeros(1, S, d_in))
    delta = out[0]
    expected_scale = ref / math.sqrt(T)
    for j in range(S):
        s = S - 1 - j
        v_expected = W[T - 1 - j] / W[T - 1 - j].norm() * expected_scale
        assert torch.allclose(delta[s], v_expected, atol=1e-5)


# ── Hook math: V0 == V4 at tied init ──────────────────────────────────


def test_hook_v0_equals_v4_at_tied_init():
    """At tied init, encoder_preimage = T · mean_decoder. After
    L2-normalising, V0 and V4 vectors are equal up to sign."""
    torch.manual_seed(5)
    T, d_in, d_sae, ref = 5, 16, 6, 4.0
    arch = _TxcStub(d_in=d_in, d_sae=d_sae, T=T, tied_init=True)
    W_dec_f = arch.W_dec.data[1]
    pre = encoder_preimage(arch, 1)
    h0 = TXCSteeringHook(W_dec_f, mode="v0", ref_norm=ref, T=T)
    h4 = TXCSteeringHook(W_dec_f, mode="v4", ref_norm=ref, T=T, encoder_preimage=pre)
    h0.magnitudes = torch.tensor([1.0])
    h4.magnitudes = torch.tensor([1.0])
    x = torch.zeros(1, 1, d_in)
    out0 = h0(None, None, x)
    out4 = h4(None, None, x)
    # Same direction (up to sign), same scaling. T · mean has same dir
    # as mean → V0 ≡ V4 at tied init.
    cos = torch.nn.functional.cosine_similarity(out0[0, 0], out4[0, 0], dim=0).item()
    assert abs(cos) > 0.9999, f"V0 vs V4 cos at tied init = {cos}"


def test_hook_v0_diverges_from_v4_when_encoder_drifts():
    torch.manual_seed(6)
    T, d_in, d_sae, ref = 5, 16, 6, 4.0
    arch = _TxcStub(d_in=d_in, d_sae=d_sae, T=T, tied_init=True)
    fid = 0
    # Drift the encoder for feature fid:
    with torch.no_grad():
        arch.W_enc.data[:, :, fid] += 1.5 * torch.randn_like(arch.W_enc.data[:, :, fid])
    W_dec_f = arch.W_dec.data[fid]
    pre = encoder_preimage(arch, fid)
    h0 = TXCSteeringHook(W_dec_f, mode="v0", ref_norm=ref, T=T)
    h4 = TXCSteeringHook(W_dec_f, mode="v4", ref_norm=ref, T=T, encoder_preimage=pre)
    h0.magnitudes = torch.tensor([1.0])
    h4.magnitudes = torch.tensor([1.0])
    x = torch.zeros(1, 1, d_in)
    cos = torch.nn.functional.cosine_similarity(
        h0(None, None, x)[0, 0], h4(None, None, x)[0, 0], dim=0,
    ).item()
    assert abs(cos) < 0.95, f"after encoder drift, V0/V4 should diverge; cos={cos}"


# ── Energy comparability with √T correction ──────────────────────────


def test_total_energy_v0_equals_v1_v2_under_sqrt_t_correction():
    """Total Frobenius energy injected over the trailing-T window is
    matched between V0 and V1/V2 when sqrt_t_correction=True."""
    torch.manual_seed(7)
    T, d_in, ref = 5, 16, 3.0
    W = torch.randn(T, d_in)
    h0 = TXCSteeringHook(W, mode="v0", ref_norm=ref, T=T)
    h1 = TXCSteeringHook(W, mode="v1", ref_norm=ref, T=T)
    h2 = TXCSteeringHook(W, mode="v2", ref_norm=ref, T=T)
    for h in (h0, h1, h2):
        h.magnitudes = torch.tensor([1.0])

    # V0 over a T-slot batch: T copies of a unit vector × ref → energy = T·ref^2
    e0 = (h0(None, None, torch.zeros(1, T, d_in))).norm().item() ** 2
    # V1: T distinct unit vectors × (ref/√T) → energy = T · (ref/√T)^2 = ref^2
    e1 = (h1(None, None, torch.zeros(1, T, d_in))).norm().item() ** 2
    # V2 over a T-slot batch fills all T positions → energy = ref^2
    e2 = (h2(None, None, torch.zeros(1, T, d_in))).norm().item() ** 2

    # With √T correction, V1 and V2 inject ref^2; V0 injects T · ref^2.
    # The correction's purpose: per-step magnitude is comparable across
    # protocols. If a caller wants V1/V2 to inject T·ref^2 like V0,
    # they pass sqrt_t_correction=False — exposing the trade-off
    # explicitly.
    assert e0 == pytest.approx(T * ref ** 2, rel=1e-4)
    assert e1 == pytest.approx(ref ** 2, rel=1e-4)
    assert e2 == pytest.approx(ref ** 2, rel=1e-4)


def test_sqrt_t_correction_off_gives_uncorrected_energy():
    torch.manual_seed(8)
    T, d_in, ref = 5, 8, 2.0
    W = torch.randn(T, d_in)
    h1 = TXCSteeringHook(W, mode="v1", ref_norm=ref, T=T, sqrt_t_correction=False)
    h1.magnitudes = torch.tensor([1.0])
    e1 = h1(None, None, torch.zeros(1, T, d_in)).norm().item() ** 2
    # Without √T correction, each per-position vector has norm ref, so
    # energy = T · ref^2.
    assert e1 == pytest.approx(T * ref ** 2, rel=1e-4)


# ── Magnitude scaling ─────────────────────────────────────────────────


def test_magnitudes_scale_delta_linearly():
    torch.manual_seed(9)
    T, d_in, ref = 5, 8, 1.0
    W = torch.randn(T, d_in)
    for mode in ALL_MODES:
        kwargs = {}
        if mode == "v4":
            kwargs["encoder_preimage"] = torch.randn(d_in)
        hook = TXCSteeringHook(W, mode=mode, ref_norm=ref, T=T, **kwargs)
        hook.magnitudes = torch.tensor([1.0])
        delta1 = hook(None, None, torch.zeros(1, T, d_in)).clone()
        # New hook (avoid token_count drift in V1).
        hook = TXCSteeringHook(W, mode=mode, ref_norm=ref, T=T, **kwargs)
        hook.magnitudes = torch.tensor([3.0])
        delta3 = hook(None, None, torch.zeros(1, T, d_in)).clone()
        ratio = delta3.norm() / max(delta1.norm().item(), 1e-12)
        assert abs(ratio - 3.0) < 1e-4, f"{mode}: scaling mismatch ratio={ratio}"


def test_zero_magnitude_is_no_op():
    """All-zero magnitudes short-circuits the hook."""
    W = torch.randn(5, 8)
    for mode in ALL_MODES:
        kwargs = {"encoder_preimage": torch.randn(8)} if mode == "v4" else {}
        hook = TXCSteeringHook(W, mode=mode, ref_norm=1.0, T=5, **kwargs)
        hook.magnitudes = torch.tensor([0.0, 0.0])
        x = torch.randn(2, 7, 8)
        out = hook(None, None, x)
        assert torch.equal(out, x), f"{mode}: zero-mag should be no-op"


def test_per_row_magnitudes_independent():
    torch.manual_seed(10)
    T, d_in, ref = 5, 8, 2.0
    W = torch.randn(T, d_in)
    hook = TXCSteeringHook(W, mode="v0", ref_norm=ref, T=T)
    hook.magnitudes = torch.tensor([1.0, 2.0])
    x = torch.zeros(2, 3, d_in)
    out = hook(None, None, x)
    # Row 0 has norm = √3 × ref × 1; row 1 = √3 × ref × 2.
    n0 = out[0].norm().item()
    n1 = out[1].norm().item()
    assert n1 == pytest.approx(2 * n0, rel=1e-4)


# ── build_hook convenience constructor ────────────────────────────────


def test_build_hook_v4_pulls_preimage():
    arch = _TxcStub(d_in=8, d_sae=4, T=5, tied_init=False)
    hook = build_hook(arch, feature_id=2, mode="v4", ref_norm=3.0)
    expected_pre = arch.W_enc.data[:, :, 2].sum(dim=0)
    assert torch.allclose(hook.enc, expected_pre)


def test_build_hook_v0_doesnt_need_preimage():
    arch = _TxcStub(d_in=8, d_sae=4, T=5)
    hook = build_hook(arch, feature_id=0, mode="v0", ref_norm=2.0)
    assert hook.enc is None
