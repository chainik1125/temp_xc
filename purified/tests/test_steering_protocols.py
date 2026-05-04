"""Tests for :mod:`temp_bench.eval.steering_protocols`.

Verifies that V3 latent-space steering returns the expected
``decode(z + α e_f) - decode(z)`` perturbation, with optional
energy normalisation."""

from __future__ import annotations

import pytest
import torch

from temp_bench.architectures.base import ArchConfig, TempBenchArch
from temp_bench.eval.steering_protocols import latent_space_steer


class _LinearTxcStub(TempBenchArch):
    """Linear TXC where decode(z) = z @ W_dec; encode(x) = x @ W_enc.
    Lets us check ``decode(z+α e_f) - decode(z) = α · W_dec[f]`` directly."""

    def __init__(self, *, d_in: int, d_sae: int, T: int, seed: int = 0):
        super().__init__()
        self.config = ArchConfig(name="lin_txc", d_in=d_in, d_sae=d_sae, k_pos=1, T=T)
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        g = torch.Generator().manual_seed(seed)
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, T, d_in, generator=g))
        self.W_enc = torch.nn.Parameter(torch.randn(T, d_in, d_sae, generator=g))

    def encode(self, x):
        # (B, T, d_in) → (B, 1, d_sae)
        return torch.einsum("btd,tds->bs", x, self.W_enc).unsqueeze(1)

    def decode(self, z):
        if z.dim() == 3:
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec)

    def decoder_directions(self):
        return self.W_dec.data.mean(dim=1).clone()

    @property
    def T(self):
        return self._T


def test_latent_space_steer_returns_alpha_times_decoder_atom():
    """For a linear TXC, decode(z+α e_f) - decode(z) = α · W_dec[f]."""
    torch.manual_seed(0)
    T, d_in, d_sae = 5, 8, 4
    arch = _LinearTxcStub(d_in=d_in, d_sae=d_sae, T=T)
    x = torch.randn(2, T, d_in)
    fid = 2
    alpha = 1.7
    delta = latent_space_steer(arch, x, feature_id=fid, magnitude=alpha)
    expected = alpha * arch.W_dec.data[fid].unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(delta, expected, atol=1e-5), (delta - expected).abs().max()


def test_latent_space_steer_per_row_magnitudes():
    torch.manual_seed(1)
    T, d_in, d_sae = 5, 8, 4
    arch = _LinearTxcStub(d_in=d_in, d_sae=d_sae, T=T)
    x = torch.randn(3, T, d_in)
    mags = torch.tensor([1.0, 2.0, 0.5])
    delta = latent_space_steer(arch, x, feature_id=0, magnitude=mags)
    for b in range(3):
        expected = mags[b] * arch.W_dec.data[0]
        assert torch.allclose(delta[b], expected, atol=1e-5)


def test_latent_space_steer_ref_norm_normalises_per_row():
    torch.manual_seed(2)
    T, d_in, d_sae = 5, 8, 4
    arch = _LinearTxcStub(d_in=d_in, d_sae=d_sae, T=T)
    x = torch.randn(3, T, d_in)
    mags = torch.tensor([1.0, 2.0, 0.5])
    ref = 4.0
    delta = latent_space_steer(arch, x, feature_id=0, magnitude=mags, ref_norm=ref)
    for b in range(3):
        target_norm = ref * mags[b].item()
        assert delta[b].norm().item() == pytest.approx(target_norm, rel=1e-4)


def test_latent_space_steer_validation():
    torch.manual_seed(3)
    arch = _LinearTxcStub(d_in=4, d_sae=2, T=5)
    with pytest.raises(ValueError, match="\\(B, T, d_in\\)"):
        latent_space_steer(arch, torch.zeros(5, 4), feature_id=0, magnitude=1.0)
    with pytest.raises(ValueError, match="!= arch T"):
        latent_space_steer(arch, torch.zeros(2, 4, 4), feature_id=0, magnitude=1.0)
