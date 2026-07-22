"""FreqFrac lens — tap extraction, DCT profiles, aggregation (freqbench port).

Covers ``explorations.synthetic.freqfrac`` (PORT.md § C):
- the DCT basis matches the ``spectral_txc`` plugin's (kept duplicated on
  purpose — this test is the sync guard) and is orthonormal,
- tap extraction handles all three panel layouts (3-D ``W_enc``, spectral
  band coefficients, 2-D token ``W_enc``),
- analytic plants: a pure-cosine atom → one-hot profile; a delta-in-time
  stacked atom → the ψ[:, τ]² row; spectral atoms → mass confined to their
  own band,
- profiles are per-atom normalized; band fractions partition to 1,
- firing weights pool the code axes and drive the weighted curve.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from explorations.synthetic import freqfrac
from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK, _dct_basis
from temp_bench.archs.stacked_batchtopk import StackedBatchTopK
from temp_bench.archs.txc_batchtopk import TXCBatchTopKPre


class _FakeTokenArch(nn.Module):
    """Minimal stand-in for the 2-D ``W_enc (d_in, d_sae)`` token layout."""

    def __init__(self, d_in: int, d_sae: int) -> None:
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae))


def test_dct_basis_matches_plugin_and_is_orthonormal() -> None:
    for T in (1, 2, 4, 8, 16):
        psi = freqfrac.dct_basis(T)
        assert torch.allclose(psi, _dct_basis(T), atol=1e-6)
        assert torch.allclose(psi @ psi.T, torch.eye(T), atol=1e-5)


def test_taps_layouts_and_profile_normalization() -> None:
    pre = TXCBatchTopKPre(d_in=6, d_sae=5, T=8, k_pos=1)
    stacked = StackedBatchTopK(d_in=6, d_sae=5, T=8, k_pos=1)
    spectral = SpectralTXCBatchTopK(d_in=8, d_sae=8, T=8, k_pos=1)
    token = _FakeTokenArch(d_in=6, d_sae=5)

    for model, H, T in ((pre, 5, 8), (stacked, 5, 8), (spectral, 8, 8),
                        (token, 5, 1)):
        taps = freqfrac.encoder_taps(model)
        assert taps.shape[0] == H and taps.shape[1] == T
        prof = freqfrac.freq_profile(model)
        assert prof.shape == (H, T)
        assert torch.allclose(prof.sum(dim=1), torch.ones(H), atol=1e-5)
        assert float(prof.min()) >= 0.0

    # T = 1: all energy at DC by construction.
    prof_tok = freqfrac.freq_profile(token)
    assert torch.allclose(prof_tok, torch.ones(5, 1), atol=1e-6)
    assert torch.allclose(freqfrac.spectral_concentration(prof_tok),
                          torch.ones(5))


def test_planted_cosine_atom_is_one_hot() -> None:
    """Atom h with taps ψ[w_h] ⊗ u must put all its energy at index w_h."""
    T, d_in, d_sae = 8, 6, 5
    model = TXCBatchTopKPre(d_in=d_in, d_sae=d_sae, T=T, k_pos=1)
    psi = freqfrac.dct_basis(T)
    freqs = [0, 1, 3, 5, 7]
    with torch.no_grad():
        for h, w in enumerate(freqs):
            u = torch.randn(d_in)
            model.W_enc.data[:, :, h] = psi[w][:, None] * u[None, :]
    prof = freqfrac.freq_profile(model)
    for h, w in enumerate(freqs):
        expect = torch.zeros(T)
        expect[w] = 1.0
        assert torch.allclose(prof[h], expect, atol=1e-5)
    # A pure tone maxes the top-2-adjacent concentration.
    conc = freqfrac.spectral_concentration(prof)
    assert float(conc.min()) > 0.999


def test_stacked_delta_atom_matches_psi_squared() -> None:
    """A per-position (delta-in-time) atom's profile is the ψ[:, τ]² column —
    broadband, the localized end of the lens."""
    T, d_in, d_sae = 8, 6, 4
    model = StackedBatchTopK(d_in=d_in, d_sae=d_sae, T=T, k_pos=1)
    positions = [0, 2, 5, 7]
    with torch.no_grad():
        model.W_enc.data.zero_()
        for h, tau in enumerate(positions):
            model.W_enc.data[tau, :, h] = torch.randn(d_in)
    prof = freqfrac.freq_profile(model)
    psi = freqfrac.dct_basis(T)
    for h, tau in enumerate(positions):
        assert torch.allclose(prof[h], psi[:, tau].pow(2), atol=1e-5)


def test_spectral_atoms_confined_to_their_band() -> None:
    model = SpectralTXCBatchTopK(d_in=8, d_sae=8, T=8, k_pos=1,
                                 bands="multiband")
    prof = freqfrac.freq_profile(model)
    all_idx = set(range(8))
    for b, (s, e) in enumerate(model.band_slices):
        outside = sorted(all_idx - set(model.bands[b]))
        if outside:
            assert float(prof[s:e][:, outside].sum()) < 1e-5
    # Band fractions on the model's own bands: each atom entirely in its band.
    frac = freqfrac.band_fractions(prof, model.bands)
    for b, (s, e) in enumerate(model.band_slices):
        assert torch.allclose(frac[s:e, b], torch.ones(e - s), atol=1e-5)
    assert torch.allclose(frac.sum(dim=1), torch.ones(8), atol=1e-5)


def test_firing_weights_and_weighted_curve() -> None:
    torch.manual_seed(0)
    T, d_in, d_sae = 4, 6, 8
    model = TXCBatchTopKPre(d_in=d_in, d_sae=d_sae, T=T, k_pos=2)
    x = torch.randn(32, T, d_in)
    mean_act, rate = freqfrac.firing_weights(model, x, batch_size=16)
    assert mean_act.shape == rate.shape == (d_sae,)
    assert float(mean_act.min()) >= 0.0
    assert float(rate.min()) >= 0.0 and float(rate.max()) <= 1.0
    # Stacked encode keeps the T axis — pooled the same way.
    stacked = StackedBatchTopK(d_in=d_in, d_sae=d_sae, T=T, k_pos=1)
    m2, r2 = freqfrac.firing_weights(stacked, x, batch_size=16)
    assert m2.shape == r2.shape == (d_sae,)

    prof = freqfrac.freq_profile(model)
    # One-hot weights recover that atom's own profile.
    one_hot = torch.zeros(d_sae)
    one_hot[3] = 1.0
    assert torch.allclose(freqfrac.arch_curve(prof, one_hot), prof[3],
                          atol=1e-6)
    # Uniform curve is the plain mean; a distribution over frequencies.
    curve = freqfrac.arch_curve(prof)
    assert curve.shape == (T,)
    assert abs(float(curve.sum()) - 1.0) < 1e-5
    # All-zero weights fall back to uniform rather than NaN.
    assert torch.allclose(freqfrac.arch_curve(prof, torch.zeros(d_sae)),
                          curve, atol=1e-6)
