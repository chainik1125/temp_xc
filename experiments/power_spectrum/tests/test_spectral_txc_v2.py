"""Focused unit tests for the experiment-local spectral TXC v2."""

from __future__ import annotations

import pytest
import torch

from experiments.power_spectrum.code.spectral_txc_v2 import (
    SpectralTXCV2,
    _fourier_bands,
    _real_fourier_basis,
)
from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK


def test_defaults_are_numerically_backward_compatible() -> None:
    kwargs = {"d_in": 7, "d_sae": 24, "T": 4, "k_pos": 1, "bands": "multiband"}
    torch.manual_seed(17)
    parent = SpectralTXCBatchTopK(**kwargs)
    torch.manual_seed(17)
    v2 = SpectralTXCV2(**kwargs)

    for name, parameter in parent.named_parameters():
        assert torch.equal(parameter, dict(v2.named_parameters())[name])

    x = torch.randn(9, 4, 7)
    assert torch.equal(parent.encode(x), v2.encode(x))
    parent_out = parent.train_step(x)
    v2_out = v2.train_step(x)
    assert parent_out.keys() == v2_out.keys()
    for key in parent_out:
        assert torch.equal(parent_out[key], v2_out[key]), key


def test_remove_dc_is_offset_invariant_and_masks_decoder_dc() -> None:
    model = SpectralTXCV2(
        d_in=5,
        d_sae=24,
        T=4,
        k_pos=1,
        bands="multiband",
        dc_mode="remove",
    )
    model.eval()
    x = torch.randn(8, 4, 5)
    offset = torch.randn(8, 1, 5)

    assert torch.allclose(model.encode(x), model.encode(x + offset), atol=2e-6)
    decoder_temporal_mean = model._dec_full().mean(dim=1)
    assert torch.allclose(
        decoder_temporal_mean,
        torch.zeros_like(decoder_temporal_mean),
        atol=2e-6,
    )
    assert model.selection_k_per_band[0] == 0
    assert sum(model.selection_k_per_band) == model.k_win


def test_real_fourier_basis_is_orthonormal_and_keeps_quadrature_pairs() -> None:
    basis = _real_fourier_basis(16)
    assert torch.allclose(basis @ basis.T, torch.eye(16), atol=2e-6)
    row_bands, frequency_bands = _fourier_bands(16, "multiband")
    assert [len(band) for band in row_bands] == [1, 4, 6, 5]
    assert frequency_bands == [[0], [1, 2], [3, 4, 5], [6, 7, 8]]
    assert sorted(index for band in row_bands for index in band) == list(range(16))


def test_fourier_model_reconstructs_exact_band_projections() -> None:
    model = SpectralTXCV2(
        d_in=3,
        d_sae=32,
        T=8,
        bands="multiband",
        temporal_basis="fourier",
        selection_mode="global",
        adaptive_frequency_alpha=0.1,
        adaptive_frequency_adversary_alpha=0.1,
    )
    x = torch.randn(6, 8, 3)
    exact = [model._project_frequencies(x, band) for band in model.bands]
    errors, _ = model._frequency_band_errors_and_power(x, exact)
    assert torch.all(errors < 1e-10)
    assert model.frequency_bin_bands == [[0], [1], [2], [3, 4]]
    output = model.train_step(x)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.frequency_weight_logits.grad is not None


def test_global_selection_removes_guaranteed_per_band_occupancy() -> None:
    common = {
        "d_in": 3,
        "d_sae": 24,
        "T": 4,
        "k_pos": 1,
        "bands": "multiband",
    }
    global_model = SpectralTXCV2(**common, selection_mode="global")
    per_band_model = SpectralTXCV2(**common, selection_mode="per_band")
    pre = torch.full((2, 24), 0.1)
    first_start, first_end = global_model.band_slices[0]
    pre[:, first_start:first_end] = 10.0

    global_code = global_model._select(pre)
    per_band_code = per_band_model._select(pre)
    for start, end in global_model.band_slices[1:]:
        assert torch.count_nonzero(global_code[:, start:end]) == 0
    for start, end in per_band_model.band_slices:
        assert torch.count_nonzero(per_band_code[:, start:end]) > 0
    assert torch.count_nonzero(global_code) == global_model.k_win * pre.shape[0]


def test_overdominance_penalty_uses_frequency_width_target() -> None:
    model = SpectralTXCV2(
        d_in=2,
        d_sae=24,
        T=4,
        bands="multiband",
        dominance_alpha=0.2,
        dominance_cap=1.1,
        dominance_target="bandwidth",
    )
    shape = (3, 4, 2)
    concentrated = [torch.ones(shape), torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)]
    loss, shares = model._dominance_loss(concentrated)
    assert loss > 0
    assert shares.argmax().item() == 0

    # T=4 multiband consists of four singleton frequencies, so equal power is
    # exactly the width-proportional target and lies under the one-sided cap.
    balanced = [torch.ones(shape) for _ in range(4)]
    balanced_loss, balanced_shares = model._dominance_loss(balanced)
    assert torch.allclose(balanced_shares, torch.full((4,), 0.25))
    assert balanced_loss == 0


def test_frequency_matryoshka_matches_bandwise_dct_projections() -> None:
    model = SpectralTXCV2(
        d_in=3,
        d_sae=24,
        T=4,
        bands="multiband",
        frequency_matryoshka_alpha=0.3,
    )
    x = torch.randn(5, 4, 3)
    exact_bands = [model._project_frequencies(x, band) for band in model.bands]
    exact_loss = model._frequency_matryoshka_loss(x, exact_bands)
    assert exact_loss < 1e-10

    zero_bands = [torch.zeros_like(x) for _ in model.bands]
    assert model._frequency_matryoshka_loss(x, zero_bands) > 0


def test_adaptive_frequency_weights_start_at_bandwidth_prior() -> None:
    model = SpectralTXCV2(
        d_in=3,
        d_sae=32,
        T=8,
        bands="multiband",
        adaptive_frequency_alpha=0.1,
        adaptive_frequency_adversary_alpha=0.1,
    )
    expected = torch.tensor([1.0, 2.0, 2.0, 3.0]) / 8.0
    assert torch.allclose(model.learned_frequency_weights(), expected)
    assert torch.allclose(model.learned_frequency_weights().sum(), torch.tensor(1.0))


def test_adaptive_frequency_loss_is_order_free_and_exact_at_projection() -> None:
    model = SpectralTXCV2(
        d_in=3,
        d_sae=32,
        T=8,
        bands="multiband",
        adaptive_frequency_alpha=0.1,
        adaptive_frequency_adversary_alpha=0.1,
    )
    x = torch.randn(7, 8, 3)
    exact = [model._project_frequencies(x, band) for band in model.bands]
    errors, _ = model._frequency_band_errors_and_power(x, exact)
    assert torch.all(errors < 1e-10)

    permutation = torch.tensor([3, 1, 0, 2])
    values = torch.tensor([0.2, 0.4, 0.8, 1.6])
    weights = model._active_frequency_weights().detach()
    assert torch.allclose(
        (weights * values).sum(),
        (weights[permutation] * values[permutation]).sum(),
    )


def test_adaptive_frequency_adversary_upweights_hard_band() -> None:
    model = SpectralTXCV2(
        d_in=3,
        d_sae=32,
        T=8,
        bands="multiband",
        adaptive_frequency_alpha=0.1,
        adaptive_frequency_adversary_alpha=1.0,
        adaptive_frequency_entropy=0.2,
        adaptive_frequency_floor=0.1,
    )
    optimizer = torch.optim.SGD([model.frequency_weight_logits], lr=0.5)
    before = model.learned_frequency_weights().detach().clone()
    normalized_errors = torch.tensor([0.1, 0.2, 0.4, 2.0])
    reward, _ = model._frequency_weight_reward(normalized_errors)
    surrogate = -(reward - reward.detach())
    optimizer.zero_grad()
    surrogate.backward()
    optimizer.step()
    after = model.learned_frequency_weights().detach()
    assert after[-1] > before[-1]
    assert after[0] < before[0]
    floor = model.adaptive_frequency_floor * model.frequency_weight_prior
    assert torch.all(after >= floor - 1e-7)
    assert torch.allclose(after.sum(), torch.tensor(1.0), atol=1e-6)


def test_adaptive_frequency_routing_changes_support_not_code_values() -> None:
    model = SpectralTXCV2(
        d_in=3,
        d_sae=32,
        T=8,
        k_pos=1,
        bands="multiband",
        selection_mode="global",
        adaptive_frequency_alpha=0.1,
        adaptive_frequency_adversary_alpha=0.1,
        adaptive_frequency_routing_strength=1.0,
    )
    with torch.no_grad():
        model.frequency_weight_logits.copy_(torch.tensor([-5.0, -5.0, -5.0, 5.0]))
    pre = torch.ones(1, 32)
    selected = model._batchtopk_global(pre)
    start, stop = model.band_slices[-1]
    assert torch.count_nonzero(selected) == model.k_win
    assert torch.count_nonzero(selected[:, start:stop]) == model.k_win
    assert torch.equal(selected[selected > 0], torch.ones(model.k_win))


def test_adaptive_frequency_noise_floor_removes_expected_white_noise_power() -> None:
    model = SpectralTXCV2(
        d_in=3,
        d_sae=16,
        T=8,
        k_pos=1,
        bands="multiband",
        selection_mode="global",
        adaptive_frequency_alpha=0.1,
        adaptive_frequency_adversary_alpha=0.1,
        adaptive_frequency_noise_variance=0.25,
    )
    generator = torch.Generator().manual_seed(17)
    samples = torch.randn(8192, 8, 3, generator=generator) * 0.5
    zeros = [torch.zeros_like(samples) for _ in range(model.n_bands)]
    errors, powers = model._frequency_band_errors_and_power(samples, zeros)
    assert torch.all(errors < 0.002)
    assert torch.all(powers < 0.002)


def test_augmented_objective_is_additive_and_differentiable() -> None:
    model = SpectralTXCV2(
        d_in=5,
        d_sae=24,
        T=4,
        k_pos=1,
        selection_mode="global",
        dominance_alpha=0.2,
        frequency_matryoshka_alpha=0.3,
        dead_threshold_tokens=10**9,
    )
    out = model.train_step(torch.randn(16, 4, 5))
    expected = (
        out["mse"]
        + model.auxk_alpha * out["auxk"]
        + model.dominance_alpha * out["dominance"]
        + model.frequency_matryoshka_alpha * out["frequency_matryoshka"]
    )
    assert torch.allclose(out["loss"].detach(), expected)
    out["loss"].backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(gradient).all() for gradient in grads)


def test_invalid_experimental_modes_fail_loudly() -> None:
    with pytest.raises(ValueError, match="T >= 2"):
        SpectralTXCV2(d_in=4, d_sae=8, T=1, dc_mode="remove")
    with pytest.raises(ValueError, match="selection_mode"):
        SpectralTXCV2(d_in=4, d_sae=8, T=2, selection_mode="adaptive")
    with pytest.raises(ValueError, match="both be positive"):
        SpectralTXCV2(
            d_in=4,
            d_sae=8,
            T=2,
            adaptive_frequency_alpha=0.1,
        )
    with pytest.raises(ValueError, match="routing requires"):
        SpectralTXCV2(
            d_in=4,
            d_sae=8,
            T=2,
            adaptive_frequency_routing_strength=1.0,
        )
