from __future__ import annotations

import torch

from experiments.power_spectrum.code.backtracking_fourier_xc import (
    FourierTopKXC,
    fourier_bands,
    fourier_parameter_count,
    matched_fourier_width,
    real_fourier_basis,
    txc_parameter_count,
)
from temp_bench.archs.txc_base import TXCBase


def test_real_fourier_basis_is_orthonormal() -> None:
    for length in (1, 2, 4, 5, 10):
        basis = real_fourier_basis(length)
        torch.testing.assert_close(
            basis @ basis.T,
            torch.eye(length),
            atol=2e-6,
            rtol=2e-6,
        )


def test_multiband_fourier_keeps_quadrature_pairs_together() -> None:
    row_bands, frequency_bands = fourier_bands(10)
    assert row_bands == [[0], [1, 2], [3, 4, 5, 6], [7, 8, 9]]
    assert frequency_bands == [[0], [1], [2, 3], [4, 5]]


def test_reported_parameter_count_matches_model() -> None:
    model = FourierTopKXC(d_in=7, d_sae=31, T=6, k_pos=2)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert actual == fourier_parameter_count(
        d_in=7,
        d_sae=31,
        window=6,
    )


def test_matched_width_is_the_closest_total_parameter_count() -> None:
    for window in (1, 2, 4, 6, 10):
        width = matched_fourier_width(
            d_in=4_096,
            txc_d_sae=32_768,
            window=window,
        )
        target = txc_parameter_count(
            d_in=4_096,
            d_sae=32_768,
            window=window,
        )
        observed = fourier_parameter_count(
            d_in=4_096,
            d_sae=width,
            window=window,
        )
        for neighbour in (width - 1, width + 1):
            if neighbour >= len(fourier_bands(window)[0]):
                candidate = fourier_parameter_count(
                    d_in=4_096,
                    d_sae=neighbour,
                    window=window,
                )
                assert abs(observed - target) <= abs(candidate - target)
        assert abs(observed - target) / target < 1e-4


def test_full_band_coefficients_equal_dense_time_kernels() -> None:
    torch.manual_seed(7)
    model = FourierTopKXC(
        d_in=3,
        d_sae=8,
        T=4,
        k_pos=1,
        bands="full",
    )
    x = torch.randn(5, 4, 3)
    basis = model.temporal_basis
    dense_encoder = torch.einsum("wt,hwd->htd", basis, model.enc_coef[0])
    dense_decoder = torch.einsum("wt,hwd->htd", basis, model.dec_coef[0])
    expected_pre = (
        torch.einsum("btd,htd->bh", x, dense_encoder) + model.b_enc[0]
    )
    torch.testing.assert_close(model.preactivations(x), expected_pre)

    values, indices = model.select_topk(expected_pre)
    code = model.dense_code(values, indices)
    expected_reconstruction = (
        torch.einsum("bh,htd->btd", code, dense_decoder) + model.b_dec
    )
    torch.testing.assert_close(model.decode(code), expected_reconstruction)


def test_train_step_uses_exact_per_example_budget_and_has_no_matryoshka() -> None:
    torch.manual_seed(11)
    model = FourierTopKXC(d_in=5, d_sae=24, T=4, k_pos=2)
    x = torch.randn(6, 4, 5)
    result = model.train_step(x)
    assert result["l0"].item() <= 8
    assert result["loss"].requires_grad
    assert not hasattr(model, "frequency_matryoshka_alpha")


def test_full_band_model_is_a_rotated_txcbase() -> None:
    torch.manual_seed(17)
    fourier = FourierTopKXC(
        d_in=4,
        d_sae=12,
        T=3,
        k_pos=2,
        bands="full",
    )
    txc = TXCBase(d_in=4, d_sae=12, T=3, k_pos=2)
    dense_encoder = torch.einsum(
        "wt,hwd->htd",
        fourier.temporal_basis,
        fourier.enc_coef[0],
    )
    dense_decoder = torch.einsum(
        "wt,hwd->htd",
        fourier.temporal_basis,
        fourier.dec_coef[0],
    )
    with torch.no_grad():
        txc.W_enc.copy_(dense_encoder.permute(1, 2, 0))
        txc.W_dec.copy_(dense_decoder)
        txc.b_enc.copy_(fourier.b_enc[0])
        txc.b_dec.copy_(fourier.b_dec)

    x = torch.randn(7, 3, 4)
    fourier_result = fourier.train_step(x)
    txc_result = txc.train_step(x)
    for name in ("loss", "mse", "l0", "auxk", "dead"):
        torch.testing.assert_close(
            fourier_result[name],
            txc_result[name],
            atol=2e-5,
            rtol=2e-5,
        )
