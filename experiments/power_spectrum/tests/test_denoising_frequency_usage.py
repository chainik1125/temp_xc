from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.power_spectrum.code import run_denoising_frequency_usage as usage
from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK


def _tiny_t2_model(*, d_sae: int = 4) -> SpectralTXCBatchTopK:
    return SpectralTXCBatchTopK(
        d_in=1,
        d_sae=d_sae,
        T=2,
        k_pos=1,
        bands="multiband",
        auxk_alpha=0.0,
    )


def test_t2_band_accounting_feature_usage_and_masks() -> None:
    model = _tiny_t2_model()
    codes = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
        ]
    )
    masks = usage.frequency_feature_masks(model)
    assert masks["full"].tolist() == [True, True, True, True]
    assert masks["dc"].tolist() == [True, True, False, False]
    assert masks["ac"].tolist() == [False, False, True, True]
    assert not masks["mixed"].any()

    feature = usage.per_feature_usage(codes)
    assert feature["fire_rate"] == [0.5, 0.5, 0.5, 0.5]
    assert feature["l1_mean"] == [0.5, 1.5, 1.0, 2.0]
    assert feature["l2_rms"] == pytest.approx(
        [2**-0.5, 3 * 2**-0.5, 2**0.5, 2 * 2**0.5]
    )

    decoded = {"per_band": [10.0, 30.0], "per_band_share": [0.25, 0.75]}
    bands = usage.per_band_usage(model, codes, decoded)
    assert [row["allocated_atoms"] for row in bands] == [2, 2]
    assert [row["allocated_k"] for row in bands] == [1, 1]
    assert [row["realized_l0"] for row in bands] == [1.0, 1.0]
    assert [row["realized_l0_share"] for row in bands] == [0.5, 0.5]
    assert [row["decoded_energy_share"] for row in bands] == [0.25, 0.75]


def test_actual_decoded_coefficient_and_bias_energy_are_separate() -> None:
    model = _tiny_t2_model(d_sae=2)
    with torch.no_grad():
        model.dec_coef[0].zero_()
        model.dec_coef[1].zero_()
        model.dec_coef[0][0, 0, 0] = 2.0
        model.dec_coef[1][0, 0, 0] = 3.0
        model.b_dec.copy_(torch.tensor([[1.0], [1.0]]))
    codes = torch.tensor([[1.0, 2.0], [2.0, 1.0]])

    decoded = usage.decoded_reconstruction_energy(model, codes, batch_size=1)
    assert decoded["dc"] == pytest.approx(10.0)
    assert decoded["ac"] == pytest.approx(22.5)
    assert decoded["total"] == pytest.approx(32.5)
    assert decoded["band_additivity_relative_error"] < 1e-6

    coefficient = usage.activation_weighted_coefficient_energy(model, codes)
    assert coefficient["dc"] == pytest.approx(10.0)
    assert coefficient["ac"] == pytest.approx(22.5)
    assert coefficient["total"] == pytest.approx(32.5)

    bias = usage.bias_spectrum(model)
    assert bias["dc"] == pytest.approx(2.0)
    assert bias["ac"] == pytest.approx(0.0, abs=1e-7)
    assert decoded["total"] == pytest.approx(32.5)


def test_ridge_probe_masks_restrict_information_deterministically() -> None:
    rng = np.random.default_rng(7)
    hidden = rng.standard_normal((300, 2)).astype(np.float32)
    codes = np.column_stack(
        [
            hidden[:, 0],
            2.0 * hidden[:, 0],
            hidden[:, 1],
            -3.0 * hidden[:, 1],
        ]
    ).astype(np.float32)
    masks = {
        "full": np.array([True, True, True, True]),
        "dc": np.array([True, True, False, False]),
        "ac": np.array([False, False, True, True]),
    }
    result = usage.ridge_r2_by_mask(
        codes,
        hidden,
        masks,
        train_fraction=0.8,
        alpha=1e-3,
    )
    assert result["full"]["per_hidden_feature_r2"] == pytest.approx(
        [1.0, 1.0], abs=1e-5
    )
    assert result["dc"]["per_hidden_feature_r2"][0] > 0.999
    assert result["dc"]["per_hidden_feature_r2"][1] < 0.1
    assert result["ac"]["per_hidden_feature_r2"][0] < 0.1
    assert result["ac"]["per_hidden_feature_r2"][1] > 0.999


def test_true_dct_power_separates_constant_and_alternating_series() -> None:
    constant = torch.ones(2, 4, 1)
    alternating = torch.tensor([1.0, -1.0, 1.0, -1.0]).reshape(1, 4, 1)

    dc = usage.dct_power(constant, T=2)
    ac = usage.dct_power(alternating, T=2)
    assert dc["dc"] == pytest.approx(2.0)
    assert dc["ac"] == pytest.approx(0.0, abs=1e-7)
    assert ac["dc"] == pytest.approx(0.0, abs=1e-7)
    assert ac["ac"] == pytest.approx(2.0)
    assert dc["parseval_relative_error"] < 1e-6
    assert ac["parseval_relative_error"] < 1e-6
