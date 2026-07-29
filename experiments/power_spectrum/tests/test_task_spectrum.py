from __future__ import annotations

import numpy as np

from experiments.power_spectrum.code.task_spectrum import (
    classification_probe,
    dc_features,
    periodogram,
    spectral_features,
    summarize_spectrum,
)


def test_periodogram_is_normalized_and_dc_removal_works() -> None:
    rng = np.random.default_rng(0)
    levels = rng.normal(size=(128, 1, 3))
    x = levels + 0.05 * rng.normal(size=(128, 32, 3))
    raw = periodogram(x, center="global")
    removed = periodogram(x, center="sequence")
    assert np.isclose(raw.normalized_power.sum(), 1.0)
    assert raw.normalized_power[0] > 0.95
    assert removed.normalized_power[0] < 1e-20


def test_slow_process_has_more_low_frequency_mass_than_white_noise() -> None:
    rng = np.random.default_rng(1)
    white = rng.normal(size=(512, 64, 2))
    slow = np.zeros_like(white)
    slow[:, 0] = rng.normal(size=(512, 2))
    for t in range(1, 64):
        slow[:, t] = 0.95 * slow[:, t - 1] + 0.2 * rng.normal(size=(512, 2))
    assert summarize_spectrum(slow).ac_low_fraction > summarize_spectrum(white).ac_low_fraction


def test_cross_spectrum_retains_rotation_direction_that_power_erases() -> None:
    rng = np.random.default_rng(2)
    n, T = 1000, 32
    labels = np.repeat([0, 1], n // 2)
    sign = 2 * labels - 1
    phase = rng.uniform(0, 2 * np.pi, size=n)
    t = np.arange(T)
    theta = phase[:, None] + sign[:, None] * (2 * np.pi * 5 / T) * t[None, :]
    x = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    x += 0.05 * rng.normal(size=x.shape)

    power = classification_probe(
        spectral_features(x, kind="power", n_components=2), labels, seed=3
    )
    cross = classification_probe(
        spectral_features(x, kind="cross", n_components=2), labels, seed=3
    )
    assert power.score_mean < 0.58
    assert cross.score_mean > 0.95


def test_dc_vector_retains_signed_stable_state() -> None:
    rng = np.random.default_rng(4)
    labels = np.repeat([-1.0, 1.0], 128)
    x = labels[:, None, None] + 0.1 * rng.normal(size=(256, 16, 3))
    score = classification_probe(dc_features(x, n_components=3), labels, seed=5)
    assert score.score_mean > 0.99


def test_spectral_projection_accepts_disjoint_calibration_data() -> None:
    rng = np.random.default_rng(6)
    calibration = rng.normal(size=(32, 8, 5))
    probe = rng.normal(size=(48, 8, 5))
    power = spectral_features(
        probe,
        kind="power",
        n_components=3,
        fit_x=calibration,
    )
    dc = dc_features(probe, n_components=3, fit_x=calibration)
    assert power.shape[0] == probe.shape[0]
    assert dc.shape == (probe.shape[0], 3)
