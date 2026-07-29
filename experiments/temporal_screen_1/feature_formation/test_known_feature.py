"""Checks for prespecified Ward-direction formation measurements."""

from __future__ import annotations

import numpy as np

from experiments.temporal_screen_1.feature_formation.known_feature import (
    curve_summary,
    paired_scalar_curve,
    project_direction_panel,
)


def test_known_direction_recovers_transient_formation() -> None:
    rng = np.random.default_rng(7)
    offsets = np.arange(-24, 9)
    n_pairs, d_model = 300, 16
    direction = rng.normal(size=d_model)
    direction /= np.linalg.norm(direction)
    panel = rng.normal(size=(n_pairs, 2, len(offsets), d_model))
    transient = 2.5 * np.exp(-((offsets + 10) / 3.0) ** 2)
    panel[:, 0] += transient[None, :, None] * direction[None, None]

    curve = project_direction_panel(panel, direction, offsets)
    summary = curve_summary(curve)

    assert -12 <= summary["peak_auc_offset"] <= -8
    assert summary["bands"]["ward"]["mean_auc"] > 0.8
    assert summary["bands"]["post"]["mean_auc"] < 0.6


def test_paired_scalar_curve_reports_sparsity_and_sign() -> None:
    event = np.asarray([[0.0, 3.0], [0.0, 2.0], [1.0, 4.0]])
    neutral = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    curve = paired_scalar_curve(event, neutral, [-1, 0])

    assert curve[0].event_nonzero_fraction == 1 / 3
    assert curve[0].neutral_nonzero_fraction == 0
    assert curve[1].paired_difference > 0
    assert curve[1].auc > 0.8
