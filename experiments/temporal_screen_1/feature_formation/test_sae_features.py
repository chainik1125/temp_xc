"""Checks for sparse conventional-SAE feature curves."""

from __future__ import annotations

import numpy as np

from experiments.temporal_screen_1.feature_formation.sae_features import (
    dense_sae_design,
    transported_sae_curve,
)


def _sparse_forming_panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_pairs, n_time, d_sae, k = 180, 9, 64, 4
    offsets = np.arange(-32, 4, 4)
    indices = np.empty((n_pairs, 2, n_time, k), dtype=np.int32)
    values = rng.exponential(size=(n_pairs, 2, n_time, k)).astype(np.float32)
    for pair in range(n_pairs):
        for arm in range(2):
            for time in range(n_time):
                indices[pair, arm, time] = rng.choice(
                    np.arange(1, d_sae),
                    size=k,
                    replace=False,
                )
    amplitude = 4.0 / (1 + np.exp(-(offsets + 12) / 2.5))
    # Reserve feature zero as a planted conventional SAE feature, but give
    # both arms the same noisy baseline so infinitesimal early amplitudes do
    # not become perfectly separable merely because the neutral is exactly
    # zero.
    indices[:, :, :, 0] = 0
    baseline = rng.exponential(size=(n_pairs, 2, n_time)).astype(np.float32)
    values[:, :, :, 0] = baseline
    values[:, 0, :, 0] += amplitude[None, :]
    return indices, values, offsets, d_sae


def test_dense_sae_design_recovers_sparse_values() -> None:
    indices, values, _, d_sae = _sparse_forming_panel()
    x, y, groups = dense_sae_design(
        indices[:2],
        values[:2],
        3,
        d_sae=d_sae,
        feature_ids=np.asarray([0, 1, 2]),
    )
    assert x.shape == (4, 3)
    assert y.tolist() == [1, 0, 1, 0]
    assert groups.tolist() == [0, 0, 1, 1]
    assert np.all(x[::2, 0] > 0)


def test_transported_sae_curve_finds_planted_forming_feature() -> None:
    indices, values, offsets, d_sae = _sparse_forming_panel(seed=4)
    result = transported_sae_curve(
        indices,
        values,
        offsets,
        discovery_band=(-12, -4),
        d_sae=d_sae,
        top_n=3,
        regularization=0.2,
        seed=8,
    )
    assert result.selected_feature_counts.get(0) == 5
    early = np.mean([row.auc for row in result.points if row.offset <= -24])
    late = np.mean([row.auc for row in result.points if row.offset >= -8])
    assert early < 0.65
    assert late > 0.9
