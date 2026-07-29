"""Focused synthetic checks for feature-formation estimators."""

from __future__ import annotations

import numpy as np

from experiments.temporal_screen_1.feature_formation.estimators import (
    paired_design,
    positionwise_curve,
    summarize_curve,
    transported_curve,
)


def _forming_panel(
    *,
    seed: int = 0,
    n_pairs: int = 180,
    offsets: np.ndarray | None = None,
    d_model: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if offsets is None:
        offsets = np.arange(-32, 9, 4)
    direction = rng.normal(size=d_model)
    direction /= np.linalg.norm(direction)
    panel = rng.normal(
        scale=1.0,
        size=(n_pairs, 2, len(offsets), d_model),
    ).astype(np.float32)
    # The same semantic direction gradually becomes locally available only in
    # the event arm.  The neutral arm retains identical noise statistics.
    amplitude = 2.4 / (1.0 + np.exp(-(offsets + 12) / 3.0))
    panel[:, 0] += amplitude[None, :, None] * direction[None, None, :]
    return panel, offsets


def test_paired_design_keeps_complete_pairs() -> None:
    panel, _ = _forming_panel(n_pairs=3)
    x, y, groups = paired_design(panel, time_index=4, width=2)
    assert x.shape == (6, 48)
    assert y.tolist() == [1, 0, 1, 0, 1, 0]
    assert groups.tolist() == [0, 0, 1, 1, 2, 2]


def test_positionwise_curve_recovers_late_local_formation() -> None:
    panel, offsets = _forming_panel()
    curve = positionwise_curve(
        panel,
        offsets,
        width=1,
        regularization=0.2,
        seed=11,
    )
    early = np.mean([row.auc for row in curve if row.offset <= -24])
    late = np.mean([row.auc for row in curve if row.offset >= -4])
    assert early < 0.62
    assert late > 0.9
    summary = summarize_curve(curve)
    assert summary.midpoint_50_offset is not None
    assert -20 <= summary.midpoint_50_offset <= -4


def test_transported_curve_tracks_one_fixed_forming_direction() -> None:
    panel, offsets = _forming_panel(seed=5)
    curve = transported_curve(
        panel,
        offsets,
        discovery_band=(-12, -4),
        width=1,
        regularization=0.2,
        seed=17,
    )
    early = np.mean([row.auc for row in curve if row.offset <= -24])
    late = np.mean([row.auc for row in curve if row.offset >= -4])
    assert early < 0.62
    assert late > 0.9


def test_null_panel_has_no_stable_formation_curve() -> None:
    rng = np.random.default_rng(9)
    offsets = np.arange(-24, 9, 4)
    panel = rng.normal(size=(220, 2, len(offsets), 20)).astype(np.float32)
    curve = transported_curve(
        panel,
        offsets,
        discovery_band=(-12, -4),
        regularization=0.05,
        seed=3,
    )
    assert max(abs(row.auc - 0.5) for row in curve) < 0.12

