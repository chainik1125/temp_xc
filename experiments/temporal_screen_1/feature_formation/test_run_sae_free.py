"""Pair-construction checks for the Ward calibration."""

from __future__ import annotations

import numpy as np

from experiments.temporal_screen_1.feature_formation.run_sae_free import (
    build_paired_panel,
)


def test_pair_builder_uses_distant_same_rollout_neutral() -> None:
    cache = {
        "sequences": [
            np.arange(320 * 3, dtype=np.float32).reshape(320, 3),
            np.ones((320, 3), dtype=np.float32),
        ],
        "event_positions": [np.asarray([150]), np.asarray([], dtype=np.int64)],
        "qids": ["a", "b"],
        "categories": ["math", "math"],
    }
    offsets = np.arange(-16, 9)
    panel, records = build_paired_panel(
        cache,
        offsets,
        exclusion_radius=40,
        preferred_separation=80,
    )
    assert panel.shape == (1, 2, len(offsets), 3)
    assert len(records) == 1
    assert abs(records[0]["neutral_minus_event"]) >= 40
    np.testing.assert_array_equal(
        panel[0, 0],
        cache["sequences"][0][150 + offsets],
    )
