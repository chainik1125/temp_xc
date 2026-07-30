from __future__ import annotations

import numpy as np

from experiments.temporal_screen_1.behavior_profiles.core import (
    binary_auc,
    nearest_length_matching,
    paired_bootstrap_curve,
    reveal_counts,
    spatial_mediation_summary,
    turn_on_summary,
)


def test_reveal_counts_are_unique_and_cover_endpoints():
    assert reveal_counts(3) == [0, 1, 2, 3]
    assert reveal_counts(0) == [0]


def test_nearest_length_matching_is_one_to_one():
    matched = nearest_length_matching([10, 30, 20], [31, 9, 100, 19])
    assert matched == [1, 0, 3]
    assert len(set(matched)) == 3


def test_binary_auc_handles_ties():
    assert binary_auc([2, 2], [1, 1]) == 1.0
    assert binary_auc([1], [1]) == 0.5


def test_turn_on_summary():
    summary = turn_on_summary([0, 0.25, 0.5, 0.75, 1], [0, 0.1, 0.6, 1, 1])
    assert summary.half_rise_fraction == 0.5
    assert summary.effective_fraction_95 == 0.75
    assert 0.5 < summary.normalized_area < 0.7


def test_flat_curve_has_no_onset():
    summary = turn_on_summary([0, 0.5, 1], [2, 2, 2])
    assert summary.half_rise_fraction is None
    assert summary.effective_fraction_95 is None


def test_paired_bootstrap_curve_shapes():
    values = np.arange(12, dtype=float).reshape(4, 3)
    result = paired_bootstrap_curve(values, n_bootstrap=20)
    assert result["mean"] == [4.5, 5.5, 6.5]
    assert len(result["ci_low"]) == 3


def test_spatial_mediation_summary_recovers_refusal_sequence_gap():
    result = spatial_mediation_summary(
        baseline_target=31 / 32,
        current_token_ablated=31 / 32,
        all_positions_ablated=0.0,
        baseline_neutral=0.0,
        current_token_added=0.0,
        all_positions_added=1.0,
    )
    assert result.current_token_necessity == 0.0
    assert result.all_positions_necessity == 31 / 32
    assert result.sequence_support_gap == 31 / 32
    assert result.sequence_sufficiency_gap == 1.0


def test_spatial_mediation_requires_complete_addition_panel():
    with np.testing.assert_raises(ValueError):
        spatial_mediation_summary(
            baseline_target=1.0,
            current_token_ablated=1.0,
            all_positions_ablated=0.0,
            baseline_neutral=0.0,
        )
