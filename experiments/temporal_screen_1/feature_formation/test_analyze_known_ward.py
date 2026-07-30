"""Checks for corrected Ward causal-result aggregation."""

from __future__ import annotations

from experiments.temporal_screen_1.feature_formation.analyze_known_ward import (
    _bootstrap_causal,
    _causal_offset,
)


def test_causal_offset_parses_only_raw_per_offset_sources() -> None:
    assert _causal_offset("base_derived_off-13_raw") == -13
    assert _causal_offset("base_derived_off+0_raw") == 0
    assert _causal_offset("base_derived_union_raw") is None
    assert _causal_offset("base_derived_off+0_normmatched") is None


def test_paired_bootstrap_preserves_promptwise_lift() -> None:
    rows = []
    for prompt in range(6):
        rows.append(
            {
                "source": "stage_a_magnitude_zero_baseline",
                "prompt_id": f"p{prompt}",
                "keyword_rate": prompt / 100,
            }
        )
        rows.append(
            {
                "source": "intervention",
                "prompt_id": f"p{prompt}",
                "keyword_rate": prompt / 100 + 0.2,
            }
        )
    result = _bootstrap_causal({"rows": rows}, n_bootstrap=500, seed=4)
    intervention = result["sources"]["intervention"]

    assert abs(intervention["paired_lift_over_baseline"] - 0.2) < 1e-12
    assert all(
        abs(bound - 0.2) < 1e-12
        for bound in intervention["paired_lift_ci95"]
    )
