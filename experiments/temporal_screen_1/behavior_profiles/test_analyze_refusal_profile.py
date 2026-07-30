from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.temporal_screen_1.behavior_profiles.analyze_refusal_profile import (
    analyze,
    paired_condition_effect,
)


HERE = Path(__file__).resolve().parent
RESULT = HERE / "results" / "refusal_prefix_profile.json"


def _row(prompt_id: str, refusal: bool, log_odds: float) -> dict:
    return {
        "prompt_id": prompt_id,
        "generated_refusal": refusal,
        "refusal_log_odds": log_odds,
        "response_sha256": f"{prompt_id}-{refusal}-{log_odds}",
    }


def test_paired_condition_effect_preserves_prompt_pairing():
    baseline = {
        "a": _row("a", True, 5.0),
        "b": _row("b", True, 4.0),
        "c": _row("c", False, -2.0),
    }
    condition = {
        "a": _row("a", False, -1.0),
        "b": _row("b", True, 3.0),
        "c": _row("c", True, 1.0),
    }
    result = paired_condition_effect(baseline, condition, seed=1)

    assert result["paired_refusal_rate_reduction"] == 0.0
    assert result["baseline_refuses_condition_does_not"] == 1
    assert result["baseline_does_not_condition_refuses"] == 1
    assert np.isclose(
        result["paired_mean_refusal_log_odds_reduction"],
        4 / 3,
    )


def test_paired_condition_effect_rejects_unmatched_prompts():
    baseline = {"a": _row("a", True, 1.0)}
    condition = {"b": _row("b", False, -1.0)}

    with pytest.raises(ValueError, match="prompt IDs"):
        paired_condition_effect(baseline, condition, seed=1)


def test_recorded_prompt_lag_localization_panel():
    analysis = analyze(json.loads(RESULT.read_text()))
    effects = {
        effect["condition"]: effect
        for effect in analysis["prompt_lag_localization"]["paired_effects"]
    }

    assert effects["prompt_lag_ablation_1_4"][
        "condition_refusal_rate"
    ] == 27 / 32
    assert effects["prompt_lag_ablation_5_8"][
        "condition_refusal_rate"
    ] == 31 / 32
    assert effects["prompt_lag_ablation_17_32"][
        "condition_refusal_rate"
    ] == 28 / 32
    all_prior = effects["prompt_lag_ablation_all_prior"]
    assert all_prior["condition_refusal_rate"] == 0.0
    assert all_prior["baseline_refuses_condition_does_not"] == 31
    assert all_prior["paired_refusal_rate_reduction_ci95"] == [
        29 / 32,
        1.0,
    ]
    assert analysis["prompt_lag_localization"][
        "largest_finite_band_refusal_rate_reduction"
    ] == 4 / 32
    assert analysis["prompt_lag_localization"][
        "sum_finite_band_marginal_refusal_rate_reductions"
    ] == 9 / 32
