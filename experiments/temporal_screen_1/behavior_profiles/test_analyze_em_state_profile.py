"""Tests for completed Medical EM profile analysis."""

from __future__ import annotations

import json
from pathlib import Path

from .analyze_em_state_profile import analyze, canonicalize_progress_zero


RESULT = Path(__file__).resolve().parent / "results" / "em_state_profile_paper7b.json"


def test_progress_zero_is_canonicalized_without_changing_terminal():
    payload = json.loads(RESULT.read_text())
    source = payload["profiles"]["instantaneous_residual"]["raw_projection"]
    corrected, audit = canonicalize_progress_zero(source)
    assert audit["raw_macro_auc"] == 0.5
    assert audit["correction_needed"] is False
    assert corrected["auc"]["macro_auc"][0] == 0.5
    assert corrected["bootstrap"]["low"][0] == 0.5
    assert corrected["bootstrap"]["high"][0] == 0.5
    assert corrected["auc"]["macro_auc"][-1] == source["auc"]["macro_auc"][-1]


def test_completed_result_supports_only_weak_inconclusive_terminal_signal():
    summary = analyze(json.loads(RESULT.read_text()))
    assert summary["sample"]["n_positive"] == 16
    assert summary["sample"]["n_negative"] == 35
    assert summary["sample"]["n_eligible_prompt_groups"] == 6
    assert summary["sample"]["n_within_prompt_pairs"] == 38
    assert summary["cross_variant"]["terminal_macro_auc_range"] == [
        0.6041666666666666,
        0.6041666666666666,
    ]
    assert summary["cross_variant"]["n_terminal_intervals_excluding_chance"] == 0
    prefix_raw = summary["families"]["profiles"]["prefix_mean_residual"][
        "raw_projection"
    ]
    assert prefix_raw["progress_points_with_pointwise_low_above_chance"] == [0.1]


def test_positionwise_profile_finds_only_isolated_mid_response_separation():
    summary = analyze(json.loads(RESULT.read_text()))
    positionwise = summary["families"]["positionwise_profiles"]
    prefix_raw = positionwise["prefix_mean_residual"]["raw_projection"]
    prefix_cosine = positionwise["prefix_mean_residual"]["cosine_projection"]
    instant_raw = positionwise["instantaneous_residual"]["raw_projection"]
    assert prefix_raw["peak_macro_auc"] == 0.8055555555555555
    assert prefix_cosine["peak_macro_auc"] == 0.8055555555555555
    assert prefix_raw["peak_progress"] == 0.30000000000000004
    assert prefix_raw["peak_prompt_bootstrap_interval"][0] == 0.5833333333333334
    assert prefix_cosine["peak_prompt_bootstrap_interval"][0] == 0.611111111111111
    assert instant_raw["peak_progress"] == 0.6000000000000001
    support = summary["cross_variant"]["paired_raw_cosine_pointwise_support"]
    assert support["positionwise_profiles"]["prefix_mean_residual"] == [
        0.30000000000000004
    ]
    assert support["positionwise_profiles"]["instantaneous_residual"] == [
        0.6000000000000001
    ]
