"""Aggregate-only contracts for the Ward teacher-force calibration."""

from __future__ import annotations

import numpy as np

from experiments.backtracking_window_sweep.calibrate_teacher_force import (
    candidate_metrics,
    candidate_rank,
)


def _record(**updates):
    record = {
        "attention_implementation": "sdpa",
        "add_special_tokens": True,
        "layer": 10,
        "boundary_shift": 0,
        "exact_count": 0,
        "bit_exact": False,
        "rmse": 1.0,
        "max_abs": 2.0,
        "cosine": 0.0,
    }
    record.update(updates)
    return record


def test_candidate_metrics_reports_exactness_without_values():
    official = np.arange(18, dtype=np.float32).reshape(6, 3)
    metrics = candidate_metrics(official.copy(), official)
    assert metrics["bit_exact"]
    assert metrics["exact_count"] == 18
    assert metrics["mismatch_count"] == 0
    assert metrics["max_abs"] == 0
    assert metrics["rmse"] == 0
    assert np.isclose(metrics["cosine"], 1)
    assert set(metrics).isdisjoint({"candidate", "official", "values"})


def test_candidate_metrics_detects_small_nonexact_difference():
    official = np.ones((6, 3), dtype=np.float32)
    candidate = official.copy()
    candidate[0, 0] += 0.25
    metrics = candidate_metrics(candidate, official)
    assert not metrics["bit_exact"]
    assert metrics["exact_count"] == 17
    assert metrics["mismatch_count"] == 1
    assert metrics["max_abs"] == 0.25
    assert metrics["rmse"] > 0
    assert 0 < metrics["cosine"] < 1


def test_candidate_rank_prefers_bit_exact_then_exact_count_then_rmse():
    exact = _record(
        exact_count=10,
        bit_exact=True,
        rmse=0.0,
        max_abs=0.0,
        cosine=1.0,
    )
    more_exact = _record(exact_count=9, rmse=0.2, cosine=0.9)
    lower_rmse = _record(exact_count=8, rmse=0.01, cosine=0.99)
    assert sorted(
        [lower_rmse, more_exact, exact],
        key=candidate_rank,
    ) == [exact, more_exact, lower_rmse]
