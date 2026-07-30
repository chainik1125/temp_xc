"""Tests for the SAE-free Medical EM state-profile estimator."""

from __future__ import annotations

import numpy as np

from .em_state_profile import (
    binary_auc,
    estimate_state_profile,
    normalize_activation_rows,
    select_coherent_extremes,
)


def test_select_coherent_extremes_excludes_ambiguous_and_incoherent_rows():
    rows = [
        {"alignment": 25, "coherence": 90, "prompt_index": 0},
        {"alignment": 85, "coherence": 80, "prompt_index": 0},
        {"alignment": 60, "coherence": 95, "prompt_index": 1},
        {"alignment": 10, "coherence": 40, "prompt_index": 1},
    ]
    selected = select_coherent_extremes(rows)
    assert selected.indices.tolist() == [0, 1]
    assert selected.labels.tolist() == [1, 0]
    assert selected.groups.tolist() == [0, 0]
    assert selected.n_positive == 1
    assert selected.n_negative == 1
    assert selected.n_excluded == 2


def test_binary_auc_gives_half_credit_for_ties():
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([2.0, 1.0, 1.0, 0.0])
    assert binary_auc(labels, scores) == 0.875


def test_crossfit_profile_detects_terminal_signal_without_prompt_leakage():
    # Four prompt groups, each with two positive and two negative rollouts.
    # Progress zero is identical within prompt (so within-prompt AUC=0.5);
    # the held-out terminal class difference is shared along dimension zero.
    groups = np.repeat(np.arange(4), 4)
    labels = np.tile(np.asarray([1, 1, 0, 0]), 4)
    values = np.zeros((16, 3, 4), dtype=float)
    for index, (group, label) in enumerate(zip(groups, labels)):
        values[index, 0] = [0.0, float(group), 0.0, 1.0]
        values[index, 1] = [0.5 if label else -0.5, float(group), 0.0, 1.0]
        values[index, 2] = [2.0 if label else -2.0, float(group), 0.0, 1.0]

    result = estimate_state_profile(
        values,
        labels,
        groups,
        [0.0, 0.5, 1.0],
        n_bootstrap=100,
    )
    assert result["auc"]["macro_auc"] == [0.5, 1.0, 1.0]
    assert result["auc"]["n_eligible_groups"] == 4
    assert result["summary"]["terminal_macro_auc"] == 1.0
    assert result["summary"]["earliest_progress_with_sustained_recovery"] == 0.5
    assert all(fold["n_test"] == 4 for fold in result["folds"].values())


def test_row_normalization_is_unit_length():
    values = np.asarray([[[3.0, 4.0], [5.0, 12.0]]])
    normalized = normalize_activation_rows(values)
    assert np.allclose(np.linalg.norm(normalized, axis=-1), 1.0)
