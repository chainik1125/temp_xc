"""Focused tests for the isolated positional-SAE sensitivity audit."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from experiments.backtracking_window_sweep.positional_sae_sensitivity import (
    OuterFold,
    grouped_inner_splits,
    paired_question_bootstrap_many,
    ranked_features,
    run_sensitivity,
    scale_stable_effect,
    validate_isolated_output,
)


def test_scale_stable_effect_is_invariant_to_column_units():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    values = np.array(
        [
            [0.0, 1.0, 0.2],
            [0.0, 2.0, 0.1],
            [0.0, 1.5, 0.3],
            [1.0, 2.0, 0.8],
            [1.1, 1.0, 0.9],
            [0.9, 1.5, 0.7],
        ],
        dtype=np.float64,
    )
    matrix = sparse.csr_matrix(values)
    rescaled = sparse.csr_matrix(values * np.array([100.0, 0.01, 7.0]))
    np.testing.assert_allclose(
        scale_stable_effect(matrix, y),
        scale_stable_effect(rescaled, y),
        rtol=1e-12,
        atol=1e-12,
    )
    assert ranked_features(matrix, y)[0] == 0


def test_grouped_inner_splits_never_split_question_groups():
    groups = np.repeat(np.asarray([f"q{index}" for index in range(12)]), 2)
    y = np.tile(np.array([0, 1], dtype=np.int8), 12)
    splits = grouped_inner_splits(y, groups, folds=3, seed=7)
    assert len(splits) == 3
    for train, valid in splits:
        assert set(groups[train]).isdisjoint(set(groups[valid]))
        assert set(np.unique(y[train])) == {0, 1}
        assert set(np.unique(y[valid])) == {0, 1}


def test_output_must_be_separate_from_primary_cell(tmp_path):
    cell = tmp_path / "full" / "cells" / "T6_seed42"
    cell.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside the primary cell"):
        validate_isolated_output(cell, cell / "sensitivity")
    validate_isolated_output(
        cell, tmp_path / "sensitivity" / "full" / "T6_seed42"
    )


def test_paired_question_bootstrap_is_deterministic():
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    indices = np.arange(len(y), dtype=np.int64)
    reference = np.array([0.2, 0.9, 0.1, 0.8, 0.3, 0.7])
    candidate = np.array([0.3, 0.8, 0.2, 0.7, 0.4, 0.6])
    outer = [
        OuterFold(
            fold=0,
            test_indices=indices,
            y=y,
            groups=groups,
            txc_probability=reference,
        )
    ]
    payload = {
        "test_indices": indices,
        "y": y,
        "groups": groups,
        "ordered": candidate,
    }
    first = paired_question_bootstrap_many(
        outer, {"S32": [payload]}, repeats=100, seed=11
    )
    second = paired_question_bootstrap_many(
        outer, {"S32": [payload]}, repeats=100, seed=11
    )
    assert first == second
    assert first["S32"]["repeats"] == 100


def test_end_to_end_audit_writes_only_separate_output(tmp_path):
    cell = tmp_path / "primary" / "full" / "cells" / "T2_seed42"
    (cell / "codes").mkdir(parents=True)
    (cell / "predictions" / "txc").mkdir(parents=True)
    groups = np.repeat(np.asarray([f"q{index}" for index in range(12)]), 2)
    y = np.tile(np.array([0, 1], dtype=np.int8), 12)
    rng = np.random.default_rng(3)
    values = rng.normal(size=(len(y), 8))
    values[:, 0] += 2.0 * y
    sparse.save_npz(
        cell / "codes" / "sae_positional_ordered.npz",
        sparse.csr_matrix(values),
    )
    folds = [np.arange(0, 12), np.arange(12, 24)]
    for fold, indices in enumerate(folds):
        np.savez_compressed(
            cell / "predictions" / "txc" / f"S4_fold{fold}.npz",
            test_indices=indices,
            y=y[indices],
            groups=groups[indices],
            ordered=(0.2 + 0.6 * y[indices]).astype(np.float32),
        )
    (cell / "result.json").write_text(
        """
{
  "status": "complete",
  "window": 2,
  "seed": 42,
  "n_rows": 24,
  "folds": 2,
  "probes": {"txc": [{"n_features": 4}]}
}
""".strip()
        + "\n"
    )
    output = tmp_path / "sensitivity" / "full" / "T2_seed42"
    result = run_sensitivity(
        cell_dir=cell,
        output_dir=output,
        s_grid=(2, 4),
        c_grid=(0.1, 1.0),
        inner_folds=2,
        bootstrap_repeats=20,
        bootstrap_seed=9,
    )
    assert result["status"] == "complete"
    assert [row["budget"] for row in result["summaries"]] == [2, 4]
    assert (output / "result.json").exists()
    assert (output / "summary.md").exists()
    assert not (cell / "summary.md").exists()
