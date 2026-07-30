from __future__ import annotations

import numpy as np

from experiments.power_spectrum.code.build_backtracking_recovery_artifact import (
    _head_matches_candidates,
    _stored_npz_memmap,
    _tail_matches_official,
    cohort_sha256,
    stratified_lowest_score,
)


def test_stratified_lowest_score_matches_targets_and_preserves_mask_order() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint8)
    scores = np.asarray([0.4, 0.5, 0.1, 0.2, 0.3, 0.1])
    selected = stratified_lowest_score(
        labels,
        scores,
        target_rows=4,
        target_positive_rows=2,
    )
    assert selected.tolist() == [False, False, True, True, True, True]
    assert labels[selected].tolist() == [0, 1, 0, 1]


def test_cohort_sha256_is_order_sensitive() -> None:
    first = cohort_sha256(["a", "b"], [0, 1])
    second = cohort_sha256(["b", "a"], [1, 0])
    assert first != second


def test_stored_npz_memmap_reads_array_without_materializing_archive(
    tmp_path,
) -> None:
    path = tmp_path / "arrays.npz"
    expected = np.arange(48, dtype=np.float32).reshape(2, 6, 4)
    np.savez(path, X=expected, labels=np.asarray([0, 1]))

    mapped = _stored_npz_memmap(path, "X")

    assert isinstance(mapped, np.memmap)
    assert np.array_equal(mapped, expected)
    assert _tail_matches_official(
        mapped,
        expected,
        np.asarray([0, 1]),
    )


def test_head_matches_selected_candidate_rows(tmp_path) -> None:
    candidate_path = tmp_path / "trace_000.npz"
    candidates = np.arange(3 * 8 * 4, dtype=np.float32).reshape(3, 8, 4)
    np.savez(
        candidate_path,
        candidate_X=candidates,
        candidate_keys=np.asarray(["a", "b", "c"]),
        candidate_is_bt=np.asarray([0, 1, 0], dtype=np.uint8),
    )
    selected = np.asarray([True, False, True])
    recovered = candidates[selected].copy()

    assert _head_matches_candidates(
        recovered,
        [candidate_path],
        selected,
    )
    recovered[1, 0, 0] += 1
    assert not _head_matches_candidates(
        recovered,
        [candidate_path],
        selected,
    )
