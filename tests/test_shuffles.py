"""Tests for :mod:`temp_bench.utils.shuffles`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from temp_bench.utils.shuffles import shuffle_within_window, shuffle_within_window_numpy


def test_shape_dtype_preserved_torch():
    x = torch.randn(7, 5, 17, dtype=torch.float32)
    y = shuffle_within_window(x, T=5, seed=0)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_shape_dtype_preserved_numpy():
    x = np.random.randn(7, 5, 17).astype(np.float32)
    y = shuffle_within_window_numpy(x, T=5, seed=0)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_per_row_permutation_is_within_window():
    """Each row's T positions are a permutation of the original."""
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    y = shuffle_within_window(x, T=5, seed=1)
    for b in range(2):
        # Sort along T axis — sorted versions should match.
        assert torch.equal(
            x[b].sum(dim=-1).sort().values,
            y[b].sum(dim=-1).sort().values,
        )


def test_seeded_determinism():
    x = torch.randn(5, 5, 8)
    y1 = shuffle_within_window(x, T=5, seed=42)
    y2 = shuffle_within_window(x, T=5, seed=42)
    assert torch.equal(y1, y2)


def test_different_seeds_differ():
    x = torch.randn(5, 5, 8)
    y1 = shuffle_within_window(x, T=5, seed=1)
    y2 = shuffle_within_window(x, T=5, seed=2)
    assert not torch.equal(y1, y2)


def test_per_row_decorrelates():
    """With per_row=True, different rows get different permutations."""
    x = torch.arange(8 * 5 * 4, dtype=torch.float32).view(8, 5, 4)
    y = shuffle_within_window(x, T=5, seed=7, per_row=True)
    perms = []
    for b in range(8):
        # Reconstruct the permutation by matching positions.
        positions = []
        for j in range(5):
            for i in range(5):
                if torch.equal(y[b, j], x[b, i]):
                    positions.append(i)
                    break
        perms.append(tuple(positions))
    # At least two rows should have different perms.
    assert len(set(perms)) > 1


def test_global_shuffle_is_consistent_across_rows():
    """With per_row=False, all rows share one permutation."""
    x = torch.arange(8 * 5 * 4, dtype=torch.float32).view(8, 5, 4)
    y = shuffle_within_window(x, T=5, seed=7, per_row=False)
    # The same permutation π implies y[b, j] == x[b, π(j)]; recovering π
    # from row 0 must explain every other row.
    perm = []
    for j in range(5):
        for i in range(5):
            if torch.equal(y[0, j], x[0, i]):
                perm.append(i)
                break
    for b in range(1, 8):
        for j in range(5):
            assert torch.equal(y[b, j], x[b, perm[j]])


def test_validation_T_mismatch():
    x = torch.randn(2, 5, 3)
    with pytest.raises(ValueError, match="T="):
        shuffle_within_window(x, T=4, seed=0)


def test_validation_2d():
    x = torch.randn(5, 3)
    with pytest.raises(ValueError, match="\\(B, T, d_in\\)"):
        shuffle_within_window(x, T=5, seed=0)


def test_torch_numpy_agree_per_row_false():
    """Per-row=False uses torch.randperm (CPU generator); numpy uses
    np.random.default_rng. They use different RNG algorithms — we don't
    require output equality there. But shapes / preservation properties
    must hold."""
    x = np.random.randn(4, 5, 3).astype(np.float32)
    y = shuffle_within_window_numpy(x, T=5, seed=0, per_row=False)
    assert y.shape == x.shape
    # Position content preserved (permutation, not perturbation).
    np.testing.assert_array_equal(np.sort(x.sum(axis=-1), axis=-1),
                                  np.sort(y.sum(axis=-1), axis=-1))
