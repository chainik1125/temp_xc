"""Tests for :mod:`temp_bench.eval.detection`.

Pure-Python tests using a tiny stub :class:`TempBenchArch` and
synthetic activations with **known temporal structure** vs **window
density** confound. Verifies:

* encode-and-pool returns the right shapes for both per-token and
  window-arch encoders;
* PR-AUC monotonically saturates as S grows on a separable cohort;
* the within-window shuffle ablation correctly distinguishes a
  genuinely-temporal signal (positive shuffle gap) from a
  position-invariant signal (≈0 shuffle gap).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from temp_bench.architectures.base import ArchConfig, TempBenchArch
from temp_bench.eval.detection import (
    DEFAULT_S_GRID,
    DetectionResult,
    detect_case_study,
    detection_table,
    encode_and_pool,
)


# ── Stub archs ────────────────────────────────────────────────────────


class _PerTokenStub(TempBenchArch):
    """Per-token SAE-style stub: encode returns (B, T, d_sae)."""

    def __init__(self, *, d_in: int, d_sae: int, T: int = 1):
        super().__init__()
        self.config = ArchConfig(name="stub_pt", d_in=d_in, d_sae=d_sae, k_pos=1, T=T)
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.W_enc = torch.nn.Parameter(torch.randn(d_in, d_sae))
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in) -> (B, T, d_sae)
        return torch.einsum("btd,ds->bts", x, self.W_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bts,sd->btd", z, self.W_dec)

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.clone()


class _WindowStub(TempBenchArch):
    """TXC-style stub: encode returns (B, 1, d_sae) (collapses T axis)."""

    def __init__(self, *, d_in: int, d_sae: int, T: int):
        super().__init__()
        self.config = ArchConfig(name="stub_w", d_in=d_in, d_sae=d_sae, k_pos=1, T=T)
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        # Per-position encoder slabs — like TXC.
        self.W_enc = torch.nn.Parameter(torch.randn(T, d_in, d_sae))
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, T, d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,tds->bs", x, self.W_enc).unsqueeze(1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec)

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.mean(dim=1).clone()


# ── encode_and_pool ───────────────────────────────────────────────────


def test_encode_and_pool_per_token_shape():
    arch = _PerTokenStub(d_in=8, d_sae=16, T=5)
    X = torch.randn(20, 5, 8)
    Y = encode_and_pool(arch, X.numpy())
    assert Y.shape == (20, 16)


def test_encode_and_pool_window_shape():
    arch = _WindowStub(d_in=8, d_sae=16, T=5)
    X = torch.randn(20, 5, 8)
    Y = encode_and_pool(arch, X.numpy())
    assert Y.shape == (20, 16)


def test_encode_and_pool_max_not_mean_over_T():
    """The pool must be amax, not mean. Mean would dilute a single
    sharp activation across T."""
    arch = _PerTokenStub(d_in=4, d_sae=2, T=3)
    # Set W_enc so feature 0 = first-input-dim, feature 1 = sum-of-dims.
    arch.W_enc.data.zero_()
    arch.W_enc.data[0, 0] = 1.0
    arch.W_enc.data[:, 1] = 1.0

    x = torch.zeros(1, 3, 4)
    x[0, 0, 0] = 5.0   # spike on position 0, dim 0
    Y = encode_and_pool(arch, x.numpy())
    # Feature 0: amax(|5, 0, 0|) = 5; mean would be 5/3 ≈ 1.67.
    assert np.isclose(Y[0, 0], 5.0)


# ── DetectionResult ───────────────────────────────────────────────────


def _make_synthetic_separable_cohort(
    n_pos: int = 60,
    n_neg: int = 60,
    d_in: int = 16,
    T: int = 5,
    n_groups: int = 12,
    feature_scale: float = 4.0,
    rng_seed: int = 0,
):
    """Cohort where positives carry a (T, d_in)-shaped signature and
    negatives don't. Returns (X, y, group_ids)."""
    rng = np.random.default_rng(rng_seed)
    n = n_pos + n_neg
    X = rng.standard_normal((n, T, d_in)).astype(np.float32)

    # The positive signature: spike at d_in dim 0 across all T positions.
    # This is detectable both before AND after within-window shuffle.
    X[:n_pos, :, 0] += feature_scale

    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    # Assign group ids so train/test folds don't trivially leak.
    group_ids = np.array([i % n_groups for i in range(n)])
    return X, y, group_ids


def test_detect_case_study_high_pr_auc_on_separable_cohort():
    arch = _PerTokenStub(d_in=16, d_sae=8, T=5)
    # Wire encoder to detect the spike directly.
    arch.W_enc.data.zero_()
    arch.W_enc.data[0, 0] = 1.0   # feature 0 reads dim 0

    X, y, gids = _make_synthetic_separable_cohort()
    res = detect_case_study(
        arch, X, y, gids,
        S_grid=(1, 2, 4),
        n_folds=4,
        shuffle_seed=None,
    )
    assert isinstance(res, DetectionResult)
    assert res.pr_auc[1] > 0.9, f"expected PR-AUC≈1 on separable; got {res.pr_auc}"
    assert res.shuffle_gap is None  # disabled


def test_detect_case_study_shape_validation():
    arch = _PerTokenStub(d_in=4, d_sae=2)
    with pytest.raises(ValueError):
        detect_case_study(arch, np.zeros((10, 5)), np.zeros(10))
    with pytest.raises(ValueError, match="labels shape"):
        detect_case_study(arch, np.zeros((10, 5, 4)), np.zeros(7))
    with pytest.raises(ValueError, match="question_ids shape"):
        detect_case_study(
            arch, np.zeros((10, 5, 4)), np.zeros(10), np.zeros(7),
        )


# ── Shuffle ablation distinguishes temporal vs window-density ──────────


def _make_temporal_signature_cohort(
    n_pos: int = 80,
    n_neg: int = 80,
    d_in: int = 16,
    T: int = 5,
    rng_seed: int = 0,
    sig_strength: float = 3.0,
):
    """Positives carry a TEMPORALLY-STRUCTURED signature: dim 0 positive
    at t=0, negative at t=T-1. Negatives have no signature.

    A position-aware encoder slab can learn this. After
    within-window shuffle the signature collapses (the same dim 0
    values still appear, but their positions are random — a
    position-aware encoder no longer sees them as the same vector).
    Window-arch with per-position W_enc[t] should show a positive
    shuffle gap.
    """
    rng = np.random.default_rng(rng_seed)
    n = n_pos + n_neg
    X = rng.standard_normal((n, T, d_in)).astype(np.float32)
    X[:n_pos, 0, 0] += sig_strength
    X[:n_pos, T - 1, 0] -= sig_strength

    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    gids = np.array([i % 10 for i in range(n)])
    return X, y, gids


def _make_density_only_cohort(
    n_pos: int = 80,
    n_neg: int = 80,
    d_in: int = 16,
    T: int = 5,
    rng_seed: int = 0,
    sig_strength: float = 3.0,
):
    """Positives carry a position-INVARIANT signature: dim 0 elevated at
    EVERY t. Within-window shuffle preserves the signal completely. A
    position-aware encoder should detect this AND its shuffled version
    equally well — shuffle gap ≈ 0."""
    rng = np.random.default_rng(rng_seed)
    n = n_pos + n_neg
    X = rng.standard_normal((n, T, d_in)).astype(np.float32)
    X[:n_pos, :, 0] += sig_strength
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    gids = np.array([i % 10 for i in range(n)])
    return X, y, gids


def test_shuffle_gap_negligible_for_position_invariant_signal():
    """Position-invariant signature → shuffle gap ≈ 0."""
    torch.manual_seed(0)
    arch = _WindowStub(d_in=16, d_sae=8, T=5)
    # Make the encoder roughly average across positions on dim 0.
    arch.W_enc.data.zero_()
    arch.W_enc.data[:, 0, 0] = 1.0  # every t, dim 0 → feature 0

    X, y, gids = _make_density_only_cohort()
    res = detect_case_study(
        arch, X, y, gids,
        S_grid=(1, 2),
        n_folds=4,
        shuffle_seed=42,
    )
    # PR-AUC remains high in both cases.
    assert res.pr_auc[1] > 0.9
    assert res.pr_auc_shuffled[1] > 0.9
    # And the gap is small. Actual gap floats around 0 (stochastic from
    # the LogReg fit + KFold split); we just need it to NOT be a large
    # positive.
    assert abs(res.shuffle_gap[1]) < 0.05, res.shuffle_gap


def test_shuffle_gap_positive_for_truly_temporal_signal():
    """Position-specific signature: detection drops materially after
    within-window shuffle."""
    torch.manual_seed(0)
    arch = _WindowStub(d_in=16, d_sae=8, T=5)
    # Make the encoder extract the temporal-contrast signature: feature
    # 0 = (x[0, 0] - x[T-1, 0]).
    arch.W_enc.data.zero_()
    arch.W_enc.data[0, 0, 0] = 1.0       # t=0, dim 0
    arch.W_enc.data[-1, 0, 0] = -1.0     # t=T-1, dim 0

    X, y, gids = _make_temporal_signature_cohort()
    res = detect_case_study(
        arch, X, y, gids,
        S_grid=(1, 2),
        n_folds=4,
        shuffle_seed=42,
    )
    # Unshuffled detection should be high.
    assert res.pr_auc[1] > 0.9, res.pr_auc
    # Shuffled detection drops (because the position-contrast signature
    # is collapsed). We require a clearly positive gap.
    assert res.shuffle_gap[1] > 0.10, res.shuffle_gap


# ── Markdown rendering ─────────────────────────────────────────────────


def test_detection_table_markdown_has_correct_headers():
    arch = _PerTokenStub(d_in=16, d_sae=4, T=5)
    arch.W_enc.data.zero_()
    arch.W_enc.data[0, 0] = 1.0
    X, y, gids = _make_synthetic_separable_cohort(n_pos=20, n_neg=20)
    res = detect_case_study(arch, X, y, gids, S_grid=(1, 2), shuffle_seed=42, n_folds=2)
    md = detection_table({"stub_pt": res}, S_grid=(1, 2))
    assert "stub_pt unshuf" in md
    assert "stub_pt shuf" in md
    assert "stub_pt gap" in md
    assert "| 1 |" in md
    assert "| 2 |" in md
