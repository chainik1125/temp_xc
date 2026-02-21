"""Tests for data generation module."""

import torch
import pytest

from src.v0_toy_model.data_generation import (
    _fix_correlation_matrix,
    create_correlation_matrix,
    generate_random_correlation_matrix,
    get_correlated_features,
    get_training_batch,
)


class TestFixCorrelationMatrix:
    def test_already_psd(self):
        """A PSD matrix should be unchanged (up to numerical precision)."""
        matrix = torch.eye(5)
        fixed = _fix_correlation_matrix(matrix)
        assert torch.allclose(fixed, matrix, atol=1e-5)

    def test_non_psd_becomes_psd(self):
        """A non-PSD matrix should have all eigenvalues >= 0 after fixing."""
        matrix = torch.eye(3)
        matrix[0, 1] = matrix[1, 0] = 0.99
        matrix[0, 2] = matrix[2, 0] = 0.99
        matrix[1, 2] = matrix[2, 1] = -0.99
        fixed = _fix_correlation_matrix(matrix)
        eigenvals = torch.linalg.eigvalsh(fixed)
        assert (eigenvals >= -1e-6).all()

    def test_diagonal_is_one(self):
        """After fixing, diagonal should be 1."""
        matrix = torch.randn(4, 4)
        matrix = (matrix + matrix.T) / 2
        fixed = _fix_correlation_matrix(matrix)
        assert torch.allclose(torch.diag(fixed), torch.ones(4), atol=1e-5)


class TestCreateCorrelationMatrix:
    def test_shape(self):
        matrix = create_correlation_matrix(5)
        assert matrix.shape == (5, 5)

    def test_identity_default(self):
        matrix = create_correlation_matrix(3)
        assert torch.allclose(matrix, torch.eye(3))

    def test_symmetric(self):
        correlations = {(0, 1): 0.5, (1, 2): -0.3}
        matrix = create_correlation_matrix(3, correlations)
        assert torch.allclose(matrix, matrix.T)

    def test_explicit_values(self):
        correlations = {(0, 1): 0.4}
        matrix = create_correlation_matrix(3, correlations)
        assert matrix[0, 1].item() == pytest.approx(0.4)
        assert matrix[1, 0].item() == pytest.approx(0.4)
        assert matrix[0, 0].item() == pytest.approx(1.0)


class TestGenerateRandomCorrelationMatrix:
    def test_shape(self):
        matrix = generate_random_correlation_matrix(10, seed=42)
        assert matrix.shape == (10, 10)

    def test_psd(self):
        matrix = generate_random_correlation_matrix(10, seed=42)
        eigenvals = torch.linalg.eigvalsh(matrix)
        assert (eigenvals >= -1e-6).all()

    def test_diagonal_is_one(self):
        matrix = generate_random_correlation_matrix(10, seed=42)
        assert torch.allclose(torch.diag(matrix), torch.ones(10), atol=1e-5)

    def test_symmetric(self):
        matrix = generate_random_correlation_matrix(10, seed=42)
        assert torch.allclose(matrix, matrix.T, atol=1e-6)

    def test_reproducible(self):
        m1 = generate_random_correlation_matrix(10, seed=42)
        m2 = generate_random_correlation_matrix(10, seed=42)
        assert torch.allclose(m1, m2)


class TestGetCorrelatedFeatures:
    def test_shape(self):
        probs = torch.tensor([0.5, 0.3, 0.7])
        corr = torch.eye(3)
        result = get_correlated_features(100, probs, corr, device=torch.device("cpu"))
        assert result.shape == (100, 3)

    def test_binary_output(self):
        probs = torch.tensor([0.5, 0.3])
        corr = torch.eye(2)
        result = get_correlated_features(1000, probs, corr, device=torch.device("cpu"))
        unique_vals = result.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_firing_rates_approximate(self):
        """Empirical firing rates should be close to specified probabilities."""
        torch.manual_seed(42)
        probs = torch.tensor([0.3, 0.6, 0.1])
        corr = torch.eye(3)
        result = get_correlated_features(50000, probs, corr, device=torch.device("cpu"))
        empirical_rates = result.mean(dim=0)
        assert torch.allclose(empirical_rates, probs, atol=0.02)

    def test_positive_correlation_increases_cooccurrence(self):
        """Positive correlation between features should increase co-occurrence."""
        torch.manual_seed(42)
        probs = torch.tensor([0.4, 0.4])
        n = 50000

        # Uncorrelated
        corr_uncorr = torch.eye(2)
        feats_uncorr = get_correlated_features(n, probs, corr_uncorr, torch.device("cpu"))
        cooccur_uncorr = (feats_uncorr[:, 0] * feats_uncorr[:, 1]).mean()

        # Positively correlated
        corr_pos = create_correlation_matrix(2, {(0, 1): 0.5})
        feats_pos = get_correlated_features(n, probs, corr_pos, torch.device("cpu"))
        cooccur_pos = (feats_pos[:, 0] * feats_pos[:, 1]).mean()

        assert cooccur_pos > cooccur_uncorr


class TestGetTrainingBatch:
    def test_shape(self):
        probs = torch.tensor([0.5, 0.3, 0.7])
        corr = torch.eye(3)
        result = get_training_batch(64, probs, corr, device=torch.device("cpu"))
        assert result.shape == (64, 3)

    def test_non_negative(self):
        """All values should be non-negative (ReLU applied)."""
        probs = torch.tensor([0.5, 0.3])
        corr = torch.eye(2)
        std_mags = torch.tensor([0.5, 0.5])
        result = get_training_batch(
            1000, probs, corr, std_magnitudes=std_mags, device=torch.device("cpu")
        )
        assert (result >= 0).all()

    def test_default_magnitudes(self):
        """With no std, active features should have magnitude ~1.0."""
        torch.manual_seed(42)
        probs = torch.tensor([1.0])  # Always fires
        corr = torch.eye(1)
        result = get_training_batch(100, probs, corr, device=torch.device("cpu"))
        # Should be very close to 1.0 since std=0
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_zero_features_when_not_firing(self):
        """Features that don't fire should have zero magnitude."""
        torch.manual_seed(42)
        probs = torch.tensor([0.0, 1.0])  # First never fires
        corr = torch.eye(2)
        result = get_training_batch(100, probs, corr, device=torch.device("cpu"))
        assert (result[:, 0] == 0).all()
