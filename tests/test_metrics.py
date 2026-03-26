"""Tests for evaluation metrics."""

import torch
import pytest

from temporal_bench.metrics import compute_nmse, compute_l0, feature_recovery


class TestNMSE:
    def test_perfect_reconstruction(self):
        x = torch.randn(10, 5, 40)
        assert compute_nmse(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_zero_reconstruction(self):
        x = torch.randn(10, 5, 40)
        x_hat = torch.zeros_like(x)
        assert compute_nmse(x, x_hat) == pytest.approx(1.0, rel=0.1)

    def test_positive(self):
        x = torch.randn(10, 5, 40)
        x_hat = x + 0.1 * torch.randn_like(x)
        assert compute_nmse(x, x_hat) > 0


class TestL0:
    def test_all_zero(self):
        latents = torch.zeros(4, 5, 20)
        assert compute_l0(latents) == 0.0

    def test_all_nonzero(self):
        latents = torch.ones(4, 5, 20)
        assert compute_l0(latents) == 20.0

    def test_sparse(self):
        latents = torch.zeros(4, 5, 20)
        latents[:, :, :3] = 1.0  # 3 nonzero per token
        assert compute_l0(latents) == pytest.approx(3.0)


class TestFeatureRecovery:
    def test_perfect_recovery(self):
        """Decoder columns = true features -> AUC should be 1.0."""
        true_features = torch.eye(10, 40)[:10]  # (10, 40)
        decoder_dirs = true_features.T  # (40, 10)
        result = feature_recovery(decoder_dirs, true_features)
        assert result["auc"] == pytest.approx(1.0, abs=0.02)
        assert result["r_at_90"] == 1.0
        assert result["mean_max_cos"] == pytest.approx(1.0, abs=1e-5)

    def test_random_recovery(self):
        """Random decoder should have low AUC."""
        torch.manual_seed(42)
        true_features = torch.randn(10, 40)
        decoder_dirs = torch.randn(40, 10)
        result = feature_recovery(decoder_dirs, true_features)
        assert result["auc"] < 0.5
        assert result["r_at_90"] < 0.3

    def test_partial_recovery(self):
        """Some features aligned, some not."""
        true_features = torch.eye(10, 40)[:10]
        decoder_dirs = torch.zeros(40, 10)
        # First 5 decoder columns match first 5 features
        for i in range(5):
            decoder_dirs[i, i] = 1.0
        # Last 5 are random
        decoder_dirs[:, 5:] = torch.randn(40, 5)
        result = feature_recovery(decoder_dirs, true_features)
        assert result["r_at_90"] == pytest.approx(0.5, abs=0.1)
