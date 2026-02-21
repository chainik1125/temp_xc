"""Tests for temporal data generation."""

import torch
import pytest

from src.v1_temporal.temporal_data_generation import (
    generate_temporal_batch,
    generate_temporal_features,
)


class TestGenerateTemporalFeatures:
    def test_shape(self):
        """Output should be (batch, T, n_global + n_local)."""
        out = generate_temporal_features(
            batch_size=64,
            seq_len=2,
            global_firing_probs=torch.tensor([0.5, 0.5, 0.5]),
            local_firing_probs=torch.tensor([0.5, 0.5]),
            device=torch.device("cpu"),
        )
        assert out.shape == (64, 2, 5)

    def test_binary_output(self):
        """All values should be 0 or 1."""
        out = generate_temporal_features(
            batch_size=256,
            seq_len=3,
            global_firing_probs=torch.tensor([0.3, 0.7]),
            local_firing_probs=torch.tensor([0.5]),
            device=torch.device("cpu"),
        )
        assert ((out == 0) | (out == 1)).all()

    def test_global_features_identical_across_positions(self):
        """Global features should have the same binary value at all positions."""
        n_global = 3
        out = generate_temporal_features(
            batch_size=512,
            seq_len=4,
            global_firing_probs=torch.tensor([0.5] * n_global),
            local_firing_probs=torch.tensor([0.5, 0.5]),
            device=torch.device("cpu"),
        )
        # Global features are first n_global columns
        global_feats = out[:, :, :n_global]  # (batch, T, n_global)
        # All positions should match position 0
        for t in range(1, 4):
            assert torch.equal(global_feats[:, 0, :], global_feats[:, t, :])

    def test_local_features_differ_across_positions(self):
        """Local features should not be identical across positions (statistically)."""
        n_global = 2
        n_local = 3
        out = generate_temporal_features(
            batch_size=2048,
            seq_len=2,
            global_firing_probs=torch.tensor([0.5] * n_global),
            local_firing_probs=torch.tensor([0.5] * n_local),
            device=torch.device("cpu"),
        )
        local_feats = out[:, :, n_global:]  # (batch, 2, n_local)
        # Not all local features should be identical across positions
        matches = (local_feats[:, 0, :] == local_feats[:, 1, :]).float().mean()
        # With p=0.5, expected match rate is 0.5 (0 matches 0 or 1 matches 1)
        assert matches < 0.8  # Should be around 0.5, definitely not 1.0

    def test_firing_rates_approximate(self):
        """Empirical firing rates should match specified probabilities."""
        probs_global = torch.tensor([0.3, 0.7])
        probs_local = torch.tensor([0.5])
        out = generate_temporal_features(
            batch_size=10_000,
            seq_len=2,
            global_firing_probs=probs_global,
            local_firing_probs=probs_local,
            device=torch.device("cpu"),
        )
        # Pool over batch and time dimensions
        empirical = out.float().mean(dim=(0, 1))
        expected = torch.cat([probs_global, probs_local])
        assert torch.allclose(empirical, expected, atol=0.05)

    def test_seq_len_1(self):
        """Should work with seq_len=1."""
        out = generate_temporal_features(
            batch_size=32,
            seq_len=1,
            global_firing_probs=torch.tensor([0.5]),
            local_firing_probs=torch.tensor([0.5]),
            device=torch.device("cpu"),
        )
        assert out.shape == (32, 1, 2)


class TestGenerateTemporalBatch:
    def test_shape(self):
        out = generate_temporal_batch(
            batch_size=64,
            seq_len=2,
            global_firing_probs=torch.tensor([0.5] * 3),
            local_firing_probs=torch.tensor([0.5] * 2),
            device=torch.device("cpu"),
        )
        assert out.shape == (64, 2, 5)

    def test_non_negative(self):
        """All values should be >= 0 (ReLU applied)."""
        out = generate_temporal_batch(
            batch_size=256,
            seq_len=2,
            global_firing_probs=torch.tensor([0.5] * 3),
            local_firing_probs=torch.tensor([0.5] * 2),
            std_magnitudes=torch.tensor([0.3] * 5),
            device=torch.device("cpu"),
        )
        assert (out >= 0).all()

    def test_default_magnitudes(self):
        """With default magnitudes (mean=1, std=0), active features should be exactly 1."""
        out = generate_temporal_batch(
            batch_size=256,
            seq_len=2,
            global_firing_probs=torch.tensor([0.5] * 3),
            local_firing_probs=torch.tensor([0.5] * 2),
            device=torch.device("cpu"),
        )
        # Active features should be exactly 1.0
        active = out[out > 0]
        assert torch.allclose(active, torch.ones_like(active))

    def test_global_features_same_magnitude_pattern(self):
        """With std=0, global features should have identical values across positions."""
        n_global = 3
        out = generate_temporal_batch(
            batch_size=256,
            seq_len=2,
            global_firing_probs=torch.tensor([0.5] * n_global),
            local_firing_probs=torch.tensor([0.5] * 2),
            device=torch.device("cpu"),
        )
        global_feats = out[:, :, :n_global]
        assert torch.equal(global_feats[:, 0, :], global_feats[:, 1, :])
