"""Tests for temporal toy model."""

import torch
import pytest

from src.v1_temporal.temporal_toy_model import TemporalToyModel


class TestTemporalToyModel:
    def test_output_shape_temporal(self):
        """(batch, T, num_features) -> (batch, T, hidden_dim)."""
        model = TemporalToyModel(10, 20, ortho_num_steps=500)
        x = torch.randn(32, 2, 10)
        out = model(x)
        assert out.shape == (32, 2, 20)

    def test_output_shape_single_token(self):
        """(batch, num_features) -> (batch, hidden_dim) should also work."""
        model = TemporalToyModel(10, 20, ortho_num_steps=500)
        x = torch.randn(32, 10)
        out = model(x)
        assert out.shape == (32, 20)

    def test_feature_directions_shape(self):
        model = TemporalToyModel(10, 20, ortho_num_steps=500)
        assert model.feature_directions.shape == (10, 20)

    def test_feature_orthogonality(self):
        """Off-diagonal cosine similarity should be near zero."""
        model = TemporalToyModel(5, 20, ortho_num_steps=2000)
        F = model.feature_directions
        F_normed = F / F.norm(dim=1, keepdim=True)
        cos_sim = F_normed @ F_normed.T
        off_diag = cos_sim - torch.eye(5)
        assert off_diag.abs().max() < 0.05

    def test_hooked_root_module(self):
        """Should inherit from HookedRootModule."""
        from transformer_lens.hook_points import HookedRootModule
        model = TemporalToyModel(5, 20, ortho_num_steps=500)
        assert isinstance(model, HookedRootModule)

    def test_gamma_zero_position_independence(self):
        """With gamma=0, output at position t depends only on input at position t."""
        model = TemporalToyModel(5, 20, gamma=0.0, ortho_num_steps=500)
        x = torch.randn(8, 3, 5)
        out = model(x)
        for t in range(3):
            out_t = model(x[:, t, :])
            assert torch.allclose(out[:, t, :], out_t, atol=1e-6)

    def test_gamma_nonzero_breaks_position_independence(self):
        """With gamma>0, output at position t>0 depends on earlier positions."""
        model = TemporalToyModel(5, 20, gamma=0.5, ortho_num_steps=500)
        x = torch.randn(8, 3, 5)
        out = model(x)
        # Position 0 should still match (no prior context)
        out_0 = model(x[:, 0, :])
        assert torch.allclose(out[:, 0, :], out_0, atol=1e-6)
        # Position 1 should NOT match single-token (it mixes with position 0)
        out_1_single = model(x[:, 1, :])
        assert not torch.allclose(out[:, 1, :], out_1_single, atol=1e-4)

    def test_gamma_single_token_still_works(self):
        """Single-token input should work regardless of gamma."""
        model = TemporalToyModel(5, 20, gamma=0.5, ortho_num_steps=500)
        x = torch.randn(8, 5)
        out = model(x)
        assert out.shape == (8, 20)

    def test_mixing_formula_numerically(self):
        """Verify the causal mean-pooling formula on a known input."""
        model = TemporalToyModel(3, 6, gamma=0.4, ortho_num_steps=500)
        x = torch.randn(2, 4, 3)
        out = model(x)

        # Compute expected output manually
        h = model.embed(x)  # (2, 4, 6)
        gamma = 0.4
        expected = torch.zeros_like(h)
        for t in range(4):
            mean_so_far = h[:, :t+1, :].mean(dim=1)  # (2, 6)
            expected[:, t, :] = (1 - gamma) * h[:, t, :] + gamma * mean_so_far

        assert torch.allclose(out, expected, atol=1e-6)
