"""Tests for orthogonalization and toy model."""

import torch
import pytest

from src.v0_toy_model.orthogonalize import orthogonalize
from src.v0_toy_model.toy_model import ToyModel


class TestOrthogonalize:
    def test_shape(self):
        result = orthogonalize(5, 20, num_steps=500)
        assert result.shape == (5, 20)

    def test_unit_norm(self):
        result = orthogonalize(5, 20, num_steps=500)
        norms = result.norm(dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-4)

    def test_near_orthogonal(self):
        """Off-diagonal cosine sims should be near zero."""
        result = orthogonalize(5, 20, num_steps=1000)
        dot_products = result @ result.T
        off_diag = dot_products - torch.eye(5)
        assert off_diag.abs().max() < 0.05

    def test_target_cos_sim(self):
        """When target_cos_sim > 0, pairwise cosine sims should approach target."""
        target = 0.3
        result = orthogonalize(3, 20, target_cos_sim=target, num_steps=2000)
        dot_products = result @ result.T
        mask = ~torch.eye(3, dtype=torch.bool)
        off_diag = dot_products[mask]
        assert off_diag.mean().item() == pytest.approx(target, abs=0.05)


class TestToyModel:
    def test_output_shape(self):
        model = ToyModel(5, 20, ortho_num_steps=500)
        x = torch.randn(32, 5)
        out = model(x)
        assert out.shape == (32, 20)

    def test_feature_directions_shape(self):
        model = ToyModel(5, 20, ortho_num_steps=500)
        directions = model.feature_directions
        assert directions.shape == (5, 20)

    def test_feature_orthogonality(self):
        """Feature directions should be nearly orthogonal."""
        model = ToyModel(5, 20, ortho_num_steps=1000)
        directions = model.feature_directions
        cos_sims = directions @ directions.T
        off_diag = cos_sims - torch.eye(5)
        assert off_diag.abs().max() < 0.05

    def test_hooked_root_module(self):
        """Model should be a valid HookedRootModule."""
        from transformer_lens.hook_points import HookedRootModule
        model = ToyModel(3, 10, ortho_num_steps=100)
        assert isinstance(model, HookedRootModule)
