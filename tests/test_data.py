"""Tests for data generation pipeline."""

import torch
import pytest

from temporal_bench.data.markov import generate_markov_support, pi_rho_to_transition
from temporal_bench.data.toy_model import ToyModel
from temporal_bench.data.pipeline import DataPipeline
from temporal_bench.config import DataConfig


class TestMarkov:
    def test_shapes(self):
        support = generate_markov_support(10, 64, pi=0.1, rho=0.5, n_sequences=32)
        assert support.shape == (32, 10, 64)

    def test_binary(self):
        support = generate_markov_support(5, 100, pi=0.1, rho=0.5, n_sequences=10)
        assert torch.all((support == 0) | (support == 1))

    def test_marginal_rate(self):
        """Empirical marginal should match pi within tolerance."""
        gen = torch.Generator().manual_seed(42)
        support = generate_markov_support(
            20, 1000, pi=0.1, rho=0.5, n_sequences=100, generator=gen
        )
        empirical_pi = support.mean().item()
        assert abs(empirical_pi - 0.1) < 0.02

    def test_iid_at_rho_zero(self):
        """At rho=0, lag-1 autocorrelation should be near zero."""
        gen = torch.Generator().manual_seed(42)
        support = generate_markov_support(
            10, 1000, pi=0.1, rho=0.0, n_sequences=100, generator=gen
        )
        # Compute lag-1 autocorrelation
        s0 = support[:, :, :-1]
        s1 = support[:, :, 1:]
        mu = support.mean()
        cov = ((s0 - mu) * (s1 - mu)).mean()
        var = ((support - mu) ** 2).mean()
        rho_empirical = (cov / var).item()
        assert abs(rho_empirical) < 0.05

    def test_persistence_at_high_rho(self):
        """At rho=0.9, lag-1 autocorrelation should be near 0.9."""
        gen = torch.Generator().manual_seed(42)
        support = generate_markov_support(
            10, 1000, pi=0.1, rho=0.9, n_sequences=100, generator=gen
        )
        s0 = support[:, :, :-1]
        s1 = support[:, :, 1:]
        mu = support.mean()
        cov = ((s0 - mu) * (s1 - mu)).mean()
        var = ((support - mu) ** 2).mean()
        rho_empirical = (cov / var).item()
        assert abs(rho_empirical - 0.9) < 0.05

    def test_transition_probabilities(self):
        alpha, beta = pi_rho_to_transition(0.1, 0.5)
        # alpha = rho*(1-pi) + pi = 0.5*0.9 + 0.1 = 0.55
        assert abs(alpha - 0.55) < 1e-10
        # beta = pi*(1-rho) = 0.1*0.5 = 0.05
        assert abs(beta - 0.05) < 1e-10


class TestToyModel:
    def test_feature_orthogonality(self):
        gen = torch.Generator().manual_seed(42)
        tm = ToyModel(10, 40, generator=gen)
        F = tm.features  # (10, 40)
        gram = F @ F.T
        off_diag = gram - torch.eye(10)
        assert off_diag.abs().max() < 1e-5

    def test_embed_shapes(self):
        gen = torch.Generator().manual_seed(42)
        tm = ToyModel(10, 40, generator=gen)
        support = torch.ones(5, 10, 8)  # all features on
        x = tm.embed(support, generator=gen)
        assert x.shape == (5, 8, 40)

    def test_embed_zeros(self):
        gen = torch.Generator().manual_seed(42)
        tm = ToyModel(10, 40, generator=gen)
        support = torch.zeros(5, 10, 8)
        x = tm.embed(support, generator=gen)
        assert x.abs().max() < 1e-10


class TestPipeline:
    def test_sample_windows(self):
        cfg = DataConfig(n_features=10, d_model=40, pi=0.1)
        pipe = DataPipeline(cfg)
        x = pipe.sample_windows(batch_size=16, T=5, rho=0.5)
        assert x.shape == (16, 5, 40)

    def test_eval_data(self):
        cfg = DataConfig(n_features=10, d_model=40, pi=0.1)
        pipe = DataPipeline(cfg)
        x = pipe.eval_data(n_sequences=100, T=5, rho=0.5)
        assert x.shape == (100, 5, 40)

    def test_true_features(self):
        cfg = DataConfig(n_features=10, d_model=40)
        pipe = DataPipeline(cfg)
        assert pipe.true_features.shape == (10, 40)
