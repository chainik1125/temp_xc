"""Tests for bench architecture specs — create, forward, eval, decoder."""

import pytest
import torch

from src.bench.architectures.topk_sae import TopKSAESpec
from src.bench.architectures.stacked_sae import StackedSAESpec
from src.bench.architectures.crosscoder import CrosscoderSpec
from src.bench.architectures.tfa import TFASpec
from src.bench.architectures import get_default_models

D_IN = 32
D_SAE = 16
K = 4
T = 2
DEVICE = torch.device("cpu")


class TestTopKSAE:
    def test_create_and_forward(self):
        spec = TopKSAESpec()
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        x = torch.randn(8, D_IN)
        loss, x_hat, z = model(x)
        assert x_hat.shape == (8, D_IN)
        assert z.shape == (8, D_SAE)
        assert (z > 0).float().sum(dim=-1).mean().item() == pytest.approx(K, abs=0.1)

    def test_eval_forward(self):
        spec = TopKSAESpec()
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        x = torch.randn(8, D_IN)
        out = spec.eval_forward(model, x)
        assert out.n_tokens == 8
        assert out.sum_se >= 0
        assert out.sum_signal > 0

    def test_decoder_directions(self):
        spec = TopKSAESpec()
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        dd = spec.decoder_directions(model)
        assert dd.shape == (D_IN, D_SAE)


class TestStackedSAE:
    def test_create_and_forward(self):
        spec = StackedSAESpec(T=T)
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        x = torch.randn(8, T, D_IN)
        loss, x_hat, z = model(x)
        assert x_hat.shape == (8, T, D_IN)
        assert z.shape == (8, T, D_SAE)

    def test_decoder_positions(self):
        spec = StackedSAESpec(T=T)
        assert spec.n_decoder_positions == T
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        dd0 = spec.decoder_directions(model, pos=0)
        dd1 = spec.decoder_directions(model, pos=1)
        dd_avg = spec.decoder_directions(model)
        assert dd0.shape == (D_IN, D_SAE)
        assert dd1.shape == (D_IN, D_SAE)
        assert dd_avg.shape == (D_IN, D_SAE)


class TestCrosscoder:
    def test_create_and_forward(self):
        spec = CrosscoderSpec(T=T)
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        x = torch.randn(8, T, D_IN)
        loss, x_hat, z = model(x)
        assert x_hat.shape == (8, T, D_IN)
        assert z.shape == (8, D_SAE)
        # k*T active latents
        expected_l0 = K * T
        actual_l0 = (z > 0).float().sum(dim=-1).mean().item()
        assert actual_l0 == pytest.approx(expected_l0, abs=0.1)

    def test_decoder_positions(self):
        spec = CrosscoderSpec(T=T)
        assert spec.n_decoder_positions == T
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        dd = spec.decoder_directions(model, pos=0)
        assert dd.shape == (D_IN, D_SAE)


class TestTFA:
    def test_create_and_forward(self):
        # TFA needs width divisible by (bottleneck_factor * n_heads)
        # With d_sae=16, n_heads=4, bottleneck_factor=1: 16/(1*4)=4, ok
        spec = TFASpec(n_heads=4, bottleneck_factor=1)
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        x = torch.randn(4, 8, D_IN)  # (B, T, d)
        recons, inter = model(x)
        assert recons.shape == (4, 8, D_IN)

    def test_eval_forward(self):
        spec = TFASpec(n_heads=4, bottleneck_factor=1)
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        x = torch.randn(4, 8, D_IN)
        out = spec.eval_forward(model, x)
        assert out.n_tokens == 4 * 8

    def test_decoder_directions(self):
        spec = TFASpec(n_heads=4, bottleneck_factor=1)
        model = spec.create(D_IN, D_SAE, K, DEVICE)
        dd = spec.decoder_directions(model)
        assert dd.shape == (D_IN, D_SAE)


class TestRegistry:
    def test_get_default_models(self):
        models = get_default_models(T_values=[2])
        names = [m.name for m in models]
        assert "TopKSAE" in names
        assert "TFA" in names
        assert "TFA-pos" in names
        assert "Stacked T=2" in names
        assert "TXCDR T=2" in names
