"""Tests for model interface contracts and basic behavior."""

import torch
import pytest

from temporal_bench.models.base import ModelOutput, TemporalAE
from temporal_bench.models.baum_welch_factorial import BaumWelchFactorialAE
from temporal_bench.models.topk_sae import TopKSAE
from temporal_bench.models.temporal_crosscoder import TemporalCrosscoder
from temporal_bench.models.per_feature_temporal import PerFeatureTemporalAE


B, T, d, m, k = 4, 5, 40, 20, 3


def _get_models():
    """Return instances of all model types."""
    return [
        ("sae", TopKSAE(d_in=d, d_sae=m, k=k)),
        ("txcdr", TemporalCrosscoder(d_in=d, d_sae=m, T=T, k_per_pos=k)),
        ("per_feature_temporal", PerFeatureTemporalAE(d_in=d, d_sae=m, T=T, k=k)),
        ("per_feature_temporal_causal", PerFeatureTemporalAE(d_in=d, d_sae=m, T=T, k=k, causal=True)),
        ("bw_factorial", BaumWelchFactorialAE(d_in=d, d_sae=m, T=T, k=k)),
    ]


class TestInterface:
    """All models must satisfy the TemporalAE interface contract."""

    @pytest.mark.parametrize("name,model", _get_models())
    def test_is_temporal_ae(self, name, model):
        assert isinstance(model, TemporalAE)

    @pytest.mark.parametrize("name,model", _get_models())
    def test_forward_output_type(self, name, model):
        x = torch.randn(B, T, d)
        out = model(x)
        assert isinstance(out, ModelOutput)

    @pytest.mark.parametrize("name,model", _get_models())
    def test_x_hat_shape(self, name, model):
        x = torch.randn(B, T, d)
        out = model(x)
        assert out.x_hat.shape == (B, T, d)

    @pytest.mark.parametrize("name,model", _get_models())
    def test_loss_is_scalar(self, name, model):
        x = torch.randn(B, T, d)
        out = model(x)
        assert out.loss.dim() == 0

    @pytest.mark.parametrize("name,model", _get_models())
    def test_loss_backward(self, name, model):
        x = torch.randn(B, T, d)
        out = model(x)
        out.loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    @pytest.mark.parametrize("name,model", _get_models())
    def test_decoder_directions_shape(self, name, model):
        D = model.decoder_directions()
        assert D.shape == (d, m)

    @pytest.mark.parametrize("name,model", _get_models())
    def test_metrics_dict(self, name, model):
        x = torch.randn(B, T, d)
        out = model(x)
        assert isinstance(out.metrics, dict)
        assert "recon_loss" in out.metrics
        assert "l0" in out.metrics


class TestTopKSAE:
    def test_l0_equals_k(self):
        model = TopKSAE(d_in=d, d_sae=m, k=k)
        x = torch.randn(B, T, d)
        out = model(x)
        # TopK should give exactly k nonzero per token
        assert abs(out.metrics["l0"] - k) < 0.01

    def test_position_independent(self):
        """SAE should produce same output for same input regardless of other positions."""
        model = TopKSAE(d_in=d, d_sae=m, k=k)
        x1 = torch.randn(1, T, d)
        x2 = x1.clone()
        x2[0, 1:, :] = torch.randn(T - 1, d)  # change other positions
        out1 = model(x1)
        out2 = model(x2)
        # Position 0 should be identical
        assert torch.allclose(out1.x_hat[0, 0], out2.x_hat[0, 0])


class TestTemporalCrosscoder:
    def test_shared_latent(self):
        model = TemporalCrosscoder(d_in=d, d_sae=m, T=T, k_per_pos=k)
        x = torch.randn(B, T, d)
        out = model(x)
        # Latents should be the same at every position (shared)
        for t in range(1, T):
            assert torch.allclose(out.latents[:, 0, :], out.latents[:, t, :])


class TestPerFeatureTemporalAE:
    def test_starts_as_sae(self):
        """With K=0, should behave like independent SAE."""
        torch.manual_seed(42)
        sae = TopKSAE(d_in=d, d_sae=m, k=k)
        torch.manual_seed(42)
        pft = PerFeatureTemporalAE(d_in=d, d_sae=m, T=T, k=k)
        # K is zero, so temporal_mix is identity
        x = torch.randn(B, T, d)
        out_sae = sae(x)
        out_pft = pft(x)
        assert torch.allclose(out_sae.x_hat, out_pft.x_hat, atol=1e-6)

    def test_causal_mask(self):
        model = PerFeatureTemporalAE(d_in=d, d_sae=m, T=T, k=k, causal=True)
        K = model._get_kernel()
        # Upper triangle should be zero
        for t in range(T):
            for s in range(t + 1, T):
                assert (K[:, t, s] == 0).all()

    def test_kernel_tracks_both_l0(self):
        model = PerFeatureTemporalAE(d_in=d, d_sae=m, T=T, k=k)
        # Set K to nonzero so temporal mixing changes activations
        with torch.no_grad():
            model.K.fill_(0.1)
        x = torch.randn(B, T, d)
        out = model(x)
        assert "pre_mix_l0" in out.metrics
        assert "l0" in out.metrics


class TestBaumWelchFactorialAE:
    def test_support_posteriors_are_probabilities(self):
        model = BaumWelchFactorialAE(d_in=d, d_sae=m, T=T, k=k)
        x = torch.randn(B, T, d)
        out = model(x)
        state_posteriors = out.aux["state_posteriors"]
        assert state_posteriors.shape == (B, T, m, 2)
        assert (state_posteriors >= 0).all()
        assert (state_posteriors <= 1).all()
        assert torch.allclose(
            state_posteriors.sum(dim=-1),
            torch.ones(B, T, m),
            atol=1e-5,
        )

    def test_metric_latents_are_binary_support(self):
        model = BaumWelchFactorialAE(d_in=d, d_sae=m, T=T, k=k)
        x = torch.randn(B, T, d)
        out = model(x)
        metric_latents = model.latents_for_metrics(out)
        assert metric_latents.shape == (B, T, m)
        assert ((metric_latents == 0) | (metric_latents == 1)).all()

    def test_topk_l0_equals_k(self):
        model = BaumWelchFactorialAE(d_in=d, d_sae=m, T=T, k=k)
        x = torch.randn(B, T, d)
        out = model(x)
        assert abs(out.metrics["l0"] - k) < 0.01
