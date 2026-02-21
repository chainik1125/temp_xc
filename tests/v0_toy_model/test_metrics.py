"""Tests for metrics module."""

import torch
import pytest

from src.v0_toy_model.metrics import (
    decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
    match_sae_latents_to_features,
    variance_explained,
)
from src.v0_toy_model.toy_model import ToyModel
from src.v0_toy_model.train_sae import create_sae
from src.v0_toy_model.initialization import init_sae_to_match_model


class TestDecoderPairwiseCosineSimilarity:
    def test_orthogonal_decoder_near_zero(self):
        """c_dec should be ~0 for an SAE initialized to orthogonal features."""
        model = ToyModel(5, 20, ortho_num_steps=1000)
        sae = create_sae(20, 5, k=2.0, device=torch.device("cpu"))
        init_sae_to_match_model(sae, model)
        cdec = decoder_pairwise_cosine_similarity(sae)
        assert cdec < 0.05

    def test_random_decoder_positive(self):
        """c_dec should be > 0 for a random (non-orthogonal) decoder."""
        sae = create_sae(5, 10, k=2.0, device=torch.device("cpu"))
        # Small d_in relative to d_sae means random decoder won't be orthogonal
        cdec = decoder_pairwise_cosine_similarity(sae)
        assert cdec > 0.01


class TestDecoderFeatureCosineSimilarity:
    def test_shape(self):
        model = ToyModel(5, 20, ortho_num_steps=500)
        sae = create_sae(20, 5, k=2.0, device=torch.device("cpu"))
        cos_sim = decoder_feature_cosine_similarity(sae, model)
        assert cos_sim.shape == (5, 5)

    def test_gt_sae_near_identity(self):
        """For a ground-truth SAE, cos sim should be near identity."""
        model = ToyModel(5, 20, ortho_num_steps=1000)
        sae = create_sae(20, 5, k=2.0, device=torch.device("cpu"))
        init_sae_to_match_model(sae, model)
        cos_sim = decoder_feature_cosine_similarity(sae, model)
        # Each latent should have high cosine sim with exactly one feature
        max_per_row = cos_sim.abs().max(dim=1).values
        assert (max_per_row > 0.95).all()


class TestVarianceExplained:
    def test_perfect_reconstruction(self):
        x = torch.randn(100, 10)
        ve = variance_explained(x, x)
        assert ve == pytest.approx(1.0, abs=1e-5)

    def test_zero_reconstruction(self):
        x = torch.randn(100, 10)
        ve = variance_explained(x, torch.zeros_like(x))
        # Should be negative or near zero
        assert ve < 0.1

    def test_partial_reconstruction(self):
        x = torch.randn(100, 10)
        noise = 0.1 * torch.randn_like(x)
        ve = variance_explained(x, x + noise)
        assert 0.5 < ve < 1.0


class TestMatchSaeLatentsToFeatures:
    def test_identity_permutation(self):
        """If cos sim is diagonal, permutation should be identity."""
        cos_sim = torch.eye(5)
        perm = match_sae_latents_to_features(cos_sim)
        assert torch.equal(perm, torch.arange(5))

    def test_reversed_permutation(self):
        """If cos sim is anti-diagonal, permutation should reverse."""
        cos_sim = torch.zeros(5, 5)
        for i in range(5):
            cos_sim[4 - i, i] = 1.0
        perm = match_sae_latents_to_features(cos_sim)
        expected = torch.tensor([4, 3, 2, 1, 0])
        assert torch.equal(perm, expected)
