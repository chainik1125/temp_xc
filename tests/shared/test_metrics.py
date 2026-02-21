"""Tests for shared metrics module (generalized interface)."""

import torch
import pytest

from src.shared.metrics import (
    decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
    match_sae_latents_to_features,
    variance_explained,
)
from src.shared.initialization import init_sae_to_features
from src.shared.train_sae import create_sae
from src.shared.orthogonalize import orthogonalize


class TestDecoderFeatureCosineSimilarityTensorInterface:
    """Test that decoder_feature_cosine_similarity accepts raw tensors."""

    def test_accepts_tensor(self):
        """Should work with a raw feature_directions tensor."""
        features = orthogonalize(5, 20, num_steps=1000)
        sae = create_sae(20, 5, k=2.0, device=torch.device("cpu"))
        init_sae_to_features(sae, features)
        cos_sim = decoder_feature_cosine_similarity(sae, features)
        assert cos_sim.shape == (5, 5)

    def test_gt_init_near_identity(self):
        """GT-initialized SAE should have near-identity cos sim with features."""
        features = orthogonalize(5, 20, num_steps=1000)
        sae = create_sae(20, 5, k=2.0, device=torch.device("cpu"))
        init_sae_to_features(sae, features)
        cos_sim = decoder_feature_cosine_similarity(sae, features)
        max_per_row = cos_sim.abs().max(dim=1).values
        assert (max_per_row > 0.95).all()


class TestInitSaeToFeatures:
    """Test that init_sae_to_features works with raw tensors."""

    def test_sets_weights(self):
        features = orthogonalize(5, 20, num_steps=500)
        sae = create_sae(20, 5, k=2.0, device=torch.device("cpu"))
        init_sae_to_features(sae, features)
        # W_dec should match features
        assert torch.allclose(sae.W_dec.data, features, atol=1e-5)
        # W_enc should match features.T
        assert torch.allclose(sae.W_enc.data, features.T, atol=1e-5)

    def test_with_noise(self):
        features = orthogonalize(5, 20, num_steps=500)
        sae = create_sae(20, 5, k=2.0, device=torch.device("cpu"))
        init_sae_to_features(sae, features, noise_level=0.1)
        # Should NOT exactly match (noise added)
        assert not torch.allclose(sae.W_dec.data, features, atol=1e-3)
        # But rows should still be unit norm
        norms = sae.W_dec.data.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
