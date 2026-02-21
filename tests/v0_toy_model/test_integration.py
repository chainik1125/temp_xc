"""Integration test: end-to-end pipeline smoke test."""

import torch
import pytest

from src.utils.seed import set_seed
from src.v0_toy_model.configs import TrainingConfig
from src.v0_toy_model.data_generation import (
    create_correlation_matrix,
    get_training_batch,
)
from src.v0_toy_model.eval_sae import eval_sae
from src.v0_toy_model.metrics import (
    decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
)
from src.v0_toy_model.toy_model import ToyModel
from src.v0_toy_model.train_sae import create_sae, train_toy_sae


class TestEndToEndPipeline:
    @pytest.mark.slow
    def test_smoke_test(self):
        """End-to-end: create model, train SAE, evaluate. Verify pipeline runs."""
        set_seed(42)
        device = torch.device("cpu")

        # Small model
        num_features = 5
        hidden_dim = 20
        firing_probs = torch.tensor([0.4] * num_features)
        corr_matrix = create_correlation_matrix(num_features)
        mean_mags = torch.ones(num_features)
        std_mags = torch.zeros(num_features)

        toy_model = ToyModel(num_features, hidden_dim, ortho_num_steps=500)
        toy_model = toy_model.to(device)

        def generate_batch(batch_size: int) -> torch.Tensor:
            return get_training_batch(
                batch_size, firing_probs, corr_matrix, mean_mags, std_mags, device
            )

        # Train with very few samples for speed
        training_cfg = TrainingConfig(
            k=2.0,
            d_sae=num_features,
            total_training_samples=50_000,
            batch_size=512,
            seed=42,
        )

        sae = create_sae(hidden_dim, num_features, k=2.0, device=device)
        trained_sae = train_toy_sae(
            sae, toy_model, generate_batch, training_cfg, device
        )

        # Evaluate
        result = eval_sae(trained_sae, toy_model, generate_batch, n_samples=5000)
        assert result.true_l0 > 0
        assert result.sae_l0 > 0
        assert result.mse >= 0

        # Metrics
        cdec = decoder_pairwise_cosine_similarity(trained_sae)
        assert isinstance(cdec, float)
        assert cdec >= 0

        cos_sim = decoder_feature_cosine_similarity(trained_sae, toy_model)
        assert cos_sim.shape == (num_features, num_features)
