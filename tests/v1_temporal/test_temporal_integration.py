"""Integration test: end-to-end v1 temporal pipeline smoke test."""

import torch
import pytest

from src.shared.configs import TrainingConfig
from src.shared.eval_sae import eval_sae
from src.shared.metrics import (
    decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
)
from src.shared.train_sae import create_sae, train_sae
from src.utils.seed import set_seed
from src.v1_temporal.temporal_data_generation import generate_temporal_batch
from src.v1_temporal.temporal_toy_model import TemporalToyModel


class TestTemporalEndToEnd:
    @pytest.mark.slow
    def test_smoke_test(self):
        """End-to-end: create temporal model with mixing, train SAE, evaluate."""
        set_seed(42)
        device = torch.device("cpu")

        n_global = 3
        n_local = 3
        num_features = n_global + n_local
        hidden_dim = 20
        seq_len = 2
        gamma = 0.3

        global_probs = torch.tensor([0.4] * n_global)
        local_probs = torch.tensor([0.4] * n_local)
        mean_mags = torch.ones(num_features)
        std_mags = torch.zeros(num_features)

        model = TemporalToyModel(
            num_features, hidden_dim, gamma=gamma, ortho_num_steps=500,
        )
        model = model.to(device)

        def generate_flattened(batch_size):
            feats = generate_temporal_batch(
                batch_size, seq_len, global_probs, local_probs,
                mean_magnitudes=mean_mags, std_magnitudes=std_mags,
                device=device,
            )
            hidden = model(feats)
            return hidden.view(-1, hidden_dim)

        true_l0 = global_probs.sum().item() + local_probs.sum().item()

        training_cfg = TrainingConfig(
            k=true_l0,
            d_sae=num_features,
            total_training_samples=50_000,
            batch_size=512,
            seed=42,
        )

        sae = create_sae(hidden_dim, num_features, k=true_l0, device=device)
        sae = train_sae(sae, generate_flattened, training_cfg, device)

        # Standard metrics only
        eval_result = eval_sae(sae, generate_flattened, n_samples=5000, true_l0=true_l0)
        assert eval_result.sae_l0 > 0
        assert eval_result.mse >= 0

        cdec = decoder_pairwise_cosine_similarity(sae)
        assert isinstance(cdec, float)
        assert cdec >= 0

        cos_sim = decoder_feature_cosine_similarity(sae, model.feature_directions)
        assert cos_sim.shape == (num_features, num_features)
