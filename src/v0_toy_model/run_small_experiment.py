"""Section 3.1 reproduction: 5 features, d=20, positive and negative correlations.

Trains SAEs at L0=2.0 (correct) and L0=1.8 (too low, initialized to ground truth)
for both positive (+0.4) and negative (-0.4) correlations between f0 and f1-f4.
Generates decoder-feature heatmaps to verify mixing patterns match Figures 2-3.
"""

import os
from functools import partial

import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v0_toy_model.configs import TrainingConfig
from src.v0_toy_model.data_generation import (
    create_correlation_matrix,
    get_training_batch,
)
from src.v0_toy_model.initialization import init_sae_to_match_model
from src.v0_toy_model.metrics import (
    decoder_feature_cosine_similarity,
    match_sae_latents_to_features,
)
from src.v0_toy_model.plotting import (
    plot_correlation_matrix,
    plot_decoder_feature_heatmap,
)
from src.v0_toy_model.toy_model import ToyModel
from src.v0_toy_model.train_sae import create_sae, train_toy_sae

RESULTS_DIR = "src/v0_toy_model/results/small_experiment"


def run_small_experiment(
    correlation_value: float,
    k_values: list[float],
    seed: int = 42,
) -> None:
    """Run the small experiment with a given correlation pattern.

    Args:
        correlation_value: Correlation between f0 and f1-f4.
        k_values: List of SAE L0 values to train.
        seed: Random seed.
    """
    set_seed(seed)
    device = DEFAULT_DEVICE
    corr_label = "positive" if correlation_value > 0 else "negative"
    print(f"\n{'='*60}")
    print(f"Small experiment: {corr_label} correlations (r={correlation_value})")
    print(f"{'='*60}")

    # Setup
    num_features = 5
    hidden_dim = 20
    firing_probs = torch.tensor([0.4] * num_features)
    true_l0 = firing_probs.sum().item()
    mean_mags = torch.ones(num_features)
    std_mags = torch.zeros(num_features)

    # Correlations: f0 <-> f1, f0 <-> f2, f0 <-> f3, f0 <-> f4
    correlations = {(0, i): correlation_value for i in range(1, num_features)}
    corr_matrix = create_correlation_matrix(num_features, correlations)

    print(f"True L0: {true_l0}")
    print(f"Correlation matrix:\n{corr_matrix}")

    # Build toy model
    toy_model = ToyModel(num_features, hidden_dim, ortho_num_steps=2000)
    toy_model = toy_model.to(device)

    # Plot correlation matrix
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_correlation_matrix(
        corr_matrix,
        f"Feature Correlations ({corr_label}, r={correlation_value})",
        os.path.join(RESULTS_DIR, f"corr_matrix_{corr_label}"),
    )

    # Batch generator
    def generate_batch(batch_size: int) -> torch.Tensor:
        return get_training_batch(
            batch_size, firing_probs, corr_matrix, mean_mags, std_mags, device
        )

    for k in k_values:
        print(f"\n--- Training SAE with k={k} ---")
        set_seed(seed)

        training_cfg = TrainingConfig(
            k=k,
            d_sae=num_features,
            total_training_samples=15_000_000,
            batch_size=1024,
            seed=seed,
        )

        sae = create_sae(hidden_dim, num_features, k=k, device=device)

        # For L0 < true L0, initialize to ground truth to show gradient pressure
        if k < true_l0:
            print(f"  Initializing SAE to ground truth (k={k} < true_l0={true_l0})")
            init_sae_to_match_model(sae, toy_model)

        sae = train_toy_sae(sae, toy_model, generate_batch, training_cfg, device)

        # Compute decoder-feature cosine similarity
        cos_sim = decoder_feature_cosine_similarity(sae, toy_model)
        perm = match_sae_latents_to_features(cos_sim)
        cos_sim_ordered = cos_sim[perm]

        k_str = str(k).replace(".", "p")
        plot_decoder_feature_heatmap(
            cos_sim_ordered,
            f"Decoder-Feature Cos Sim (k={k}, {corr_label} r={correlation_value})",
            os.path.join(RESULTS_DIR, f"heatmap_{corr_label}_k{k_str}"),
        )
        print(f"  Decoder-feature cos sim (ordered):")
        print(f"  {cos_sim_ordered.numpy()}")


def main() -> None:
    print("Small Experiment: Reproducing Chanin et al. Section 3.1")
    print(f"Device: {DEFAULT_DEVICE}")

    # Positive correlations (Figure 2)
    run_small_experiment(
        correlation_value=0.4,
        k_values=[2.0, 1.8],
        seed=42,
    )

    # Negative correlations (Figure 3)
    run_small_experiment(
        correlation_value=-0.4,
        k_values=[2.0, 1.8],
        seed=42,
    )

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
