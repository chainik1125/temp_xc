"""Sections 3.2/3.4/3.5 reproduction: 50 features, d=100, L0 sweep.

Trains BatchTopK SAEs at various k values with 5 seeds each. Generates:
- c_dec vs L0 plot (Figure 6)
- Sparsity-reconstruction tradeoff (Figure 4)
- Representative heatmaps at k=5, 11, 18 (Figure 1)
"""

import os
from functools import partial

import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v0_toy_model.configs import TrainingConfig
from src.v0_toy_model.data_generation import (
    generate_random_correlation_matrix,
    get_training_batch,
)
from src.v0_toy_model.eval_sae import eval_sae
from src.v0_toy_model.initialization import init_sae_to_match_model
from src.v0_toy_model.metrics import (
    decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
    match_sae_latents_to_features,
)
from src.v0_toy_model.plotting import (
    plot_cdec_vs_l0,
    plot_correlation_matrix,
    plot_decoder_feature_heatmap,
    plot_sparsity_reconstruction_tradeoff,
)
from src.v0_toy_model.toy_model import ToyModel
from src.v0_toy_model.train_sae import create_sae, train_toy_sae

RESULTS_DIR = "src/v0_toy_model/results/large_experiment"

# Experiment parameters
NUM_FEATURES = 50
HIDDEN_DIM = 100
STD_MAGNITUDE = 0.15
CORR_SEED = 42
K_VALUES = [1, 2, 3, 5, 7, 9, 11, 13, 15, 18, 21, 25]
NUM_SEEDS = 5
HEATMAP_K_VALUES = [5, 11, 18]


def setup_experiment(device: torch.device) -> tuple:
    """Create the toy model and data generation parameters.

    Returns:
        (toy_model, firing_probs, corr_matrix, mean_mags, std_mags, true_l0)
    """
    # Power-law firing probabilities: linearly from 0.345 to 0.05
    firing_probs = torch.linspace(0.345, 0.05, NUM_FEATURES)
    true_l0 = firing_probs.sum().item()
    print(f"True L0: {true_l0:.2f}")

    mean_mags = torch.ones(NUM_FEATURES)
    std_mags = torch.full((NUM_FEATURES,), STD_MAGNITUDE)

    # Random correlation matrix
    corr_matrix = generate_random_correlation_matrix(
        NUM_FEATURES,
        positive_ratio=0.5,
        correlation_strength_range=(0.3, 0.9),
        sparsity=0.3,
        seed=CORR_SEED,
    )

    # Build toy model
    set_seed(42)
    toy_model = ToyModel(NUM_FEATURES, HIDDEN_DIM, ortho_num_steps=3000)
    toy_model = toy_model.to(device)

    return toy_model, firing_probs, corr_matrix, mean_mags, std_mags, true_l0


def make_batch_fn(
    firing_probs: torch.Tensor,
    corr_matrix: torch.Tensor,
    mean_mags: torch.Tensor,
    std_mags: torch.Tensor,
    device: torch.device,
):
    """Create a batch generation function."""
    def generate_batch(batch_size: int) -> torch.Tensor:
        return get_training_batch(
            batch_size, firing_probs, corr_matrix, mean_mags, std_mags, device
        )
    return generate_batch


def run_l0_sweep(
    toy_model: ToyModel,
    generate_batch_fn,
    true_l0: float,
    device: torch.device,
) -> dict:
    """Run the L0 sweep experiment.

    Returns:
        Dict with cdec_results, ve_results, gt_ve_results, and heatmap data.
    """
    cdec_results: dict[float, list[float]] = {}
    ve_results: dict[float, list[float]] = {}
    gt_ve_results: dict[float, list[float]] = {}
    heatmap_data: dict[float, torch.Tensor] = {}

    for k in K_VALUES:
        print(f"\n{'='*40}")
        print(f"k = {k}")
        print(f"{'='*40}")

        cdec_results[k] = []
        ve_results[k] = []
        gt_ve_results[k] = []

        for seed_idx in range(NUM_SEEDS):
            seed = 42 + seed_idx
            set_seed(seed)
            print(f"  Seed {seed_idx + 1}/{NUM_SEEDS} (seed={seed})")

            # Train learned SAE
            training_cfg = TrainingConfig(
                k=float(k),
                d_sae=NUM_FEATURES,
                total_training_samples=15_000_000,
                batch_size=1024,
                seed=seed,
            )

            sae = create_sae(HIDDEN_DIM, NUM_FEATURES, k=float(k), device=device)
            sae = train_toy_sae(
                sae, toy_model, generate_batch_fn, training_cfg, device
            )

            # c_dec
            cdec = decoder_pairwise_cosine_similarity(sae)
            cdec_results[k].append(cdec)
            print(f"    c_dec = {cdec:.4f}")

            # Evaluation
            eval_result = eval_sae(sae, toy_model, generate_batch_fn)
            ve_results[k].append(eval_result.ve)
            print(f"    VE = {eval_result.ve:.4f}, MSE = {eval_result.mse:.4f}")

            # Ground-truth SAE evaluation at this k
            gt_sae = create_sae(HIDDEN_DIM, NUM_FEATURES, k=float(k), device=device)
            init_sae_to_match_model(gt_sae, toy_model)
            gt_eval = eval_sae(gt_sae, toy_model, generate_batch_fn)
            gt_ve_results[k].append(gt_eval.ve)
            print(f"    GT VE = {gt_eval.ve:.4f}")

            # Save heatmap for representative k values (first seed only)
            if k in HEATMAP_K_VALUES and seed_idx == 0:
                cos_sim = decoder_feature_cosine_similarity(sae, toy_model)
                perm = match_sae_latents_to_features(cos_sim)
                heatmap_data[k] = cos_sim[perm]

    return {
        "cdec": cdec_results,
        "ve": ve_results,
        "gt_ve": gt_ve_results,
        "heatmaps": heatmap_data,
    }


def main() -> None:
    print("Large Experiment: Reproducing Chanin et al. Sections 3.2, 3.4, 3.5")
    print(f"Device: {DEFAULT_DEVICE}")

    device = DEFAULT_DEVICE
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Setup
    toy_model, firing_probs, corr_matrix, mean_mags, std_mags, true_l0 = (
        setup_experiment(device)
    )

    # Save correlation matrix
    plot_correlation_matrix(
        corr_matrix,
        "Feature Correlation Matrix (random, seed=42)",
        os.path.join(RESULTS_DIR, "corr_matrix"),
    )

    generate_batch_fn = make_batch_fn(
        firing_probs, corr_matrix, mean_mags, std_mags, device
    )

    # Run the sweep
    results = run_l0_sweep(toy_model, generate_batch_fn, true_l0, device)

    # Plot c_dec vs L0 (Figure 6)
    plot_cdec_vs_l0(
        results["cdec"],
        true_l0,
        os.path.join(RESULTS_DIR, "cdec_vs_l0"),
        title=f"c_dec vs L0 (true L0 ≈ {true_l0:.1f})",
    )

    # Plot sparsity-reconstruction tradeoff (Figure 4)
    plot_sparsity_reconstruction_tradeoff(
        results["ve"],
        results["gt_ve"],
        true_l0,
        os.path.join(RESULTS_DIR, "sparsity_recon_tradeoff"),
    )

    # Plot representative heatmaps (Figure 1)
    for k, cos_sim in results["heatmaps"].items():
        plot_decoder_feature_heatmap(
            cos_sim,
            f"Decoder-Feature Cos Sim (k={k})",
            os.path.join(RESULTS_DIR, f"heatmap_k{k}"),
        )

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"True L0: {true_l0:.2f}")
    for k in K_VALUES:
        cdec_mean = sum(results["cdec"][k]) / len(results["cdec"][k])
        ve_mean = sum(results["ve"][k]) / len(results["ve"][k])
        print(f"  k={k:3d}: c_dec={cdec_mean:.4f}, VE={ve_mean:.4f}")

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
