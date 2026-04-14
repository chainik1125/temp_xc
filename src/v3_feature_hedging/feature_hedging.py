"""Feature hedging reproduction (Chanin et al., arXiv 2508.16560 Fig 2).

Demonstrates that when SAE L0 < true L0, SAEs learn "sparse but wrong"
features via feature hedging. Correlated features get mixed in decoder
directions to improve MSE at the cost of monosemanticity.

Part 1: 5-feature reproduction (exact paper setup)
Part 2: 50-feature version at scale
"""

# %% Imports

from pathlib import Path

import torch

from src.shared.configs import TrainingConfig
from src.shared.correlation import (
    create_correlation_matrix,
    generate_random_correlation_matrix,
)
from src.shared.initialization import init_sae_to_features
from src.shared.magnitude_sampling import get_training_batch
from src.shared.metrics import (
    decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
    match_sae_latents_to_features,
)
from src.shared.orthogonalize import orthogonalize
from src.shared.plotting import plot_cdec_vs_l0
from src.v3_feature_hedging.plotting import (
    plot_correlation_matrix,
    plot_decoder_feature_heatmap,
)
from src.shared.train_sae import create_sae, train_sae
from src.utils.device import DEFAULT_DEVICE
from src.utils.logging import log
from src.utils.seed import set_seed

# %% Paths

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = str(_PROJECT_ROOT / "results" / "feature_hedging")

# %% Configuration — Part 1: 5-feature reproduction

SEED = 42
NUM_FEATURES_5 = 5
HIDDEN_DIM_5 = 20
FIRING_PROB_5 = 0.4  # true L0 = 5 * 0.4 = 2.0
TRUE_L0_5 = NUM_FEATURES_5 * FIRING_PROB_5
CORRELATION_RHO = 0.4  # f0 correlated with f1-f4

# %% Configuration — Part 2: 50-feature version

NUM_FEATURES_50 = 50
HIDDEN_DIM_50 = 100
FIRING_PROB_50 = 0.22  # true L0 ≈ 11
TRUE_L0_50 = NUM_FEATURES_50 * FIRING_PROB_50
K_VALUES_50 = [5, 8, 11, 14, 17]

# ============================================================================
# Part 1: 5-feature reproduction of Fig 2
# ============================================================================

# %% Setup 5-feature toy model

log("part1", "starting 5-feature hedging reproduction")
set_seed(SEED)

feature_directions_5 = orthogonalize(NUM_FEATURES_5, HIDDEN_DIM_5, target_cos_sim=0.0)
feature_directions_5 = feature_directions_5.to(DEFAULT_DEVICE)

# Correlation: f0 positively correlated with f1-f4 at rho=0.4
correlations_5 = {(0, i): CORRELATION_RHO for i in range(1, NUM_FEATURES_5)}
corr_matrix_5 = create_correlation_matrix(NUM_FEATURES_5, correlations_5)

log("part1", "correlation matrix built",
    num_features=NUM_FEATURES_5, rho=CORRELATION_RHO)

# %% Plot correlation matrix

plot_correlation_matrix(
    corr_matrix_5,
    title="Feature Correlation Matrix (5 features, ρ=0.4)",
    save_path=f"{RESULTS_DIR}/correlation_5feat",
)
log("plot", "saved correlation_5feat")

# %% Data generation function for 5-feature model

firing_probs_5 = torch.full((NUM_FEATURES_5,), FIRING_PROB_5)


def generate_hidden_acts_5(batch_size: int) -> torch.Tensor:
    """Generate hidden activations for the 5-feature model."""
    feature_acts = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=firing_probs_5,
        correlation_matrix=corr_matrix_5,
        device=DEFAULT_DEVICE,
    )
    return feature_acts @ feature_directions_5


# %% Train SAE with k=2 (matching true L0=2.0)

log("part1", "training SAE k=2 (matching true L0)")
sae_5_k2 = create_sae(d_in=HIDDEN_DIM_5, d_sae=NUM_FEATURES_5, k=2)
init_sae_to_features(sae_5_k2, feature_directions_5)
cfg_5_k2 = TrainingConfig(k=2, d_sae=NUM_FEATURES_5)
sae_5_k2 = train_sae(sae_5_k2, generate_hidden_acts_5, cfg_5_k2)

# %% Evaluate and plot k=2

cos_sim_5_k2 = decoder_feature_cosine_similarity(sae_5_k2, feature_directions_5)
perm_5_k2 = match_sae_latents_to_features(cos_sim_5_k2)
cos_sim_5_k2 = cos_sim_5_k2[perm_5_k2]

plot_decoder_feature_heatmap(
    cos_sim_5_k2,
    title="Decoder vs Features (5feat, k=2, matching L0)",
    save_path=f"{RESULTS_DIR}/decoder_vs_features_5feat_k2",
)
log("plot", "saved decoder_vs_features_5feat_k2")

# %% Train SAE with k=1 (below true L0, forces hedging)

log("part1", "training SAE k=1 (below true L0, hedging expected)")
sae_5_k1 = create_sae(d_in=HIDDEN_DIM_5, d_sae=NUM_FEATURES_5, k=1)
init_sae_to_features(sae_5_k1, feature_directions_5)
cfg_5_k1 = TrainingConfig(k=1, d_sae=NUM_FEATURES_5)
sae_5_k1 = train_sae(sae_5_k1, generate_hidden_acts_5, cfg_5_k1)

# %% Evaluate and plot k=1

cos_sim_5_k1 = decoder_feature_cosine_similarity(sae_5_k1, feature_directions_5)
perm_5_k1 = match_sae_latents_to_features(cos_sim_5_k1)
cos_sim_5_k1 = cos_sim_5_k1[perm_5_k1]

plot_decoder_feature_heatmap(
    cos_sim_5_k1,
    title="Decoder vs Features (5feat, k=1, hedging)",
    save_path=f"{RESULTS_DIR}/decoder_vs_features_5feat_k1",
)
log("plot", "saved decoder_vs_features_5feat_k1")

# %% Part 1 summary

log("part1", "5-feature experiment complete",
    k2_max_diag=cos_sim_5_k2.diag().abs().mean().item(),
    k1_max_diag=cos_sim_5_k1.diag().abs().mean().item())

# ============================================================================
# Part 2: 50-feature version
# ============================================================================

# %% Setup 50-feature toy model

log("part2", "starting 50-feature hedging experiment")
set_seed(SEED)

feature_directions_50 = orthogonalize(NUM_FEATURES_50, HIDDEN_DIM_50, target_cos_sim=0.0)
feature_directions_50 = feature_directions_50.to(DEFAULT_DEVICE)

corr_matrix_50 = generate_random_correlation_matrix(NUM_FEATURES_50, seed=SEED)

log("part2", "random correlation matrix built", num_features=NUM_FEATURES_50)

# %% Data generation function for 50-feature model

firing_probs_50 = torch.full((NUM_FEATURES_50,), FIRING_PROB_50)


def generate_hidden_acts_50(batch_size: int) -> torch.Tensor:
    """Generate hidden activations for the 50-feature model."""
    feature_acts = get_training_batch(
        batch_size=batch_size,
        firing_probabilities=firing_probs_50,
        correlation_matrix=corr_matrix_50,
        device=DEFAULT_DEVICE,
    )
    return feature_acts @ feature_directions_50


# %% Train SAEs at multiple k values

cdec_results: dict[float, list[float]] = {}

for k in K_VALUES_50:
    log("part2", f"training SAE k={k}", true_l0=TRUE_L0_50)
    set_seed(SEED)

    sae_50 = create_sae(d_in=HIDDEN_DIM_50, d_sae=NUM_FEATURES_50, k=k)
    init_sae_to_features(sae_50, feature_directions_50)
    cfg_50 = TrainingConfig(k=k, d_sae=NUM_FEATURES_50)
    sae_50 = train_sae(sae_50, generate_hidden_acts_50, cfg_50)

    # Compute decoder-feature cosine similarity and reorder
    cos_sim_50 = decoder_feature_cosine_similarity(sae_50, feature_directions_50)
    perm_50 = match_sae_latents_to_features(cos_sim_50)
    cos_sim_50 = cos_sim_50[perm_50]

    plot_decoder_feature_heatmap(
        cos_sim_50,
        title=f"Decoder vs Features (50feat, k={k})",
        save_path=f"{RESULTS_DIR}/decoder_vs_features_50feat_k{k}",
    )
    log("plot", f"saved decoder_vs_features_50feat_k{k}")

    # Compute c_dec for this k
    c_dec = decoder_pairwise_cosine_similarity(sae_50)
    cdec_results[float(k)] = [c_dec]
    log("part2", f"k={k}", c_dec=c_dec)

# %% Plot c_dec vs L0

plot_cdec_vs_l0(
    cdec_results,
    true_l0=TRUE_L0_50,
    save_path=f"{RESULTS_DIR}/cdec_vs_l0_50feat",
    title="Decoder Pairwise Cosine Similarity vs L0 (50 features)",
)
log("plot", "saved cdec_vs_l0_50feat")

# %% Final summary

log("summary", "feature hedging experiment complete",
    results_dir=RESULTS_DIR,
    part1_k_values="[1, 2]",
    part2_k_values=str(K_VALUES_50))
