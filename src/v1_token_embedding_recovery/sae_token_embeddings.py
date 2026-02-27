"""Can SAEs recover true features from token embeddings?

Reproduces notebooks/sae_token_embeddings.ipynb using our SAE pipeline.

If an SAE only sees frozen feature combinations in embeddings, can it still
learn the underlying true features? We test this in a toy model with 50 true
features and 25 "token embeddings". We create these token embeddings by
sampling 25 activations and freezing them. The SAE is then trained just from
these 25 frozen activations.

Original finding: SAEs learn the token embeddings, not the true features.
"""

# %% Imports

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

from src.shared.configs import TrainingConfig
from src.shared.magnitude_sampling import get_training_batch
from src.shared.orthogonalize import orthogonalize
from src.shared.train_sae import DataIterator, create_sae, train_sae
from src.shared.plotting import save_figure
from src.utils.cos_sim import cos_sims
from src.utils.device import DEFAULT_DEVICE
from src.utils.logging import log
from src.utils.seed import set_seed

# %% Configuration

NUM_FEATURES = 50
HIDDEN_DIM = 100
NUM_TOKENS = 25
FIRING_PROB = 11 / 50  # ~0.22, gives avg 11 features firing
FIRING_MAG_MEAN = 1.0
FIRING_MAG_STD = 0.15
TARGET_COS_SIM = 0.0
TOPK_K = 22
SEED = 42

# Project-root-relative results directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = str(_PROJECT_ROOT / "results" / "sae_token_embeddings")

# %% Plotting helpers


def plot_sae_feature_cos_sims(
    sae_W_enc: torch.Tensor,
    sae_W_dec: torch.Tensor,
    feature_directions: torch.Tensor,
    title_suffix: str,
    reorder_features: bool = True,
    dtick: int = 5,
    save_path: str | None = None,
) -> None:
    """Plot encoder and decoder cosine similarity with true features.

    Args:
        sae_W_enc: Encoder weights, shape (d_in, d_sae).
        sae_W_dec: Decoder weights, shape (d_sae, d_in).
        feature_directions: True features, shape (num_features, hidden_dim).
        title_suffix: Label for the plot title.
        reorder_features: Whether to sort latents by best feature match.
        dtick: Tick spacing for axes.
        save_path: Base path (without extension) for saving .png.
    """
    # cos_sims expects (d, n) column-vectors
    # W_dec.T: (d_in, d_sae), feature_directions.T: (hidden_dim, num_features)
    dec_sims = torch.round(cos_sims(sae_W_dec.T, feature_directions.T) * 100) / 100 + 0.0
    enc_sims = torch.round(cos_sims(sae_W_enc, feature_directions.T) * 100) / 100 + 0.0

    if reorder_features:
        best_matches = torch.argmax(torch.abs(dec_sims), dim=1)
        sorted_idx = torch.argsort(best_matches)
        dec_sims = dec_sims[sorted_idx]
        enc_sims = enc_sims[sorted_idx]

    enc_data = enc_sims.detach().cpu().numpy()
    dec_data = dec_sims.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cosine Similarity with True Features ({title_suffix})", fontsize=13)

    sns.heatmap(
        enc_data, vmin=-1, vmax=1, center=0, cmap="RdBu_r",
        cbar=False, ax=ax1,
    )
    ax1.set_title("SAE encoder")
    ax1.set_xlabel("True feature")
    ax1.set_ylabel("SAE Latent")
    ax1.xaxis.set_major_locator(plt.MultipleLocator(dtick))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(dtick))

    sns.heatmap(
        dec_data, vmin=-1, vmax=1, center=0, cmap="RdBu_r",
        cbar_kws={"label": "cos sim"}, ax=ax2,
    )
    ax2.set_title("SAE decoder")
    ax2.set_xlabel("True feature")
    ax2.set_ylabel("SAE Latent")
    ax2.xaxis.set_major_locator(plt.MultipleLocator(dtick))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(dtick))

    plt.tight_layout()
    plt.show()
    if save_path is not None:
        save_figure(fig, save_path)


def plot_decoder_vs_token_embeddings(
    sae_W_dec: torch.Tensor,
    token_hidden_acts: torch.Tensor,
    title: str,
    save_path: str | None = None,
) -> None:
    """Plot cosine similarity between SAE decoder and token embeddings.

    Args:
        sae_W_dec: Decoder weights, shape (d_sae, d_in).
        token_hidden_acts: Token embeddings in hidden space, shape (n_tokens, d_in).
        title: Plot title.
        save_path: Base path (without extension) for saving .png.
    """
    # cos_sims expects (d, n) column-vectors
    sims = cos_sims(sae_W_dec.T, token_hidden_acts.T)
    data = sims.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data, vmin=-1, vmax=1, center=0, cmap="RdBu_r",
        cbar_kws={"label": "cos sim"}, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Token embedding")
    ax.set_ylabel("SAE Latent")
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        save_figure(fig, save_path)


# %% Create toy model and token embeddings

set_seed(SEED)

log("info", "initializing toy_model", num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM,
    target_cos_sim=TARGET_COS_SIM)

# Orthogonal feature directions: (num_features, hidden_dim)
feature_directions = orthogonalize(NUM_FEATURES, HIDDEN_DIM, target_cos_sim=TARGET_COS_SIM)
feature_directions = feature_directions.to(DEFAULT_DEVICE)

# Simple linear embedding: feature_acts @ feature_directions -> hidden_acts
# This replaces the notebook's ToyModel(nn.Linear)
def embed(feature_acts: torch.Tensor) -> torch.Tensor:
    """Map feature activations to hidden space: (batch, num_features) -> (batch, hidden_dim)."""
    return feature_acts.to(DEFAULT_DEVICE) @ feature_directions

# %% Generate 25 frozen token embeddings

firing_probs = torch.full((NUM_FEATURES,), FIRING_PROB)
corr_matrix = torch.eye(NUM_FEATURES)  # no feature correlations

token_feats = get_training_batch(
    batch_size=NUM_TOKENS,
    firing_probabilities=firing_probs,
    correlation_matrix=corr_matrix,
    std_magnitudes=torch.full((NUM_FEATURES,), FIRING_MAG_STD),
)
token_feats = token_feats.to(DEFAULT_DEVICE)

avg_nnz = (token_feats > 0).float().sum(dim=1).mean().item()
log("data", f"generated {NUM_TOKENS} token_embeddings", firing_prob=FIRING_PROB,
    avg_nnz=avg_nnz, firing_mag=f"N({FIRING_MAG_MEAN}, {FIRING_MAG_STD})")

# Token embeddings in hidden space (what the SAE actually sees)
token_hidden_acts = embed(token_feats)


def generate_batch(batch_size: int) -> torch.Tensor:
    """Sample uniformly from frozen token embeddings (in hidden space)."""
    idx = torch.randint(0, NUM_TOKENS, (batch_size,))
    return token_hidden_acts[idx]


# %% Train TopK SAE with 50 latents (same as number of true features)

log("info", "training topk sae", d_sae=NUM_FEATURES, top_k=TOPK_K)

topk_sae_50 = create_sae(d_in=HIDDEN_DIM, d_sae=NUM_FEATURES, k=TOPK_K)
topk_training_cfg = TrainingConfig(k=TOPK_K, d_sae=NUM_FEATURES)
topk_sae_50 = train_sae(topk_sae_50, generate_batch, topk_training_cfg)

# %% Evaluate TopK SAE (50 latents) — decoder vs true features

plot_sae_feature_cos_sims(
    topk_sae_50.W_enc.data, topk_sae_50.W_dec.data, feature_directions,
    "TopK 50-latent SAE",
    save_path=f"{RESULTS_DIR}/topk_k{TOPK_K}_50_latent_enc_dec_vs_features",
)

# %% Evaluate TopK SAE (50 latents) — decoder vs token embeddings

plot_decoder_vs_token_embeddings(
    topk_sae_50.W_dec.data, token_hidden_acts,
    "cos sim between SAE decoder and token embeddings (TopK 50-latent SAE)",
    save_path=f"{RESULTS_DIR}/topk_k{TOPK_K}_50_latent_dec_vs_token_embeddings",
)

# %% Summary stats for TopK 50-latent SAE

dec_vs_features = cos_sims(topk_sae_50.W_dec.T, feature_directions.T)
dec_vs_tokens = cos_sims(topk_sae_50.W_dec.T, token_hidden_acts.T)
log("eval", "topk_sae_50",
    max_feature_cos_sim=dec_vs_features.abs().max(dim=1).values.mean().item(),
    max_token_cos_sim=dec_vs_tokens.abs().max(dim=1).values.mean().item())

# %% Train L1 SAE with 50 latents (original notebook configuration)

log("info", "training l1 sae (original notebook config)", d_sae=NUM_FEATURES,
    l1_coefficient=1.0, l1_warm_up_steps=5000)

l1_cfg_50 = StandardTrainingSAEConfig(
    l1_coefficient=1.0,
    l1_warm_up_steps=5_000,
    d_in=HIDDEN_DIM,
    d_sae=NUM_FEATURES,
)
l1_sae_50 = StandardTrainingSAE(l1_cfg_50)

data_iter = DataIterator(generate_batch, topk_training_cfg.batch_size)
from sae_lens import SAETrainer
from sae_lens.config import LoggingConfig, SAETrainerConfig

l1_trainer_cfg = SAETrainerConfig(
    n_checkpoints=0,
    checkpoint_path=None,
    save_final_checkpoint=False,
    total_training_samples=topk_training_cfg.total_training_samples,
    device=str(DEFAULT_DEVICE),
    autocast=False,
    lr=topk_training_cfg.lr,
    lr_end=topk_training_cfg.lr,
    lr_scheduler_name="constant",
    lr_warm_up_steps=0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_decay_steps=0,
    n_restart_cycles=1,
    train_batch_size_samples=topk_training_cfg.batch_size,
    dead_feature_window=1000,
    feature_sampling_window=2000,
    logger=LoggingConfig(log_to_wandb=False),
)
trainer = SAETrainer(cfg=l1_trainer_cfg, sae=l1_sae_50, data_provider=data_iter)
l1_sae_50 = trainer.fit()

# %% Evaluate L1 SAE (50 latents) — decoder vs true features

plot_sae_feature_cos_sims(
    l1_sae_50.W_enc.data, l1_sae_50.W_dec.data, feature_directions,
    "L1 50-latent SAE (original notebook)",
    save_path=f"{RESULTS_DIR}/l1_50_latent_enc_dec_vs_features",
)

# %% Evaluate L1 SAE (50 latents) — decoder vs token embeddings

plot_decoder_vs_token_embeddings(
    l1_sae_50.W_dec.data, token_hidden_acts,
    "cos sim between SAE decoder and token embeddings (L1 50-latent SAE)",
    save_path=f"{RESULTS_DIR}/l1_50_latent_dec_vs_token_embeddings",
)

# %% Summary stats for L1 50-latent SAE

dec_vs_features = cos_sims(l1_sae_50.W_dec.T, feature_directions.T)
dec_vs_tokens = cos_sims(l1_sae_50.W_dec.T, token_hidden_acts.T)
log("eval", "l1_sae_50",
    max_feature_cos_sim=dec_vs_features.abs().max(dim=1).values.mean().item(),
    max_token_cos_sim=dec_vs_tokens.abs().max(dim=1).values.mean().item())

# %% Train TopK SAE with 25 latents (same as number of tokens)

log("info", "training topk sae", d_sae=NUM_TOKENS, top_k=TOPK_K)

topk_sae_25 = create_sae(d_in=HIDDEN_DIM, d_sae=NUM_TOKENS, k=TOPK_K)
topk_training_cfg_25 = TrainingConfig(k=TOPK_K, d_sae=NUM_TOKENS)
topk_sae_25 = train_sae(topk_sae_25, generate_batch, topk_training_cfg_25)

# %% Evaluate TopK SAE (25 latents) — decoder vs true features

plot_sae_feature_cos_sims(
    topk_sae_25.W_enc.data, topk_sae_25.W_dec.data, feature_directions,
    "TopK 25-latent SAE",
    save_path=f"{RESULTS_DIR}/topk_k{TOPK_K}_25_latent_enc_dec_vs_features",
)

# %% Evaluate TopK SAE (25 latents) — decoder vs token embeddings

plot_decoder_vs_token_embeddings(
    topk_sae_25.W_dec.data, token_hidden_acts,
    "cos sim between SAE decoder and token embeddings (TopK 25-latent SAE)",
    save_path=f"{RESULTS_DIR}/topk_k{TOPK_K}_25_latent_dec_vs_token_embeddings",
)

# %% Train L1 SAE with 25 latents (original notebook configuration)

log("info", "training l1 sae (original notebook config)", d_sae=NUM_TOKENS,
    l1_coefficient=1.0, l1_warm_up_steps=5000)

l1_cfg_25 = StandardTrainingSAEConfig(
    l1_coefficient=1.0,
    l1_warm_up_steps=5_000,
    d_in=HIDDEN_DIM,
    d_sae=NUM_TOKENS,
)
l1_sae_25 = StandardTrainingSAE(l1_cfg_25)

data_iter_25 = DataIterator(generate_batch, topk_training_cfg.batch_size)
trainer_25 = SAETrainer(cfg=l1_trainer_cfg, sae=l1_sae_25, data_provider=data_iter_25)
l1_sae_25 = trainer_25.fit()

# %% Evaluate L1 SAE (25 latents) — decoder vs true features

plot_sae_feature_cos_sims(
    l1_sae_25.W_enc.data, l1_sae_25.W_dec.data, feature_directions,
    "L1 25-latent SAE (original notebook)",
    save_path=f"{RESULTS_DIR}/l1_25_latent_enc_dec_vs_features",
)

# %% Evaluate L1 SAE (25 latents) — decoder vs token embeddings

plot_decoder_vs_token_embeddings(
    l1_sae_25.W_dec.data, token_hidden_acts,
    "cos sim between SAE decoder and token embeddings (L1 25-latent SAE)",
    save_path=f"{RESULTS_DIR}/l1_25_latent_dec_vs_token_embeddings",
)

# %% Final summary comparison

log("summary", "all results")

for name, sae in [
    ("topk_sae_50", topk_sae_50),
    ("l1_sae_50", l1_sae_50),
    ("topk_sae_25", topk_sae_25),
    ("l1_sae_25", l1_sae_25),
]:
    dec_feat = cos_sims(sae.W_dec.T, feature_directions.T)
    dec_tok = cos_sims(sae.W_dec.T, token_hidden_acts.T)
    log("result", name,
        mean_max_feature_cos_sim=dec_feat.abs().max(dim=1).values.mean().item(),
        mean_max_token_cos_sim=dec_tok.abs().max(dim=1).values.mean().item())

# %% Conclusion
# Expected: all SAEs learn the token embeddings (high token cos sim) rather than
# the true features (low feature cos sim). This holds for both TopK and L1
# sparsity mechanisms, confirming the original notebook finding.
