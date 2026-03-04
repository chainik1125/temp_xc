"""
config.py — Central configuration for Temporal Crosscoder sweep experiments.

All hyperparameters, sweep grids, wandb settings, and paths live here.
Import this everywhere; never hardcode constants elsewhere.
"""

import os
import torch

# ─── Experiment identity ────────────────────────────────────────────────────────
PROJECT_NAME = "temporal-crosscoders-sweep"
EXPERIMENT_TAG = "v1"  # bump when changing setup

# ─── Wandb ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT = PROJECT_NAME
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)  # set via env or leave None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")   # "online" | "offline" | "disabled"
WANDB_TAGS = [EXPERIMENT_TAG, "sweep"]

# ─── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

# ─── Toy model geometry ─────────────────────────────────────────────────────────
NUM_FEATS = 100       # number of ground-truth features (also d_sae)
HIDDEN_DIM = 300      # representation dimension d
FEAT_PROB = 0.05      # Bernoulli activation probability
FEAT_MEAN = 1.0       # magnitude mean
FEAT_STD = 0.15       # magnitude std

# ─── Data generation schemes ────────────────────────────────────────────────────
# Only iid and markov (Scheme C) per user request
DATASETS = ["iid", "markov"]

# Markov chain parameters (Scheme C)
MARKOV_ALPHA = 0.95   # stay-on probability
MARKOV_BETA = 0.03    # turn-on probability

# ─── Training ───────────────────────────────────────────────────────────────────
TRAIN_STEPS = 200_000   # 1M steps for convergence
LOG_INTERVAL = 1_000      # log metrics every N steps
EVAL_BATCH = 2048         # larger batch for stable eval
BATCH_SIZE = 64           # training batch size (both SAE and TXCDR)
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
SEED = 42

# ─── Sweep grid ─────────────────────────────────────────────────────────────────
SWEEP_K = [1, 2, 4, 8]           # base active latents per token-position
SWEEP_T = [1, 4, 8, 10]       # window lengths

def should_skip(k: int, T: int) -> bool:
    """Skip configurations where k*T >= NUM_FEATS (underdetermined)."""
    return k * T >= NUM_FEATS

# ─── Model sizing ───────────────────────────────────────────────────────────────
D_SAE = NUM_FEATS     # SAE / crosscoder latent dimension = number of true features

def sae_effective_k(k: int) -> int:
    """Active latents for the SAE.  k = per-token budget."""
    return k

def txcdr_effective_k(k: int, T: int) -> int:
    """Active latents for the crosscoder.  k per position × T positions."""
    return k * T

# ─── Paths ──────────────────────────────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "logs")
VIZ_DIR = os.environ.get("VIZ_DIR", "viz_outputs")
CHECKPOINT_DIR = os.environ.get("CKPT_DIR", "checkpoints")

# ─── Convenience: full wandb config dict (logged at run start) ──────────────────
def make_wandb_config(model_type: str, dataset: str, k: int, T: int) -> dict:
    """Return a flat dict suitable for wandb.init(config=...)."""
    eff_k = (sae_effective_k(k) if model_type == "sae"
             else txcdr_effective_k(k, T))
    return dict(
        model_type=model_type,
        dataset=dataset,
        k=k,
        T=T,
        effective_k=eff_k,
        d_sae=D_SAE,
        num_feats=NUM_FEATS,
        hidden_dim=HIDDEN_DIM,
        feat_prob=FEAT_PROB,
        feat_mean=FEAT_MEAN,
        feat_std=FEAT_STD,
        train_steps=TRAIN_STEPS,
        log_interval=LOG_INTERVAL,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        seed=SEED,
        markov_alpha=MARKOV_ALPHA,
        markov_beta=MARKOV_BETA,
        experiment_tag=EXPERIMENT_TAG,
    )
