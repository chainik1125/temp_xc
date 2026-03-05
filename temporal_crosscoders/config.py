"""
config.py — Central configuration for Temporal Crosscoder sweep experiments.

All hyperparameters, sweep grids, wandb settings, and paths live here.
Import this everywhere; never hardcode constants elsewhere.
"""

import os
import torch

# ─── Experiment identity ────────────────────────────────────────────────────────
PROJECT_NAME = "temporal-crosscoders-sweep-final"
EXPERIMENT_TAG = "v7"  # bump when changing setup

# ─── Wandb ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT = PROJECT_NAME
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)  # set via env or leave None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")   # "online" | "offline" | "disabled"
WANDB_TAGS = [EXPERIMENT_TAG, "sweep"]

# ─── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

# ─── Toy model geometry ─────────────────────────────────────────────────────────
NUM_FEATS = 50      # number of ground-truth features (also d_sae)
HIDDEN_DIM = 100      # representation dimension d
FEAT_PROB = 0.05     # Bernoulli activation probability
FEAT_MEAN = 1.0       # magnitude mean
FEAT_STD = 0.15       # magnitude std

# ─── Data generation schemes ────────────────────────────────────────────────────
# Only iid and markov (Scheme C) per user request
DATASETS = ["markov", "iid"]
#DATASETS = ['markov']  # for quick testing; comment out to restore full sweep

# Markov chain parameters (Scheme C)
MARKOV_ALPHA = 0.95   # stay-on probability
MARKOV_BETA = 0.03    # turn-on probability

# ─── Training ───────────────────────────────────────────────────────────────────
TRAIN_STEPS = 80_000 # 1m steps for convergence
LOG_INTERVAL = 1_500      # log metrics every N steps
EVAL_BATCH = 128         # larger batch for stable eval
BATCH_SIZE = 1          # training batch size (both SAE and TXCDR)
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
SEED = 42

# ─── Sweep grid ─────────────────────────────────────────────────────────────────
SWEEP_K = [2, 5, 10, 25][::-1]  # base active latents per token-position
SWEEP_T = [2, 5][::-1]       # window lengths

def should_skip(k: int, T: int) -> bool:
    """Skip configurations where k*T >= NUM_FEATS (underdetermined)."""
    return txcdr_effective_k(k, T) > NUM_FEATS

# ─── Model sizing ───────────────────────────────────────────────────────────────
D_SAE = NUM_FEATS     # SAE / crosscoder latent dimension = number of true features

def sae_effective_k(k: int) -> int:
    """Active latents for the SAE.  k = per-token budget."""
    return k

def txcdr_effective_k(k: int, T: int) -> int:
    """Active latents for the crosscoder.  k per position × T positions."""
    return k # k*T

# ─── Data cache ─────────────────────────────────────────────────────────────────
# Pre-generate long chains; serve sliding windows from them.
# This ensures all T values see the same underlying process and the markov
# chain has time to express deep temporal structure.
NUM_CHAINS = 256           # number of independent long chains
CHAIN_LENGTH = 2048        # steps per chain (>> max T)
CACHE_REFRESH_INTERVAL = 50_000  # regenerate chains every N training steps (0 = never)

# ─── Paths ──────────────────────────────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "logs")
VIZ_DIR = os.environ.get("VIZ_DIR", "viz_outputs_kT")
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
