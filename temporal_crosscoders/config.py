"""
config.py — Central configuration for Temporal Crosscoder sweep experiments.

All hyperparameters, sweep grids, wandb settings, and paths live here.
Import this everywhere; never hardcode constants elsewhere.
"""

import os
import torch

# ─── Experiment identity ────────────────────────────────────────────────────────
PROJECT_NAME = "temporal-crosscoders-sweep-stacked"
EXPERIMENT_TAG = "v8"  # v8: stacked SAE baseline, correlation sweep

# ─── Wandb ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT = PROJECT_NAME
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_TAGS = [EXPERIMENT_TAG, "sweep"]

# ─── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

# ─── Toy model geometry ─────────────────────────────────────────────────────────
NUM_FEATS = 128        # number of ground-truth features (also d_sae)
HIDDEN_DIM = 256      # representation dimension d
FEAT_PROB = 0.05      # stationary activation probability
FEAT_MEAN = 1.0       # magnitude mean
FEAT_STD = 0.15       # magnitude std

# ─── Correlation parametrization ─────────────────────────────────────────────────
# We parametrize temporal structure by rho = lag-1 autocorrelation of the
# binary support process (2-state Markov chain).
#
#   rho = alpha - beta     (lag-1 autocorrelation)
#   p_stat = FEAT_PROB     (stationary firing probability)
#
# Given rho and p_stat, the Markov parameters are:
#   beta  = p_stat * (1 - rho)
#   alpha = rho + beta = rho * (1 - p_stat) + p_stat
#
# rho = 0 recovers IID (alpha = beta = p_stat).
# rho > 0 gives temporal persistence; higher = stickier features.

SWEEP_RHO = [0.0, 0.6, 0.9]  # correlation levels to sweep

# ─── HMM emission sweep ────────────────────────────────────────────────────────
# gamma = autocorrelation amplitude prefactor. Controlled by (p_A, p_B, q) where
# q = stationary probability of hidden state B (= FEAT_PROB for MC case).
# gamma = q(1-q)(p_B-p_A)^2 / [mu(1-mu)], mu = (1-q)*p_A + q*p_B.
# All configs maintain mu = FEAT_PROB.
SWEEP_EMISSION = [
    {"p_A": 0.0, "p_B": 1.0, "q": FEAT_PROB, "label": "MC"},        # gamma=1.0
    {"p_A": 0.0, "p_B": 0.5, "q": 2*FEAT_PROB, "label": "gamma=0.47"},
    {"p_A": 0.0, "p_B": 0.25, "q": 4*FEAT_PROB, "label": "gamma=0.21"},
]


def markov_params(rho: float, p_stat: float = FEAT_PROB) -> tuple[float, float]:
    """Return (alpha, beta) for a given lag-1 autocorrelation rho."""
    beta = p_stat * (1.0 - rho)
    alpha = rho * (1.0 - p_stat) + p_stat
    return alpha, beta


# ─── Training ───────────────────────────────────────────────────────────────────
TRAIN_STEPS = 65_000
LOG_INTERVAL = 1_500
EVAL_BATCH = 128
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
SEED = 42

# ─── Sweep grid ─────────────────────────────────────────────────────────────────
SWEEP_K = [2, 5, 10, 25][::-1]   # active latents per token-position
SWEEP_T = [2, 5][::-1]           # window lengths


def should_skip(k: int, T: int) -> bool:
    """Skip configurations where k*T > NUM_FEATS (underdetermined for stacked SAE)."""
    return k * T >= NUM_FEATS


# ─── Model sizing ───────────────────────────────────────────────────────────────
D_SAE = NUM_FEATS  # SAE / crosscoder latent dimension = number of true features

# ─── Data cache ─────────────────────────────────────────────────────────────────
NUM_CHAINS = 128
CHAIN_LENGTH = 1024
CACHE_REFRESH_INTERVAL = 50_000

# ─── Paths ──────────────────────────────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "logs")
VIZ_DIR = os.environ.get("VIZ_DIR", "viz_outputs_v8")
CHECKPOINT_DIR = os.environ.get("CKPT_DIR", "checkpoints")


# ─── Convenience: full wandb config dict (logged at run start) ──────────────────
def make_wandb_config(model_type: str, rho: float, k: int, T: int) -> dict:
    """Return a flat dict suitable for wandb.init(config=...)."""
    alpha, beta = markov_params(rho)
    return dict(
        model_type=model_type,
        rho=rho,
        k=k,
        T=T,
        d_sae=D_SAE,
        num_feats=NUM_FEATS,
        hidden_dim=HIDDEN_DIM,
        feat_prob=FEAT_PROB,
        feat_mean=FEAT_MEAN,
        feat_std=FEAT_STD,
        markov_alpha=alpha,
        markov_beta=beta,
        train_steps=TRAIN_STEPS,
        log_interval=LOG_INTERVAL,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        seed=SEED,
        experiment_tag=EXPERIMENT_TAG,
    )
