"""
config.py — Central configuration for NLP temporal crosscoder sweep over Gemma 2 2B.

All hyperparameters, sweep grids, wandb settings, and paths live here.
Import this everywhere; never hardcode constants elsewhere.
"""

import os
import torch

# ─── Experiment identity ────────────────────────────────────────────────────────
PROJECT_NAME = "temporal-crosscoders-nlp-gemma2-it"
EXPERIMENT_TAG = "nlp-v1"

# ─── Wandb ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT = PROJECT_NAME
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_TAGS: list[str] = [EXPERIMENT_TAG, "nlp-sweep", "gemma2-2b"]

# ─── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-2b-it"
D_MODEL = 2304           # Gemma 2 2B hidden dimension
NUM_LAYERS = 26           # total transformer layers (0-indexed: 0..25)

# ─── Layer extraction points ────────────────────────────────────────────────────
# component: "resid" for residual stream (output of the full layer),
#            "attn"  for attention output (before MLP, after o_proj)
LAYER_SPECS: dict[str, dict] = {
    "mid_res": {
        "layer": NUM_LAYERS // 2,     # layer 13
        "component": "resid",
        "hook_name": "blocks.{layer}.hook_resid_post",
        "d_act": D_MODEL,
        "label": f"resid_L{NUM_LAYERS // 2}",
    },
    "final_res": {
        "layer": NUM_LAYERS - 1,      # layer 25
        "component": "resid",
        "hook_name": "blocks.{layer}.hook_resid_post",
        "d_act": D_MODEL,
        "label": f"resid_L{NUM_LAYERS - 1}",
    },
    "mid_attn": {
        "layer": NUM_LAYERS // 2,     # layer 13
        "component": "attn",
        "hook_name": "blocks.{layer}.attn.hook_z",
        "d_act": D_MODEL,
        "label": f"attn_L{NUM_LAYERS // 2}",
    },
    "final_attn": {
        "layer": NUM_LAYERS - 1,      # layer 25
        "component": "attn",
        "hook_name": "blocks.{layer}.attn.hook_z",
        "d_act": D_MODEL,
        "label": f"attn_L{NUM_LAYERS - 1}",
    },
}

# ─── Dictionary sizing ─────────────────────────────────────────────────────────
EXPANSION_FACTOR = 8
D_SAE = D_MODEL * EXPANSION_FACTOR  # 2304 * 8 = 18432

# ─── Sweep grid ────────────────────────────────────────────────────────────────
SWEEP_LAYERS: list[str] = list(LAYER_SPECS.keys())
SWEEP_K: list[int] = [50, 100]
SWEEP_T: list[int] = [2, 5][::-1]
SWEEP_ARCHITECTURES: list[str] = ["stacked_sae", "txcdr"]

# ─── Training ──────────────────────────────────────────────────────────────────
TRAIN_STEPS = 10_000
LOG_INTERVAL = 200
BATCH_SIZE = 1024             # default; overridden per-run by batch_size_for_T()
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
SEED = 42


def batch_size_for_T(T: int) -> int:
    """Scale batch size based on T to stay within 48GB GPU memory.

    Model params scale linearly with T (one SAE per position).
    T=5: ~5GB model+optim, plenty of room → B=1024
    T=10: ~10GB model+optim → B=512
    T=25: ~25GB model+optim → B=64
    """
    return 256

# ─── Activation caching ───────────────────────────────────────────────────────
NUM_CHAINS = 24_000           # number of sequences to cache
SEQ_LENGTH = 32              # tokens per sequence
CACHE_BATCH_SIZE = 512         # batch size for model forward pass during caching
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_SUBSET = "sample-10BT"
DATASET_SPLIT = "train"

# ─── Paths ─────────────────────────────────────────────────────────────────────
NLP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("NLP_CACHE_DIR", os.path.join(NLP_DIR, "cached_activations"))
LOG_DIR = os.environ.get("NLP_LOG_DIR", os.path.join(NLP_DIR, "logs"))
VIZ_DIR = os.environ.get("NLP_VIZ_DIR", os.path.join(NLP_DIR, "viz_outputs"))
CHECKPOINT_DIR = os.environ.get("NLP_CKPT_DIR", os.path.join(NLP_DIR, "checkpoints"))

# ─── Autointerp ───────────────────────────────────────────────────────────────
AUTOINTERP_MODEL = "claude-haiku-4-5-20251001"
AUTOINTERP_MAX_EXAMPLES = 20     # max activating examples per feature
AUTOINTERP_TOP_FEATURES = 50     # number of top features to interpret
AUTOINTERP_BATCH_SIZE = 10       # features per API batch


def make_wandb_config(
    model_type: str, layer: str, k: int, T: int,
) -> dict:
    """Return a flat dict suitable for wandb.init(config=...)."""
    spec = LAYER_SPECS[layer]
    return dict(
        model_type=model_type,
        layer=layer,
        layer_index=spec["layer"],
        component=spec["component"],
        d_act=spec["d_act"],
        k=k,
        T=T,
        d_sae=D_SAE,
        expansion_factor=EXPANSION_FACTOR,
        d_model=D_MODEL,
        lm_model=MODEL_NAME,
        train_steps=TRAIN_STEPS,
        log_interval=LOG_INTERVAL,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        seed=SEED,
        num_chains=NUM_CHAINS,
        seq_length=SEQ_LENGTH,
        experiment_tag=EXPERIMENT_TAG,
    )


def run_name(model_type: str, layer: str, k: int, T: int) -> str:
    """Canonical run name for logging and checkpoints."""
    return f"{model_type}__{layer}__k{k}__T{T}"
