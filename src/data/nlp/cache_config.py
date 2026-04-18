"""NLP pipeline config — model-agnostic.

Everything model-specific flows from `src.data.nlp.models.get_model_config`.
This module holds only *training/caching/sweep* hyperparameters that are
independent of the subject LM.

Andre's original config (hardcoded Gemma 2 2B constants) is superseded by the
model registry. If you need Andre's old values, call
`get_model_config("gemma-2-2b-it")`.
"""

from __future__ import annotations

import os

import torch

from src.data.nlp.models import ModelConfig, get_model_config

# ─── Experiment identity ────────────────────────────────────────────────────────
PROJECT_NAME = "temporal-crosscoders-nlp"
EXPERIMENT_TAG = "sprint-v1"

# ─── Wandb ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT = PROJECT_NAME
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_TAGS: list[str] = [EXPERIMENT_TAG, "nlp-sweep"]

# ─── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Default model (overridable via --model CLI arg everywhere) ─────────────────
DEFAULT_MODEL = os.environ.get("TXC_DEFAULT_MODEL", "deepseek-r1-distill-llama-8b")

# ─── Dictionary sizing ─────────────────────────────────────────────────────────
EXPANSION_FACTOR = 8
# D_SAE is model-dependent — compute via d_sae_for(model_name) instead of a const.

def d_sae_for(model_name: str) -> int:
    return get_model_config(model_name).d_model * EXPANSION_FACTOR

# ─── Layer extraction points ────────────────────────────────────────────────────
# Component keys: "resid" (post-block residual) or "attn" (self-attn output).
# Layer indices are resolved from the model registry's default_layer_indices
# unless overridden on the command line.
LAYER_COMPONENTS: tuple[str, ...] = ("resid",)  # Add "attn" to cache attention out.


def build_layer_specs(
    model_name: str,
    layer_indices: tuple[int, ...] | None = None,
    components: tuple[str, ...] = LAYER_COMPONENTS,
) -> dict[str, dict]:
    """Build the runtime layer spec dict from a model registry entry.

    Returns a dict keyed by short layer label (e.g. "resid_L12") mapping to
    { layer, component, d_act, label }. Consumers should not hardcode keys.
    """
    cfg = get_model_config(model_name)
    idxs = layer_indices if layer_indices is not None else cfg.default_layer_indices
    specs: dict[str, dict] = {}
    for idx in idxs:
        for comp in components:
            key = f"{comp}_L{idx}"
            specs[key] = {
                "layer": int(idx),
                "component": comp,
                "d_act": cfg.d_model,
                "label": key,
                "family": cfg.architecture_family,
            }
    return specs

# ─── Sweep grid (architecture/hparams only — NOT model-specific) ───────────────
SWEEP_K: list[int] = [50, 100]
SWEEP_T: list[int] = [5, 2]
SWEEP_ARCHITECTURES: list[str] = ["topk_sae", "stacked_sae", "crosscoder", "tfa"]

# ─── Training ──────────────────────────────────────────────────────────────────
TRAIN_STEPS = 10_000
LOG_INTERVAL = 200
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
SEED = 42


def batch_size_for_T(T: int, model_name: str = DEFAULT_MODEL) -> int:
    """Scale batch size based on T and model width to stay under GPU memory.

    A rough upper bound: param memory ≈ T * 2 * d_model * d_sae * 4 bytes.
    For 8B at d_model=4096 with 8x expansion, one SAE per T costs ~1GB params +
    optim. Err on the side of small batches; tune later.
    """
    d_model = get_model_config(model_name).d_model
    if d_model >= 4096:
        return max(64, 256 // max(1, T // 2))
    return max(128, 512 // max(1, T // 2))


# ─── Activation caching defaults ───────────────────────────────────────────────
NUM_CHAINS = 24_000
SEQ_LENGTH = 32
CACHE_BATCH_SIZE = 128
DATASET_NAME_DEFAULT = "HuggingFaceFW/fineweb"
DATASET_SUBSET_DEFAULT = "sample-10BT"
DATASET_SPLIT_DEFAULT = "train"

# ─── Paths ─────────────────────────────────────────────────────────────────────
NLP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(NLP_DIR))
DATA_ROOT = os.environ.get("TXC_DATA_ROOT", os.path.join(REPO_DIR, "data"))

def cache_dir_for(model_name: str, dataset_name: str) -> str:
    return os.path.join(DATA_ROOT, "cached_activations", model_name, dataset_name)

LOG_DIR = os.environ.get("NLP_LOG_DIR", os.path.join(NLP_DIR, "logs"))
VIZ_DIR = os.environ.get("NLP_VIZ_DIR", os.path.join(NLP_DIR, "viz_outputs"))
CHECKPOINT_DIR = os.environ.get("NLP_CKPT_DIR", os.path.join(NLP_DIR, "checkpoints"))

# ─── Autointerp ───────────────────────────────────────────────────────────────
AUTOINTERP_API_MODEL = "claude-haiku-4-5-20251001"
AUTOINTERP_MAX_EXAMPLES = 20
AUTOINTERP_TOP_FEATURES = 50
AUTOINTERP_BATCH_SIZE = 10


def make_wandb_config(
    model_type: str,
    model_name: str,
    layer_key: str,
    layer_spec: dict,
    k: int,
    T: int,
) -> dict:
    cfg = get_model_config(model_name)
    return dict(
        arch=model_type,
        subject_model=model_name,
        subject_hf_path=cfg.hf_path,
        layer_key=layer_key,
        layer_index=layer_spec["layer"],
        component=layer_spec["component"],
        d_act=layer_spec["d_act"],
        d_model=cfg.d_model,
        k=k,
        T=T,
        d_sae=d_sae_for(model_name),
        expansion_factor=EXPANSION_FACTOR,
        train_steps=TRAIN_STEPS,
        log_interval=LOG_INTERVAL,
        lr=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        seed=SEED,
        num_chains=NUM_CHAINS,
        seq_length=SEQ_LENGTH,
        experiment_tag=EXPERIMENT_TAG,
    )


def run_name(model_type: str, model_name: str, layer_key: str, k: int, T: int) -> str:
    return f"{model_type}__{model_name}__{layer_key}__k{k}__T{T}"
