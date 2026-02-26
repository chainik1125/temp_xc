import torch
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE

from bill.temporal_autocorrelation.config import ExperimentConfig


def load_model_and_sae(
    config: ExperimentConfig,
    device: str = "cpu",
) -> tuple[HookedTransformer, SAE]:
    """Load GPT-2 Small and the pretrained SAE."""
    model = HookedTransformer.from_pretrained(config.model_name, device=device)
    sae, _, _ = SAE.from_pretrained(
        release=config.sae_release,
        sae_id=config.sae_id,
        device=device,
    )
    return model, sae


def extract_sae_features_batch(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
    hook_point: str,
) -> np.ndarray:
    """Run a batch of tokens through the model and encode with the SAE.

    Returns post-ReLU SAE feature activations: non-negative and sparse.
    A feature is "active" at a position when its value is > 0.

    Args:
        model: HookedTransformer instance.
        sae: Pretrained SAE instance.
        tokens: Token IDs of shape [B, T].
        hook_point: Name of the hook to cache (e.g. "blocks.8.hook_resid_pre").

    Returns:
        SAE feature activations as numpy array of shape [B, T, D].
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_point,
        )
        residuals = cache[hook_point]  # [B, T, d_model]
        feature_acts = sae.encode(residuals)  # [B, T, D]

    result = feature_acts.cpu().numpy()

    # Free GPU memory
    del cache, residuals, feature_acts
    if tokens.device.type == "cuda":
        torch.cuda.empty_cache()

    return result
