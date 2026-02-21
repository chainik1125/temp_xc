"""v0 metrics -- backward-compatible wrappers around shared metrics."""

import torch
from sae_lens import BatchTopKTrainingSAE

from src.shared.metrics import (
    decoder_feature_cosine_similarity as _decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
    match_sae_latents_to_features,
    variance_explained,
)
from src.v0_toy_model.toy_model import ToyModel

__all__ = [
    "decoder_feature_cosine_similarity",
    "decoder_pairwise_cosine_similarity",
    "match_sae_latents_to_features",
    "variance_explained",
]


def decoder_feature_cosine_similarity(
    sae: BatchTopKTrainingSAE,
    toy_model_or_directions: ToyModel | torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible wrapper: accepts ToyModel or raw feature_directions tensor."""
    if isinstance(toy_model_or_directions, ToyModel):
        feature_directions = toy_model_or_directions.feature_directions
    else:
        feature_directions = toy_model_or_directions
    return _decoder_feature_cosine_similarity(sae, feature_directions)
