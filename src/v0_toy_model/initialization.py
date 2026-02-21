"""v0 initialization -- backward-compatible wrapper."""

from sae_lens import BatchTopKTrainingSAE

from src.shared.initialization import init_sae_to_features
from src.v0_toy_model.toy_model import ToyModel

__all__ = ["init_sae_to_match_model"]


def init_sae_to_match_model(
    sae: BatchTopKTrainingSAE,
    toy_model: ToyModel,
    noise_level: float = 0.0,
) -> None:
    """Backward-compatible: delegates to shared init_sae_to_features."""
    init_sae_to_features(sae, toy_model.feature_directions, noise_level)
