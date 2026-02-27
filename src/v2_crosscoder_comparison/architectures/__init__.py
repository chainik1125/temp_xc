"""Architecture factory for crosscoder comparison experiment."""

from src.v2_crosscoder_comparison.architectures.base import BaseArchitecture
from src.v2_crosscoder_comparison.architectures.crosscoder import Crosscoder
from src.v2_crosscoder_comparison.architectures.naive_sae import NaiveSAE
from src.v2_crosscoder_comparison.architectures.stacked_sae import StackedSAE
from src.v2_crosscoder_comparison.configs import ArchitectureConfig


def create_architecture(
    config: ArchitectureConfig,
    d_model: int,
) -> BaseArchitecture:
    """Factory to create an architecture from config.

    Args:
        config: Architecture configuration.
        d_model: Model hidden dimension.

    Returns:
        Architecture instance.
    """
    if config.arch_type == "naive_sae":
        return NaiveSAE(config, d_model)
    elif config.arch_type == "stacked_sae":
        return StackedSAE(config, d_model)
    elif config.arch_type == "crosscoder":
        return Crosscoder(config, d_model)
    else:
        raise ValueError(f"Unknown architecture type: {config.arch_type}")
