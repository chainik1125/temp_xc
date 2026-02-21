"""v0 train_sae -- backward-compatible wrappers."""

from collections.abc import Callable

import torch
from sae_lens import BatchTopKTrainingSAE

from src.shared.configs import TrainingConfig
from src.shared.train_sae import DataIterator, create_sae, train_sae
from src.utils.device import DEFAULT_DEVICE

__all__ = ["DataIterator", "create_sae", "train_toy_sae", "train_sae"]


def train_toy_sae(
    sae: BatchTopKTrainingSAE,
    toy_model: torch.nn.Module,
    generate_batch_fn: Callable[[int], torch.Tensor],
    training_config: TrainingConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> BatchTopKTrainingSAE:
    """Backward-compatible wrapper: composes toy_model + generate_batch_fn into a single callable."""

    def generate_hidden_acts(batch_size: int) -> torch.Tensor:
        with torch.no_grad():
            feature_acts = generate_batch_fn(batch_size)
            return toy_model(feature_acts)

    return train_sae(sae, generate_hidden_acts, training_config, device)
