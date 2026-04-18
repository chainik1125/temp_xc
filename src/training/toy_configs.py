"""Shared configuration dataclasses."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for SAE training."""

    k: float = 11.0
    d_sae: int | None = None  # Defaults to num_features if None
    lr: float = 3e-4
    total_training_samples: int = 15_000_000
    batch_size: int = 1024
    dead_feature_window: int = 1000
    feature_sampling_window: int = 2000
    seed: int = 42
