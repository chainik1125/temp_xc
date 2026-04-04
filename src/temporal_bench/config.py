"""Configuration dataclasses for data, training, and experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Synthetic data generation parameters."""

    n_features: int = 50
    d_model: int = 100
    pi: float = 0.1  # marginal firing probability
    magnitude_mean: float = 1.0
    magnitude_std: float = 0.15
    seed: int = 42


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    n_steps: int = 30_000
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float | None = None
    grad_clip: float = 1.0
    normalize_decoder_every: int = 1
    log_every: int = 0
    eval_every: int = 5_000
    optimizer: str = "adam"
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 0
    lr_schedule: str = "constant"
    grouped_weight_decay: bool = False
    seed: int = 42


@dataclass
class SweepConfig:
    """Experiment sweep parameters."""

    models: list[str] = field(default_factory=lambda: ["sae", "txcdr", "per_feature_temporal"])
    rho_values: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.9])
    k_values: list[int] = field(default_factory=lambda: [2, 5, 10])
    T_values: list[int] = field(default_factory=lambda: [2, 5])
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    n_seeds: int = 1
    output_dir: str = "results"
