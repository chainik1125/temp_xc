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
    # Stochastic HMM emissions (report §2.1.1). Defaults correspond to the
    # deterministic case s = h used in the core rho x k x T sweep.
    p_A: float = 0.0  # P(s=1 | h=0) — false-positive emission rate
    p_B: float = 1.0  # P(s=1 | h=1) — true-positive emission rate
    # Optional heterogeneous lag-1 autocorrelation per feature. If provided
    # (non-empty list), it overrides any scalar rho passed into data methods
    # and disables the per-rho cache. Length must equal n_features.
    rho_per_feature: list[float] = field(default_factory=list)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    n_steps: int = 30_000
    batch_size: int = 64
    lr: float = 3e-4
    grad_clip: float = 1.0
    eval_every: int = 5_000
    seed: int = 42


@dataclass
class SweepConfig:
    """Experiment sweep parameters."""

    models: list[str] = field(default_factory=lambda: ["regular_sae", "stacked_sae", "txcdr", "regular_sae_kT"])
    rho_values: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.9])
    k_values: list[int] = field(default_factory=lambda: [2, 5, 10])
    T_values: list[int] = field(default_factory=lambda: [2, 5])
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    n_seeds: int = 1
    output_dir: str = "results"
