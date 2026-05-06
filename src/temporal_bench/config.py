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
    # Coupled-features mode (Aniket Level 3 / Dmitry exp1c3). When
    # n_hidden > 0 and n_parents > 0, K=n_hidden hidden Markov chains drive
    # M=n_features emission features through a binary coupling matrix where
    # each emission has exactly n_parents parent hidden chains. The
    # observation x is built from the M emission features; the metrics also
    # report decoder-cosine AUC against the K aggregated "hidden" feature
    # directions (gAUC / global recovery).
    n_hidden: int = 0  # K. 0 disables coupled mode.
    n_parents: int = 0  # parents per emission. 0 disables coupled mode.


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
