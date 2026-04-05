"""Configuration dataclasses for the benchmarking framework.

All experiment parameters are defined here as composable dataclasses.
Import these everywhere; never hardcode constants elsewhere.
"""

from dataclasses import dataclass, field


@dataclass
class ToyModelConfig:
    """Configuration for the synthetic toy model."""

    num_features: int = 128
    hidden_dim: int = 256
    target_cos_sim: float = 0.0


@dataclass
class MarkovConfig:
    """Temporal correlation parametrization.

    Each feature has an independent 2-state Markov chain controlling its
    binary support (on/off). Parametrized by:
        pi: stationary firing probability
        rho: lag-1 autocorrelation (0 = IID, 1 = deterministic)

    Transition probabilities:
        beta  = pi * (1 - rho)        (off -> on)
        alpha = rho * (1 - pi) + pi   (on -> off complement)
    """

    pi: float = 0.05
    rho: float = 0.0

    @property
    def alpha(self) -> float:
        return self.rho * (1.0 - self.pi) + self.pi

    @property
    def beta(self) -> float:
        return self.pi * (1.0 - self.rho)


@dataclass
class DataConfig:
    """Complete specification of the synthetic data setup."""

    toy_model: ToyModelConfig = field(default_factory=ToyModelConfig)
    markov: MarkovConfig = field(default_factory=MarkovConfig)
    seq_len: int = 64
    d_sae: int = 128  # dictionary width (often = num_features)
    seed: int = 42
    eval_n_seq: int = 2000


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    total_steps: int = 30_000
    batch_size: int = 2048
    lr: float = 3e-4
    grad_clip: float = 1.0
    log_every: int = 500
    seed: int = 42


@dataclass
class SweepConfig:
    """What to sweep over in an experiment."""

    k_values: list[int] = field(default_factory=lambda: [2, 5, 10, 25])
    rho_values: list[float] = field(default_factory=lambda: [0.0, 0.6, 0.9])
    T_values: list[int] = field(default_factory=lambda: [2, 5])
    seeds: list[int] = field(default_factory=lambda: [42])
