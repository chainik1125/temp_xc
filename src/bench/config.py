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
        delta: leak parameter for leaky reset (0 = standard, >0 = stickier)

    Transition probabilities (standard reset, delta=0):
        beta  = pi * (1 - rho)        (off -> on)
        alpha = rho * (1 - pi) + pi   (on -> off complement)

    With leaky reset (delta>0):
        rho_eff = 1 - (1-rho) * (1-delta)
    """

    pi: float = 0.05
    rho: float = 0.0
    delta: float = 0.0  # leak parameter (0 = standard reset)

    @property
    def lam(self) -> float:
        """Mixing rate (1 - rho)."""
        return 1.0 - self.rho

    @property
    def rho_eff(self) -> float:
        """Effective autocorrelation accounting for leak."""
        return 1.0 - self.lam * (1.0 - self.delta)

    @property
    def alpha(self) -> float:
        return self.rho_eff * (1.0 - self.pi) + self.pi

    @property
    def beta(self) -> float:
        return self.pi * (1.0 - self.rho_eff)


@dataclass
class CouplingConfig:
    """Configuration for coupled features (many-to-many hidden->emission).

    K hidden states produce M > K emission features through a binary
    coupling matrix. Each emission has exactly n_parents parent hidden states.

    When set on DataConfig, switches to coupled-feature data generation.
    """

    K_hidden: int = 10
    M_emission: int = 20
    n_parents: int = 2
    emission_mode: str = "or"  # "or" (deterministic) or "sigmoid"
    sigmoid_alpha: float = 5.0
    sigmoid_beta: float = -2.0


@dataclass
class DataConfig:
    """Complete specification of the synthetic data setup.

    When `coupling` is None (default), uses simple independent-feature
    Markov chains with num_features features. When `coupling` is set,
    uses K hidden states mapped to M emission features via a coupling matrix.
    """

    toy_model: ToyModelConfig = field(default_factory=ToyModelConfig)
    markov: MarkovConfig = field(default_factory=MarkovConfig)
    coupling: CouplingConfig | None = None
    seq_len: int = 64
    d_sae: int = 128  # dictionary width (often = num_features or M_emission)
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
