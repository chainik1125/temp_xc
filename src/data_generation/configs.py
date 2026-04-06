"""Configuration dataclasses for the data generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class TransitionConfig:
    """Configuration for the 2x2 Markov chain transition matrix.

    Can be constructed either from a raw matrix or via the reset process
    parameterization (lam, p).
    """

    matrix: torch.Tensor = field(default_factory=lambda: torch.eye(2))
    stationary_on_prob: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, torch.Tensor):
            self.matrix = torch.tensor(self.matrix, dtype=torch.float32)
        # Deferred import to avoid circular dependency: configs <- transition <- configs
        from src.data_generation.transition import validate_transition_matrix

        validate_transition_matrix(self.matrix)

    @classmethod
    def from_reset_process(cls, lam: float, p: float) -> TransitionConfig:
        """Construct from the reset process parameterization.

        Args:
            lam: Mixing parameter in [0, 1]. 0 = perfect memory, 1 = i.i.d.
            p: Stationary firing probability.
        """
        # Deferred import to avoid circular dependency: configs <- transition <- configs
        from src.data_generation.transition import build_transition_matrix

        matrix = build_transition_matrix(lam, p)
        return cls(matrix=matrix, stationary_on_prob=p)

    @classmethod
    def from_leaky_reset(
        cls, lam: float, p: float, delta: float
    ) -> TransitionConfig:
        """Construct from the leaky reset parameterization.

        Like the standard reset, but "reset" events are biased toward the
        current state by a leak parameter delta in [0,1].

        Args:
            lam: Mixing parameter in [0, 1].
            p: Stationary firing probability.
            delta: Leak parameter in [0, 1]. 0 = standard reset, 1 = absorbing.
        """
        from src.data_generation.transition import build_leaky_transition_matrix

        matrix = build_leaky_transition_matrix(lam, p, delta)
        return cls(matrix=matrix, stationary_on_prob=p)


@dataclass
class EmissionConfig:
    """HMM emission probabilities for the two hidden states.

    Controls stochastic emissions from the hidden Markov chain:
        s_t | z_t = A ~ Bernoulli(p_A)
        s_t | z_t = B ~ Bernoulli(p_B)

    With defaults (p_A=0, p_B=1), the observation equals the hidden state
    (deterministic emission), recovering the original MC behavior.
    """

    p_A: float = 0.0
    p_B: float = 1.0


@dataclass
class MagnitudeConfig:
    """Configuration for magnitude sampling."""

    distribution: str = "half_normal"
    mu: float = 0.0
    sigma: float = 1.0


@dataclass
class FeatureConfig:
    """Configuration for ground-truth feature directions."""

    k: int = 10
    d: int = 64
    orthogonal: bool = True
    target_cos_sim: float = 0.0


@dataclass
class SequenceConfig:
    """Configuration for sequence generation."""

    T: int = 128
    n_sequences: int = 1


@dataclass
class CouplingConfig:
    """Configuration for coupled features (many-to-many hidden->emission).

    K hidden states produce M > K emission features through a binary
    coupling matrix. Each emission has exactly n_parents parent hidden states.
    """

    K_hidden: int = 10
    M_emission: int = 20
    n_parents: int = 2
    emission_mode: str = "or"  # "or" (deterministic) or "sigmoid"
    sigmoid_alpha: float = 5.0  # sharpness (sigmoid mode only)
    sigmoid_beta: float = -2.0  # bias (sigmoid mode only)


@dataclass
class DataGenerationConfig:
    """Top-level configuration combining all pipeline parameters.

    For per-feature temporal persistence, set ``per_feature_pi`` and
    ``per_feature_rho`` (each a list of length k).  When these are set,
    the ``transition`` config is ignored for hidden-state generation
    and each feature gets its own Markov chain parameters.
    """

    transition: TransitionConfig = field(default_factory=TransitionConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    magnitude: MagnitudeConfig = field(default_factory=MagnitudeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    seed: int = 42
    per_feature_pi: list[float] | None = None
    per_feature_rho: list[float] | None = None


@dataclass
class CoupledDataGenerationConfig:
    """Configuration for coupled-feature data generation.

    Uses K hidden states mapped to M emission features via a coupling matrix.
    The transition config applies to each of the K hidden chains independently.
    """

    transition: TransitionConfig = field(default_factory=TransitionConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    magnitude: MagnitudeConfig = field(default_factory=MagnitudeConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    hidden_dim: int = 64  # d, dimensionality of observation space
    target_cos_sim: float = 0.0
    seed: int = 42
