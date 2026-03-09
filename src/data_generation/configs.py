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
class DataGenerationConfig:
    """Top-level configuration combining all pipeline parameters."""

    transition: TransitionConfig = field(default_factory=TransitionConfig)
    magnitude: MagnitudeConfig = field(default_factory=MagnitudeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    seed: int = 42
