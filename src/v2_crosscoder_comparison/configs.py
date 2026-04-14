"""Configuration dataclasses for the crosscoder comparison experiment."""

from dataclasses import dataclass, field


@dataclass
class ToyModelConfig:
    """Configuration for the two-position toy model."""

    num_features: int = 50
    hidden_dim: int = 100
    target_cos_sim: float = 0.0


@dataclass
class DataConfig:
    """Configuration for data generation."""

    num_features: int = 50
    firing_prob: float = 0.22
    rho: float = 0.0
    n_positions: int = 2
    mean_magnitude: float = 1.0
    std_magnitude: float = 0.15


@dataclass
class ArchitectureConfig:
    """Configuration for an SAE architecture."""

    arch_type: str = "naive_sae"  # naive_sae, stacked_sae, crosscoder
    d_sae: int = 100
    top_k: int = 11
    l1_coefficient: float = 0.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    toy_model: ToyModelConfig = field(default_factory=ToyModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    # Training params
    training_steps: int = 15_000
    batch_size: int = 4096
    lr: float = 3e-4
    seed: int = 42

    # Output dirs
    results_dir: str = "results"
    plots_dir: str = "plots"

    @property
    def total_training_samples(self) -> int:
        return self.training_steps * self.batch_size
