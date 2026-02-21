"""v0-specific configuration dataclasses."""

from dataclasses import dataclass, field

from src.shared.configs import TrainingConfig

__all__ = ["ToyModelConfig", "DataConfig", "TrainingConfig", "ExperimentConfig"]


@dataclass
class ToyModelConfig:
    """Configuration for the toy model."""

    num_features: int
    hidden_dim: int
    target_cos_sim: float = 0.0
    ortho_lr: float = 0.01
    ortho_num_steps: int = 1000


@dataclass
class DataConfig:
    """Configuration for data generation."""

    num_features: int
    firing_probabilities: list[float] = field(default_factory=list)
    mean_magnitudes: list[float] | None = None
    std_magnitudes: list[float] | None = None
    correlations: dict[tuple[int, int], float] | None = None
    random_correlation: bool = False
    positive_ratio: float = 0.5
    correlation_strength_range: tuple[float, float] = (0.3, 0.8)
    correlation_sparsity: float = 0.3
    correlation_seed: int | None = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    toy_model: ToyModelConfig
    data: DataConfig
    training: TrainingConfig
    name: str = "experiment"
    results_dir: str = "src/v0_toy_model/results"
