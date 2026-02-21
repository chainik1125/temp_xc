"""v1 temporal experiment configuration dataclasses."""

from dataclasses import dataclass, field

from src.shared.configs import TrainingConfig


@dataclass
class TemporalToyModelConfig:
    """Configuration for the temporal toy model."""

    num_global_features: int
    num_local_features: int
    hidden_dim: int
    seq_len: int = 2
    target_cos_sim: float = 0.0
    ortho_lr: float = 0.01
    ortho_num_steps: int = 1000

    @property
    def num_features(self) -> int:
        return self.num_global_features + self.num_local_features


@dataclass
class TemporalDataConfig:
    """Configuration for temporal data generation."""

    num_global_features: int
    num_local_features: int
    seq_len: int = 2
    global_firing_probabilities: list[float] = field(default_factory=list)
    local_firing_probabilities: list[float] = field(default_factory=list)
    mean_magnitudes: list[float] | None = None
    std_magnitudes: list[float] | None = None


@dataclass
class TemporalExperimentConfig:
    """Top-level temporal experiment configuration."""

    model: TemporalToyModelConfig
    data: TemporalDataConfig
    training: TrainingConfig
    name: str = "temporal_baseline"
    results_dir: str = "src/v1_temporal/results"
