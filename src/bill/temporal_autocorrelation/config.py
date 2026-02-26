from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    # Model and SAE
    model_name: str = "gpt2-small"
    sae_release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.8.hook_resid_pre"
    hook_point: str = "blocks.8.hook_resid_pre"

    # Data
    dataset_name: str = "openwebtext"
    num_sequences: int = 4096
    seq_length: int = 128
    batch_size: int = 16

    # Autocorrelation
    max_lag: int = 10
    min_activations_for_autocorr: int = 2

    # Output
    output_dir: Path = Path("results")

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
