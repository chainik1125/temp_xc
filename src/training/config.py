"""Configuration dataclasses for the benchmarking framework.

All experiment parameters are defined here as composable dataclasses.
Import these everywhere; never hardcode constants elsewhere.
"""

from dataclasses import dataclass, field

from src.data.toy.configs import CouplingConfig


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
class DataConfig:
    """Complete specification of a benchmarking data source.

    Two top-level data sources, selected by `dataset_type`:

    - `markov`  (default): synthetic Markov chain toy model, as originally
                designed. Parametrized by `toy_model`, `markov`, `coupling`.
                Used for controlled architecture comparison on known ground
                truth features.

    - `cached_activations`: load pre-cached real-LM activations produced by
                `src/bench/cache_activations.py`. Specify the
                `model_name` (routes through `src.data.nlp.models`), the
                `cached_dataset` (e.g. "fineweb", "gsm8k", "coding"), and the
                `cached_layer_key` (e.g. "resid_L12"). Used for the sprint's
                real-LM comparison track.

    `shuffle_within_sequence` is the critical temporal-control knob: when
    True, the sequence axis of every emitted activation window is randomly
    permuted. A temporal architecture's advantage over a standard SAE should
    *disappear* under shuffling if the advantage is genuinely temporal; if it
    survives, it's exploiting some non-temporal structure (the TFA "free
    dense channel" confound).
    """

    # Synthetic-toy fields (used when dataset_type == "markov") -------------
    toy_model: ToyModelConfig = field(default_factory=ToyModelConfig)
    markov: MarkovConfig = field(default_factory=MarkovConfig)
    coupling: CouplingConfig | None = None
    seq_len: int = 64
    d_sae: int = 128
    seed: int = 42
    eval_n_seq: int = 2000

    # Real-LM fields (used when dataset_type == "cached_activations") -------
    dataset_type: str = "markov"          # "markov" | "cached_activations"
    model_name: str = "deepseek-r1-distill-llama-8b"
    cached_dataset: str = "fineweb"       # subdir under data/cached_activations/<model>/
    cached_layer_key: str = "resid_L12"   # which <key>.npy to load
    cached_root: str | None = None        # override default path; None = use config default

    # Common ----------------------------------------------------------------
    shuffle_within_sequence: bool = False  # temporal shuffled control


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
