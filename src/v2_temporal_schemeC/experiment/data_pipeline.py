"""Unified data pipeline for temporal crosscoder experiments.

Provides DataConfig, DataPipeline, and build_data_pipeline() to replace
the copy-pasted compute_scaling_factor / make_*_gen functions.
"""

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations


@dataclass
class EventConfig:
    """Event-structured data generation (factorial HMM).

    When set on DataConfig, switches generators from independent Markov
    chains to event-grouped features. pi and rho on the parent DataConfig
    are ignored; event-level pi/rho are used instead.
    """
    n_events: int
    features_per_event: int
    pi_events: list[float]
    rho_events: list[float]
    membership: torch.Tensor | None = None  # (n_features, n_events); None = block


@dataclass
class DataConfig:
    """Complete specification of the synthetic data setup."""
    num_features: int
    hidden_dim: int
    seq_len: int
    pi: list[float]
    rho: list[float]
    dict_width: int
    seed: int = 42
    eval_n_seq: int = 2000
    event_config: EventConfig | None = None  # set to use event-structured data


@dataclass
class DataPipeline:
    """Pre-built data generators and eval data. Created once, shared across models."""
    toy_model: ToyModel
    scaling_factor: float
    true_features: torch.Tensor
    eval_hidden: torch.Tensor
    gen_flat: Callable[[int], torch.Tensor]
    gen_seq: Callable[[int], torch.Tensor]
    gen_seq_shuffled: Callable[[int], torch.Tensor]
    gen_windows: dict[int, Callable[[int], torch.Tensor]] = field(default_factory=dict)
    config: DataConfig = None


def _generate_raw_activations(
    n_seq: int, config: DataConfig, device: torch.device,
) -> torch.Tensor:
    """Generate raw feature activations using either independent Markov or event mode."""
    if config.event_config is not None:
        from src.v2_temporal_schemeC.factorial_hmm import (
            generate_event_activations,
            generate_event_activations_general,
        )
        ec = config.event_config
        pi_e = torch.tensor(ec.pi_events)
        rho_e = torch.tensor(ec.rho_events)
        if ec.membership is not None:
            acts, _, _ = generate_event_activations_general(
                n_seq, config.seq_len, pi_e, rho_e, ec.membership, device=device,
            )
        else:
            acts, _, _ = generate_event_activations(
                n_seq, config.seq_len, ec.n_events, ec.features_per_event,
                pi_e, rho_e, device=device,
            )
    else:
        pi_t = torch.tensor(config.pi)
        rho_t = torch.tensor(config.rho)
        acts, _ = generate_markov_activations(
            n_seq, config.seq_len, pi_t, rho_t, device=device,
        )
    return acts


def compute_scaling_factor(
    model: ToyModel,
    config: DataConfig,
    device: torch.device,
    n_samples: int = 10000,
) -> float:
    """Compute scaling factor: sqrt(hidden_dim) / mean(||hidden||)."""
    with torch.no_grad():
        acts = _generate_raw_activations(n_samples, config, device)
        hidden = model(acts)
        mean_norm = hidden.reshape(-1, config.hidden_dim).norm(dim=-1).mean().item()
        return math.sqrt(config.hidden_dim) / mean_norm


def _make_flat_gen(
    model: ToyModel, config: DataConfig,
    device: torch.device, sf: float,
) -> Callable[[int], torch.Tensor]:
    def gen(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // config.seq_len)
        acts = _generate_raw_activations(n_seq, config, device)
        return (model(acts) * sf).reshape(-1, config.hidden_dim)[:batch_size]
    return gen


def _make_seq_gen(
    model: ToyModel, config: DataConfig,
    device: torch.device, sf: float, shuffle: bool = False,
) -> Callable[[int], torch.Tensor]:
    def gen(n_seq: int) -> torch.Tensor:
        acts = _generate_raw_activations(n_seq, config, device)
        if shuffle:
            for i in range(n_seq):
                acts[i] = acts[i, torch.randperm(config.seq_len, device=device)]
        return model(acts) * sf
    return gen


def _make_window_gen(
    model: ToyModel, config: DataConfig,
    device: torch.device, sf: float, T: int,
) -> Callable[[int], torch.Tensor]:
    def gen(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // (config.seq_len - T + 1)) + 1
        acts = _generate_raw_activations(n_seq, config, device)
        hidden = model(acts) * sf
        windows = []
        for t in range(config.seq_len - T + 1):
            windows.append(hidden[:, t:t + T, :])
        all_w = torch.cat(windows, dim=0)
        idx = torch.randperm(all_w.shape[0], device=device)[:batch_size]
        return all_w[idx]
    return gen


def build_data_pipeline(
    config: DataConfig,
    device: torch.device,
    window_sizes: list[int] | None = None,
) -> DataPipeline:
    """Build all data generators and eval data from a DataConfig.

    Args:
        config: Data configuration.
        device: Torch device.
        window_sizes: List of T values for crosscoder window generators.

    Returns:
        Fully constructed DataPipeline.
    """
    # Build toy model (deterministic)
    set_seed(config.seed)
    model = ToyModel(
        num_features=config.num_features,
        hidden_dim=config.hidden_dim,
    ).to(device)
    model.eval()
    true_features = model.feature_directions

    # Compute scaling factor
    sf = compute_scaling_factor(model, config, device)

    # Build generators
    gen_flat = _make_flat_gen(model, config, device, sf)
    gen_seq = _make_seq_gen(model, config, device, sf, shuffle=False)
    gen_seq_shuffled = _make_seq_gen(model, config, device, sf, shuffle=True)

    gen_windows = {}
    for T in (window_sizes or []):
        gen_windows[T] = _make_window_gen(model, config, device, sf, T)

    # Build eval data
    set_seed(config.seed + 100)
    acts_eval = _generate_raw_activations(config.eval_n_seq, config, device)
    eval_hidden = model(acts_eval) * sf

    return DataPipeline(
        toy_model=model,
        scaling_factor=sf,
        true_features=true_features,
        eval_hidden=eval_hidden,
        gen_flat=gen_flat,
        gen_seq=gen_seq,
        gen_seq_shuffled=gen_seq_shuffled,
        gen_windows=gen_windows,
        config=config,
    )
