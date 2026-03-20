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


def compute_scaling_factor(
    model: ToyModel,
    config: DataConfig,
    device: torch.device,
    n_samples: int = 10000,
) -> float:
    """Compute scaling factor: sqrt(hidden_dim) / mean(||hidden||)."""
    with torch.no_grad():
        pi_t = torch.tensor(config.pi)
        rho_t = torch.tensor(config.rho)
        acts, _ = generate_markov_activations(
            n_samples, config.seq_len, pi_t, rho_t, device=device,
        )
        hidden = model(acts)
        mean_norm = hidden.reshape(-1, config.hidden_dim).norm(dim=-1).mean().item()
        return math.sqrt(config.hidden_dim) / mean_norm


def _make_flat_gen(
    model: ToyModel, pi_t: torch.Tensor, rho_t: torch.Tensor,
    device: torch.device, sf: float, hidden_dim: int, seq_len: int,
) -> Callable[[int], torch.Tensor]:
    def gen(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // seq_len)
        acts, _ = generate_markov_activations(n_seq, seq_len, pi_t, rho_t, device=device)
        return (model(acts) * sf).reshape(-1, hidden_dim)[:batch_size]
    return gen


def _make_seq_gen(
    model: ToyModel, pi_t: torch.Tensor, rho_t: torch.Tensor,
    device: torch.device, sf: float, seq_len: int, shuffle: bool = False,
) -> Callable[[int], torch.Tensor]:
    def gen(n_seq: int) -> torch.Tensor:
        acts, _ = generate_markov_activations(n_seq, seq_len, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                acts[i] = acts[i, torch.randperm(seq_len, device=device)]
        return model(acts) * sf
    return gen


def _make_window_gen(
    model: ToyModel, pi_t: torch.Tensor, rho_t: torch.Tensor,
    device: torch.device, sf: float, seq_len: int, T: int,
) -> Callable[[int], torch.Tensor]:
    def gen(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // (seq_len - T + 1)) + 1
        acts, _ = generate_markov_activations(n_seq, seq_len, pi_t, rho_t, device=device)
        hidden = model(acts) * sf
        windows = []
        for t in range(seq_len - T + 1):
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
    pi_t = torch.tensor(config.pi)
    rho_t = torch.tensor(config.rho)

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
    gen_flat = _make_flat_gen(model, pi_t, rho_t, device, sf,
                              config.hidden_dim, config.seq_len)
    gen_seq = _make_seq_gen(model, pi_t, rho_t, device, sf, config.seq_len,
                             shuffle=False)
    gen_seq_shuffled = _make_seq_gen(model, pi_t, rho_t, device, sf,
                                      config.seq_len, shuffle=True)

    gen_windows = {}
    for T in (window_sizes or []):
        gen_windows[T] = _make_window_gen(model, pi_t, rho_t, device, sf,
                                           config.seq_len, T)

    # Build eval data
    set_seed(config.seed + 100)
    acts_eval, _ = generate_markov_activations(
        config.eval_n_seq, config.seq_len, pi_t, rho_t, device=device,
    )
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
