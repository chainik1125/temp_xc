"""Unified data pipeline for architecture comparison experiments.

Builds a toy model, data generators, and evaluation data from a DataConfig.
Adapted from Han's data_pipeline.py pattern, using the shared orthogonalize
and Markov chain utilities.

Usage:
    pipeline = build_data_pipeline(config, device, window_sizes=[2, 5])
    gen_flat = pipeline.gen_flat      # for TopKSAE
    gen_seq = pipeline.gen_seq        # for TFA
    gen_win = pipeline.gen_windows[2] # for Stacked/Crosscoder with T=2
"""

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from src.bench.config import DataConfig
from src.shared.orthogonalize import orthogonalize


class ToyModel(nn.Module):
    """Linear embedding toy model with orthogonal feature directions.

    Maps sparse feature activations (n_features,) to a dense hidden
    representation (hidden_dim,) via a fixed linear embedding.
    """

    def __init__(self, num_features: int, hidden_dim: int, target_cos_sim: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        embeddings = orthogonalize(
            num_vectors=num_features,
            vector_len=hidden_dim,
            target_cos_sim=target_cos_sim,
        )
        # Store as buffer (not parameter) — we don't train this
        self.register_buffer("W", embeddings)  # (num_features, hidden_dim)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """Map activations to hidden space.

        Args:
            activations: (..., num_features) sparse activation coefficients.

        Returns:
            (..., hidden_dim) dense hidden representations.
        """
        return activations @ self.W

    @property
    def feature_directions(self) -> torch.Tensor:
        """(hidden_dim, num_features) feature direction matrix."""
        return self.W.T


def _generate_markov_activations(
    n_seq: int,
    seq_len: int,
    num_features: int,
    pi: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate binary support via independent 2-state Markov chains + magnitudes.

    Returns activations of shape (n_seq, seq_len, num_features).
    """
    # Markov transition probabilities
    p01 = pi * (1.0 - rho)  # off -> on
    p10 = (1.0 - pi) * (1.0 - rho)  # on -> off

    # Initialize from stationary distribution
    support = torch.zeros(n_seq, seq_len, num_features, device=device)
    support[:, 0, :] = (torch.rand(n_seq, num_features, device=device) < pi).float()

    for t in range(1, seq_len):
        prev = support[:, t - 1, :]
        u = torch.rand(n_seq, num_features, device=device)
        turn_on = (prev == 0) & (u < p01)
        stay_on = (prev == 1) & (u >= p10)
        support[:, t, :] = (turn_on | stay_on).float()

    # Magnitudes: half-normal (|N(0,1)|)
    magnitudes = torch.randn(n_seq, seq_len, num_features, device=device).abs()
    return support * magnitudes


@dataclass
class DataPipeline:
    """Pre-built data generators and eval data. Created once, shared across models."""

    toy_model: ToyModel
    scaling_factor: float
    true_features: torch.Tensor  # (hidden_dim, num_features)
    eval_hidden: torch.Tensor  # (n_seq, seq_len, hidden_dim)
    gen_flat: Callable[[int], torch.Tensor]
    gen_seq: Callable[[int], torch.Tensor]
    gen_windows: dict[int, Callable[[int], torch.Tensor]] = field(
        default_factory=dict
    )


def _compute_scaling_factor(
    toy_model: ToyModel,
    config: DataConfig,
    device: torch.device,
    n_samples: int = 10_000,
) -> float:
    """Compute scaling factor: sqrt(hidden_dim) / mean(||hidden||)."""
    with torch.no_grad():
        acts = _generate_markov_activations(
            n_samples,
            config.seq_len,
            config.toy_model.num_features,
            config.markov.pi,
            config.markov.rho,
            device,
        )
        hidden = toy_model(acts)
        mean_norm = hidden.reshape(-1, config.toy_model.hidden_dim).norm(dim=-1).mean().item()
        return math.sqrt(config.toy_model.hidden_dim) / mean_norm


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
        Fully constructed DataPipeline ready for sweep use.
    """
    torch.manual_seed(config.seed)
    toy_model = ToyModel(
        num_features=config.toy_model.num_features,
        hidden_dim=config.toy_model.hidden_dim,
        target_cos_sim=config.toy_model.target_cos_sim,
    ).to(device)
    toy_model.eval()

    true_features = toy_model.feature_directions  # (hidden_dim, num_features)
    sf = _compute_scaling_factor(toy_model, config, device)

    num_f = config.toy_model.num_features
    hidden_dim = config.toy_model.hidden_dim
    seq_len = config.seq_len
    pi = config.markov.pi
    rho = config.markov.rho

    def gen_flat(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // seq_len)
        acts = _generate_markov_activations(n_seq, seq_len, num_f, pi, rho, device)
        return (toy_model(acts) * sf).reshape(-1, hidden_dim)[:batch_size]

    def gen_seq(n_seq: int) -> torch.Tensor:
        acts = _generate_markov_activations(n_seq, seq_len, num_f, pi, rho, device)
        return toy_model(acts) * sf

    gen_windows: dict[int, Callable[[int], torch.Tensor]] = {}
    for T in window_sizes or []:
        def _make_window_gen(T_: int) -> Callable[[int], torch.Tensor]:
            def gen(batch_size: int) -> torch.Tensor:
                n_seq = max(1, batch_size // (seq_len - T_ + 1)) + 1
                acts = _generate_markov_activations(
                    n_seq, seq_len, num_f, pi, rho, device
                )
                hidden = toy_model(acts) * sf
                windows = []
                for t in range(seq_len - T_ + 1):
                    windows.append(hidden[:, t : t + T_, :])
                all_w = torch.cat(windows, dim=0)
                idx = torch.randperm(all_w.shape[0], device=device)[:batch_size]
                return all_w[idx]
            return gen
        gen_windows[T] = _make_window_gen(T)

    # Build eval data
    torch.manual_seed(config.seed + 100)
    eval_acts = _generate_markov_activations(
        config.eval_n_seq, seq_len, num_f, pi, rho, device
    )
    eval_hidden = toy_model(eval_acts) * sf

    return DataPipeline(
        toy_model=toy_model,
        scaling_factor=sf,
        true_features=true_features,
        eval_hidden=eval_hidden,
        gen_flat=gen_flat,
        gen_seq=gen_seq,
        gen_windows=gen_windows,
    )
