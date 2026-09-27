"""Config dataclasses for colored-source synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ColoredSourceConfig:
    """Parameters for a delayed temporally-colored source dataset.

    Attributes:
        N: Number of latent sources / ground-truth feature directions.
        d: Ambient observation dimension. Stages 1/2 require d == N.
        D: Delay (number of independent residue classes mod D). D=1 is the lag-1 base case.
        sigma: Observation noise standard deviation.
        rho_min: Smallest autocorrelation in the per-coordinate AR(1) schedule.
        rho_max: Largest autocorrelation.
        n_seq: Number of independent sequences (chains).
        T_chain: Length of each chain.
        seed: RNG seed for reproducibility.
    """

    N: int = 128
    d: int = 128
    D: int = 1
    sigma: float = 0.1
    rho_min: float = 0.1
    rho_max: float = 0.9
    n_seq: int = 128
    T_chain: int = 512
    seed: int = 0

    @classmethod
    def stage1_default(cls) -> "ColoredSourceConfig":
        return cls()

    @classmethod
    def stage2_default(cls, D: int) -> "ColoredSourceConfig":
        T_chain = 1024 if D >= 4 else 512
        n_seq = 256 if D >= 4 else 128
        return cls(D=D, T_chain=T_chain, n_seq=n_seq)


def rho_schedule(N: int, rho_min: float, rho_max: float) -> torch.Tensor:
    """Linearly spaced autocorrelations on [rho_min, rho_max].

    Distinct rho_i is required for the temporal-recovery claim.
    """
    if N < 2:
        return torch.tensor([rho_min], dtype=torch.float64)
    return torch.linspace(rho_min, rho_max, N, dtype=torch.float64)


def eigengap(rho: torch.Tensor) -> float:
    """gamma = min_{i != j} |rho_i - rho_j|. Used in the temporal sample-complexity bound."""
    sorted_rho, _ = torch.sort(rho)
    diffs = torch.diff(sorted_rho)
    return diffs.abs().min().item()
