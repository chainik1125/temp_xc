"""Abstract base class for all architectures in ``temp-bench``.

Every TXC and SAE in this paper exposes the same minimal interface:
``encode(x) -> z`` and ``decode(z) -> x_hat``. This unifies probing,
qualitative analysis, steering, and case-study evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ArchConfig:
    """Common config keys. Specific archs subclass and add fields."""
    name: str
    d_in: int
    d_sae: int
    k_pos: int          # sparsity per (token, position) — 1 for SAEs, T*k_pos for window total
    T: int = 1          # window length; 1 for per-token SAEs


class TempBenchArch(ABC, nn.Module):
    """Unified interface across SAE / T-SAE / TFA / MLC / TXC-base / TXC-pro.

    Shape conventions:

        x:      (B, T, d_in)        — windowed input (T=1 for per-token SAEs)
        z:      (B, T, d_sae)       — latent codes (T may be 1 for shared-z TXCs;
                                       implementations document their own shape)
        x_hat:  (B, T, d_in)        — reconstruction
    """

    config: ArchConfig

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @property
    def d_sae(self) -> int:
        return self.config.d_sae

    @property
    def T(self) -> int:
        return self.config.T

    def decoder_directions(self) -> torch.Tensor:
        """Return decoder columns as ``(d_sae, d_in)`` (T-averaged for window archs).

        Used by C1/C2 feature-recovery AUC and C4 qualitative ranking.
        Subclasses override if averaging is not the right reduction.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement decoder_directions()"
        )
