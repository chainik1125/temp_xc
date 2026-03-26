"""Base model interface for all temporal autoencoder architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Standard output from every model's forward pass.

    Attributes:
        x_hat: (B, T, d) reconstructed activations.
        latents: (B, T, m) latent codes (post any temporal mixing).
        loss: Scalar training loss.
        metrics: Per-step metrics dict (recon_loss, l0, etc.).
    """

    x_hat: torch.Tensor
    latents: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class TemporalAE(ABC, nn.Module):
    """Abstract base class for temporal autoencoder architectures.

    All models take (B, T, d) input and return ModelOutput.
    Even independent SAEs receive (B, T, d) and reshape internally.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Full forward pass.

        Args:
            x: Input activations, shape (B, T, d).

        Returns:
            ModelOutput with reconstruction, latents, loss, and metrics.
        """
        ...

    @abstractmethod
    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        """Extract decoder weight columns for feature recovery evaluation.

        Args:
            pos: If not None, return decoder for a specific position.
                 If None, return a single (d, m) matrix (shared or averaged).

        Returns:
            (d, m) tensor where columns are decoder atoms.
        """
        ...

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Optional: project decoder columns to unit norm. Override if needed."""
        pass

    @property
    def n_positions(self) -> int | None:
        """Number of distinct decoder positions, or None if shared/flat."""
        return None
