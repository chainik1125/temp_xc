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
        latents: (B, T, m) primary latent representation used for decoding.
            For SAE-style models this is the sparse code itself. For models
            with explicit temporal inference (e.g. forward-backward or
            Baum-Welch-style updates), this can instead be a posterior mean or
            another decoded feature-space summary.
        metric_latents: Optional latent tensor to use for sparsity metrics.
            This lets models expose dense posterior means in ``latents`` while
            still reporting sparsity on a thresholded / MAP / support-like
            proxy that is closer to the intended latent ontology.
        loss: Scalar training loss.
        metrics: Per-step metrics dict (recon_loss, l0, etc.).
        aux: Optional extra tensors from the temporal inference step such as
            local evidence, state marginals, pairwise marginals, or transition
            statistics. Evaluation code ignores this field, but it provides a
            clean place for models with structured inference to expose their
            internals.
    """

    x_hat: torch.Tensor
    latents: torch.Tensor
    loss: torch.Tensor
    metric_latents: torch.Tensor | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


class TemporalAE(ABC, nn.Module):
    """Abstract base class for temporal autoencoder architectures.

    All models take (B, T, d) input and return ModelOutput.
    Even independent SAEs receive (B, T, d) and reshape internally.
    """

    def __init__(self):
        super().__init__()
        # Training can disable eager metric extraction to avoid per-step
        # host/device synchronization on small models.
        self.collect_metrics = True

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

    def latents_for_metrics(self, out: ModelOutput) -> torch.Tensor:
        """Return the latent tensor that evaluation should use for sparsity.

        Models with structured temporal inference can override this if they
        need evaluation to use something other than ``out.latents``.
        """
        if out.metric_latents is not None:
            return out.metric_latents
        return out.latents

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Optional: project decoder columns to unit norm. Override if needed."""
        pass

    def reconstruction_bias(self, pos: int | None = None) -> torch.Tensor | None:
        """Optional: return the decoder bias at a position for analysis code."""
        return None

    @property
    def n_positions(self) -> int | None:
        """Number of distinct decoder positions, or None if shared/flat."""
        return None
