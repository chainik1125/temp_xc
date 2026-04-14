"""Base architecture interface for crosscoder comparison."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.v2_crosscoder_comparison.configs import ArchitectureConfig


@dataclass
class ArchLoss:
    """Loss components from an architecture forward pass."""

    total_loss: float
    recon_loss: float
    sparsity_loss: float
    l0: float


class BaseArchitecture(ABC):
    """Abstract base class for all architectures in the comparison."""

    def __init__(self, config: ArchitectureConfig, d_model: int):
        self.config = config
        self.d_model = d_model

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent activations.

        Args:
            x: Input tensor of shape (batch, n_positions, d_model).

        Returns:
            Latent activations.
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent activations to reconstructed input.

        Args:
            z: Latent activations.

        Returns:
            Reconstructed input of shape (batch, n_positions, d_model).
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode.

        Args:
            x: Input of shape (batch, n_positions, d_model).

        Returns:
            Reconstruction of shape (batch, n_positions, d_model).
        """
        ...

    @abstractmethod
    def get_losses(self, x: torch.Tensor) -> ArchLoss:
        """Compute losses for a batch.

        Args:
            x: Input of shape (batch, n_positions, d_model).

        Returns:
            ArchLoss with loss components.
        """
        ...

    @abstractmethod
    def get_decoder_weights(self) -> torch.Tensor:
        """Get decoder weight matrix for evaluation.

        Returns:
            Decoder weights. Shape depends on architecture:
            - SAE: (d_sae, d_model)
            - Crosscoder: (d_sae, n_positions, d_model)
        """
        ...
