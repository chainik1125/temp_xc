"""Toy model: linear embedding of sparse features into activation space.

Generates activation vectors x_t = sum_k a_{t,k} * f_k where:
  - f_k are orthogonal unit-norm feature directions in R^d
  - a_{t,k} = s_{t,k} * m_{t,k} (support * magnitude)
"""

from __future__ import annotations

import torch

from ..utils import orthogonalize


class ToyModel:
    """Linear toy model mapping sparse features to activation vectors."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        *,
        generator: torch.Generator | None = None,
    ):
        self.n_features = n_features
        self.d_model = d_model
        self.features = orthogonalize(n_features, d_model, generator=generator)

    def to(self, device: torch.device) -> ToyModel:
        self.features = self.features.to(device)
        return self

    def embed(
        self,
        support: torch.Tensor,
        magnitude_mean: float = 1.0,
        magnitude_std: float = 0.15,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Embed binary support into activation space.

        Args:
            support: (n_seq, n_features, T) binary tensor.
            magnitude_mean: Mean of half-normal magnitude distribution.
            magnitude_std: Std of magnitude distribution.
            generator: Optional torch random generator.

        Returns:
            x: (n_seq, T, d_model) activation vectors.
        """
        n_seq, k, T = support.shape
        # Magnitudes: |N(mean, std^2)|
        magnitudes = (
            torch.randn(n_seq, k, T, generator=generator) * magnitude_std
            + magnitude_mean
        ).abs()

        activations = support * magnitudes  # (n_seq, k, T)

        # Move to same device as features
        activations = activations.to(self.features.device)

        # x_t = sum_k a_{t,k} * f_k
        # activations: (n_seq, k, T), features: (k, d)
        x = torch.einsum("nkt,kd->ntd", activations, self.features)
        return x
