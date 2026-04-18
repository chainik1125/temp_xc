"""Toy model with orthogonal feature directions.

Embeds g features into a d-dimensional space via a linear layer
with orthogonalized weights. Each activation is a sparse linear
combination of these feature directions.
"""

from typing import Any

import torch
from transformer_lens.hook_points import HookedRootModule

from src.utils.orthogonalize import orthogonalize


class ToyModel(HookedRootModule):
    """Linear embedding toy model with orthogonal feature directions.

    Input: (batch, T, num_features) or (batch, num_features)
    Output: (batch, T, hidden_dim) or (batch, hidden_dim)

    Args:
        num_features: Number of features.
        hidden_dim: Embedding dimension.
        target_cos_sim: Target pairwise cosine similarity.
        bias: Whether to include bias in the linear layer.
        ortho_lr: Learning rate for orthogonalization.
        ortho_num_steps: Steps for orthogonalization.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        target_cos_sim: float = 0.0,
        bias: bool = False,
        ortho_lr: float = 0.01,
        ortho_num_steps: int = 1000,
    ):
        super().__init__()
        self.num_feats = num_features
        self.hidden_dim = hidden_dim
        self.embed = torch.nn.Linear(num_features, hidden_dim, bias=bias)
        embeddings = orthogonalize(
            num_features,
            hidden_dim,
            target_cos_sim=target_cos_sim,
            lr=ortho_lr,
            num_steps=ortho_num_steps,
        )
        self.embed.weight.data = embeddings.T
        self.setup()

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply embedding at each position.

        Args:
            x: (batch, T, num_features) or (batch, num_features).

        Returns:
            (batch, T, hidden_dim) or (batch, hidden_dim).
        """
        return self.embed(x)

    @property
    def feature_directions(self) -> torch.Tensor:
        """Return feature direction matrix of shape (num_features, hidden_dim)."""
        return self.embed.weight.data.T
