"""Toy model following the Linear Representation Hypothesis.

The model embeds g features into a d-dimensional space via a linear layer
with orthogonalized weights. It inherits from HookedRootModule for
compatibility with SAE-lens training.
"""

from typing import Any

import torch
from transformer_lens.hook_points import HookedRootModule

from src.shared.orthogonalize import orthogonalize


class ToyModel(HookedRootModule):
    """Linear embedding toy model with orthogonalized feature directions.

    Args:
        num_feats: Number of ground-truth features (g).
        hidden_dim: Embedding dimension (d). Should satisfy d >= g for
            perfect orthogonality.
        target_cos_sim: Target pairwise cosine similarity between feature
            embeddings. 0 = orthogonal.
        bias: Whether to include bias in the linear layer.
        ortho_lr: Learning rate for orthogonalization.
        ortho_num_steps: Number of optimization steps for orthogonalization.
    """

    def __init__(
        self,
        num_feats: int,
        hidden_dim: int,
        target_cos_sim: float = 0.0,
        bias: bool = False,
        ortho_lr: float = 0.01,
        ortho_num_steps: int = 1000,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.hidden_dim = hidden_dim
        self.embed = torch.nn.Linear(num_feats, hidden_dim, bias=bias)
        embeddings = orthogonalize(
            num_feats,
            hidden_dim,
            target_cos_sim=target_cos_sim,
            lr=ortho_lr,
            num_steps=ortho_num_steps,
        )
        self.embed.weight.data = embeddings.T
        self.setup()

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.embed(x)

    @property
    def feature_directions(self) -> torch.Tensor:
        """Return feature direction matrix of shape (num_feats, hidden_dim)."""
        return self.embed.weight.data.T
