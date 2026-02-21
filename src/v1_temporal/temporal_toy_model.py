"""Temporal toy model with optional causal mean-pooling mixing.

The model embeds g features into a d-dimensional space via a linear layer
with orthogonalized weights, then optionally applies causal mean-pooling
to simulate attention-like cross-position mixing.

When gamma > 0, the post-mixing representation at position t is:
    x_t_post = (1 - gamma) * x_t + gamma * (1/t) * sum_{t'=1}^{t} x_{t'}

This reinforces global features (which contribute the same direction at
every attended position) while diluting local features.
"""

from typing import Any

import torch
from transformer_lens.hook_points import HookedRootModule

from src.shared.orthogonalize import orthogonalize


class TemporalToyModel(HookedRootModule):
    """Linear embedding toy model with optional causal mixing.

    Input: (batch, T, num_features) -- feature activations at each position
    Output: (batch, T, hidden_dim) -- hidden activations at each position

    Args:
        num_features: Total number of features (global + local).
        hidden_dim: Embedding dimension.
        gamma: Mixing strength for causal mean-pooling. 0 = no mixing.
        target_cos_sim: Target pairwise cosine similarity.
        bias: Whether to include bias in the linear layer.
        ortho_lr: Learning rate for orthogonalization.
        ortho_num_steps: Steps for orthogonalization.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        gamma: float = 0.0,
        target_cos_sim: float = 0.0,
        bias: bool = False,
        ortho_lr: float = 0.01,
        ortho_num_steps: int = 1000,
    ):
        super().__init__()
        self.num_feats = num_features
        self.hidden_dim = hidden_dim
        self.gamma = gamma
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
        """Apply embedding at each position, then optionally mix.

        Args:
            x: (batch, T, num_features) or (batch, num_features).

        Returns:
            (batch, T, hidden_dim) or (batch, hidden_dim).
        """
        h = self.embed(x)
        if self.gamma > 0 and h.dim() == 3:
            h = self._causal_mix(h)
        return h

    def _causal_mix(self, h: torch.Tensor) -> torch.Tensor:
        """Apply causal mean-pooling mixing.

        x_t_post = (1 - gamma) * x_t + gamma * mean(x_1, ..., x_t)

        Args:
            h: (batch, T, hidden_dim)

        Returns:
            Mixed tensor of same shape.
        """
        T = h.shape[1]
        mixed = torch.zeros_like(h)
        cumsum = torch.zeros_like(h[:, 0, :])  # (batch, d)
        for t in range(T):
            cumsum = cumsum + h[:, t, :]
            mean_so_far = cumsum / (t + 1)
            mixed[:, t, :] = (1 - self.gamma) * h[:, t, :] + self.gamma * mean_so_far
        return mixed

    @property
    def feature_directions(self) -> torch.Tensor:
        """Return feature direction matrix of shape (num_features, hidden_dim)."""
        return self.embed.weight.data.T
