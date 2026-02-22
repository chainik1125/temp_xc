"""Two-position toy model with shared embedding, no cross-position mixing."""

import torch
import torch.nn as nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from src.shared.orthogonalize import orthogonalize
from src.v2_crosscoder_comparison.configs import ToyModelConfig


class TwoPositionToyModel(HookedRootModule):
    """Toy model that maps feature activations to hidden activations per position.

    Uses a shared linear embedding across positions with no cross-position mixing.
    Input: (batch, n_positions, num_features)
    Output: (batch, n_positions, hidden_dim)
    """

    def __init__(self, cfg: ToyModelConfig):
        super().__init__()
        self.cfg = cfg

        # Shared embedding: (num_features, hidden_dim) with near-orthogonal rows
        embedding = orthogonalize(
            cfg.num_features, cfg.hidden_dim, target_cos_sim=cfg.target_cos_sim
        )
        self.embedding = nn.Parameter(embedding, requires_grad=False)

        self.hook_hidden = HookPoint()
        self.setup()

    def forward(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Map feature activations to hidden activations.

        Args:
            feature_acts: (batch, n_positions, num_features)

        Returns:
            Hidden activations: (batch, n_positions, hidden_dim)
        """
        # (batch, n_positions, num_features) @ (num_features, hidden_dim)
        # -> (batch, n_positions, hidden_dim)
        hidden = torch.einsum("bpf,fh->bph", feature_acts, self.embedding)
        hidden = self.hook_hidden(hidden)
        return hidden
