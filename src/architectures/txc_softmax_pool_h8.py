"""Galaxy 11 — Soft-Max-Pool merge TXC + H8 multi-distance contrastive stack.

Combines the two strongest architectural ideas:
  - Galaxy 8 soft-max-pool encoder (learnable per-feature τ)
  - H8 multi-distance contrastive InfoNCE loss (W's contribution)

Y's hypothesis (2026-05-01): if soft-max-pool alone (Galaxy 8) hits
Δ=+1.089 at coh ≥ 1.75 PP and max-pool + H8 (W's TXCMaxPoolMergeH8)
hits Δ=+0.811, then soft-max-pool + H8 should match or exceed both.

Implementation: subclass TXCBareMultiDistanceContrastiveAntidead;
override `_pre_activation` to use softmax-weighted-sum across T
positions with per-feature learnable temperature τ.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
)


class TXCSoftMaxPoolMergeH8(TXCBareMultiDistanceContrastiveAntidead):
    """Soft-max-pool merge encoder; H8 multi-distance stack everywhere else.

    Encoder forward:
        pre_pos[b, t, s] = (x[b, t, :] @ W_enc[t, :, s])
        weights[b, t, s] = softmax_t(pre_pos[b, :, s] / τ_s)
        merged[b, s]     = Σ_t weights[b, t, s] * pre_pos[b, t, s] + b_enc[s]
        z[b, s]          = ReLU(top_k(merged))

    τ is parameterized as exp(log_tau); one log_tau per feature, init at 0.
    """

    def __init__(self, d_in, d_sae, T, k, **kwargs):
        super().__init__(d_in, d_sae, T, k, **kwargs)
        # log_tau initialized at 0 (τ=1.0); one per feature
        self.log_tau = nn.Parameter(torch.zeros(d_sae))

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        pre_pos = torch.einsum("btd,tds->bts", x, self.W_enc)         # (B, T, d_sae)
        tau = self.log_tau.exp().clamp(min=0.05, max=20.0)
        weights = torch.softmax(pre_pos / tau.view(1, 1, -1), dim=1)  # (B, T, d_sae)
        merged = (weights * pre_pos).sum(dim=1)                        # (B, d_sae)
        return merged + self.b_enc
