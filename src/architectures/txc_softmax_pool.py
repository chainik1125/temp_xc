"""Galaxy 8 — Soft-Max-Pool TXC with learnable temperature.

Generalizes both Galaxy 6 (hard max-pool) AND TXCBareAntidead (additive
sum) into a single architecture. Each feature learns its own pooling
temperature τ.

Mechanism:
    For each feature j:
        pre_pos[t, j] = x[t] @ W_enc[t][:, j]
        weights[t, j] = softmax(pre_pos[:, j] / τ_j)[t]
        z[j] = Σ_t weights[t, j] * pre_pos[t, j]

When τ → 0:   z[j] → max_t pre_pos[t, j]   (max-pool, Galaxy 6)
When τ → ∞:   z[j] → mean_t pre_pos[t, j]  (additive average; up to
                                            scale = additive sum from
                                            TXCBareAntidead)

The temperature is parameterized as τ = exp(log_tau) with one log_tau
per feature, initialized at log(1.0) = 0. Per-feature τ allows each
feature to pick its preferred pooling style — e.g. concept-specific
features can be sharp (small τ) while broad context features can be
diffuse (large τ).

Decoder: identical to TXCBareAntidead (per-position write-back).

Hypothesis: this should match or exceed Galaxy 6 and TXCBareAntidead
because it can choose either limit, and may find non-trivial sweet
spots in between.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_antidead import TXCBareAntidead


class TXCSoftMaxPool(TXCBareAntidead):
    """Soft-max-pool encoder with learnable per-feature temperature."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__(d_in, d_sae, T, k)
        # log_tau initialized at 0 (τ=1.0); one per feature
        self.log_tau = nn.Parameter(torch.zeros(d_sae))

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) pre-ReLU, pre-TopK via softmax-weighted-sum.

        weights[b, t, j] = softmax_t(pre_pos[b, t, j] / τ_j)
        z[b, j] = Σ_t weights[b, t, j] * pre_pos[b, t, j]
        """
        # Per-position pre-activation: (B, T, d_sae)
        pre_pos = torch.einsum("btd,tds->bts", x, self.W_enc)
        # Per-feature temperature: (d_sae,) -> (1, 1, d_sae) for broadcast
        tau = self.log_tau.exp().clamp(min=0.05, max=20.0)
        # Softmax-weighted sum over T positions
        weights = torch.softmax(pre_pos / tau.view(1, 1, -1), dim=1)
        pooled = (weights * pre_pos).sum(dim=1)
        return pooled + self.b_enc
