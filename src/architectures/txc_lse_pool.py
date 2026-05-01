"""Galaxy 20 — LogSumExp Pool TXC with learnable temperature.

Y's GIGABRAIN architectural proposal #3 (2026-05-01): yet another
non-additive aggregation. Where Galaxy 6 is hard-max and Galaxy 8 is
softmax-weighted-MEAN, Galaxy 20 is **softmax-LSE** — preserves
unnormalized mass, behaves differently in the τ → ∞ limit.

Mechanism:
    pre_pos[b, t, s] = (x[b, t, :] @ W_enc[t, :, s])
    z[b, s]          = τ_s * log Σ_t exp(pre_pos[b, t, s] / τ_s) + b_enc[s]

Limits:
    τ → 0:  z → max_t pre_pos[t]                       (hard max-pool, Galaxy 6)
    τ → ∞:  z → mean_t pre_pos[t] + τ * log T          (mean + log T)

So at large τ, Galaxy 20 → mean + constant offset (NOT additive sum).
At small τ, Galaxy 20 → max (Galaxy 6).

Compared to Galaxy 8 (softmax-weighted-mean), Galaxy 20 preserves more
of the original signal magnitude when multiple positions co-fire (since
LSE > max when there are multiple high values, while softmax-weighted-
mean ≤ max).

Hypothesis: Galaxy 20 might capture concepts where MULTIPLE positions
need to co-fire — a feature is on if ANY position is highly active OR
if MANY positions are moderately active.

Decoder identical to TXCBareAntidead.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.architectures.txc_bare_antidead import TXCBareAntidead


class TXCLogSumExpPool(TXCBareAntidead):
    """LogSumExp pool encoder with learnable per-feature temperature."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__(d_in, d_sae, T, k)
        # log_tau initialized at 0 (τ=1.0); one per feature
        self.log_tau = nn.Parameter(torch.zeros(d_sae))

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) pre-ReLU, pre-TopK via LogSumExp.

        z[b, s] = τ_s * log Σ_t exp(pre_pos[b, t, s] / τ_s)
        """
        pre_pos = torch.einsum("btd,tds->bts", x, self.W_enc)  # (B, T, d_sae)
        tau = self.log_tau.exp().clamp(min=0.05, max=20.0)
        scaled = pre_pos / tau.view(1, 1, -1)                   # (B, T, d_sae)
        # torch.logsumexp is numerically stable
        lse = torch.logsumexp(scaled, dim=1) * tau              # (B, d_sae)
        return lse + self.b_enc
