"""Galaxy 6 — Max-Pool TXC.

Y's GIGABRAIN architectural proposal (2026-05-01): instead of summing
per-position encoder outputs, take the MAX activation across positions
per feature.

Mechanism:
    For each feature j:
        pre_pos[t] = x[t] @ W_enc[t][:, j]  (per-position pre-activation)
        z[j] = max_t(pre_pos[t])           (max across positions)
        z[j] = ReLU(z[j])                  (ReLU)
    Then TopK across features.

Decoder identical to TXCBareAntidead: x_hat[t] = z @ W_dec[:, t, :].

Why this might help: the max-pool selects the position where each feature
is strongest, breaking the "diluted by averaging" failure mode at large T.
At T=2 it should behave similar to sum (peak picking dominant of two);
at T=5+ it should preserve concentrated signals.

Hypothesis: lifts coh ≥ 1.75 win at higher T (T=5+) where sum dilutes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_antidead import TXCBareAntidead, geometric_median


class TXCMaxPool(TXCBareAntidead):
    """Max-pool encoder; everything else inherited from TXCBareAntidead.

    Replace sum-over-T pre-activation with max-over-T.
    """

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) pre-ReLU, pre-TopK using max-over-T."""
        # Per-position pre-activation: (B, T, d_sae)
        pre_pos = torch.einsum("btd,tds->bts", x, self.W_enc)
        # Max over T positions: (B, d_sae)
        max_per_feat = pre_pos.max(dim=1).values
        return max_per_feat + self.b_enc
