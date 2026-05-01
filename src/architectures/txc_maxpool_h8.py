"""MYSTERY: Max-pool merge TXC + H8 stack.

Variant of multiplicative-merge: instead of product, use MAX across
positions. Captures "concept active at SOME position in the window"
(disjunctive feature).

For steering, max-pool features represent "concept somewhere in window".
Less position-specific, more recall-oriented.

Implementation: subclass TXCBareMultiDistanceContrastiveAntidead, override
_pre_activation to use max instead of sum.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
)


class TXCMaxPoolMergeH8(TXCBareMultiDistanceContrastiveAntidead):
    """Max-pool merge encoder; H8 stack everywhere else.

    Encoder forward:
        z_per_pos[b, t, s] = (x[b, t, :] @ W_enc[t, :, s])
        pre[b, s]          = max_t z_per_pos[b, t, s] + b_enc[s]
        z[b, s]            = ReLU(top_k(pre))
    """

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        z_per_pos = torch.einsum("btd,tds->bts", x, self.W_enc)  # (B, T, d_sae)
        merged = z_per_pos.max(dim=1).values                      # (B, d_sae)
        return merged + self.b_enc
