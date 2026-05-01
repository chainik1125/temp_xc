"""MYSTERY: Multiplicative-merge TXC + H8 stack.

Hypothesis (Han, 2026-04-30): "what if instead of summing the encoder
result at each position to get the final latent, we do something
mysterious?"

Canonical TXC encoder: z[s] = ReLU(sum_t (x[t] @ W_enc[t,:,s]) + b[s]).
This is "additive" merge: position contributions sum.

THIS arch: z[s] = ReLU(prod_t softplus(x[t] @ W_enc[t,:,s]) + b[s] - threshold).
Multiplicative merge: feature only fires when ALL T positions activate it.
Conjunctive feature detector — much sparser, higher-confidence features.

For steering, the multiplicative feature represents "concept consistent
across the entire window". Clamping such a feature writes a coherent-
across-positions signal — directly addressing the coherence-cliff failure
mode.

Implementation: subclass TXCBareMultiDistanceContrastiveAntidead, override
_pre_activation. Everything else (matryoshka decoder, contrastive loss,
anti-dead stack) inherits.

Note: softplus instead of ReLU per-position to avoid hard-zeros
(which would zero out gradient through product).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
)


class TXCMultiplicativeMergeH8(TXCBareMultiDistanceContrastiveAntidead):
    """Multiplicative-merge encoder; H8 stack everywhere else.

    Encoder forward:
        z_per_pos[b, t, s] = softplus(x[b, t, :] @ W_enc[t, :, s])
        pre[b, s]          = prod_t z_per_pos[b, t, s] + b_enc[s]
        z[b, s]            = ReLU(top_k(pre))

    All else (decoder, anti-dead, matryoshka, contrastive) inherits.
    """

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) pre-ReLU, pre-TopK.

        Multiplicative merge: per-position softplus, then product, with
        log-domain stability (sum log to avoid product underflow).
        """
        # Per-position pre-activation: (B, T, d_sae)
        z_per_pos = torch.einsum("btd,tds->bts", x, self.W_enc)
        # Softplus per-position (ensures positive, smooth, non-zero gradient).
        # Using softplus directly on z gives values ~log(2) ≈ 0.69 at z=0.
        a_per_pos = F.softplus(z_per_pos)                   # (B, T, d_sae), positive
        # Multiplicative merge across positions via log-sum-exp for stability:
        #   prod_t a_per_pos[t] = exp(sum_t log(a_per_pos[t]))
        # log(softplus(x)) is well-defined since softplus > 0.
        merged = a_per_pos.log().sum(dim=1).exp()           # (B, d_sae)
        return merged + self.b_enc
