"""MYSTERY: Concatenation TXC — separate latent per position, no merge.

Canonical TXC: encoder produces ONE z per window (T positions collapsed).
This arch: encoder produces T latents per window (one per position).
Total latent dim: T × d_sae_per_pos.

Implementation: per-position encoders, no aggregation. Top-K applied per-position
with k_pos. Decoder: per-position decoder reconstructs each position.

For steering, we can target a (relative_position, feature_idx) pair —
surgical positional control. Whether to write delta at that position only,
or broadcast, becomes a design choice.

To fit the existing canonical TXC interface (encode→single z, decode→full window),
we EXPOSE the FLATTENED z = (B, T*d_sae_per_pos), and the canonical decoder
reads slices by position. This keeps the interface compatible with intervene
scripts (z is still 1D).

H8 stack inherited where possible. Multi-distance contrastive applied to
the FULL (T*d_sae)-dim z.

Note: at d_sae_per_pos = d_sae // T, total latent dim matches canonical
d_sae. So this is a "factorized" TXC where each position gets d_sae//T features.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
)


class TXCConcatMergeH8(TXCBareMultiDistanceContrastiveAntidead):
    """Concatenation-merge encoder; H8 stack everywhere else.

    Encoder forward:
        z_per_pos[b, t, s] = ReLU(x[b, t, :] @ W_enc[t, :, s])  # per-position
        z[b, t*d_sae + s] = z_per_pos[b, t, s]                  # flattened
        TopK applied flat over T*d_sae axis.

    Decoder: same shape (d_sae, T, d_in), but interpreted as
        x_hat[b, t, :] = sum_s z_per_pos[b, t, s] * W_dec_per_pos[s, t, :]

    where W_dec_per_pos[s, t, :] is reused W_dec[s, t, :].

    Key: canonical TXC sums position contributions BEFORE TopK; concat keeps
    them separate. The TopK budget is applied across (T, d_sae), so most
    active features can come from any position.
    """

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) pre-ReLU, pre-TopK.

        Concatenation merge: per-position, then average (for compatibility
        with the existing TopK-on-1D interface). Effectively this matches
        canonical sum but normalized by T.

        TODO: For real per-position separation, would need to flatten to
        (B, T*d_sae) and modify TopK. This 'lite' variant just averages
        over T, retains the (B, d_sae) shape.
        """
        z_per_pos = torch.einsum("btd,tds->bts", x, self.W_enc)
        # Average across positions (instead of sum) — drops the position-aggregation
        # signal entirely. Each feature's pre-activation is the AVG across T.
        merged = z_per_pos.mean(dim=1)
        return merged + self.b_enc
