"""Matryoshka-TXCDR + multi-scale InfoNCE with same-sequence hard negatives.

Agentic cycle 06 candidate (`agentic_txc_06`). Extends
`MatryoshkaTXCDRContrastiveMultiscale` (cycle 02 winning config at
n_contr_scales=3, γ=0.5) by augmenting the InfoNCE denominator with
K same-sequence hard negatives per anchor.

Motivation: cycles 03-05 showed the multi-scale axis is sharply peaked
at (n=3, γ=0.5). To improve further, need to change the contrastive
SIGNAL QUALITY, not its weight or depth. Hard negatives from the same
sequence but far positions force scale-1/2/3 features to discriminate
*within* a sequence — the current in-batch negatives are mostly
cross-sequence and therefore "easy" (different sequences are trivially
distinguishable from context).

Data shape: expects (B, 2+K, T, d) pairs from
`make_pair_hardneg_window_gen_gpu`:
    [:, 0]     = anchor window
    [:, 1]     = positive (adjacent, +1 token offset)
    [:, 2:2+K] = K same-sequence hard negatives, |gap|≥min_gap

For each scale s in [1, n_contr_scales], computes InfoNCE over the
scale-s prefix, using:
    positive col = scale-s prefix of positive[i] (i.e., same row)
    negatives    = scale-s prefixes of all other in-batch positives
                   (B-1) + all B*K hard-neg prefixes

Total contrastive loss = Σ_s γ^s · InfoNCE_s(anchor_s, pos_s, hn_s).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
    MatryoshkaTXCDRContrastiveMultiscale,
)


def _info_nce_with_hardneg(
    z_a: torch.Tensor,      # (B, d_prefix) anchor
    z_p: torch.Tensor,      # (B, d_prefix) positive (row i matches row i)
    z_hn: torch.Tensor,     # (B, K, d_prefix) hard negs (per-anchor)
) -> torch.Tensor:
    """InfoNCE with per-anchor hard negatives added to the denominator."""
    B, d = z_a.shape
    K = z_hn.shape[1]
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_p = F.normalize(z_p, dim=-1, eps=1e-8)
    z_hn = F.normalize(z_hn, dim=-1, eps=1e-8)

    # Cross-anchor similarity to all positives — row i's positive is col i
    sim_pos = z_a @ z_p.t()                     # (B, B)
    # Per-anchor similarity to its own K hard negs
    sim_hn = (z_a.unsqueeze(1) * z_hn).sum(-1)  # (B, K)

    # Concatenate: columns [0..B-1] = positives (correct col = i),
    # columns [B..B+K-1] = per-anchor hard negs (all wrong for anchor i).
    sim = torch.cat([sim_pos, sim_hn], dim=1)   # (B, B+K)
    labels = torch.arange(B, device=z_a.device)
    # Symmetric loss: also reverse direction (positive queried against anchors)
    sim_rev = z_p @ z_a.t()                     # (B, B)
    # For the reverse direction we don't use hard negs (they anchor on z_a).
    return 0.5 * (F.cross_entropy(sim, labels)
                  + F.cross_entropy(sim_rev, labels))


class MatryoshkaTXCDRContrastiveHardneg(MatryoshkaTXCDRContrastiveMultiscale):
    """Multiscale matryoshka + adjacent-window contrastive + hard negs.

    Accepts (B, 2+K, T, d) input. Forward uses multi-scale InfoNCE
    with hard-neg-augmented denominator at each scale.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        n_contr_scales: int = 3,
        gamma: float = 0.5,
        K_hardneg: int = 4,
        latent_splits: tuple[int, ...] | None = None,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            n_contr_scales=n_contr_scales, gamma=gamma,
            latent_splits=latent_splits,
        )
        self.K_hardneg = K_hardneg

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """Input shape (B, 2+K, T, d) — anchor, positive, K hard negs."""
        if x.ndim != 4:
            # Fallback to parent's single-window path if (B, T, d)
            return super().forward(x, alpha=alpha)

        assert x.shape[1] == 2 + self.K_hardneg, (
            f"Expected (B, 2+K={2+self.K_hardneg}, T, d), "
            f"got shape {tuple(x.shape)}"
        )

        x_prev = x[:, 0]                      # (B, T, d)
        x_cur = x[:, 1]                       # (B, T, d)
        x_hn = x[:, 2:]                       # (B, K, T, d)
        B, K, T, d_in = x_hn.shape

        z_prev = self.encode(x_prev)          # (B, d_sae)
        z_cur = self.encode(x_cur)            # (B, d_sae)
        # Encode hard negs: flatten to (B*K, T, d)
        z_hn_flat = self.encode(x_hn.reshape(B * K, T, d_in))  # (B*K, d_sae)
        z_hn = z_hn_flat.reshape(B, K, -1)    # (B, K, d_sae)

        l_matr = (self._matryoshka_loss(x_prev, z_prev)
                  + self._matryoshka_loss(x_cur, z_cur))

        l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
        for s in range(self.n_contr_scales):
            prefix_s = self.prefix_sum[s]
            z_prev_s = z_prev[:, :prefix_s]
            z_cur_s = z_cur[:, :prefix_s]
            z_hn_s = z_hn[:, :, :prefix_s]
            l_contr_s = _info_nce_with_hardneg(z_cur_s, z_prev_s, z_hn_s)
            l_contr = l_contr + (self.gamma ** s) * l_contr_s

        total = l_matr + alpha * l_contr
        x_hat_cur = self.decode_scale(z_cur, self.T - 1)
        return total, x_hat_cur, z_cur
