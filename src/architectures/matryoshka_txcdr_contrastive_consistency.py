"""Matryoshka-TXCDR + multi-scale cosine CONSISTENCY (not InfoNCE).

Agentic cycle 07 candidate (`agentic_txc_07`). Subclasses
`MatryoshkaTXCDRContrastiveMultiscale` (cycle 02 winning config) but
replaces the discriminative InfoNCE with a generative cosine consistency
loss at each scale.

Consistency loss for scale s:
    L_cons_s = 2 - 2 * mean(cos_sim(z_cur_s, stop_grad(z_prev_s)))

This pulls z_cur_s toward z_prev_s without "pushing" other windows apart.
Tests whether cycle 02's gain came from the pull-together mechanism or
the push-apart of cross-sequence negatives in InfoNCE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
    MatryoshkaTXCDRContrastiveMultiscale,
)


def _cosine_consistency(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Pull z_a toward stop_grad(z_b) via 2 - 2 · mean(cos_sim)."""
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b_sg = F.normalize(z_b.detach(), dim=-1, eps=1e-8)
    cos = (z_a * z_b_sg).sum(-1)        # (B,)
    return 2.0 - 2.0 * cos.mean()


class MatryoshkaTXCDRContrastiveConsistency(MatryoshkaTXCDRContrastiveMultiscale):
    """Multiscale matryoshka + COSINE CONSISTENCY (generative) at each scale."""

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        if x.ndim == 3:
            z = self.encode(x)
            loss = self._matryoshka_loss(x, z)
            x_hat = self.decode_scale(z, self.T - 1)
            return loss, x_hat, z

        if x.ndim == 4 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]
            z_prev = self.encode(x_prev)
            z_cur = self.encode(x_cur)

            l_matr = (self._matryoshka_loss(x_prev, z_prev)
                      + self._matryoshka_loss(x_cur, z_cur))

            l_cons = torch.zeros((), device=x.device, dtype=x.dtype)
            for s in range(self.n_contr_scales):
                prefix_s = self.prefix_sum[s]
                # Symmetric: pull cur→prev and prev→cur
                l_s = 0.5 * (
                    _cosine_consistency(z_cur[:, :prefix_s], z_prev[:, :prefix_s])
                    + _cosine_consistency(z_prev[:, :prefix_s], z_cur[:, :prefix_s])
                )
                l_cons = l_cons + (self.gamma ** s) * l_s

            total = l_matr + alpha * l_cons
            x_hat_cur = self.decode_scale(z_cur, self.T - 1)
            return total, x_hat_cur, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )
