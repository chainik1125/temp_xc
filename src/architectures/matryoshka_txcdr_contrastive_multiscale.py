"""Matryoshka-TXCDR + multi-scale InfoNCE.

Agentic cycle 02 candidate (`agentic_txc_02`). Extends
`MatryoshkaTXCDRContrastive` by computing InfoNCE at multiple nested
prefix scales (not just scale-1), with geometric decay γ^s so the
scale-1 term remains dominant.

Motivation: A3 α=1.0 hit a local optimum for probing. Cycle 01 showed
adding more pressure (orthogonality) hurt. Changing the signal rather
than the pressure is the next move. Scales 2..T latents currently
receive no contrastive gradient — only reconstruction. Adding
decaying-weight InfoNCE at scales 2, 3 tests whether those latents
benefit from cross-token consistency pressure.

Contrastive loss:
    L_contr = Σ_{s=0}^{S-1} γ^s · InfoNCE(z_cur[:, :prefix_s+1],
                                          z_prev[:, :prefix_s+1])

where S = n_contr_scales (default 3), γ default 0.5. This nests:
scale-1 gets weight 1, scale-2 weight 0.5, scale-3 weight 0.25.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.matryoshka_txcdr_contrastive import (
    MatryoshkaTXCDRContrastive,
    _info_nce,
)


class MatryoshkaTXCDRContrastiveMultiscale(MatryoshkaTXCDRContrastive):
    """Matryoshka TXCDR + InfoNCE across multiple prefix scales.

    Args: same as parent, plus:
        n_contr_scales: number of scales at which to apply InfoNCE.
            Must be ≤ T. Default 3.
        gamma: per-scale decay weight. Default 0.5.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        n_contr_scales: int = 3,
        gamma: float = 0.5,
        latent_splits: tuple[int, ...] | None = None,
    ):
        super().__init__(d_in, d_sae, T, k, latent_splits=latent_splits)
        assert 1 <= n_contr_scales <= T, (
            f"n_contr_scales={n_contr_scales} must be in [1, T={T}]"
        )
        self.n_contr_scales = n_contr_scales
        self.gamma = gamma

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """Same dispatch as parent. Pair-window path replaces the single
        scale-1 InfoNCE with a multi-scale weighted sum."""
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

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for s in range(self.n_contr_scales):
                prefix_s = self.prefix_sum[s]
                z_h_prev = z_prev[:, :prefix_s]
                z_h_cur = z_cur[:, :prefix_s]
                l_contr = l_contr + (self.gamma ** s) * _info_nce(z_h_cur, z_h_prev)

            total = l_matr + alpha * l_contr
            x_hat_cur = self.decode_scale(z_cur, self.T - 1)
            return total, x_hat_cur, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )
