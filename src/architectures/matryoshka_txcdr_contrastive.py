"""Position-nested Matryoshka TXCDR + InfoNCE on adjacent windows.

Phase 5.7 autoresearch candidate A3. Combines:
  - `PositionMatryoshkaTXCDR` (position-nested prefix decoders: scale-1
    reconstructs each position alone, scale-T reconstructs full window).
  - Ye et al. 2025 InfoNCE on adjacent samples, ported here to adjacent
    T-windows (shift = 1 token).

Rationale (plan §"Notes on re-exploring Matryoshka"): the nested
prefix-structure gives contrastive an obvious H/L assignment. The
scale-1 prefix (first `m_1 = d_sae // T` latents) is trained to
reconstruct each individual position, so it should be largely
shift-invariant for the T-1 positions shared between two adjacent
windows. Contrasting on the scale-1 prefix enforces that adjacent
windows encode overlapping positions into compatible scale-1 latents.

Inference API matches PositionMatryoshkaTXCDR so probing reuses the
matryoshka encode dispatch in run_probing.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.matryoshka_txcdr import PositionMatryoshkaTXCDR


class MatryoshkaTXCDRContrastive(PositionMatryoshkaTXCDR):
    """Matryoshka TXCDR + adjacent-window InfoNCE on scale-1 prefix.

    Args:
        d_in, d_sae, T, k: same as PositionMatryoshkaTXCDR.
        latent_splits: optional latent-split tuple.
        contr_prefix: number of latents to use for the InfoNCE head.
            Defaults to self.prefix_sum[0] (the scale-1 prefix,
            m_1 = d_sae // T).
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        latent_splits: tuple[int, ...] | None = None,
        contr_prefix: int | None = None,
    ):
        super().__init__(d_in, d_sae, T, k, latent_splits=latent_splits)
        self.contr_prefix = (
            contr_prefix if contr_prefix is not None else self.prefix_sum[0]
        )

    def _matryoshka_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        losses = []
        for t_idx in range(self.T):
            t_size = t_idx + 1
            x_center = self._window_center(x, t_size)
            x_hat = self.decode_scale(z, t_idx)
            loss_scale = (x_hat - x_center).pow(2).sum(dim=-1).mean()
            losses.append(loss_scale)
        return torch.stack(losses).mean()

    def forward(self, x: torch.Tensor, alpha: float = 0.1):
        """Dispatch on input shape.

        - `(B, T, d)`: Matryoshka loss only (single-window fallback).
        - `(B, 2, T, d)`: adjacent-window pair — Matryoshka(prev) +
          Matryoshka(cur) + α · InfoNCE(scale-1-prefix).

        Returns `(loss, x_hat_cur_fullscale, z_cur)`.
        """
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

            z_h_prev = z_prev[:, : self.contr_prefix]
            z_h_cur = z_cur[:, : self.contr_prefix]
            l_contr = _info_nce(z_h_cur, z_h_prev)

            total = l_matr + alpha * l_contr
            x_hat_cur = self.decode_scale(z_cur, self.T - 1)
            return total, x_hat_cur, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
