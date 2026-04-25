"""Matryoshka-TXCDR multi-scale contrastive + BatchTopK + AuxK loss.

Phase 6.1 Cycle H candidate (`agentic_txc_11_stack`). Combines:

    - Phase 5.7 cycle-02 recipe: matryoshka + multi-scale InfoNCE
      (γ=0.5, n_scales=3).
    - Phase 5.7 experiment (ii): BatchTopK sparsity (variable per-
      sample budget; EMA-thresholded inference).
    - Phase 6.1 Cycle F discovery: BatchTopK is the load-bearing
      piece for TXC qualitative (0.80 alive, 7/8 semantic labels on
      the concat A+B autointerp protocol).
    - Phase 6.1 Cycle A AuxK port: parallel reconstruction from
      top-`aux_k` DEAD features recruits them back into the active
      set.

Goal: see whether stacking AuxK ON TOP of BatchTopK (both are anti-
dead mechanisms but act on different axes) pushes Cycle F's 7/8 to
a full 8/8 semantic. BatchTopK revives dead features by allowing
them to fire per-sample; AuxK adds a gradient signal that directly
shapes their decoder directions toward the reconstruction residual.

Expected marginal: small (BatchTopK is already at 0.80 alive and
dominates the top-8-by-variance ranking). Worst case: no change.
Best case: the remaining 1 punctuation feature in Cycle F is
displaced by a semantic one.

Inheritance:
    MatryoshkaTXCDRContrastiveMultiscaleBatchTopK  (Cycle F base)
        ↑
    MatryoshkaTXCDRContrastiveMultiscaleBatchTopKAuxK  (this class)

AuxK logic mirrors `matryoshka_txcdr_contrastive_multiscale_auxk.py`
but trims any TopK-specific code path. `pre` is the flat (B, d_sae)
pre-activation; dead features use the same `num_tokens_since_fired`
tracker.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures._batchtopk_variants import (
    MatryoshkaTXCDRContrastiveMultiscaleBatchTopK,
)
from src.architectures.matryoshka_txcdr_contrastive import _info_nce


class MatryoshkaTXCDRContrastiveMultiscaleBatchTopKAuxK(
    MatryoshkaTXCDRContrastiveMultiscaleBatchTopK
):
    """Cycle F base + AuxK residual-reconstruction loss.

    Args (in addition to parent):
        aux_k: budget of dead features per-sample for AuxK.
        dead_threshold_tokens: tokens-since-fired threshold.
        auxk_alpha: weight on the AuxK term.
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
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            n_contr_scales=n_contr_scales, gamma=gamma,
            latent_splits=latent_splits,
        )
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count", torch.tensor(0, dtype=torch.long))

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-ReLU, pre-BatchTopK (B, d_sae)."""
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def _auxk_loss(
        self,
        x_cur: torch.Tensor,        # (B, T, d_in)
        z_cur: torch.Tensor,        # (B, d_sae) post-sparsity
        pre_cur: torch.Tensor,      # (B, d_sae) pre-ReLU, pre-sparsity
        x_hat_full: torch.Tensor,   # (B, T, d_in)
    ) -> torch.Tensor:
        active_mask = (z_cur > 0).any(dim=0)
        n_tokens = x_cur.shape[0] * x_cur.shape[1]
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0

        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)
        if n_dead == 0:
            self.last_auxk_loss.fill_(0.0)
            return torch.zeros((), device=x_cur.device, dtype=x_cur.dtype)

        k_aux = min(self.aux_k, n_dead)
        auxk_pre = F.relu(pre_cur).masked_fill(~dead_mask.unsqueeze(0), 0.0)
        vals, idx = auxk_pre.topk(k_aux, dim=-1, sorted=False)
        aux_buf = torch.zeros_like(auxk_pre)
        aux_buf.scatter_(-1, idx, vals)

        W_full = self.W_decs[self.T - 1]                    # (d_sae, T, d_in)
        aux_decode = torch.einsum("bs,std->btd", aux_buf, W_full)

        residual = x_cur - x_hat_full.detach()
        l2 = (residual - aux_decode).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        loss = (l2 / denom.clamp(min=1e-8)).nan_to_num(0.0)
        self.last_auxk_loss.fill_(float(loss.detach()))
        return loss

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """(B, T, d): matryoshka-only; (B, 2, T, d): full + AuxK on x_cur."""
        if x.ndim == 3:
            z = self.encode(x)
            loss = self._matryoshka_loss(x, z)
            x_hat = self.decode_scale(z, self.T - 1)
            return loss, x_hat, z

        if x.ndim == 4 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]

            # Pre-activations (need pre_cur for AuxK)
            pre_cur = self._pre_activation(x_cur)
            # z_cur via the class's BatchTopK sparsity module
            z_cur = self.sparsity(pre_cur) if self.k is not None else F.relu(pre_cur)
            z_prev = self.encode(x_prev)

            l_matr = (self._matryoshka_loss(x_prev, z_prev)
                      + self._matryoshka_loss(x_cur, z_cur))

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for s in range(self.n_contr_scales):
                prefix_s = self.prefix_sum[s]
                l_contr = l_contr + (self.gamma ** s) * _info_nce(
                    z_cur[:, :prefix_s], z_prev[:, :prefix_s]
                )

            x_hat_full = self.decode_scale(z_cur, self.T - 1)
            l_auxk = self._auxk_loss(x_cur, z_cur, pre_cur, x_hat_full)

            total = l_matr + alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat_full, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )
