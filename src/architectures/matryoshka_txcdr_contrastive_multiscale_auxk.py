"""Matryoshka-TXCDR multi-scale contrastive + AuxK loss for dead features.

Phase 6.1 agentic cycle A candidate (`agentic_txc_09_auxk`). Extends
`MatryoshkaTXCDRContrastiveMultiscale` (the Phase 5.7 winner,
`agentic_txc_02`) by porting the paper's AuxK auxiliary-reconstruction
loss from `tsae_paper.py`.

Rationale (Phase 6.1 briefing, 2026-04-23):

    agentic_txc_02 is top on Phase 5 sparse probing (0.80 AUC) but
    bottom on Phase 6 qualitative autointerp (2 / 8 semantic labels,
    alive fraction 0.39). Diagnosis: tsae_paper's 0.73 alive fraction
    comes mostly from its **AuxK auxiliary loss** — a parallel
    reconstruction from the top-k DEAD features that pulls them back
    into the active set. Our matryoshka-contrastive stack does not
    have this mechanism; features that die early stay dead.

This class preserves the full Phase 5.7 recipe (matryoshka + multi-
scale contrastive, γ=0.5, n_scales=3) and adds the AuxK term as a
cheap residual-reconstruction penalty on the current batch.

AuxK loss (adapted to the window-based TXC encoder):

    1. Track `num_tokens_since_fired` per-feature (d_sae,) buffer.
    2. After encoding x_cur, identify features that haven't fired in
       the last `dead_threshold_tokens` tokens (default 10M).
    3. Compute pre-ReLU activations of those dead features on x_cur.
    4. Top-aux_k those dead features per sample.
    5. Decode through the full-scale decoder W_decs[T-1] (no bias):
       aux_decode shape (B, T, d_in).
    6. Loss: MSE(x - x_hat_full, aux_decode) / variance(x - x_hat_full).
    7. Add `auxk_alpha * aux_loss` to the total.

Everything else (matryoshka reconstruction, multi-scale InfoNCE)
is inherited from the parent and unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
    MatryoshkaTXCDRContrastiveMultiscale,
)
from src.architectures.matryoshka_txcdr_contrastive import _info_nce


class MatryoshkaTXCDRContrastiveMultiscaleAuxK(MatryoshkaTXCDRContrastiveMultiscale):
    """agentic_txc_02 recipe + AuxK loss on dead features.

    Args (in addition to parent):
        aux_k: budget of dead features to recruit into the residual
            reconstruction per sample. Paper uses d_in // 2; we use
            a fixed 512 by default (a reasonable mid-point; d_sae
            is 18 432 and d_in is 2 304 so 512 / 18432 ≈ 2.8 %).
        dead_threshold_tokens: tokens-since-fired threshold for
            "dead" classification. Paper default 10 000 000.
        auxk_alpha: weight on the AuxK term in the total loss.
            Paper default 1 / 32.
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
        self.register_buffer(
            "last_auxk_loss", torch.tensor(-1.0)
        )
        self.register_buffer(
            "last_dead_count", torch.tensor(0, dtype=torch.long)
        )

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-TopK, pre-ReLU activations. Shape (B, d_sae).

        Mirrors `PositionMatryoshkaTXCDR.encode` up to but excluding
        the top-k + ReLU step.
        """
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def _auxk_loss(
        self,
        x_cur: torch.Tensor,        # (B, T, d_in)
        z_cur: torch.Tensor,        # (B, d_sae)
        pre_cur: torch.Tensor,      # (B, d_sae)  pre-TopK, pre-ReLU
        x_hat_full: torch.Tensor,   # (B, T, d_in) full-scale reconstruction
    ) -> torch.Tensor:
        """Compute AuxK auxiliary-reconstruction loss for dead features.

        Also updates `num_tokens_since_fired` in-place (side effect).
        """
        # Features that fired anywhere in this batch
        active_mask = (z_cur > 0).any(dim=0)
        n_tokens_in_batch = x_cur.shape[0] * x_cur.shape[1]

        # Update dead tracker (all features advance by n_tokens_in_batch,
        # features that fired this batch reset to 0).
        self.num_tokens_since_fired += n_tokens_in_batch
        self.num_tokens_since_fired[active_mask] = 0

        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)
        if n_dead == 0:
            self.last_auxk_loss.fill_(0.0)
            return torch.zeros((), device=x_cur.device, dtype=x_cur.dtype)

        k_aux = min(self.aux_k, n_dead)

        # ReLU the pre-activations (matches tsae_paper's post_relu),
        # then zero-out non-dead features.
        auxk_pre = F.relu(pre_cur)
        auxk_pre = auxk_pre.masked_fill(~dead_mask.unsqueeze(0), 0.0)

        # Top-k per-sample over dead features
        vals, idx = auxk_pre.topk(k_aux, dim=-1, sorted=False)
        auxk_buf = torch.zeros_like(auxk_pre)
        auxk_buf.scatter_(-1, idx, vals)

        # Decode through the full-scale decoder (no bias, per tsae_paper).
        W_full = self.W_decs[self.T - 1]            # (d_sae, T, d_in)
        aux_decode = torch.einsum("bs,std->btd", auxk_buf, W_full)

        # Residual of the primary full-scale reconstruction.
        # Detach so AuxK gradients don't flow through the primary decoder
        # via x_hat_full (the primary loss already shapes it).
        residual = x_cur - x_hat_full.detach()

        l2 = (residual - aux_decode).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        loss = (l2 / denom.clamp(min=1e-8)).nan_to_num(0.0)

        self.last_auxk_loss.fill_(float(loss.detach()))
        return loss

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """Same dispatch as parent; pair-window path adds AuxK on x_cur.

        - (B, T, d): matryoshka loss only, no AuxK (inference / warmup).
        - (B, 2, T, d): matryoshka(prev+cur) + α·multi_scale_InfoNCE + AuxK(cur).
        """
        if x.ndim == 3:
            z = self.encode(x)
            loss = self._matryoshka_loss(x, z)
            x_hat = self.decode_scale(z, self.T - 1)
            return loss, x_hat, z

        if x.ndim == 4 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]

            # Parent's encode() handles topk+ReLU; we also need the pre-
            # activation of x_cur for AuxK, so compute it once here.
            pre_cur = self._pre_activation(x_cur)
            if self.k is not None:
                vals, idx = pre_cur.topk(self.k, dim=-1)
                z_cur = torch.zeros_like(pre_cur)
                z_cur.scatter_(1, idx, F.relu(vals))
            else:
                z_cur = F.relu(pre_cur)
            z_prev = self.encode(x_prev)

            # Matryoshka reconstruction loss
            l_matr = (self._matryoshka_loss(x_prev, z_prev)
                      + self._matryoshka_loss(x_cur, z_cur))

            # Multi-scale contrastive
            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for s in range(self.n_contr_scales):
                prefix_s = self.prefix_sum[s]
                z_h_prev = z_prev[:, :prefix_s]
                z_h_cur = z_cur[:, :prefix_s]
                l_contr = l_contr + (self.gamma ** s) * _info_nce(z_h_cur, z_h_prev)

            # Full-scale reconstruction for AuxK residual
            x_hat_full = self.decode_scale(z_cur, self.T - 1)

            # AuxK on x_cur (mutates num_tokens_since_fired; detaches x_hat_full)
            l_auxk = self._auxk_loss(x_cur, z_cur, pre_cur, x_hat_full)

            total = l_matr + alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat_full, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )
