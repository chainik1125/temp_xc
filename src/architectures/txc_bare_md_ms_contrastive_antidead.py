"""Part B H13 (mdms = multi-distance × multi-scale).

Stacks H8's multi-distance contrastive with H7's multi-scale prefix
contrastive: at every shift distance s, compute InfoNCE at multiple
matryoshka prefix lengths {h, 2h, 3h, ...}.

Total contrastive = Σ_s Σ_p w_s · γ^p · InfoNCE(z_anchor[:, :p_len], z_pos_s[:, :p_len])

  - shifts ∈ {1, 2, ...}     — distance-pair diversity
  - p_len = (p+1) · h_size   — matryoshka prefix granularity
  - w_s = 1/(1+s)            — inverse-distance weighting
  - γ^p                      — multi-scale geometric decay (0.5 default)

Why this might top H8:

H7 = multi-scale at shift=1 only (3 terms).
H8 = single-scale at shifts={1,2} (2 terms).
H13 = multi-scale at shifts={1,2}     (6 terms).

If multi-distance and multi-scale are orthogonal contrastive pressures
(local distance-invariance vs cross-scale feature consistency), stacking
them should add information without redundancy.

Risk: the 6 InfoNCE terms could dominate the recon loss; α may need
re-tuning, or the multi-scale γ decay may need to be steeper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.txc_bare_matryoshka_contrastive_antidead import (
    TXCBareMatryoshkaContrastiveAntidead, _info_nce,
)


class TXCBareMDxMSContrastiveAntidead(TXCBareMatryoshkaContrastiveAntidead):
    """Multi-distance × multi-scale contrastive + matryoshka recon + anti-dead.

    Forward signature matches `TXCBareMultiDistanceContrastiveAntidead`:
    accepts (B, 1+K, T, d) where K = len(shifts). For each shift k,
    reconstruct anchor + positive + add a multi-scale InfoNCE chain.
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int,
        shifts: tuple[int, ...] = (1, 2),
        weights: tuple[float, ...] | None = None,
        n_contr_scales: int = 3,
        gamma: float = 0.5,
        matryoshka_h_size: int | None = None,
        alpha: float = 1.0,
        contr_prefix: int | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            matryoshka_h_size=matryoshka_h_size,
            alpha=alpha,
            contr_prefix=contr_prefix,
            aux_k=aux_k,
            dead_threshold_tokens=dead_threshold_tokens,
            auxk_alpha=auxk_alpha,
        )
        self.shifts = tuple(shifts)
        if weights is None:
            weights = tuple(1.0 / (1.0 + s) for s in shifts)
        self.loss_weights = tuple(weights)
        assert len(self.loss_weights) == len(self.shifts)
        self.n_contr_scales = int(n_contr_scales)
        self.gamma = float(gamma)
        # Validate scales fit within d_sae.
        max_p = self.n_contr_scales * self.contr_prefix
        if max_p > d_sae:
            raise ValueError(
                f"n_contr_scales={n_contr_scales} × contr_prefix="
                f"{self.contr_prefix} = {max_p} exceeds d_sae={d_sae}"
            )

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 3:
            return super().forward(x, alpha=alpha)
        if x.ndim == 4 and x.shape[1] == 2:
            return super().forward(x, alpha=alpha)
        if not (x.ndim == 4 and x.shape[1] > 2):
            raise ValueError(
                f"Expected (B, T, d), (B, 2, T, d) or (B, 1+K, T, d); got {tuple(x.shape)}"
            )

        K = x.shape[1] - 1
        if K != len(self.shifts):
            raise ValueError(
                f"expected {1+len(self.shifts)} window slots for shifts="
                f"{self.shifts}, got {x.shape[1]}"
            )

        x_anchor = x[:, 0]
        pre_a = self._pre_activation(x_anchor)
        vals_a, idx_a = pre_a.topk(self.k, dim=-1)
        z_anchor = torch.zeros_like(pre_a)
        z_anchor.scatter_(1, idx_a, F.relu(vals_a))

        l_recon = self._recon_loss(x_anchor, z_anchor)

        l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
        h = self.contr_prefix
        gamma_powers = [self.gamma ** s for s in range(self.n_contr_scales)]
        for k_idx, w_s in enumerate(self.loss_weights):
            x_pos = x[:, 1 + k_idx]
            pre_p = self._pre_activation(x_pos)
            vals_p, idx_p = pre_p.topk(self.k, dim=-1)
            z_pos = torch.zeros_like(pre_p)
            z_pos.scatter_(1, idx_p, F.relu(vals_p))
            l_recon = l_recon + self._recon_loss(x_pos, z_pos)
            if eff_alpha > 0.0:
                for p in range(self.n_contr_scales):
                    p_len = (p + 1) * h
                    l_contr = l_contr + w_s * gamma_powers[p] * _info_nce(
                        z_anchor[:, :p_len], z_pos[:, :p_len]
                    )

        x_hat = self.decode(z_anchor)
        l_auxk = self._update_dead_and_auxk(x_anchor, x_hat, pre_a, z_anchor)
        total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
        return total, x_hat, z_anchor
