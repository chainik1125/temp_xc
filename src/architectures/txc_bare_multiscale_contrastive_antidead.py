"""Part B H7b: bare TXC + anti-dead stack + matryoshka H/L + **MULTI-SCALE** InfoNCE.

Combines the winning recipes from Phase 5.7 and Phase 6.2:
- **Phase 5.7 agentic_txc_02 (cycle 02)**: matryoshka_txcdr_contrastive
  with InfoNCE at N nested prefix lengths (scales 1, 2, 3 tokens at
  T=5) with γ geometric decay. Best mean_pool SAE in Phase 5.7.
- **Phase 6.2 C3 / Track 2**: bare TXC encoder + anti-dead stack
  (AuxK loss on dead features, unit-norm decoder, decoder-parallel
  gradient removal, geometric-median b_dec init). Reaches 73% alive
  features vs 30% in naive port. Phase 6.2 C3 added single-scale
  contrastive — hit 0.7834 lp, beating agentic_txc_02 (0.7749 3s lp).

This arch fuses them: takes Phase 6.2 C3's base and replaces the
single-scale InfoNCE with agentic_txc_02's multi-scale InfoNCE
(scales 1, 2, 3 matryoshka sub-windows with γ=0.5 decay).

Hypothesis: if anti-dead alone improves +0.009 lp over agentic_txc_02,
and multi-scale contrastive was agentic_txc_02's key recipe, the
combination should push further — potentially topping the benchmark.

Inherits everything from TXCBareMatryoshkaContrastiveAntidead:
- per-position W_enc + W_dec (same encoder as TemporalCrosscoder)
- full anti-dead stack
- matryoshka H/L recon (can be disabled)
- geometric-median b_dec init

Overrides the contrastive term to use multi-scale InfoNCE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.txc_bare_matryoshka_contrastive_antidead import (
    TXCBareMatryoshkaContrastiveAntidead, _info_nce,
)


class TXCBareMultiscaleContrastiveAntidead(TXCBareMatryoshkaContrastiveAntidead):
    """Bare TXC + anti-dead stack + matryoshka H/L + multi-scale InfoNCE.

    Args:
        d_in, d_sae, T, k: same as parent.
        matryoshka_h_size: H prefix for matryoshka recon. Default d_sae//5.
        alpha: outer contrastive loss weight. 0 disables.
        n_contr_scales: number of matryoshka scales to compute InfoNCE on.
            Default min(3, T). Scales are matryoshka sub-window sizes
            1, 2, ..., n_contr_scales.
        gamma: geometric decay across scales. loss_s = γ^s · InfoNCE(s).
            Default 0.5 (Phase 5.7 cycle 02 optimum).
        scale_prefix_fraction: fraction of d_sae used as prefix for each
            contrastive scale. Default 0.2 (d_sae // 5), matching the
            matryoshka_txcdr_contrastive_multiscale convention.
        aux_k, dead_threshold_tokens, auxk_alpha: same as parent.
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int,
        matryoshka_h_size: int | None = None,
        alpha: float = 1.0,
        n_contr_scales: int = 3,
        gamma: float = 0.5,
        scale_prefix_fraction: float = 0.2,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        if matryoshka_h_size is None:
            matryoshka_h_size = int(d_sae * scale_prefix_fraction)
        super().__init__(
            d_in, d_sae, T, k,
            matryoshka_h_size=matryoshka_h_size,
            alpha=alpha,
            contr_prefix=matryoshka_h_size,
            aux_k=aux_k,
            dead_threshold_tokens=dead_threshold_tokens,
            auxk_alpha=auxk_alpha,
        )
        self.n_contr_scales = min(int(n_contr_scales), int(T))
        self.gamma = float(gamma)
        # Contrastive prefixes follow matryoshka_txcdr_contrastive_multiscale
        # convention: prefix_s = (s+1) * d_sae / T_scales. But we need an
        # arbitrary cap at n_contr_scales. For simplicity we use the first
        # n_contr_scales equal slices of size d_sae // n_contr_scales.
        # Alternatively, nested prefixes at matryoshka_h_size, 2·h_size, 3·h_size.
        self._contr_prefixes = [
            min(d_sae, (s + 1) * matryoshka_h_size)
            for s in range(self.n_contr_scales)
        ]

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Accepts (B, T, d) single-window or (B, 2, T, d) pair.

        Same structure as parent but with multi-scale InfoNCE:
            L_contr = Σ_{s=0}^{n_contr_scales-1} γ^s · InfoNCE(
                z_cur[:, :prefix_s], z_prev[:, :prefix_s]
            )
        """
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 3:
            # Single-window path (no pair): same as parent's single-window
            # path — matryoshka recon + AuxK, contrastive skipped.
            pre = self._pre_activation(x)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            x_hat = self.decode(z)
            l_recon = self._recon_loss(x, z)
            l_auxk = self._update_dead_and_auxk(x, x_hat, pre, z)
            total = l_recon + self.auxk_alpha * l_auxk
            return total, x_hat, z

        if x.ndim == 4 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]

            pre_prev = self._pre_activation(x_prev)
            vals_p, idx_p = pre_prev.topk(self.k, dim=-1)
            z_prev = torch.zeros_like(pre_prev)
            z_prev.scatter_(1, idx_p, F.relu(vals_p))

            pre_cur = self._pre_activation(x_cur)
            vals_c, idx_c = pre_cur.topk(self.k, dim=-1)
            z_cur = torch.zeros_like(pre_cur)
            z_cur.scatter_(1, idx_c, F.relu(vals_c))

            l_recon = (
                self._recon_loss(x_prev, z_prev)
                + self._recon_loss(x_cur, z_cur)
            )

            # Multi-scale InfoNCE
            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            if eff_alpha > 0.0 and self.n_contr_scales > 0:
                for s, prefix in enumerate(self._contr_prefixes):
                    w = self.gamma ** s
                    l_contr = l_contr + w * _info_nce(
                        z_cur[:, :prefix], z_prev[:, :prefix]
                    )

            x_hat_cur = self.decode(z_cur)
            l_auxk = self._update_dead_and_auxk(x_cur, x_hat_cur, pre_cur, z_cur)

            total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat_cur, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )
