"""MLC + Matryoshka H/L partition + InfoNCE contrastive — Ye et al. 2025 port.

Directly generalises `TemporalContrastiveSAE` (single-token SAE) to a
layer-axis crosscoder base. The architecture is MLC's (shared latent across
L residual-stream layers) with two additions:

  - **Matryoshka partition**: the first `h = d_sae // 2` latent indices
    are "high-level"; remaining are "low-level" (index convention only,
    no separate encoder).
  - **Contrastive loss**: symmetric InfoNCE between adjacent-token pairs'
    high-level latents. At training, inputs come in pairs `(x_t, x_{t-1})`
    both shaped `(B, L, d_in)`. The InfoNCE operates on the B×B similarity
    matrix of L2-normalised `z_cur[:, :h]` vs `z_prev[:, :h]`.

Encode is identical to MLC's (for inference and probing): `(B, L, d_in)
→ (B, d_sae)` with TopK. Probing therefore reuses the MLC adapter path in
`experiments/phase5_downstream_utility/probing/run_probing.py`.

Reference: Ye et al. 2025 §3.2, adapted from single-token SAE → 5-layer
crosscoder. Hypothesis (speculative, per Han 2026-04-20): adjacent-token
InfoNCE on layer-crosscoder high-level latents might carry the
"semantic continuity" signal while keeping MLC's layer-redundancy
advantage.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.mlc import MultiLayerCrosscoder


class MLCContrastive(MultiLayerCrosscoder):
    """Layer crosscoder with Matryoshka H/L partition + per-token InfoNCE.

    Inference API identical to MultiLayerCrosscoder so probing reuses the
    MLC encoder dispatch in run_probing.py.

    Args:
        d_in: residual-stream width (Gemma d_model = 2304).
        d_sae: dictionary size (18432 default in Phase 5).
        n_layers: layers in window (5 for L11-L15).
        k: window-level TopK budget (100 default).
        h: "high-level" latent prefix size. Defaults to d_sae // 2.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        n_layers: int,
        k: int | None,
        h: int | None = None,
    ):
        super().__init__(d_in, d_sae, n_layers, k)
        self.h = h if h is not None else d_sae // 2

    def decode_high_only(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from the first h latents only.

        z: (B, d_sae) -> (B, L, d_in)
        """
        z_h = z[:, : self.h]
        W_h = self.W_dec[: self.h, :, :]  # (h, L, d_in)
        return torch.einsum("bs,sld->bld", z_h, W_h) + self.b_dec

    def forward(self, x: torch.Tensor, alpha: float = 0.1):
        """Dispatch on input shape.

        - `(B, L, d_in)`: matryoshka loss only (single-token fallback).
        - `(B, 2, L, d_in)`: paired adjacent tokens [prev, cur] —
          full matryoshka + InfoNCE contrastive loss.

        Returns `(loss, x_hat_cur, z_cur)` to match the training-loop
        convention.
        """
        if x.ndim == 3:
            z = self.encode(x)
            x_hat_full = self.decode(z)
            x_hat_high = self.decode_high_only(z)
            per_tok = x.shape[-1]
            loss_full = (x_hat_full - x).pow(2).sum(dim=-1).mean() / per_tok
            loss_high = (x_hat_high - x).pow(2).sum(dim=-1).mean() / per_tok
            return loss_full + loss_high, x_hat_full, z

        if x.ndim == 4 and x.shape[1] == 2:
            x_prev = x[:, 0]  # (B, L, d)
            x_cur = x[:, 1]
            z_prev = self.encode(x_prev)
            z_cur = self.encode(x_cur)
            per_tok = x.shape[-1]

            l_full_prev = (self.decode(z_prev) - x_prev).pow(2).sum(-1).mean() / per_tok
            l_high_prev = (self.decode_high_only(z_prev) - x_prev).pow(2).sum(-1).mean() / per_tok
            l_full_cur = (self.decode(z_cur) - x_cur).pow(2).sum(-1).mean() / per_tok
            l_high_cur = (self.decode_high_only(z_cur) - x_cur).pow(2).sum(-1).mean() / per_tok
            l_matr = l_full_prev + l_high_prev + l_full_cur + l_high_cur

            z_h_prev = z_prev[:, : self.h]
            z_h_cur = z_cur[:, : self.h]
            l_contr = _info_nce(z_h_cur, z_h_prev)

            total = l_matr + alpha * l_contr
            return total, self.decode(z_cur), z_cur

        raise ValueError(
            f"Expected (B, L, d) or (B, 2, L, d), got {tuple(x.shape)}"
        )


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE on L2-normalised (B, h) latents."""
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
