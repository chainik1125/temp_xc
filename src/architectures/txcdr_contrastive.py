"""TXCDR + Matryoshka H/L partition + InfoNCE on adjacent windows.

Phase 5.7 autoresearch candidate A2. Combines the two strongest wins
from the 25-arch benchmark:
  - TXCDR-T5 (best at mean_pool: 0.8064; best mean across both
    aggregations among "TXCDR" family).
  - Ye et al. 2025 InfoNCE contrastive loss, ported to MLC as
    `mlc_contrastive` and there gave the +0.8 pp lift that put MLC at
    the top of last_position.

Architecture (T = 5 default):
  - Base = TemporalCrosscoder(d_in, d_sae, T, k=k_win): window encoder
    (T, d_in) -> (d_sae) shared latent; per-position decoder W_dec
    (d_sae, T, d_in).
  - Matryoshka partition: first `h = d_sae // 2` latents are "high";
    the remainder are "low". No separate encoder — index convention.
  - Decode-high reconstruction adds a second MSE term using only z[:, :h]
    and W_dec[:h, :, :].
  - InfoNCE between adjacent T-windows: a pair generator yields
    (W_{t-1}, W_t) where W_t = x[t-T+1:t+1] and W_{t-1} = x[t-T:t]
    (one-token shift). Symmetric InfoNCE on L2-normalised z[:, :h]
    similarities. Same form as `mlc_contrastive`, just with "T-window"
    in place of "single token".

Inference is identical to vanilla TXCDR-T5 (a `(B, T, d_in)` window
encodes to `(B, d_sae)` via TopK), so the existing TXCDR probing path
in `run_probing.py::_encode_txcdr` works without changes.

Hypothesis: contrastive on adjacent windows pushes nearby windows to
share their high-level latents, complementary to TXCDR's per-position
decoder structure (which already lets a single latent fire at varying
strengths across T positions). If a feature truly encodes a
multi-token concept, both windows that span it should activate the
same high-level latent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.crosscoder import TemporalCrosscoder


class TXCDRContrastive(TemporalCrosscoder):
    """TXCDR with Matryoshka H/L partition + adjacent-window InfoNCE.

    Inference API identical to TemporalCrosscoder so probing reuses
    the txcdr encoder dispatch in run_probing.py.

    Args:
        d_in: residual-stream width (Gemma d_model = 2304).
        d_sae: dictionary size (18432 default in Phase 5).
        T: window size.
        k: window-level TopK budget.
        h: high-level latent prefix size. Defaults to d_sae // 2.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        h: int | None = None,
    ):
        super().__init__(d_in, d_sae, T, k)
        self.h = h if h is not None else d_sae // 2

    def decode_high_only(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from the first h latents only.

        z: (B, d_sae) -> (B, T, d_in)
        """
        z_h = z[:, : self.h]
        W_h = self.W_dec[: self.h, :, :]  # (h, T, d_in)
        return torch.einsum("bs,std->btd", z_h, W_h) + self.b_dec

    def forward(self, x: torch.Tensor, alpha: float = 0.1):
        """Dispatch on input shape.

        - `(B, T, d_in)`: matryoshka loss only (window-pair fallback).
        - `(B, 2, T, d_in)`: adjacent window pair [prev, cur] —
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
            x_prev = x[:, 0]  # (B, T, d)
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
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE on L2-normalised (B, h) latents."""
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
