"""Time×Layer crosscoder + InfoNCE on adjacent-window latent slabs.

Phase 5.7 autoresearch candidate A10. Tests whether the contrastive-
on-adjacent-latents win from A2 (`txcdr_contrastive_t5`) and
A3 (`matryoshka_txcdr_contrastive_t5`) generalises to the joint
(T, L) latent space used by `time_layer_crosscoder_t5`.

Base = TimeLayerCrosscoder: encode produces `(B, T, L, d_sae)` with
a global TopK over the flattened `(T, L, d_sae)` grid. Decoder is
per-(t, l) so each of the T × L cells gets its own reconstruction.

Contrastive design:
  - Pair generator yields adjacent T-windows `(W_prev, W_cur)` shaped
    `(B, 2, T, L, d_in)` with `W_cur` shifted by 1 token vs W_prev.
  - Encode each to its full `(B, T, L, d_sae)` latent.
  - InfoNCE target: the first `h` features along the feature axis,
    aggregated over (t, l). Concretely we average the latent slab
    `z[:, :, :, :h]` over (T, L) to get a single (B, h) "high-level"
    summary per window, then sym InfoNCE on that.

Why the average-over-(T, L)? The contrastive needs a per-window
"semantic summary" that should be shift-1 stable. A feature that fires
at (t, l) in W_prev is likely to fire at (t-1, l) in W_cur (shift
by 1); averaging over t collapses that shift so the two latent
summaries compare meaningfully. Averaging over l collapses the
layer-axis noise.

Note: d_sae=8192 for time_layer (vs 18432 for MLC/TXCDR) due to VRAM
constraints on the base arch. With h = d_sae // 2 = 4096 the
contrastive head is the same shape as for A2/A3 modulo the prefix
size.

Inference encode API matches TimeLayerCrosscoder so probing reuses
the existing time_layer path in run_probing.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.time_layer_crosscoder import TimeLayerCrosscoder


class TimeLayerContrastive(TimeLayerCrosscoder):
    """Time×Layer crosscoder + InfoNCE on (T, L)-averaged high prefix.

    Args:
        d_in, d_sae, T, L, k: same as TimeLayerCrosscoder.
        h: prefix size for the contrastive head (first h features).
            Defaults to d_sae // 2.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        L: int,
        k: int | None,
        h: int | None = None,
    ):
        super().__init__(d_in, d_sae, T, L, k)
        self.h = h if h is not None else d_sae // 2

    def _contrastive_summary(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, L, d_sae) -> (B, h). Average over (T, L) on high prefix."""
        return z[:, :, :, : self.h].mean(dim=(1, 2))

    def forward(self, x: torch.Tensor, alpha: float = 0.1):
        """Dispatch on input shape.

        - `(B, T, L, d_in)`: plain reconstruction loss (no pair available).
        - `(B, 2, T, L, d_in)`: adjacent-window pair [prev, cur] —
          recon(prev) + recon(cur) + α · InfoNCE.

        Returns `(loss, x_hat_cur, z_cur)`.
        """
        if x.ndim == 4:
            z = self.encode(x)
            x_hat = self.decode(z)
            loss = (x_hat - x).pow(2).sum(dim=-1).mean()
            return loss, x_hat, z

        if x.ndim == 5 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]
            z_prev = self.encode(x_prev)
            z_cur = self.encode(x_cur)
            x_hat_prev = self.decode(z_prev)
            x_hat_cur = self.decode(z_cur)
            l_recon = (
                (x_hat_prev - x_prev).pow(2).sum(dim=-1).mean()
                + (x_hat_cur - x_cur).pow(2).sum(dim=-1).mean()
            )
            s_prev = self._contrastive_summary(z_prev)
            s_cur = self._contrastive_summary(z_cur)
            l_contr = _info_nce(s_cur, s_prev)
            total = l_recon + alpha * l_contr
            return total, x_hat_cur, z_cur

        raise ValueError(
            f"Expected (B, T, L, d) or (B, 2, T, L, d), got {tuple(x.shape)}"
        )


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
