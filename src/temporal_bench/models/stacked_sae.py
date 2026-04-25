"""Stacked SAE baseline: T independent TopK SAEs, one per window position.

Each position has its own encoder/decoder pair. Total window sparsity
L0 = k * T (k active latents per position). Parameter count scales
with T, and there is no information sharing across positions at either
training or inference time.

This matches the *per-position TopK* variant described in the midterm
report §2.2: "T independent TopK SAE modules, each selecting k active
latents per position." Both existing implementations that produced the
report's Fig 5 data use the same per-position-TopK design:
    origin/andre:src/v2_crosscoder_comparison/architectures/stacked_sae.py
    origin/aniket-runpod:src/bench/architectures/stacked_sae.py

Variants NOT implemented here (noted for future reference):

* **Concat-then-TopK** — T per-position encoders produce a length-(T*d_sae)
  concatenated latent; a single TopK(k*T) is taken over that concatenation.
  This lets the model redistribute its sparsity budget across positions
  (a position with low activation energy can receive 0 active latents
  while another receives 2k). Same parameter count as this Stacked SAE
  and same matched window L0 = k*T, but strictly more expressive.
  Useful as an intermediate between Stacked SAE and TXCDR — it tests
  whether cross-position *budget* alone helps, independent of TXCDR's
  cross-position *feature pooling*. Not in the report.

* **BatchTopK** — TopK is applied over a flattened batch dimension rather
  than per-token, so the *mean* L0 is k but the per-token L0 fluctuates.
  Implementation on `origin/wip/aliased-benchmark-runpod` in
  `src/temporal_bench/models/batchtopk_sae.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ModelOutput, TemporalAE
from .topk_sae import TopKSAE


class StackedSAE(TemporalAE):
    """T independent TopK SAEs, one per position in the window."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.saes = nn.ModuleList([TopKSAE(d_in, d_sae, k) for _ in range(T)])

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        assert T == self.T, f"Expected T={self.T}, got T={T}"

        x_hats = []
        zs = []
        losses = []
        l0s = []

        for t, sae in enumerate(self.saes):
            # Reuse TopKSAE forward by passing a (B, 1, d) slice.
            out_t = sae(x[:, t : t + 1, :])
            x_hats.append(out_t.x_hat[:, 0, :])
            zs.append(out_t.latents[:, 0, :])
            losses.append(out_t.loss)
            l0s.append(out_t.metrics["l0"])

        x_hat = torch.stack(x_hats, dim=1)
        z = torch.stack(zs, dim=1)
        loss = torch.stack(losses).mean()
        l0_mean = sum(l0s) / T

        return ModelOutput(
            x_hat=x_hat,
            latents=z,
            loss=loss,
            metrics={"recon_loss": loss.item(), "l0": l0_mean},
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        """(d, m) decoder columns.

        pos=None averages per-position decoders into a single matrix,
        matching the convention used by TemporalCrosscoder on this branch.
        pos=t returns the t-th position's decoder directly.
        """
        if pos is not None:
            return self.saes[pos].W_dec.T
        stacked = torch.stack([sae.W_dec for sae in self.saes], dim=0)  # (T, m, d)
        return stacked.mean(dim=0).T  # (d, m)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        for sae in self.saes:
            sae.normalize_decoder()

    @property
    def n_positions(self) -> int:
        return self.T
