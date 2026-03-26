"""Temporal crosscoder (ckkissane-style shared-latent architecture).

Encodes a window of T positions into a single shared sparse latent vector,
then decodes back to T positions using per-position decoder weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


class TemporalCrosscoder(TemporalAE):
    """Shared-latent temporal crosscoder.

    Architecture:
        z = TopK(sum_t W_enc[t] @ x_t + b_enc, k_total)
        x_hat_t = W_dec[t] @ z + b_dec[t]

    The encoder sums per-position projections into a single latent.
    Total active latents = k_total (typically k * T for fair comparison).
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k_per_pos: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k_per_pos = k_per_pos
        self.k_total = k_per_pos * T

        # Per-position encoder weights: (T, d_in, d_sae)
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Per-position decoder weights: (T, d_sae, d_in)
        self.W_dec = nn.Parameter(torch.empty(T, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc[t], a=math.sqrt(5))
            with torch.no_grad():
                self.W_dec.data[t] = self.W_enc.data[t].T
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        # Joint norm across (T, d) for each atom
        # W_dec: (T, d_sae, d_in) -> norm over (T, d_in) for each latent
        norms = self.W_dec.data.pow(2).sum(dim=(0, 2), keepdim=True).sqrt().clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        # Encode: sum per-position projections
        # x: (B, T, d), W_enc: (T, d, m) -> (B, T, m) -> sum over T -> (B, m)
        pre = torch.einsum("btd,tdm->bm", x, self.W_enc) + self.b_enc
        pre = F.relu(pre)

        # TopK on shared latent
        _, topk_idx = pre.topk(self.k_total, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, 1.0)
        z = pre * mask  # (B, m)

        # Decode per-position
        # z: (B, m), W_dec: (T, m, d) -> (B, T, d)
        x_hat = torch.einsum("bm,tmd->btd", z, self.W_dec) + self.b_dec

        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l0 = (z != 0).float().sum(dim=-1).mean().item()

        return ModelOutput(
            x_hat=x_hat,
            latents=z.unsqueeze(1).expand(B, T, self.d_sae),
            loss=recon_loss,
            metrics={"recon_loss": recon_loss.item(), "l0": l0},
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        if pos is not None:
            return self.W_dec[pos].T  # (d, m)
        # Average across positions
        return self.W_dec.mean(dim=0).T  # (d, m)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()

    @property
    def n_positions(self) -> int:
        return self.T
