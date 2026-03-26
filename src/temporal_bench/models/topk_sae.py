"""Independent TopK SAE baseline.

Processes each position independently. Receives (B, T, d) and reshapes
to (B*T, d) internally.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


class TopKSAE(TemporalAE):
    """Standard TopK sparse autoencoder (position-independent baseline).

    Architecture:
        z = TopK(ReLU(W_enc @ (x - b_dec) + b_enc), k)
        x_hat = W_dec @ z + b_dec
    """

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def _encode_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Encode a flat (N, d) tensor to sparse (N, m) codes."""
        pre = F.relu((x_flat - self.b_dec) @ self.W_enc + self.b_enc)
        _, topk_idx = pre.topk(self.k, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, 1.0)
        return pre * mask

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)

        z = self._encode_flat(x_flat)
        x_hat_flat = z @ self.W_dec + self.b_dec

        recon_loss = (x_flat - x_hat_flat).pow(2).sum(dim=-1).mean()
        l0 = (z != 0).float().sum(dim=-1).mean().item()

        return ModelOutput(
            x_hat=x_hat_flat.reshape(B, T, d),
            latents=z.reshape(B, T, self.d_sae),
            loss=recon_loss,
            metrics={"recon_loss": recon_loss.item(), "l0": l0},
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        return self.W_dec.T  # (d, m)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()
