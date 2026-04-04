"""BatchTopK SAE baseline.

Training keeps the top k activations per token across the flattened token
batch. Evaluation switches to a learned threshold so the model can encode
tokens independently.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


class BatchTopKSAE(TemporalAE):
    """Tokenwise BatchTopK sparse autoencoder."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        *,
        min_act_ema: float = 0.999,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.min_act_ema = min_act_ema

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("expected_min_act", torch.zeros(1))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def _encode_flat(self, x_flat: torch.Tensor) -> torch.Tensor:
        pre = F.relu((x_flat - self.b_dec) @ self.W_enc + self.b_enc)

        if self.training:
            total_k = min(self.k * x_flat.shape[0], pre.numel())
            flat_pre = pre.flatten()
            topk_vals, topk_idx = flat_pre.topk(total_k, dim=-1)
            sparse = torch.zeros_like(flat_pre)
            sparse.scatter_(0, topk_idx, topk_vals)

            with torch.no_grad():
                min_activation = (
                    topk_vals.min()
                    if topk_vals.numel() > 0
                    else torch.tensor(0.0, device=pre.device, dtype=pre.dtype)
                )
                self.expected_min_act.mul_(self.min_act_ema).add_(
                    (1.0 - self.min_act_ema) * min_activation
                )

            return sparse.view_as(pre)

        return pre * (pre > self.expected_min_act)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)

        z = self._encode_flat(x_flat)
        x_hat_flat = z @ self.W_dec + self.b_dec
        recon_loss = (x_flat - x_hat_flat).pow(2).sum(dim=-1).mean()
        metrics = {}
        if self.collect_metrics:
            metrics = {
                "recon_loss": recon_loss.item(),
                "l0": (z > 0).float().sum(dim=-1).mean().item(),
            }

        return ModelOutput(
            x_hat=x_hat_flat.reshape(B, T, d),
            latents=z.reshape(B, T, self.d_sae),
            loss=recon_loss,
            metrics=metrics,
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        return self.W_dec.T

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()

    def reconstruction_bias(self, pos: int | None = None) -> torch.Tensor:
        return self.b_dec
