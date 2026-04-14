"""Crosscoder: custom nn.Module with per-position encoder and shared latent space."""

import torch
import torch.nn as nn

from src.v2_crosscoder_comparison.architectures.base import ArchLoss, BaseArchitecture
from src.v2_crosscoder_comparison.configs import ArchitectureConfig


class CrosscoderModule(nn.Module):
    """Custom crosscoder implementation.

    Architecture (reference: ckkissane/crosscoder-model-diff-replication):
        W_enc: (n_positions, d_model, d_sae) - per-position encoder
        W_dec: (d_sae, n_positions, d_model) - per-position decoder
        b_enc: (d_sae,) - shared encoder bias
        b_dec: (n_positions, d_model) - per-position decoder bias

    Encode: z = TopK(einsum("btd,tds->bs", x, W_enc) + b_enc)
    Decode: x_hat = einsum("bs,std->btd", z, W_dec) + b_dec
    """

    def __init__(
        self,
        n_positions: int,
        d_model: int,
        d_sae: int,
        top_k: int,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.d_model = d_model
        self.d_sae = d_sae
        self.top_k = top_k

        self.W_enc = nn.Parameter(
            torch.randn(n_positions, d_model, d_sae) * (1.0 / d_model**0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, n_positions, d_model) * (1.0 / d_sae**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(n_positions, d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with TopK activation.

        Args:
            x: (batch, n_positions, d_model)

        Returns:
            z: (batch, d_sae) - sparse latent activations
        """
        # Sum over positions and project: (batch, d_sae)
        pre_act = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

        # TopK: keep top_k activations, zero out the rest
        top_vals, top_idx = pre_act.topk(self.top_k, dim=-1)
        z = torch.zeros_like(pre_act)
        z.scatter_(1, top_idx, torch.relu(top_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to per-position reconstructions.

        Args:
            z: (batch, d_sae)

        Returns:
            x_hat: (batch, n_positions, d_model)
        """
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class Crosscoder(BaseArchitecture):
    """Crosscoder architecture wrapper."""

    def __init__(self, config: ArchitectureConfig, d_model: int, n_positions: int = 2):
        super().__init__(config, d_model)
        self.module = CrosscoderModule(
            n_positions=n_positions,
            d_model=d_model,
            d_sae=config.d_sae,
            top_k=config.top_k,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.module.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.module.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def get_losses(self, x: torch.Tensor) -> ArchLoss:
        z = self.module.encode(x)
        x_hat = self.module.decode(z)

        recon_loss = (x - x_hat).pow(2).sum(dim=(-2, -1)).mean().item()
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        sparsity_loss = self.config.l1_coefficient * z.abs().sum(dim=-1).mean().item()

        return ArchLoss(
            total_loss=recon_loss + sparsity_loss,
            recon_loss=recon_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    def get_decoder_weights(self) -> torch.Tensor:
        """Get decoder weights: (d_sae, n_positions, d_model)."""
        return self.module.W_dec.data.detach()
