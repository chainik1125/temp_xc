"""Naive SAE: single SAE Lens TopKTrainingSAE treating all positions independently."""

import torch

from src.shared.train_sae import create_sae
from src.v2_crosscoder_comparison.architectures.base import ArchLoss, BaseArchitecture
from src.v2_crosscoder_comparison.configs import ArchitectureConfig


class NaiveSAE(BaseArchitecture):
    """Single TopK SAE applied to all positions by flattening.

    Flattens (batch, n_positions, d) -> (batch*n_positions, d), runs through
    a single SAE Lens TopKTrainingSAE, then reshapes back.
    """

    def __init__(self, config: ArchitectureConfig, d_model: int):
        super().__init__(config, d_model)
        self.sae = create_sae(
            d_in=d_model,
            d_sae=config.d_sae,
            k=config.top_k,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_pos, d = x.shape
        flat = x.reshape(batch * n_pos, d)
        return self.sae.encode(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch*n_pos, d_sae)
        # Decode: W_dec @ z + b_dec
        return z @ self.sae.W_dec + self.sae.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_pos, d = x.shape
        flat = x.reshape(batch * n_pos, d)
        recon = self.sae(flat)
        return recon.reshape(batch, n_pos, d)

    def get_losses(self, x: torch.Tensor) -> ArchLoss:
        batch, n_pos, d = x.shape
        flat = x.reshape(batch * n_pos, d)
        recon = self.sae(flat)
        z = self.sae.encode(flat)

        recon_loss = (flat - recon).pow(2).sum(dim=-1).mean().item()
        l0 = (z > 0).float().sum(dim=-1).mean().item()

        return ArchLoss(
            total_loss=recon_loss,
            recon_loss=recon_loss,
            sparsity_loss=0.0,
            l0=l0,
        )

    def get_decoder_weights(self) -> torch.Tensor:
        return self.sae.W_dec.data.detach()
