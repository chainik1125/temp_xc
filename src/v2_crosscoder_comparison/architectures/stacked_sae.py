"""Stacked SAE: two independent SAE Lens TopKTrainingSAEs, one per position."""

import torch

from src.shared.train_sae import create_sae
from src.v2_crosscoder_comparison.architectures.base import ArchLoss, BaseArchitecture
from src.v2_crosscoder_comparison.configs import ArchitectureConfig


class StackedSAE(BaseArchitecture):
    """Two independent TopK SAEs, one per position.

    Position 0 -> SAE 0, Position 1 -> SAE 1.
    Each trained independently via SAE Lens.
    """

    def __init__(self, config: ArchitectureConfig, d_model: int):
        super().__init__(config, d_model)
        self.saes = [
            create_sae(d_in=d_model, d_sae=config.d_sae, k=config.top_k),
            create_sae(d_in=d_model, d_sae=config.d_sae, k=config.top_k),
        ]

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode each position independently.

        Returns list of latent activations, one per position.
        """
        return [self.saes[t].encode(x[:, t, :]) for t in range(2)]

    def decode(self, z_list: list[torch.Tensor]) -> torch.Tensor:
        recons = []
        for t in range(2):
            recon_t = z_list[t] @ self.saes[t].W_dec + self.saes[t].b_dec
            recons.append(recon_t)
        return torch.stack(recons, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recons = []
        for t in range(2):
            recon_t = self.saes[t](x[:, t, :])
            recons.append(recon_t)
        return torch.stack(recons, dim=1)

    def get_losses(self, x: torch.Tensor) -> ArchLoss:
        total_recon = 0.0
        total_l0 = 0.0
        for t in range(2):
            x_t = x[:, t, :]
            recon_t = self.saes[t](x_t)
            z_t = self.saes[t].encode(x_t)
            total_recon += (x_t - recon_t).pow(2).sum(dim=-1).mean().item()
            total_l0 += (z_t > 0).float().sum(dim=-1).mean().item()

        avg_recon = total_recon / 2
        avg_l0 = total_l0 / 2

        return ArchLoss(
            total_loss=avg_recon,
            recon_loss=avg_recon,
            sparsity_loss=0.0,
            l0=avg_l0,
        )

    def get_decoder_weights(self) -> list[torch.Tensor]:
        """Get decoder weights for each position.

        Returns:
            List of decoder weight tensors, each (d_sae, d_model).
        """
        return [sae.W_dec.data.detach() for sae in self.saes]
