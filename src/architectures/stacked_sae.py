"""Stacked SAE — T independent TopK SAEs, one per token position.

The "per-layer SAEs" baseline from the crosscoders paper, adapted to
the temporal setting. Each position has its own SAE with independent
weights. Window-level L0 = k * T.

Ported from Aniket's original crosscoder models and Han's stacked_sae.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.base import ArchSpec, EvalOutput
from src.architectures.topk_sae import TopKSAE


class StackedSAE(nn.Module):
    """T independent TopK SAEs, one per position in a window.

    Input:  x in R^{B x T x d}
    Output: (recon_loss, x_hat, z)
        x_hat: (B, T, d) per-position reconstructions
        z: (B, T, h) per-position latent codes
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.saes = nn.ModuleList([TopKSAE(d_in, d_sae, k) for _ in range(T)])

    @torch.no_grad()
    def _normalize_decoder(self):
        for sae in self.saes:
            sae._normalize_decoder()

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) -> (recon_loss, x_hat, z)"""
        losses, x_hats, zs = [], [], []
        for t, sae in enumerate(self.saes):
            loss_t, x_hat_t, z_t = sae(x[:, t, :])
            losses.append(loss_t)
            x_hats.append(x_hat_t)
            zs.append(z_t)

        x_hat = torch.stack(x_hats, dim=1)
        z = torch.stack(zs, dim=1)
        loss = torch.stack(losses).mean()
        return loss, x_hat, z

    @property
    def decoder_directions(self) -> torch.Tensor:
        """(d_in, d_sae) decoder columns averaged across all T SAEs."""
        return torch.stack([sae.W_dec.data for sae in self.saes]).mean(dim=0)

    def decoder_directions_at(self, pos: int) -> torch.Tensor:
        """(d_in, d_sae) decoder columns for a specific position."""
        return self.saes[pos].W_dec.data


class StackedSAESpec(ArchSpec):
    """ArchSpec for the Stacked SAE (T independent SAEs)."""

    data_format = "window"

    def __init__(self, T: int):
        self.T = T
        self.name = f"Stacked T={T}"

    @property
    def n_decoder_positions(self):
        return self.T

    def create(self, d_in, d_sae, k, device):
        return StackedSAE(d_in, d_sae, self.T, k).to(device)

    def train(self, model, gen_fn, total_steps, batch_size, lr, device,
              log_every=500, grad_clip=1.0):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        log = {"loss": [], "l0": []}
        model.train()

        for step in range(total_steps):
            x = gen_fn(batch_size).to(device)
            loss, _, z = model(x)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model._normalize_decoder()

            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    l0_per_pos = (z > 0).float().sum(dim=-1).mean().item()
                    window_l0 = l0_per_pos * self.T
                log["loss"].append(loss.item())
                log["l0"].append(window_l0)

        model.eval()
        return log

    def eval_forward(self, model, x):
        loss, x_hat, z = model(x)
        se = (x_hat - x).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).mean(dim=1).sum().item()
        return EvalOutput(sum_se=se, sum_signal=signal, sum_l0=l0,
                          n_tokens=x.shape[0])

    def decoder_directions(self, model, pos=None):
        if pos is None:
            return model.decoder_directions
        return model.decoder_directions_at(pos)
