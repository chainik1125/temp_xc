"""Stacked SAE: T independent SAEs, one per token position.

The "per-layer SAEs" baseline from the crosscoders paper, adapted to the
temporal setting. Each position has its own TopKSAE with independent weights.
Each position gets k active latents, so window-level L0 = k * T.

Ported from Andre Shportko's temporal_crosscoders/models.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class _PositionSAE(nn.Module):
    """Single-position TopK SAE (internal, used by StackedSAE)."""

    def __init__(self, d_in: int, d_sae: int, k: int | None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def forward(self, x: torch.Tensor):
        """x: (B, d) → (x_hat, z)"""
        x_c = x - self.b_dec
        pre = x_c @ self.W_enc.T + self.b_enc
        if self.k is not None:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, topk_idx, F.relu(topk_vals))
        else:
            z = F.relu(pre)
        x_hat = z @ self.W_dec.T + self.b_dec
        return x_hat, z


class StackedSAE(nn.Module):
    """T independent SAEs, one per position in a window.

    Input:  x ∈ R^{B × T × d}
    Output: (recon_loss, x_hat, z)
        recon_loss: scalar mean reconstruction loss
        x_hat: (B, T, d) per-position reconstructions
        z: (B, T, h) per-position latent codes
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.saes = nn.ModuleList([_PositionSAE(d_in, d_sae, k) for _ in range(T)])

    @torch.no_grad()
    def _normalize_decoder(self):
        for sae in self.saes:
            sae._normalize_decoder()

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) → (recon_loss, x_hat, z)"""
        B, T, d = x.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        x_hats, zs, losses = [], [], []
        for t, sae in enumerate(self.saes):
            x_hat_t, z_t = sae(x[:, t, :])
            x_hats.append(x_hat_t)
            zs.append(z_t)
            losses.append((x[:, t, :] - x_hat_t).pow(2).sum(dim=-1).mean())

        x_hat = torch.stack(x_hats, dim=1)
        z = torch.stack(zs, dim=1)
        loss = torch.stack(losses).mean()
        return loss, x_hat, z

    def decoder_directions(self, pos: int = 0) -> torch.Tensor:
        """(d, h) decoder columns for a specific position's SAE."""
        return self.saes[pos].W_dec.data  # already (d_in, d_sae)


@dataclass
class StackedSAETrainingConfig:
    total_steps: int = 30_000
    batch_size: int = 2048
    lr: float = 3e-4
    l1_coeff: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 500


def train_stacked_sae(
    model: StackedSAE,
    generate_batch_fn,
    config: StackedSAETrainingConfig,
    device: torch.device,
) -> tuple[StackedSAE, dict]:
    """Train a StackedSAE. generate_batch_fn(batch_size) → (B, T, d) windows."""
    import sys
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    log = {"loss": []}

    for step in range(config.total_steps):
        x = generate_batch_fn(config.batch_size).to(device)
        loss, x_hat, z = model(x)

        if config.l1_coeff > 0:
            l1 = z.abs().sum(dim=-1).mean()
            loss = loss + config.l1_coeff * l1

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        model._normalize_decoder()

        if step % config.log_every == 0 or step == config.total_steps - 1:
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            print(f"  step {step:>5}/{config.total_steps} | loss={loss.item():.6f} | L0={l0:.2f}",
                  flush=True, file=sys.stderr if step < config.total_steps - 1 else sys.stdout)
            log["loss"].append({"step": step, "loss": loss.item(), "l0": l0})

    return model, log
