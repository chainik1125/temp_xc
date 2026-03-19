"""Temporal Crosscoder (ckkissane-style shared-latent architecture).

A crosscoder that encodes a window of T consecutive tokens into a single
shared latent vector z with k active features, then decodes back to T
positions using per-position decoder weights.

Architecture:
    W_enc: (T, d, h) — per-position encoder projections
    W_dec: (h, T, d) — per-position decoder projections
    b_enc: (h,)      — shared encoder bias
    b_dec: (T, d)    — per-position decoder bias

    Encode: z = TopK(sum_t(x_t @ W_enc[t]) + b_enc)  → (B, h), k non-zeros
    Decode: x_hat_t = z @ W_dec[:, t, :].T + b_dec[t]  → (B, T, d)

Ported from Andre Shportko's temporal_crosscoders/models.py.
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.device import DEFAULT_DEVICE


class TemporalCrosscoder(nn.Module):
    """Shared-latent temporal crosscoder with TopK or ReLU sparsity."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None = None):
        """
        Args:
            d_in: Input dimension per position.
            d_sae: Latent dimension (number of dictionary atoms).
            T: Window size (number of positions).
            k: If set, use TopK sparsity. If None, use ReLU (for L1 mode).
        """
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in ** 0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, T, d_in) * (1.0 / d_sae ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        """Normalize each latent's decoder across all positions jointly."""
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → z: (B, h)."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, topk_idx, F.relu(topk_vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, h) → x_hat: (B, T, d)."""
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) → (recon_loss, x_hat, z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z

    def decoder_directions(self, pos: int = 0) -> torch.Tensor:
        """(d, h) decoder columns for a given position.

        W_dec is (h, T, d) → [:, pos, :] is (h, d) → transpose to (d, h).
        """
        return self.W_dec[:, pos, :].T


# ── Training ─────────────────────────────────────────────────────────


@dataclass
class CrosscoderTrainingConfig:
    """Configuration for crosscoder training."""
    total_steps: int = 30_000
    batch_size: int = 128  # number of windows per batch
    lr: float = 3e-4
    grad_clip: float = 1.0
    log_every: int = 5000
    l1_coeff: float = 0.0  # L1 penalty on latent codes (for ReLU mode)


def train_crosscoder(
    model: TemporalCrosscoder,
    generate_batch_fn: Callable[[int], torch.Tensor],
    config: CrosscoderTrainingConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[TemporalCrosscoder, dict[str, list[float]]]:
    """Train a temporal crosscoder.

    Args:
        model: The crosscoder to train.
        generate_batch_fn: Callable that takes batch_size and returns
            windows of shape (batch_size, T, d_in).
        config: Training configuration.
        device: Torch device.

    Returns:
        Tuple of (trained model, training log).
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr
    )
    model.train()
    log = {"loss": []}

    for step in range(config.total_steps):
        batch = generate_batch_fn(config.batch_size).to(device)
        recon_loss, x_hat, z = model(batch)

        loss = recon_loss
        if config.l1_coeff > 0:
            l1_penalty = z.abs().sum(dim=-1).mean()
            loss = loss + config.l1_coeff * l1_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        model._normalize_decoder()

        if step % config.log_every == 0 or step == config.total_steps - 1:
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            print(f"  step {step:5d}/{config.total_steps} | "
                  f"loss={recon_loss.item():.6f} | L0={l0:.2f}")
            log["loss"].append(recon_loss.item())

    model.eval()
    return model, log
