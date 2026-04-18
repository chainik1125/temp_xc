"""Custom SAE supporting both ReLU+L1 and TopK sparsity.

Architecture:
    z = ReLU(W_enc @ (x - b_dec) + b_enc)
    z = TopK(z, k)            [if topk mode]
    x_hat = W_dec @ z + b_dec
    loss = MSE(x, x_hat) + l1_coeff * mean(||z||_1)  [L1 only in relu mode]

Decoder columns are projected to unit norm after each optimizer step.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable

from src.utils.device import DEFAULT_DEVICE


class ReLUSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int | None = None):
        """
        Args:
            d_in: Input dimension.
            d_sae: Dictionary size.
            k: If set, use per-token TopK sparsity. If None, use ReLU (for L1 mode).
        """
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Kaiming init
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        # Init decoder as transpose of encoder (then normalize)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self):
        """Project decoder columns to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(x @ self.W_enc + self.b_enc)
        if self.k is not None:
            _, topk_idx = torch.topk(z, self.k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, topk_idx, 1)
            z = z * mask
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z)."""
        z = self.encode(x - self.b_dec)
        x_hat = self.decode(z)
        return x_hat, z


@dataclass
class ReLUSAETrainingConfig:
    total_steps: int = 30_000
    batch_size: int = 4096
    lr: float = 3e-4
    l1_coeff: float = 1e-2
    weight_decay: float = 0.0
    warmup_steps: int = 200
    log_every: int = 5000
    seed: int = 42


def train_relu_sae(
    sae: ReLUSAE,
    generate_batch_fn: Callable[[int], torch.Tensor],
    config: ReLUSAETrainingConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[ReLUSAE, dict[str, list[float]]]:
    """Train a ReLU SAE with L1 regularization."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    sae.train()

    log = {"loss": [], "mse": [], "l1": [], "l0": []}

    for step in range(config.total_steps):
        # LR warmup
        if step < config.warmup_steps:
            lr = config.lr * (step + 1) / config.warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Generate batch
        x = generate_batch_fn(config.batch_size).to(device)

        # Forward
        x_hat, z = sae(x)

        # Loss
        mse = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1 = z.abs().sum(dim=-1).mean()
        loss = mse + config.l1_coeff * l1

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize decoder
        sae._normalize_decoder()

        # Log
        if step % config.log_every == 0 or step == config.total_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            log["loss"].append(loss.item())
            log["mse"].append(mse.item())
            log["l1"].append(l1.item())
            log["l0"].append(l0)
            print(f"  step {step:5d}/{config.total_steps} | "
                  f"MSE={mse.item():.6f} | L1={l1.item():.4f} | "
                  f"L0={l0:.2f} | loss={loss.item():.6f}")

    sae.eval()
    return sae, log
