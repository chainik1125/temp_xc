"""Stacked SAE training loop (config dataclass + train function).

The StackedSAE architecture class lives in
src/bench/architectures/stacked_sae.py — import it from there. This
file retains only the v2 training utilities, kept separate from the
bench spec-based training loop because toy experiments use dataclass
configs instead of kwargs.
"""

import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.architectures.stacked_sae import StackedSAE


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
            print(
                f"  step {step:>5}/{config.total_steps} | "
                f"loss={loss.item():.6f} | L0={l0:.2f}",
                flush=True,
                file=sys.stderr if step < config.total_steps - 1 else sys.stdout,
            )
            log["loss"].append({"step": step, "loss": loss.item(), "l0": l0})

    return model, log
