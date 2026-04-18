"""Crosscoder training loop (config dataclass + train function).

The TemporalCrosscoder architecture class lives in
src/bench/architectures/crosscoder.py — import it from there. This
file retains only the v2 training utilities, kept separate from the
bench spec-based training loop because toy experiments use dataclass
configs instead of kwargs.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from src.architectures.crosscoder import TemporalCrosscoder
from src.utils.device import DEFAULT_DEVICE


@dataclass
class CrosscoderTrainingConfig:
    total_steps: int = 30_000
    batch_size: int = 128
    lr: float = 3e-4
    grad_clip: float = 1.0
    log_every: int = 5000
    l1_coeff: float = 0.0


def train_crosscoder(
    model: TemporalCrosscoder,
    generate_batch_fn: Callable[[int], torch.Tensor],
    config: CrosscoderTrainingConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[TemporalCrosscoder, dict[str, list[float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
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
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        model._normalize_decoder()

        if step % config.log_every == 0 or step == config.total_steps - 1:
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            print(f"  step {step:5d}/{config.total_steps} | "
                  f"loss={recon_loss.item():.6f} | L0={l0:.2f}")
            log["loss"].append(recon_loss.item())

    model.eval()
    return model, log
