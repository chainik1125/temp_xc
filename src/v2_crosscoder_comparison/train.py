"""Training functions for SAE Lens architectures and custom crosscoder."""

import time
from collections.abc import Callable

import torch

from src.shared.configs import TrainingConfig
from src.shared.train_sae import train_sae
from src.utils.device import DEFAULT_DEVICE
from src.utils.logging import log
from src.v2_crosscoder_comparison.architectures.crosscoder import Crosscoder
from src.v2_crosscoder_comparison.architectures.naive_sae import NaiveSAE
from src.v2_crosscoder_comparison.architectures.stacked_sae import StackedSAE
from src.v2_crosscoder_comparison.configs import ExperimentConfig


def train_with_sae_lens(
    arch: NaiveSAE | StackedSAE,
    generate_hidden_fn: Callable[[int], torch.Tensor],
    config: ExperimentConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> None:
    """Train SAE Lens architectures using shared/train_sae.

    For NaiveSAE: trains one SAE on flattened data.
    For StackedSAE: trains each position's SAE independently.

    Args:
        arch: NaiveSAE or StackedSAE instance.
        generate_hidden_fn: Callable taking batch_size, returning hidden acts
            of shape (batch, n_positions, d_model).
        config: Experiment configuration.
        device: Torch device.
    """
    training_cfg = TrainingConfig(
        k=config.architecture.top_k,
        d_sae=config.architecture.d_sae,
        lr=config.lr,
        total_training_samples=config.total_training_samples,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    if isinstance(arch, NaiveSAE):
        def gen_flat(batch_size: int) -> torch.Tensor:
            hidden = generate_hidden_fn(batch_size)
            batch, n_pos, d = hidden.shape
            return hidden.reshape(batch * n_pos, d)

        log("info", f"training naive_sae via sae_lens | d_sae={config.architecture.d_sae} | top_k={config.architecture.top_k}")
        arch.sae = train_sae(arch.sae, gen_flat, training_cfg, device)

    elif isinstance(arch, StackedSAE):
        for t in range(2):
            def gen_pos(batch_size: int, pos=t) -> torch.Tensor:
                hidden = generate_hidden_fn(batch_size)
                return hidden[:, pos, :]

            log("info", f"training stacked_sae position {t} via sae_lens | d_sae={config.architecture.d_sae} | top_k={config.architecture.top_k}")
            arch.saes[t] = train_sae(arch.saes[t], gen_pos, training_cfg, device)


def train_crosscoder(
    arch: Crosscoder,
    generate_hidden_fn: Callable[[int], torch.Tensor],
    config: ExperimentConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> None:
    """Train crosscoder with custom Adam loop.

    Uses same optimizer settings as SAE Lens (lr=3e-4, betas=(0.9, 0.999)).

    Args:
        arch: Crosscoder architecture.
        generate_hidden_fn: Callable taking batch_size, returning hidden acts
            of shape (batch, n_positions, d_model).
        config: Experiment configuration.
        device: Torch device.
    """
    module = arch.module.to(device)
    module.train()

    optimizer = torch.optim.Adam(
        module.parameters(),
        lr=config.lr,
        betas=(0.9, 0.999),
    )

    log("info", f"training crosscoder | d_sae={config.architecture.d_sae} | top_k={config.architecture.top_k} | steps={config.training_steps}")

    start_time = time.time()
    log_interval = max(config.training_steps // 3, 1)

    for step in range(1, config.training_steps + 1):
        hidden = generate_hidden_fn(config.batch_size).to(device)

        x_hat = module(hidden)
        recon_loss = (hidden - x_hat).pow(2).sum(dim=(-2, -1)).mean()

        loss = recon_loss
        if config.architecture.l1_coefficient > 0:
            z = module.encode(hidden)
            l1_loss = config.architecture.l1_coefficient * z.abs().sum(dim=-1).mean()
            loss = loss + l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == config.training_steps:
            elapsed = time.time() - start_time
            z = module.encode(hidden)
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            log(
                "train",
                f"step {step:05d}/{config.training_steps}",
                recon_loss=recon_loss.item(),
                l0=l0,
                time=f"{elapsed:.1f}s",
            )

    module.eval()


def train_architecture(
    arch_type: str,
    arch: object,
    generate_hidden_fn: Callable[[int], torch.Tensor],
    config: ExperimentConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> None:
    """Dispatch training to the appropriate trainer.

    Args:
        arch_type: One of 'naive_sae', 'stacked_sae', 'crosscoder'.
        arch: Architecture instance.
        generate_hidden_fn: Hidden activation generator.
        config: Experiment configuration.
        device: Torch device.
    """
    if arch_type in ("naive_sae", "stacked_sae"):
        train_with_sae_lens(arch, generate_hidden_fn, config, device)
    elif arch_type == "crosscoder":
        train_crosscoder(arch, generate_hidden_fn, config, device)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")
