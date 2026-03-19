"""Custom training loop for TemporalSAE on synthetic toy data.

Adapted from src/TemporalFeatureAnalysis/train_temporal_saes.py but uses
our toy data generator instead of precomputed LLM activation files.
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from src.v2_temporal_schemeC.tfa import TemporalSAE
from src.utils.device import DEFAULT_DEVICE


@dataclass
class TFATrainingConfig:
    """Configuration for TFA training."""

    total_steps: int = 5000
    batch_size: int = 64  # sequences per batch
    lr: float = 1e-3
    min_lr: float = 9e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    log_every: int = 500
    l1_coeff: float = 0.0  # L1 penalty on novel codes (0 = disabled)


def create_tfa(
    dimin: int,
    width: int,
    k: int,
    n_heads: int = 4,
    n_attn_layers: int = 1,
    bottleneck_factor: int = 1,
    tied_weights: bool = True,
    use_pos_encoding: bool = False,
    device: torch.device = DEFAULT_DEVICE,
) -> TemporalSAE:
    """Create a TemporalSAE configured for TopK sparsity."""
    tfa = TemporalSAE(
        dimin=dimin,
        width=width,
        n_heads=n_heads,
        sae_diff_type="topk",
        kval_topk=k,
        tied_weights=tied_weights,
        n_attn_layers=n_attn_layers,
        bottleneck_factor=bottleneck_factor,
        use_pos_encoding=use_pos_encoding,
    )
    return tfa.to(device)


def _configure_optimizer(
    tfa: TemporalSAE, config: TFATrainingConfig
) -> torch.optim.AdamW:
    """Create AdamW optimizer with weight decay only on 2D+ parameters.

    Following the pattern from the original TFA training code.
    """
    decay_params = []
    no_decay_params = []
    for name, param in tfa.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )


def _get_lr(step: int, config: TFATrainingConfig) -> float:
    """Cosine warmup learning rate schedule."""
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    decay_ratio = (step - config.warmup_steps) / max(
        1, config.total_steps - config.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def train_tfa(
    tfa: TemporalSAE,
    generate_batch_fn: Callable[[int], torch.Tensor],
    config: TFATrainingConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> tuple[TemporalSAE, dict[str, list[float]]]:
    """Train TFA on toy model hidden activations.

    Args:
        tfa: The TemporalSAE to train.
        generate_batch_fn: Callable that takes n_sequences and returns
            hidden activations of shape (n_sequences, seq_len, hidden_dim).
        config: TFA training configuration.
        device: Torch device.

    Returns:
        Tuple of (trained TFA, training_log dict).
    """
    optimizer = _configure_optimizer(tfa, config)
    tfa.train()

    log = {"loss": [], "mse": [], "rel_energy_pred": []}

    for step in range(config.total_steps):
        # Update learning rate
        lr = _get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Generate batch
        batch = generate_batch_fn(config.batch_size)  # (B, T, dimin)
        batch = batch.to(device)

        # Forward
        recons, intermediates = tfa(batch)

        # Loss: MSE averaged over tokens
        batch_flat = batch.reshape(-1, batch.shape[-1])
        recons_flat = recons.reshape(-1, recons.shape[-1])
        n_tokens = batch_flat.shape[0]
        mse_loss = F.mse_loss(recons_flat, batch_flat, reduction="sum") / n_tokens

        # Optional L1 penalty on novel codes
        loss = mse_loss
        if config.l1_coeff > 0:
            novel_codes = intermediates["novel_codes"]
            l1_penalty = novel_codes.abs().sum() / n_tokens
            loss = loss + config.l1_coeff * l1_penalty

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tfa.parameters(), config.grad_clip)
        optimizer.step()

        # Log periodically
        if step % config.log_every == 0 or step == config.total_steps - 1:
            with torch.no_grad():
                mse_val = mse_loss.item()

                pred_norm = intermediates["pred_recons"].norm(dim=-1).pow(2).mean()
                novel_norm = intermediates["novel_recons"].norm(dim=-1).pow(2).mean()
                total_norm = pred_norm + novel_norm + 1e-12
                rel_energy = (pred_norm / total_norm).item()

                novel_codes = intermediates["novel_codes"]
                novel_l0 = (novel_codes > 0).float().sum(dim=-1).mean().item()

            log["loss"].append(loss.item())
            log["mse"].append(mse_val)
            log["rel_energy_pred"].append(rel_energy)

            extra = ""
            if config.l1_coeff > 0:
                extra = f" | novel_L0={novel_l0:.2f}"
            print(
                f"  step {step:5d}/{config.total_steps} | "
                f"lr={lr:.2e} | MSE={mse_val:.6f} | "
                f"pred_energy={rel_energy:.3f}{extra}"
            )

    tfa.eval()
    return tfa, log
