"""Generic training loop for all TemporalAE models."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .config import TrainConfig
from .metrics import EvalMetrics, evaluate
from .models.base import TemporalAE


def _build_optimizer(model: TemporalAE, config: TrainConfig) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam

    if config.grouped_weight_decay:
        decay_params = []
        no_decay_params = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return optimizer_cls(
            param_groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

    return optimizer_cls(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def _get_lr(step: int, config: TrainConfig) -> float:
    min_lr = config.lr if config.min_lr is None else config.min_lr

    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.lr * float(step + 1) / float(config.warmup_steps)

    if config.lr_schedule == "cosine":
        denom = max(1, config.n_steps - config.warmup_steps)
        progress = max(0.0, float(step - config.warmup_steps) / float(denom))
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr + coeff * (config.lr - min_lr)

    return config.lr


def train(
    model: TemporalAE,
    data_fn: Callable[[int], torch.Tensor],
    config: TrainConfig,
    eval_data: torch.Tensor | None = None,
    true_features: torch.Tensor | None = None,
    callback: Callable[[int, EvalMetrics], None] | None = None,
    silent: bool = False,
) -> list[EvalMetrics]:
    """Train any TemporalAE model with a generic loop.

    Args:
        model: The model to train (must implement TemporalAE interface).
        data_fn: Callable that takes batch_size and returns (B, T, d) tensor.
        config: Training hyperparameters.
        eval_data: Optional (n_seq, T, d) held-out data for periodic evaluation.
        true_features: Optional (n_features, d) ground truth for feature recovery.
        callback: Optional function called at each eval step with (step, metrics).
        silent: If True, suppress progress bar.

    Returns:
        List of EvalMetrics from periodic evaluations.
    """
    optimizer = _build_optimizer(model, config)
    history: list[EvalMetrics] = []
    prev_collect_metrics = model.collect_metrics

    model.train()
    model.collect_metrics = False
    pbar = tqdm(range(config.n_steps), disable=silent, desc="Training", mininterval=1.0)

    try:
        for step in pbar:
            lr = _get_lr(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x = data_fn(config.batch_size)
            out = model(x)

            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if config.normalize_decoder_every > 0:
                if (step + 1) % config.normalize_decoder_every == 0 or step == config.n_steps - 1:
                    model.normalize_decoder()

            if not silent and config.log_every > 0:
                if step % config.log_every == 0 or step == config.n_steps - 1:
                    loss_value = out.loss.detach().item()
                    pbar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.2e}")

            if eval_data is not None and true_features is not None:
                if step % config.eval_every == 0 or step == config.n_steps - 1:
                    metrics = evaluate(model, eval_data, true_features)
                    history.append(metrics)
                    if callback:
                        callback(step, metrics)
    finally:
        model.collect_metrics = prev_collect_metrics

    return history
