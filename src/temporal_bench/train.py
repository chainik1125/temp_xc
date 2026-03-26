"""Generic training loop for all TemporalAE models."""

from __future__ import annotations

from typing import Callable

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .config import TrainConfig
from .metrics import EvalMetrics, evaluate
from .models.base import TemporalAE


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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[EvalMetrics] = []

    model.train()
    pbar = tqdm(range(config.n_steps), disable=silent, desc="Training")

    for step in pbar:
        x = data_fn(config.batch_size)
        out = model(x)

        optimizer.zero_grad()
        out.loss.backward()
        clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        model.normalize_decoder()

        # Progress bar
        pbar.set_postfix(loss=f"{out.metrics.get('recon_loss', 0):.4f}")

        # Periodic evaluation
        if eval_data is not None and true_features is not None:
            if step % config.eval_every == 0 or step == config.n_steps - 1:
                metrics = evaluate(model, eval_data, true_features)
                history.append(metrics)
                if callback:
                    callback(step, metrics)

    return history
