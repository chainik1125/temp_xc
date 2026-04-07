"""
train.py — Training loops for StackedSAE and TemporalCrosscoder on NLP activations.

Mirrors the toy-model training loops but operates on cached Gemma 2 2B activations.
Loss function: MSE reconstruction with TopK sparsity (no explicit L1 penalty),
matching the toy model experiments exactly.

Logs entropy of feature activations alongside loss, L0, and FVU.
"""

import os
import json
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import StackedSAE, TemporalCrosscoder

from config import (
    D_SAE, BATCH_SIZE, LEARNING_RATE,
    ADAM_BETAS, GRAD_CLIP, TRAIN_STEPS, LOG_INTERVAL, DEVICE,
    LOG_DIR, CHECKPOINT_DIR, LAYER_SPECS, make_wandb_config, run_name,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, WANDB_TAGS,
)
from data import CachedActivationSource, WindowIterator


def _init_wandb(
    model_type: str, layer: str, k: int, T: int, rn: str,
):
    """Initialize a wandb run. Returns the run object or None."""
    if wandb is None or WANDB_MODE == "disabled":
        return None
    cfg = make_wandb_config(model_type, layer, k, T)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=rn,
        config=cfg,
        tags=WANDB_TAGS + [model_type, layer],
        mode=WANDB_MODE,
        reinit=True,
    )
    return run


def _save_history(history: list[dict], path: str) -> None:
    """Append-safe JSON log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=1)


def _compute_fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained."""
    total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().item()
    resid_var = (x_hat - x).pow(2).sum().item()
    return resid_var / max(total_var, 1e-8)


def _compute_activation_entropy(u: torch.Tensor) -> float:
    """Compute entropy of the binary support distribution across features.

    For a (B, ..., h) activation tensor, computes per-feature firing
    probability p_j = P(u_j > 0), then returns H = -sum(p*log(p) + (1-p)*log(1-p)).
    Normalized by log(2) to get bits. Returned as mean entropy per feature.
    """
    # Flatten to (N, h)
    flat = u.reshape(-1, u.shape[-1])
    p = (flat > 0).float().mean(dim=0)  # (h,)
    # Clamp to avoid log(0)
    eps = 1e-8
    p = p.clamp(eps, 1 - eps)
    entropy_per_feat = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)
    return entropy_per_feat.mean().item()


# ─── Stacked SAE training ───────────────────────────────────────────────────────

def train_stacked_sae(
    layer: str,
    k: int,
    T: int,
    source: CachedActivationSource,
    *,
    n_steps: int = TRAIN_STEPS,
    save_checkpoint: bool = True,
) -> tuple:
    """
    Train a StackedSAE on cached NLP activations.
    Returns (model, history).
    """
    d_act = LAYER_SPECS[layer]["d_act"]
    rn = run_name("stacked_sae", layer, k, T)
    wb_run = _init_wandb("stacked_sae", layer, k, T, rn)

    iterator = WindowIterator(source, BATCH_SIZE, T=T)

    model = StackedSAE(d_in=d_act, d_sae=D_SAE, T=T, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS,
    )

    history: list[dict] = []
    log_path = os.path.join(LOG_DIR, f"{rn}.json")
    pbar = tqdm(range(n_steps), desc=rn, leave=True, mininterval=2.0)

    for step in pbar:
        x = next(iterator)
        loss, _, u = model(x)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        model._normalize_decoder()

        if step % LOG_INTERVAL == 0 or step == n_steps - 1:
            with torch.no_grad():
                x_eval = next(iterator)
                eval_loss, x_hat, u_eval = model(x_eval)

                l0 = (u_eval > 0).float().sum(dim=-1).mean().item()
                window_l0 = l0 * T
                fvu = _compute_fvu(x_eval, x_hat)
                entropy = _compute_activation_entropy(u_eval)

            row = {
                "step": step,
                "loss": eval_loss.item(),
                "l0": l0,
                "window_l0": window_l0,
                "fvu": fvu,
                "entropy": entropy,
            }
            history.append(row)

            if wb_run is not None:
                wandb.log(row, step=step)

            pbar.set_postfix(
                loss=f"{eval_loss.item():.4f}",
                L0=f"{window_l0:.0f}",
                fvu=f"{fvu:.4f}",
                H=f"{entropy:.3f}",
            )

    _save_history(history, log_path)

    if save_checkpoint:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{rn}.pt")
        torch.save(model.state_dict(), ckpt_path)

    if wb_run is not None:
        wandb.finish()

    return model, history


# ─── Temporal Crosscoder training ────────────────────────────────────────────────

def train_txcdr(
    layer: str,
    k: int,
    T: int,
    source: CachedActivationSource,
    *,
    n_steps: int = TRAIN_STEPS,
    save_checkpoint: bool = True,
) -> tuple:
    """
    Train a TemporalCrosscoder on cached NLP activations.
    Returns (model, history).
    """
    d_act = LAYER_SPECS[layer]["d_act"]
    rn = run_name("txcdr", layer, k, T)
    wb_run = _init_wandb("txcdr", layer, k, T, rn)

    iterator = WindowIterator(source, BATCH_SIZE, T=T)

    model = TemporalCrosscoder(d_in=d_act, d_sae=D_SAE, T=T, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS,
    )

    history: list[dict] = []
    log_path = os.path.join(LOG_DIR, f"{rn}.json")
    pbar = tqdm(range(n_steps), desc=rn, leave=True, mininterval=2.0)

    for step in pbar:
        x = next(iterator)
        loss, _, z = model(x)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        model._normalize_decoder()

        if step % LOG_INTERVAL == 0 or step == n_steps - 1:
            with torch.no_grad():
                x_eval = next(iterator)
                eval_loss, x_hat, z_eval = model(x_eval)

                window_l0 = (z_eval > 0).float().sum(dim=-1).mean().item()
                fvu = _compute_fvu(x_eval, x_hat)
                entropy = _compute_activation_entropy(z_eval)

            row = {
                "step": step,
                "loss": eval_loss.item(),
                "l0": window_l0,
                "window_l0": window_l0,
                "fvu": fvu,
                "entropy": entropy,
            }
            history.append(row)

            if wb_run is not None:
                wandb.log(row, step=step)

            pbar.set_postfix(
                loss=f"{eval_loss.item():.4f}",
                L0=f"{window_l0:.0f}",
                fvu=f"{fvu:.4f}",
                H=f"{entropy:.3f}",
            )

    _save_history(history, log_path)

    if save_checkpoint:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{rn}.pt")
        torch.save(model.state_dict(), ckpt_path)

    if wb_run is not None:
        wandb.finish()

    return model, history
