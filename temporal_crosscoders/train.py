"""
train.py — Training loops for Stacked SAE and Temporal Crosscoder.

Each function returns (model, history_list) and logs to wandb + local JSON.
Both models take (B, T, d) windowed input.
"""

import os
import json
import torch
import torch.nn as nn
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from config import (
    HIDDEN_DIM, D_SAE, BATCH_SIZE, LEARNING_RATE,
    ADAM_BETAS, GRAD_CLIP, TRAIN_STEPS, LOG_INTERVAL, DEVICE,
    LOG_DIR, make_wandb_config,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, WANDB_TAGS,
)
from models import StackedSAE, TemporalCrosscoder
from data import ToyModel, CachedDataSource, CachedWindowIterator
from metrics import feature_recovery_score


def _init_wandb(model_type: str, rho: float, k: int, T: int, run_name: str):
    """Initialize a wandb run. Returns the run object or None."""
    if wandb is None or WANDB_MODE == "disabled":
        return None
    cfg = make_wandb_config(model_type, rho, k, T)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=cfg,
        tags=WANDB_TAGS + [model_type, f"rho={rho}"],
        mode=WANDB_MODE,
        reinit=True,
    )
    return run


def _save_history(history: list[dict], path: str):
    """Append-safe JSON log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=1)


def _rho_label(rho: float) -> str:
    """Short string for rho in filenames."""
    return f"rho{rho:.1f}".replace(".", "p")


# ─── Stacked SAE training ───────────────────────────────────────────────────────

def train_stacked_sae(
    rho: float,
    k: int,
    T: int,
    toy_model: ToyModel,
    true_features: torch.Tensor,
    cache: CachedDataSource,
    *,
    n_steps: int = TRAIN_STEPS,
) -> tuple:
    """
    Train a Stacked SAE: SAE(k) applied independently to each of T positions.

    Both stacked SAE and TXCDR now see the same (B, T, d) windows.
    Stacked SAE has L0 = k * T (k per position).

    Returns (model, history).
    """
    rl = _rho_label(rho)
    run_name = f"stacked_sae__{rl}__k{k}__T{T}"
    wb_run = _init_wandb("stacked_sae", rho, k, T, run_name)

    iterator = CachedWindowIterator(cache, BATCH_SIZE, T=T)

    model = StackedSAE(d_in=HIDDEN_DIM, d_sae=D_SAE, T=T, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    history = []
    log_path = os.path.join(LOG_DIR, f"{run_name}.json")
    pbar = tqdm(range(n_steps), desc=run_name, leave=True, mininterval=2.0)

    for step in pbar:
        x = next(iterator).to(DEVICE)
        loss, _, u = model(x)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        model._normalize_decoder()

        if step % LOG_INTERVAL == 0 or step == n_steps - 1:
            with torch.no_grad():
                x_eval = next(iterator).to(DEVICE)
                eval_loss, _, u_eval = model(x_eval)
                l0 = (u_eval > 0).float().sum(dim=-1).mean().item()  # per-position L0
                window_l0 = l0 * T  # total across window
                m = feature_recovery_score(model.decoder_directions, true_features)

            row = {
                "step": step,
                "loss": eval_loss.item(),
                "l0": l0,
                "window_l0": window_l0,
                "auc": m["auc"],
                "recovery_90": m["frac_recovered_90"],
                "recovery_80": m["frac_recovered_80"],
                "mean_max_cos_sim": m["mean_max_cos_sim"],
            }
            history.append(row)

            if wb_run is not None:
                wandb.log(row, step=step)

            pbar.set_postfix(
                loss=f'{eval_loss.item():.4f}',
                auc=f'{m["auc"]:.3f}',
                L0=f'{window_l0:.0f}',
            )

    _save_history(history, log_path)
    if wb_run is not None:
        wandb.finish()

    return model, history


# ─── Crosscoder training ────────────────────────────────────────────────────────

def train_txcdr(
    rho: float,
    k: int,
    T: int,
    toy_model: ToyModel,
    true_features: torch.Tensor,
    cache: CachedDataSource,
    *,
    n_steps: int = TRAIN_STEPS,
) -> tuple:
    """
    Train a Temporal Crosscoder.

    Samples sliding windows of length T. Shared latent with k active latents
    across all T positions, so window-level L0 = k.

    Returns (model, history).
    """
    rl = _rho_label(rho)
    run_name = f"txcdr__{rl}__k{k}__T{T}"
    wb_run = _init_wandb("txcdr", rho, k, T, run_name)

    iterator = CachedWindowIterator(cache, BATCH_SIZE, T=T)

    cc = TemporalCrosscoder(d_in=HIDDEN_DIM, d_sae=D_SAE, T=T, k=k).to(DEVICE)
    optimizer = torch.optim.Adam(cc.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    history = []
    log_path = os.path.join(LOG_DIR, f"{run_name}.json")
    pbar = tqdm(range(n_steps), desc=run_name, leave=True, mininterval=2.0)

    for step in pbar:
        x = next(iterator).to(DEVICE)
        loss, _, z = cc(x)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(cc.parameters(), GRAD_CLIP)
        optimizer.step()
        cc._normalize_decoder()

        if step % LOG_INTERVAL == 0 or step == n_steps - 1:
            with torch.no_grad():
                x_eval = next(iterator).to(DEVICE)
                eval_loss, _, z_eval = cc(x_eval)
                window_l0 = (z_eval > 0).float().sum(dim=-1).mean().item()
                m = feature_recovery_score(cc.decoder_directions, true_features)

            row = {
                "step": step,
                "loss": eval_loss.item(),
                "l0": window_l0,  # shared latent, so l0 = window_l0
                "window_l0": window_l0,
                "auc": m["auc"],
                "recovery_90": m["frac_recovered_90"],
                "recovery_80": m["frac_recovered_80"],
                "mean_max_cos_sim": m["mean_max_cos_sim"],
            }
            history.append(row)

            if wb_run is not None:
                wandb.log(row, step=step)

            pbar.set_postfix(
                loss=f'{eval_loss.item():.4f}',
                auc=f'{m["auc"]:.3f}',
                L0=f'{window_l0:.0f}',
            )

    _save_history(history, log_path)
    if wb_run is not None:
        wandb.finish()

    return cc, history
