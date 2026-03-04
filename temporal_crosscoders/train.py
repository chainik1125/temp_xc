"""
train.py — Training loops for SAE and Temporal Crosscoder.

Each function returns (model, history_list) and logs to wandb + local JSON.
"""

import os
import json
import time
import torch
import torch.nn as nn
from typing import Callable
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from config import (
    HIDDEN_DIM, D_SAE, BATCH_SIZE, EVAL_BATCH, LEARNING_RATE,
    ADAM_BETAS, GRAD_CLIP, TRAIN_STEPS, LOG_INTERVAL, DEVICE,
    LOG_DIR, sae_effective_k, txcdr_effective_k, make_wandb_config,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, WANDB_TAGS,
)
from models import TopKSAE, TemporalCrosscoder
from data import (
    ToyModel, ShuffledDataIterator, SequentialWindowIterator, get_seq_gen_fn,
)
from metrics import feature_recovery_score


def _init_wandb(model_type: str, dataset: str, k: int, T: int, run_name: str):
    """Initialize a wandb run. Returns the run object or None."""
    if wandb is None or WANDB_MODE == "disabled":
        return None
    cfg = make_wandb_config(model_type, dataset, k, T)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=cfg,
        tags=WANDB_TAGS + [model_type, dataset],
        mode=WANDB_MODE,
        reinit=True,
    )
    return run


def _save_history(history: list[dict], path: str):
    """Append-safe JSON log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=1)


# ─── SAE training ───────────────────────────────────────────────────────────────

def train_sae(
    dataset: str,
    k: int,
    T: int,
    toy_model: ToyModel,
    true_features: torch.Tensor,
    *,
    n_steps: int = TRAIN_STEPS,
) -> tuple:
    """
    Train a TopK SAE.

    SAE is T-independent (sees single tokens), but we still need T for the
    data generator window.  We use shuffled sampling to get i.i.d. marginals.

    Returns (model, history).
    """
    run_name = f"sae__{dataset}__k{k}__T{T}"
    wb_run = _init_wandb("sae", dataset, k, T, run_name)

    seq_gen_fn = get_seq_gen_fn(dataset)
    eff_k = sae_effective_k(k)
    iterator = ShuffledDataIterator(toy_model, seq_gen_fn, BATCH_SIZE, T=max(T, 2))

    sae = TopKSAE(d_in=HIDDEN_DIM, d_sae=D_SAE, k=eff_k).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    history = []
    log_path = os.path.join(LOG_DIR, f"{run_name}.json")
    pbar = tqdm(range(n_steps), desc=run_name, leave=True, mininterval=2.0)

    for step in pbar:
        x = next(iterator).to(DEVICE)
        loss, _, _ = sae(x)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), GRAD_CLIP)
        optimizer.step()
        sae._normalize_decoder()

        if step % LOG_INTERVAL == 0 or step == n_steps - 1:
            with torch.no_grad():
                x_eval = next(iterator).to(DEVICE)
                eval_loss, _, _ = sae(x_eval)
                m = feature_recovery_score(sae.decoder_directions, true_features)

            row = {
                "step": step,
                "loss": eval_loss.item(),
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
                r90=f'{m["frac_recovered_90"]:.2f}',
            )

    _save_history(history, log_path)
    if wb_run is not None:
        wandb.finish()

    return sae, history


# ─── Crosscoder training ────────────────────────────────────────────────────────

def train_txcdr(
    dataset: str,
    k: int,
    T: int,
    toy_model: ToyModel,
    true_features: torch.Tensor,
    *,
    n_steps: int = TRAIN_STEPS,
) -> tuple:
    """
    Train a Temporal Crosscoder.

    Returns (model, history).
    """
    run_name = f"txcdr__{dataset}__k{k}__T{T}"
    wb_run = _init_wandb("txcdr", dataset, k, T, run_name)

    seq_gen_fn = get_seq_gen_fn(dataset)
    eff_k = txcdr_effective_k(k, T)
    iterator = SequentialWindowIterator(toy_model, seq_gen_fn, BATCH_SIZE, T=T)

    cc = TemporalCrosscoder(d_in=HIDDEN_DIM, d_sae=D_SAE, T=T, k=eff_k).to(DEVICE)
    optimizer = torch.optim.Adam(cc.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

    history = []
    log_path = os.path.join(LOG_DIR, f"{run_name}.json")
    pbar = tqdm(range(n_steps), desc=run_name, leave=True, mininterval=2.0)

    for step in pbar:
        x = next(iterator).to(DEVICE)
        loss, _, u = cc(x)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(cc.parameters(), GRAD_CLIP)
        optimizer.step()
        cc._normalize_decoder()

        if step % LOG_INTERVAL == 0 or step == n_steps - 1:
            with torch.no_grad():
                x_eval = next(iterator).to(DEVICE)
                eval_loss, _, _ = cc(x_eval)
                m = feature_recovery_score(
                    cc.decoder_directions(pos=0), true_features
                )

            row = {
                "step": step,
                "loss": eval_loss.item(),
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
                r90=f'{m["frac_recovered_90"]:.2f}',
            )

    _save_history(history, log_path)
    if wb_run is not None:
        wandb.finish()

    return cc, history
