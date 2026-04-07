"""
train.py — Training loops for StackedSAE and TemporalCrosscoder on NLP activations.

Performance:
  - Vectorized models (no Python for-loop over T)
  - torch.compile for kernel fusion
  - fp16 autocast (Turing SM 7.5 has fp16 tensor cores)
  - Batch size scales with T to stay within VRAM
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

from fast_models import FastStackedSAE, FastTemporalCrosscoder

from config import (
    D_SAE, LEARNING_RATE, batch_size_for_T,
    ADAM_BETAS, GRAD_CLIP, TRAIN_STEPS, LOG_INTERVAL, DEVICE,
    LOG_DIR, CHECKPOINT_DIR, LAYER_SPECS, make_wandb_config, run_name,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, WANDB_TAGS,
)
from data import CachedActivationSource, WindowIterator


# ─── Detect fp16 support (Turing+ = SM >= 7.0) ────────────────────────────────
_USE_FP16 = (
    DEVICE.type == "cuda"
    and torch.cuda.get_device_capability()[0] >= 7
)


def _init_wandb(
    model_type: str, layer: str, k: int, T: int, rn: str,
):
    if wandb is None or WANDB_MODE == "disabled":
        return None
    cfg = make_wandb_config(model_type, layer, k, T)
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=rn,
        config=cfg,
        tags=WANDB_TAGS + [model_type, layer],
        mode=WANDB_MODE,
        reinit=True,
    )


def _save_history(history: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=1)


def _compute_fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    total_var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().item()
    resid_var = (x_hat - x).pow(2).sum().item()
    return resid_var / max(total_var, 1e-8)


def _compute_activation_entropy(u: torch.Tensor) -> float:
    flat = u.reshape(-1, u.shape[-1])
    p = (flat > 0).float().mean(dim=0)
    eps = 1e-8
    p = p.clamp(eps, 1 - eps)
    h = -(p * p.log() + (1 - p) * (1 - p).log()) / math.log(2)
    return h.mean().item()


def _train_loop(
    model: nn.Module,
    model_type: str,
    layer: str,
    k: int,
    T: int,
    source: CachedActivationSource,
    n_steps: int,
    save_checkpoint: bool,
) -> tuple[nn.Module, list[dict]]:
    """Shared training loop for both architectures."""
    rn = run_name(model_type, layer, k, T)
    wb_run = _init_wandb(model_type, layer, k, T, rn)

    bs = batch_size_for_T(T)
    iterator = WindowIterator(source, bs, T=T)

    model = model.to(DEVICE)

    # torch.compile for kernel fusion (reduce mode for stability)
    compiled = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=_USE_FP16)

    history: list[dict] = []
    log_path = os.path.join(LOG_DIR, f"{rn}.json")
    pbar = tqdm(range(n_steps), desc=rn, leave=True, mininterval=2.0)

    for step in pbar:
        x = next(iterator)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=_USE_FP16):
            loss, _, activations = compiled(x)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        model._normalize_decoder()

        if step % LOG_INTERVAL == 0 or step == n_steps - 1:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=_USE_FP16):
                x_eval = next(iterator)
                eval_loss, x_hat, act_eval = compiled(x_eval)

                # L0
                if model_type == "stacked_sae":
                    l0 = (act_eval > 0).float().sum(dim=-1).mean().item()
                    window_l0 = l0 * T
                else:
                    window_l0 = (act_eval > 0).float().sum(dim=-1).mean().item()
                    l0 = window_l0

                fvu = _compute_fvu(x_eval.float(), x_hat.float())
                entropy = _compute_activation_entropy(act_eval)

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
            )

    _save_history(history, log_path)

    if save_checkpoint:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{rn}.pt")
        torch.save(model.state_dict(), ckpt_path)

    if wb_run is not None:
        wandb.finish()

    return model, history


# ─── Public API ───────────────────────────────────────────────────────────────

def train_stacked_sae(
    layer: str,
    k: int,
    T: int,
    source: CachedActivationSource,
    *,
    n_steps: int = TRAIN_STEPS,
    save_checkpoint: bool = True,
) -> tuple:
    d_act = LAYER_SPECS[layer]["d_act"]
    model = FastStackedSAE(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    return _train_loop(model, "stacked_sae", layer, k, T, source, n_steps, save_checkpoint)


def train_txcdr(
    layer: str,
    k: int,
    T: int,
    source: CachedActivationSource,
    *,
    n_steps: int = TRAIN_STEPS,
    save_checkpoint: bool = True,
) -> tuple:
    d_act = LAYER_SPECS[layer]["d_act"]
    model = FastTemporalCrosscoder(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    return _train_loop(model, "txcdr", layer, k, T, source, n_steps, save_checkpoint)
