"""Phase 7 training utilities — Gemma-2-2b base, L12 anchor, L10-L14 MLC.

Adapted from Phase 5B's `_train_utils.py`. Differences:
- Subject model: gemma-2-2b base (not -it). Cache lives at
  `data/cached_activations/gemma-2-2b/fineweb/`.
- Anchor layer: 0-indexed L12 (not L13). MLC layers: L10-L14 (not L11-L15).
- PRELOAD_SEQS bumped to 24_000 to leverage H200's 188 GB RAM (Phase 5B
  used 6_000 to fit on 5090's 32 GB system RAM).
- Reuses GPU sample generators from `phase5b._train_utils` via re-export
  (no behavioural change; just centralising path config in `_paths.py`).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    CACHE_DIR, ANCHOR_LAYER_KEY, MLC_LAYER_KEYS, PRELOAD_SEQS,
)

# Re-export the GPU sample generators from phase5b — identical behaviour,
# just lifted into Phase 7's namespace so callers don't need to know about
# phase5b internally. If a generator semantic changes for Phase 7 we can
# override here.
from experiments.phase5b_t_scaling_explore._train_utils import (
    make_flat_gen_gpu,
    make_window_gen_gpu,
    make_pair_window_gen_gpu,
    make_multidistance_pair_gen_gpu,
    make_strided_window_gen_gpu,
    make_paired_input_gen_gpu,
    make_paired_pair_window_gen_gpu,
    make_paired_multidistance_gen_gpu,
    compute_plateau,
)


# ────────────────────────────── data preload (Phase 7 paths)


def preload_single(layer_key: str = ANCHOR_LAYER_KEY,
                   device: torch.device | None = None,
                   n_seqs: int = PRELOAD_SEQS) -> torch.Tensor:
    """Load the L12 anchor activation cache as a (n_seqs, ctx_len, d_in)
    fp16 tensor on `device`. Phase 7 default n_seqs=24_000 (full cache).
    """
    if device is None:
        device = torch.device("cuda")
    arr = np.load(CACHE_DIR / f"{layer_key}.npy", mmap_mode="r")
    sub = np.asarray(arr[:n_seqs], dtype=np.float16)
    return torch.from_numpy(sub).to(device)


def preload_multilayer(device: torch.device | None = None,
                       n_seqs: int = PRELOAD_SEQS) -> torch.Tensor:
    """Load the 5-layer MLC stack at L10-L14, returning
    (n_seqs, ctx_len, n_layers=5, d_in) fp16 on `device`.
    """
    if device is None:
        device = torch.device("cuda")
    arrs = []
    for lk in MLC_LAYER_KEYS:
        a = np.asarray(
            np.load(CACHE_DIR / f"{lk}.npy", mmap_mode="r")[:n_seqs],
            dtype=np.float16,
        )
        arrs.append(torch.from_numpy(a))
    stacked = torch.stack(arrs, dim=2)
    return stacked.to(device)


# ────────────────────────────── training loop config


@dataclass
class TrainCfg:
    """Phase 7 training defaults.

    Differences from Phase 5B:
    - batch_size bumped 1024 → 4096 to leverage H200 141 GB VRAM
      (smoke test must verify convergence is preserved; if not, fall
      back to 2048).
    - max_steps unchanged at 25_000 (same convergence budget per arch).
    """
    lr: float = 3e-4
    batch_size: int = 4096
    max_steps: int = 25_000
    log_every: int = 200
    grad_clip: float = 1.0
    plateau_threshold: float = 0.02
    min_steps: int = 3_000
    seed: int = 42


def iterate_train(
    model: nn.Module,
    gen_fn: Callable[[int], torch.Tensor],
    cfg: TrainCfg,
    device: torch.device,
    normalize_decoder: Callable | None = None,
    grad_post_hook: Callable | None = None,
    latent_l0_fn: Callable[[torch.Tensor], float] | None = None,
) -> dict[str, Any]:
    """Plain reconstruction-loss training loop with plateau early-stop.

    Mirrors phase5b._train_utils.iterate_train but instantiated under
    Phase 7's TrainCfg defaults. Returns a log dict with loss / l0 /
    convergence telemetry.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses, l0s, steps_logged = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen_fn(cfg.batch_size)
        loss, _, z = model(x)
        opt.zero_grad()
        loss.backward()
        if grad_post_hook is not None:
            grad_post_hook()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if normalize_decoder is not None:
            normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (
                    latent_l0_fn(z) if latent_l0_fn is not None
                    else (z > 0).float().sum(dim=-1).mean().item()
                )
            losses.append(loss.item())
            l0s.append(l0)
            steps_logged.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if (
                plateau_val is not None
                and plateau_val < cfg.plateau_threshold
                and step >= cfg.min_steps
            ):
                converged = True
                break
    elapsed = time.time() - t0
    model.eval()
    return {
        "loss": losses,
        "l0": l0s,
        "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged,
        "plateau_last": plateau_val,
        "elapsed_s": elapsed,
    }
