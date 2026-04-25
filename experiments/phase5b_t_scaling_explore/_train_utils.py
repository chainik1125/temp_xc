"""Phase 5B training utilities — adapted from Phase 5's train_primary_archs.py.

Differences:
- Local-path aware via _paths.py.
- Smaller default PRELOAD_SEQS (memory-conscious for 5090's 32GB).
- Adds subsequence-pair generators for new arch families.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase5b_t_scaling_explore._paths import (
    CACHE_DIR, ANCHOR_LAYER_KEY, MLC_LAYER_KEYS, PRELOAD_SEQS,
)


# ─────────────────────────────────────────── data preload


def preload_single(layer_key: str = ANCHOR_LAYER_KEY,
                   device: torch.device | None = None,
                   n_seqs: int = PRELOAD_SEQS) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda")
    arr = np.load(CACHE_DIR / f"{layer_key}.npy", mmap_mode="r")
    sub = np.asarray(arr[:n_seqs], dtype=np.float16)
    return torch.from_numpy(sub).to(device)


def preload_multilayer(device: torch.device | None = None,
                       n_seqs: int = PRELOAD_SEQS) -> torch.Tensor:
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


# ─────────────────────────────────────────── GPU sample generators


def make_flat_gen_gpu(buf: torch.Tensor):
    N, L, d = buf.shape

    def gen(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, N, (batch_size,), device=buf.device)
        tok = torch.randint(0, L, (batch_size,), device=buf.device)
        return buf[idx, tok].float()
    return gen


def make_window_gen_gpu(buf: torch.Tensor, T: int):
    """Sliding T-token window. (B, T, d)."""
    N, L, d = buf.shape
    n_wins = L - T + 1

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, n_wins, (batch_size,), device=buf.device)
        rng = torch.arange(T, device=buf.device)
        pos = off.unsqueeze(1) + rng.unsqueeze(0)
        return buf[seq.unsqueeze(1).expand(-1, T), pos].float()
    return gen


def make_pair_window_gen_gpu(buf: torch.Tensor, T: int):
    """Two adjacent sliding windows of length T (shift 1). (B, 2, T, d).

    Mirrors Phase 5's pair generator — used for shift=1 InfoNCE.
    """
    N, L, d = buf.shape

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, L - T - 1, (batch_size,), device=buf.device)
        rng = torch.arange(T, device=buf.device)

        pos_prev = off.unsqueeze(1) + rng.unsqueeze(0)
        pos_cur = (off + 1).unsqueeze(1) + rng.unsqueeze(0)

        win_prev = buf[seq.unsqueeze(1).expand(-1, T), pos_prev].float()
        win_cur = buf[seq.unsqueeze(1).expand(-1, T), pos_cur].float()
        return torch.stack([win_prev, win_cur], dim=1)
    return gen


def make_multidistance_pair_gen_gpu(buf: torch.Tensor, T: int,
                                     shifts: list[int]):
    """Multi-shift contrastive sampler for H8-family. (B, 1+K, T, d)."""
    N, L, d = buf.shape
    max_shift = max(shifts)
    assert L >= T + max_shift, f"need L>={T+max_shift}; got L={L}"

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, L - T - max_shift, (batch_size,), device=buf.device)
        rng = torch.arange(T, device=buf.device)
        outs = []
        for s in [0] + list(shifts):
            pos = (off + s).unsqueeze(1) + rng.unsqueeze(0)
            w = buf[seq.unsqueeze(1).expand(-1, T), pos].float()
            outs.append(w)
        return torch.stack(outs, dim=1)
    return gen


def make_strided_window_gen_gpu(buf: torch.Tensor, T_eff: int, stride: int):
    """Strided window: T_eff tokens sampled every `stride` from a span of
    T_eff*stride raw tokens. Output (B, T_eff, d).
    """
    N, L, d = buf.shape
    span = T_eff * stride
    assert L >= span, f"need L>={span}; got L={L}"
    n_wins = L - span + 1

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, n_wins, (batch_size,), device=buf.device)
        rng = torch.arange(0, span, stride, device=buf.device)  # (T_eff,)
        pos = off.unsqueeze(1) + rng.unsqueeze(0)               # (B, T_eff)
        return buf[seq.unsqueeze(1).expand(-1, T_eff), pos].float()
    return gen


def make_paired_input_gen_gpu(buf: torch.Tensor, T_paired: int):
    """Pair-summed input: each "position" is the sum of 2 adjacent raw tokens.
    Window of T_paired pairs spans 2*T_paired raw tokens.
    Output: (B, T_paired, d).
    """
    N, L, d = buf.shape
    span = 2 * T_paired
    assert L >= span, f"need L>={span}; got L={L}"
    n_wins = L - span + 1

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, n_wins, (batch_size,), device=buf.device)
        # raw tokens at positions off, off+1, ..., off+span-1
        rng = torch.arange(span, device=buf.device)
        pos = off.unsqueeze(1) + rng.unsqueeze(0)                 # (B, span)
        x = buf[seq.unsqueeze(1).expand(-1, span), pos].float()   # (B, span, d)
        # pair-sum: reshape to (B, T_paired, 2, d) and sum over the inner-2
        x_pairs = x.view(batch_size, T_paired, 2, d).sum(dim=2)
        return x_pairs                                             # (B, T_paired, d)
    return gen


def make_paired_pair_window_gen_gpu(buf: torch.Tensor, T_paired: int):
    """Pair-summed input for contrastive pair training (H8 base on paired).
    Two adjacent shift-1 pair-summed windows. Output (B, 2, T_paired, d).

    "Adjacent" means the second window is shifted by 1 PAIR (= 2 raw tokens),
    matching the H8 InfoNCE convention applied at the paired level.
    """
    N, L, d = buf.shape
    span = 2 * T_paired
    # Need 1 pair shift = 2 raw tokens of additional context
    needed = span + 2
    assert L >= needed, f"need L>={needed}; got L={L}"
    n_wins = L - needed + 1

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, n_wins, (batch_size,), device=buf.device)
        rng = torch.arange(span, device=buf.device)
        outs = []
        for s_pairs in (0, 1):
            s = 2 * s_pairs
            pos = (off + s).unsqueeze(1) + rng.unsqueeze(0)         # (B, span)
            x = buf[seq.unsqueeze(1).expand(-1, span), pos].float()
            x_pairs = x.view(batch_size, T_paired, 2, d).sum(dim=2)
            outs.append(x_pairs)
        return torch.stack(outs, dim=1)                              # (B, 2, T_paired, d)
    return gen


def make_paired_multidistance_gen_gpu(buf: torch.Tensor, T_paired: int,
                                        shifts_in_pairs: list[int]):
    """Pair-summed input for H8 multi-distance contrastive training.

    `shifts_in_pairs` = list of pair-level shifts. Each shift s means the
    positive window starts s pairs (= 2s raw tokens) after the anchor.
    Output: (B, 1+K, T_paired, d).
    """
    N, L, d = buf.shape
    span = 2 * T_paired
    max_shift_pairs = max(shifts_in_pairs)
    needed = span + 2 * max_shift_pairs
    assert L >= needed, f"need L>={needed}; got L={L}"

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, L - needed + 1, (batch_size,), device=buf.device)
        rng = torch.arange(span, device=buf.device)
        outs = []
        for s_pairs in [0] + list(shifts_in_pairs):
            s = 2 * s_pairs
            pos = (off + s).unsqueeze(1) + rng.unsqueeze(0)
            x = buf[seq.unsqueeze(1).expand(-1, span), pos].float()
            x_pairs = x.view(batch_size, T_paired, 2, d).sum(dim=2)
            outs.append(x_pairs)
        return torch.stack(outs, dim=1)
    return gen


# ─────────────────────────────────────────── training loop


def compute_plateau(losses: list[float], window: int = 5) -> float | None:
    if len(losses) < 2 * window:
        return None
    recent = sum(losses[-window:]) / window
    prior = sum(losses[-2 * window:-window]) / window
    if prior == 0:
        return None
    return (prior - recent) / abs(prior)


@dataclass
class TrainCfg:
    lr: float = 3e-4
    batch_size: int = 1024
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
    """Plain reconstruction-loss training loop.

    `grad_post_hook`: called after `loss.backward()` and BEFORE optimiser
        step — for things like remove_gradient_parallel_to_decoder.
    `normalize_decoder`: called AFTER opt.step().
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    losses: list[float] = []
    l0s: list[float] = []
    steps_logged: list[int] = []
    converged = False
    plateau_val: float | None = None

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
