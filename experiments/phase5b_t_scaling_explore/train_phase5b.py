"""Phase 5B training driver.

Per-arch training functions + a `run_one(arch_id, **kwargs)` dispatcher.
Each function returns (model, training_log_dict). Caller saves ckpt +
appends to `training_index.jsonl`.

All arch_ids prefixed with `phase5b_` to avoid collision with Phase 5
registry. Output paths under `experiments/phase5b_t_scaling_explore/results/`.

Run examples (from repo root):

    .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
        --arch phase5b_strided_t5_s2 --seed 42

    .venv/bin/python -m experiments.phase5b_t_scaling_explore.train_phase5b \
        --arch phase5b_track2_paired --T_paired 5 --seed 42

`run_arch_list()` is the bulk-runner used by orchestration scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase5b_t_scaling_explore._paths import (
    CKPT_DIR, LOGS_DIR, INDEX_PATH, OUT_DIR, DEFAULT_D_SAE,
)
from experiments.phase5b_t_scaling_explore._train_utils import (
    TrainCfg, iterate_train, compute_plateau,
    preload_single,
    make_window_gen_gpu, make_pair_window_gen_gpu,
    make_multidistance_pair_gen_gpu,
    make_strided_window_gen_gpu,
    make_paired_input_gen_gpu, make_paired_pair_window_gen_gpu,
    make_paired_multidistance_gen_gpu,
)


# ─────────────────────────────────────────── candidate trainers


def train_phase5b_strided_track2(
    cfg, device, T_eff: int, stride: int,
    k_pos: int = 100, d_sae: int = DEFAULT_D_SAE, buf=None,
):
    """D1/D2 base: Track 2 anti-dead stack on a strided window."""
    from src.architectures.phase5b_strided_txcdr import StridedTXCBareAntidead
    buf = buf if buf is not None else preload_single(device=device)
    k_eff = k_pos * T_eff
    model = StridedTXCBareAntidead(
        buf.shape[-1], d_sae, T_eff=T_eff, k=k_eff, stride=stride,
    ).to(device)
    gen = make_strided_window_gen_gpu(buf, T_eff=T_eff, stride=stride)

    # Init b_dec from a sample
    x0 = gen(cfg.batch_size)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    model.init_b_dec_geometric_median(x0)

    losses, l0s, steps = [], [], []
    converged = False
    plateau_val = None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x)
        opt.zero_grad()
        loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item())
            l0s.append(l0)
            steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if (plateau_val is not None and plateau_val < cfg.plateau_threshold
                    and step >= cfg.min_steps):
                converged = True
                break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T_eff": T_eff, "stride": stride,
    }


def train_phase5b_strided_h8(
    cfg, device, T_eff: int, stride: int,
    k_pos: int = 100, d_sae: int = DEFAULT_D_SAE,
    alpha: float = 1.0,
    matryoshka_h_size: int | None = None,
    buf=None,
):
    """D3: H8 (multi-distance contrastive) on strided window.

    The strided sampling is applied to ALL windows (anchor + each shift)
    consistently, so contrastive shift semantics are preserved at the
    POST-stride level.
    """
    from src.architectures.phase5b_strided_txcdr import StridedH8
    buf = buf if buf is not None else preload_single(device=device)
    k_eff = k_pos * T_eff
    if matryoshka_h_size is None:
        matryoshka_h_size = int(d_sae * 0.2)
    shifts = tuple(s for s in (1, max(1, T_eff // 4), max(1, T_eff // 2))
                    if 1 <= s <= T_eff - 1)
    shifts = tuple(sorted(set(shifts)))

    model = StridedH8(
        buf.shape[-1], d_sae, T_eff=T_eff, k=k_eff, stride=stride,
        shifts=shifts, weights=None,
        matryoshka_h_size=matryoshka_h_size, alpha=alpha,
    ).to(device)

    # For training, build a multi-distance generator that returns post-stride
    # windows (B, 1+K, T_eff, d). We'll do this by sampling raw windows of
    # appropriate span for each shift, then strided-sampling each.
    span = T_eff * stride
    needed = span + max(shifts) * stride  # shifts are in POST-stride units
    N, L, d = buf.shape
    assert L >= needed, f"need L>={needed} for T_eff={T_eff} stride={stride} max_shift={max(shifts)}"

    def gen(B: int) -> torch.Tensor:
        seq = torch.randint(0, N, (B,), device=buf.device)
        off = torch.randint(0, L - needed + 1, (B,), device=buf.device)
        rng_span = torch.arange(0, span, stride, device=buf.device)
        outs = []
        for s in [0] + list(shifts):
            base = off + s * stride                          # raw token shift = post-stride shift × stride
            pos = base.unsqueeze(1) + rng_span.unsqueeze(0)  # (B, T_eff)
            w = buf[seq.unsqueeze(1).expand(-1, T_eff), pos].float()
            outs.append(w)
        return torch.stack(outs, dim=1)                       # (B, 1+K, T_eff, d)

    x0 = gen(cfg.batch_size)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    model.init_b_dec_geometric_median(x0[:, 0])
    losses, l0s, steps = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x, alpha=alpha)
        opt.zero_grad(); loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(l0); steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True
                break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed, "T_eff": T_eff, "stride": stride,
        "shifts": list(shifts), "alpha": alpha,
    }


def train_phase5b_subseq_track2(
    cfg, device, T_max: int, t_sample: int,
    contiguous: bool = False,
    k_pos: int = 100, k_win: int | None = None,
    d_sae: int = DEFAULT_D_SAE, buf=None,
):
    """B1/B2/B3 base: Track 2 + subsequence sampling.

    `k_win` (if given) overrides the default `k_pos * t_sample` so 2D sweeps
    over t_sample can be done with fixed sparsity. Default keeps the
    Phase 5 convention (k_pos × T_window).
    """
    from src.architectures.phase5b_subseq_sampling_txcdr import SubseqTXCBareAntidead
    buf = buf if buf is not None else preload_single(device=device)
    k_eff = k_win if k_win is not None else k_pos * t_sample
    model = SubseqTXCBareAntidead(
        buf.shape[-1], d_sae, T_max=T_max, k=k_eff,
        t_sample=t_sample, contiguous=contiguous,
    ).to(device)
    gen = make_window_gen_gpu(buf, T_max)

    x0 = gen(cfg.batch_size)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    model.init_b_dec_geometric_median(x0)
    losses, l0s, steps = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x)
        opt.zero_grad(); loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(l0); steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T_max": T_max, "t_sample": t_sample, "contiguous": contiguous,
    }


def train_phase5b_subseq_h8(
    cfg, device, T_max: int, t_sample: int, contiguous: bool = False,
    k_pos: int = 100, k_win: int | None = None,
    d_sae: int = DEFAULT_D_SAE,
    alpha: float = 1.0, matryoshka_h_size: int | None = None,
    buf=None,
):
    """B4: H8 + subsequence sampling. Multi-distance contrastive on the
    summed-subset latent.

    `k_win` (if given) overrides `k_pos * t_sample` for fixed-sparsity
    t_sample sweeps.
    """
    from src.architectures.phase5b_subseq_sampling_txcdr import SubseqH8
    buf = buf if buf is not None else preload_single(device=device)
    k_eff = k_win if k_win is not None else k_pos * t_sample
    if matryoshka_h_size is None:
        matryoshka_h_size = int(d_sae * 0.2)
    shifts = tuple(s for s in (1, max(1, T_max // 4), max(1, T_max // 2))
                    if 1 <= s <= T_max - 1)
    shifts = tuple(sorted(set(shifts)))

    model = SubseqH8(
        buf.shape[-1], d_sae, T_max=T_max, k=k_eff,
        t_sample=t_sample, contiguous=contiguous,
        shifts=shifts, weights=None,
        matryoshka_h_size=matryoshka_h_size, alpha=alpha,
    ).to(device)

    gen = make_multidistance_pair_gen_gpu(buf, T_max, list(shifts))
    x0 = gen(cfg.batch_size)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    model.init_b_dec_geometric_median(x0[:, 0])
    losses, l0s, steps = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x, alpha=alpha)
        opt.zero_grad(); loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(l0); steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T_max": T_max, "t_sample": t_sample, "contiguous": contiguous,
        "shifts": list(shifts), "alpha": alpha,
    }


def train_phase5b_token_subseq(
    cfg, device, t_sample: int, pos_mode: str = "none",
    L_max: int = 128,
    k_pos: int = 100, d_sae: int = DEFAULT_D_SAE, buf=None,
):
    """C1/C2/C3: token-level encoder + sparse sum over t_sample positions.

    pos_mode: "none"=C1, "sinusoidal"=C2, "learned"=C3.
    """
    from src.architectures.phase5b_token_subseq_sae import TokenSubseqSAE
    buf = buf if buf is not None else preload_single(device=device)
    model = TokenSubseqSAE(
        buf.shape[-1], d_sae, L_max=L_max, k=k_pos,
        t_sample=t_sample, pos_mode=pos_mode,
    ).to(device)

    N, L, d = buf.shape
    assert L >= L_max, f"buf has L={L} < L_max={L_max}"

    def gen(B: int) -> torch.Tensor:
        # Sample t_sample positions per row (with replacement allowed for simplicity)
        seq = torch.randint(0, N, (B,), device=buf.device)
        # Use last L_max positions (closest to "informative")
        # Sample subset positions via per-row randperm (no replacement)
        keys = torch.rand(B, L_max, device=buf.device)
        _, idx = keys.topk(t_sample, dim=-1)
        idx, _ = idx.sort(dim=-1)
        # Map sampled indices into the BUF's coordinate system: just position in [0, L_max)
        # since we use first L_max positions of buf (or last; either works)
        gather_idx = idx                                           # (B, t_sample)
        gi = gather_idx.unsqueeze(-1).expand(-1, -1, d)
        x = buf[seq.unsqueeze(1).expand(-1, t_sample), gather_idx].float()
        return x, gather_idx

    # Init geometric median once on a sample's first position (proxy)
    # Skip — TokenSubseqSAE has no init_b_dec method by design

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses, l0s, steps = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x, positions = gen(cfg.batch_size)
        loss, _, z_sum = model(x, positions=positions)
        opt.zero_grad(); loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z_sum > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(l0); steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "L_max": L_max, "t_sample": t_sample, "pos_mode": pos_mode,
        "use_pos": pos_mode == "sinusoidal",
    }


def train_phase5b_paired_track2(
    cfg, device, T_paired: int,
    k_pos: int = 100, d_sae: int = DEFAULT_D_SAE, buf=None,
):
    """E1: Track 2 on pair-summed input (T_paired pairs, T_eff = 2 * T_paired)."""
    from src.architectures.txc_bare_antidead import TXCBareAntidead
    buf = buf if buf is not None else preload_single(device=device)
    k_eff = k_pos * T_paired  # match per-position sparsity convention (k_win = k_pos × T)
    model = TXCBareAntidead(
        buf.shape[-1], d_sae, T_paired, k_eff,
    ).to(device)
    gen = make_paired_input_gen_gpu(buf, T_paired)

    x0 = gen(cfg.batch_size)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    model.init_b_dec_geometric_median(x0)
    losses, l0s, steps = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x)
        opt.zero_grad(); loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(l0); steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T_paired": T_paired, "T_eff": 2 * T_paired,
    }


def train_phase5b_paired_h8(
    cfg, device, T_paired: int,
    k_pos: int = 100, d_sae: int = DEFAULT_D_SAE,
    alpha: float = 1.0,
    matryoshka_h_size: int | None = None,
    buf=None,
):
    """E2: H8 (multi-distance contrastive) on pair-summed input."""
    from src.architectures.txc_bare_multidistance_contrastive_antidead import (
        TXCBareMultiDistanceContrastiveAntidead,
    )
    buf = buf if buf is not None else preload_single(device=device)
    k_eff = k_pos * T_paired
    if matryoshka_h_size is None:
        matryoshka_h_size = int(d_sae * 0.2)
    # H8 shifts in PAIR space — i.e., per pair-step.
    shifts = tuple(s for s in (1, max(1, T_paired // 4), max(1, T_paired // 2))
                    if 1 <= s <= T_paired - 1)
    shifts = tuple(sorted(set(shifts)))

    model = TXCBareMultiDistanceContrastiveAntidead(
        buf.shape[-1], d_sae, T_paired, k_eff,
        shifts=shifts, weights=None,
        matryoshka_h_size=matryoshka_h_size, alpha=alpha,
    ).to(device)

    gen = make_paired_multidistance_gen_gpu(buf, T_paired, list(shifts))
    x0 = gen(cfg.batch_size)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    model.init_b_dec_geometric_median(x0[:, 0])
    losses, l0s, steps = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x, alpha=alpha)
        opt.zero_grad(); loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(l0); steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T_paired": T_paired, "T_eff": 2 * T_paired,
        "shifts": list(shifts), "alpha": alpha,
    }


def train_phase5b_pps_track2(
    cfg, device, T: int, n_scales: int = 2,
    alpha: float = 0.0, contr_shifts: tuple[int, ...] = (1,),
    k_pos: int = 100, d_sae: int = DEFAULT_D_SAE, buf=None,
):
    """A1: per-(pos, scale) matryoshka, anti-dead, optional contrastive."""
    from src.architectures.phase5b_per_pos_scale_matryoshka import PerPosScaleMatryoshkaTXC
    buf = buf if buf is not None else preload_single(device=device)
    k_eff = k_pos * T
    model = PerPosScaleMatryoshkaTXC(
        buf.shape[-1], d_sae, T, k_eff,
        n_scales=n_scales,
        alpha=alpha, contr_shifts=contr_shifts,
    ).to(device)

    if alpha > 0.0 and len(contr_shifts) > 0:
        gen = make_multidistance_pair_gen_gpu(buf, T, list(contr_shifts))
    elif alpha > 0.0:
        gen = make_pair_window_gen_gpu(buf, T)
    else:
        gen = make_window_gen_gpu(buf, T)

    x0 = gen(cfg.batch_size)
    x0_single = x0[:, 0] if x0.ndim == 4 else x0

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    model.init_b_dec_geometric_median(x0_single)
    losses, l0s, steps = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x, alpha=alpha) if alpha > 0 else model(x)
        opt.zero_grad(); loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(l0); steps.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    return model, {
        "loss": losses, "l0": l0s, "steps_logged": steps,
        "final_step": steps[-1] if steps else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T": T, "n_scales": n_scales, "alpha": alpha,
        "contr_shifts": list(contr_shifts),
    }


# ─────────────────────────────────────────── orchestration


def _save_run(model, log, run_id: str, meta: dict):
    """Save ckpt + log + index row."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    # Save fp16 to save space; convert back at load time.
    state_fp16 = {k: v.half() if v.dtype == torch.float32 else v
                   for k, v in model.state_dict().items()}
    torch.save(state_fp16, ckpt_path)

    log_path = LOGS_DIR / f"{run_id}.json"
    with open(log_path, "w") as f:
        json.dump({**log, **meta, "run_id": run_id}, f, indent=2)

    # Append index row
    INDEX_PATH.touch(exist_ok=True)
    with open(INDEX_PATH, "a") as f:
        f.write(json.dumps({
            "run_id": run_id, **meta,
            "final_step": log.get("final_step"),
            "converged": log.get("converged"),
            "elapsed_s": log.get("elapsed_s"),
            "ckpt": str(ckpt_path),
        }) + "\n")
    print(f"  [save] {run_id} → {ckpt_path}")


def run_one(arch_id: str, seed: int = 42, max_steps: int = 25_000,
            **kwargs) -> dict:
    """Run a single Phase 5B candidate. Saves ckpt + log + index row.

    Returns the meta dict that was logged.
    """
    cfg = TrainCfg(seed=seed, max_steps=max_steps)
    device = torch.device("cuda")
    buf = preload_single(device=device)
    print(f"=== {arch_id} seed={seed} ===")
    print(f"  [data] anchor preloaded: {tuple(buf.shape)}")

    if arch_id.startswith("phase5b_strided_track2"):
        T_eff = kwargs.get("T_eff", 5)
        stride = kwargs.get("stride", 2)
        model, log = train_phase5b_strided_track2(
            cfg, device, T_eff=T_eff, stride=stride, buf=buf,
        )
        meta = dict(arch="strided_track2", T_eff=T_eff, stride=stride, seed=seed,
                    k_pos=100, k_win=100*T_eff, layer=13)

    elif arch_id.startswith("phase5b_strided_h8"):
        T_eff = kwargs.get("T_eff", 5)
        stride = kwargs.get("stride", 2)
        model, log = train_phase5b_strided_h8(
            cfg, device, T_eff=T_eff, stride=stride, buf=buf,
        )
        meta = dict(arch="strided_h8", T_eff=T_eff, stride=stride, seed=seed,
                    k_pos=100, k_win=100*T_eff, alpha=1.0, layer=13)

    elif arch_id.startswith("phase5b_subseq_track2"):
        T_max = kwargs.get("T_max", 10)
        t_sample = kwargs.get("t_sample", 5)
        contig = kwargs.get("contiguous", False)
        k_win_override = kwargs.get("k_win")
        model, log = train_phase5b_subseq_track2(
            cfg, device, T_max=T_max, t_sample=t_sample, contiguous=contig,
            k_win=k_win_override, buf=buf,
        )
        k_win_used = k_win_override if k_win_override is not None else 100 * t_sample
        meta = dict(arch="subseq_track2", T_max=T_max, t_sample=t_sample,
                    contiguous=contig, seed=seed, k_pos=100,
                    k_win=k_win_used, layer=13)

    elif arch_id.startswith("phase5b_subseq_h8"):
        T_max = kwargs.get("T_max", 10)
        t_sample = kwargs.get("t_sample", 5)
        contig = kwargs.get("contiguous", False)
        k_win_override = kwargs.get("k_win")
        model, log = train_phase5b_subseq_h8(
            cfg, device, T_max=T_max, t_sample=t_sample, contiguous=contig,
            k_win=k_win_override, buf=buf,
        )
        k_win_used = k_win_override if k_win_override is not None else 100 * t_sample
        meta = dict(arch="subseq_h8", T_max=T_max, t_sample=t_sample,
                    contiguous=contig, seed=seed, k_pos=100,
                    k_win=k_win_used, alpha=1.0, layer=13)

    elif arch_id.startswith("phase5b_token_subseq"):
        t_sample = kwargs.get("t_sample", 5)
        pos_mode = kwargs.get("pos_mode") or (
            "sinusoidal" if kwargs.get("use_pos") else "none"
        )
        L_max = kwargs.get("L_max", 128)
        model, log = train_phase5b_token_subseq(
            cfg, device, t_sample=t_sample, pos_mode=pos_mode,
            L_max=L_max, buf=buf,
        )
        meta = dict(arch="token_subseq", L_max=L_max, t_sample=t_sample,
                    pos_mode=pos_mode,
                    use_pos=(pos_mode == "sinusoidal"),
                    seed=seed, k_pos=100, layer=13)
        # Disambiguate C1/C2/C3 ckpts via run_id suffix
        if not arch_id.endswith(f"_{pos_mode}"):
            arch_id = f"{arch_id}_{pos_mode}"

    elif arch_id.startswith("phase5b_track2_paired"):
        T_paired = kwargs.get("T_paired", 5)
        model, log = train_phase5b_paired_track2(
            cfg, device, T_paired=T_paired, buf=buf,
        )
        meta = dict(arch="paired_track2", T_paired=T_paired, T_eff=2*T_paired,
                    seed=seed, k_pos=100, k_win=100*T_paired, layer=13)

    elif arch_id.startswith("phase5b_h8_paired"):
        T_paired = kwargs.get("T_paired", 5)
        model, log = train_phase5b_paired_h8(
            cfg, device, T_paired=T_paired, buf=buf,
        )
        meta = dict(arch="paired_h8", T_paired=T_paired, T_eff=2*T_paired,
                    seed=seed, k_pos=100, k_win=100*T_paired, alpha=1.0, layer=13)

    elif arch_id.startswith("phase5b_pps_track2"):
        T = kwargs.get("T", 5)
        n_scales = kwargs.get("n_scales", 2)
        alpha = kwargs.get("alpha", 0.0)
        contr_shifts = tuple(kwargs.get("contr_shifts", (1,)))
        model, log = train_phase5b_pps_track2(
            cfg, device, T=T, n_scales=n_scales, alpha=alpha,
            contr_shifts=contr_shifts, buf=buf,
        )
        meta = dict(arch="pps_matryoshka", T=T, n_scales=n_scales, alpha=alpha,
                    contr_shifts=list(contr_shifts), seed=seed, k_pos=100,
                    k_win=100*T, layer=13)

    else:
        raise ValueError(f"Unknown arch_id: {arch_id}")

    _save_run(model, log, arch_id + (f"__seed{seed}"), meta)
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=25_000)
    p.add_argument("--T_eff", type=int, default=5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--T_max", type=int, default=10)
    p.add_argument("--t_sample", type=int, default=5)
    p.add_argument("--T_paired", type=int, default=5)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--n_scales", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--use_pos", action="store_true")
    p.add_argument("--pos_mode", type=str, default=None,
                    choices=["none", "sinusoidal", "learned", None])
    p.add_argument("--contiguous", action="store_true")
    p.add_argument("--k_win", type=int, default=None,
                    help="Override k_win (default: k_pos * t_sample). Use to "
                          "decouple sparsity from t_sample in subseq sweeps.")
    args = p.parse_args()

    kwargs = {
        "T_eff": args.T_eff, "stride": args.stride,
        "T_max": args.T_max, "t_sample": args.t_sample,
        "T_paired": args.T_paired, "T": args.T,
        "n_scales": args.n_scales, "alpha": args.alpha,
        "use_pos": args.use_pos, "pos_mode": args.pos_mode,
        "contiguous": args.contiguous, "k_win": args.k_win,
    }
    run_one(args.arch, seed=args.seed, max_steps=args.max_steps, **kwargs)
