"""Phase 7 training driver — canonical_archs.json driven.

Outer loop: seed in (42, 1, 2). Inner loop: 49 archs from
`canonical_archs.json`. After each seed-batch completes, posts a
`seed{N}_complete.json` marker (locally + HF) signalling Agent B.

Per-arch routing: arch_id → trainer function. 12 trainer fns cover the
12 distinct (model_class, training-pattern) combinations in the
canonical set. Each returns (model, log_dict). Caller saves ckpt +
training_log JSON + appends to training_index.jsonl + uploads to HF.

Subject-model + anchor-layer are baked into every saved meta dict so
the HF uploader can verify (`scripts/hf_upload_phase7_ckpts.py`).

Run examples (from repo root):

    .venv/bin/python -m experiments.phase7_unification.train_phase7 \\
        --arch txcdr_t5 --seed 42

    # Run the whole canonical set, single seed:
    .venv/bin/python -m experiments.phase7_unification.train_phase7 \\
        --canonical --seed 42

    # Full 3-seed campaign (outer-loop seed):
    .venv/bin/python -m experiments.phase7_unification.train_phase7 --all
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

from experiments.phase7_unification._paths import (
    CANONICAL_ARCHS_JSON, CKPT_DIR, LOGS_DIR, OUT_DIR, INDEX_PATH,
    SEED_MARKER_DIR, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL,
    DEFAULT_D_IN, DEFAULT_D_SAE, SEEDS, HF_CKPT_REPO, banner,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, iterate_train, compute_plateau,
    preload_single, preload_multilayer,
    make_window_gen_gpu, make_pair_window_gen_gpu,
    make_multidistance_pair_gen_gpu,
)


# ════════════════════════════════════════════════════════════════════
# Canonical arch loading
# ════════════════════════════════════════════════════════════════════


def load_canonical() -> dict:
    """Load canonical_archs.json. Returns the parsed dict."""
    return json.loads(CANONICAL_ARCHS_JSON.read_text())


def find_arch(canonical: dict, arch_id: str) -> dict:
    """Look up an arch by arch_id. Raises if missing."""
    for arch in canonical["archs"]:
        if arch["arch_id"] == arch_id:
            return arch
    raise ValueError(f"arch_id={arch_id} not found in canonical_archs.json")


# ════════════════════════════════════════════════════════════════════
# Per-arch trainers — dispatched by group/recipe
# ════════════════════════════════════════════════════════════════════


def _flat_train(model, gen_fn, cfg, init_x_for_geom_median=None) -> dict:
    """Plain reconstruction loop with optional b_dec geometric-median init
    + decoder unit-norm post-step. Used by TopK / TXCDR / Track-2 / MLC.
    """
    if init_x_for_geom_median is not None and hasattr(model, "init_b_dec_geometric_median"):
        model.init_b_dec_geometric_median(init_x_for_geom_median)
    grad_post = (
        model.remove_gradient_parallel_to_decoder
        if hasattr(model, "remove_gradient_parallel_to_decoder") else None
    )
    norm_dec = (
        model._normalize_decoder
        if hasattr(model, "_normalize_decoder") else None
    )
    return iterate_train(
        model, gen_fn, cfg, torch.device("cuda"),
        normalize_decoder=norm_dec, grad_post_hook=grad_post,
    )


def _contrastive_train(model, gen_fn, cfg, alpha: float = 1.0,
                       init_x_for_geom_median=None) -> dict:
    """Pair / multi-distance contrastive loop. model(x, alpha=alpha) returns
    (loss, _, z). Sample generator emits (B, 1+K, T, d) for multidistance
    or (B, 2, T, d) for single shift.
    """
    device = torch.device("cuda")
    if init_x_for_geom_median is not None and hasattr(model, "init_b_dec_geometric_median"):
        model.init_b_dec_geometric_median(init_x_for_geom_median)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses, l0s, steps_logged = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen_fn(cfg.batch_size)
        loss, _, z = model(x, alpha=alpha)
        opt.zero_grad()
        loss.backward()
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
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
        "loss": losses, "l0": l0s, "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
    }


# ────── Group 1: per-token / non-TXC ─────────────────────────────


def train_topk_sae(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Row 1: TopKSAE — per-token TopK SAE baseline."""
    from src.architectures.topk_sae import TopKSAE
    from experiments.phase7_unification._train_utils import make_flat_gen_gpu
    model = TopKSAE(DEFAULT_D_IN, DEFAULT_D_SAE, k=arch["k_pos"]).to("cuda")
    gen = make_flat_gen_gpu(buf_anchor)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=gen(cfg.batch_size))
    return model, log


def train_tsae_paper(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Rows 2, 3: TemporalMatryoshkaBatchTopKSAE — T-SAE port (Ye et al. 2025).

    Per-token SAE with Matryoshka BatchTopK + temporal InfoNCE + AuxK.
    Uses TemporalMatryoshkaBatchTopKTrainerLite per src/architectures/tsae_paper.py
    (the trainer owns the loss + threshold EMA + dead-feature aux loss).
    Adapted from experiments/phase6_qualitative_latents/train_tsae_paper.py.
    """
    import math as _math
    from src.architectures.tsae_paper import (
        TemporalMatryoshkaBatchTopKSAE,
        TemporalMatryoshkaBatchTopKTrainerLite,
        geometric_median,
    )
    d_sae = DEFAULT_D_SAE
    # Phase 6 default: 2 groups at 20%/80% split. group_weights = equal.
    group_fractions = [0.2, 0.8]
    group_sizes = [int(f * d_sae) for f in group_fractions[:-1]]
    group_sizes.append(d_sae - sum(group_sizes))
    group_weights = [1.0 / len(group_sizes)] * len(group_sizes)
    k = arch["k_pos"]
    model = TemporalMatryoshkaBatchTopKSAE(
        activation_dim=DEFAULT_D_IN, dict_size=d_sae,
        k=k, group_sizes=group_sizes,
    ).to("cuda")
    # Paper's lr scaling law: 2e-4 / sqrt(d_sae / 16384).
    lr = 2e-4 / _math.sqrt(d_sae / 16384)
    decay_start = int(0.8 * cfg.max_steps)
    trainer = TemporalMatryoshkaBatchTopKTrainerLite(
        model, group_weights=group_weights, total_steps=cfg.max_steps, lr=lr,
        warmup_steps=1000, decay_start=decay_start,
        device=torch.device("cuda"),
    )
    # Pair sampler: adjacent token pairs (B, 2, d).
    N, L, d = buf_anchor.shape
    n_wins = L - 1
    def pair_gen(_step: int) -> torch.Tensor:
        seq = torch.randint(0, N, (cfg.batch_size,), device=buf_anchor.device)
        off = torch.randint(0, n_wins, (cfg.batch_size,), device=buf_anchor.device)
        a = buf_anchor[seq, off]
        b = buf_anchor[seq, off + 1]
        return torch.stack([a, b], dim=1).float()    # (B, 2, d)

    # b_dec geometric median init from a sample.
    x0 = pair_gen(0)[:, 0]                            # (B, d)
    bdec = geometric_median(x0)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    losses, l0s, auxks, deads, steps_logged = [], [], [], [], []
    t0 = time.time()
    converged, plateau_val = False, None
    for step in range(cfg.max_steps):
        x_pair = pair_gen(step)
        stats = trainer.update(step, x_pair, b_dec_init=bdec if step == 0 else None)
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                z, _, _ = model.encode(x_pair[:, 0], return_active=True, use_threshold=False)
                l0 = float((z > 0).sum(dim=-1).float().mean().item())
            losses.append(float(stats["l2"]))
            auxks.append(float(stats["auxk"]))
            deads.append(int(stats["dead"]))
            l0s.append(l0)
            steps_logged.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if (plateau_val is not None and plateau_val < cfg.plateau_threshold
                    and step >= cfg.min_steps):
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    log = {
        "loss": losses, "l0": l0s, "auxk": auxks, "dead": deads,
        "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "group_sizes": group_sizes, "group_weights": group_weights,
        "lr_used": lr,
    }
    return model, log


def train_mlc(arch: dict, cfg: TrainCfg, buf_multilayer) -> tuple:
    """Row 4: MultiLayerCrosscoder — per-token TopK over 5 layers."""
    from src.architectures.mlc import MultiLayerCrosscoder
    n_layers = arch.get("n_layers", len(MLC_LAYERS))
    model = MultiLayerCrosscoder(
        DEFAULT_D_IN, DEFAULT_D_SAE, n_layers=n_layers, k=arch["k_win"],
    ).to("cuda")
    # MLC ingests (B, n_layers, d). Sample a single position per row:
    N, L, n_lay, d = buf_multilayer.shape
    def gen(B: int) -> torch.Tensor:
        seq = torch.randint(0, N, (B,), device=buf_multilayer.device)
        tok = torch.randint(0, L, (B,), device=buf_multilayer.device)
        return buf_multilayer[seq, tok].float()      # (B, n_lay, d)
    init_x = gen(cfg.batch_size)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=init_x)
    return model, log


def train_mlc_contrastive(arch: dict, cfg: TrainCfg, buf_multilayer) -> tuple:
    """Row 5: MLCContrastive — Matryoshka H/L + temporal InfoNCE (alpha=1.0)
    + BatchTopK. Pair-window sampling at the multi-layer cube.
    """
    from src.architectures.mlc_contrastive import MLCContrastive
    h = int(DEFAULT_D_SAE * 0.2)  # Matryoshka head ratio
    n_layers = arch.get("n_layers", len(MLC_LAYERS))
    model = MLCContrastive(
        DEFAULT_D_IN, DEFAULT_D_SAE, n_layers=n_layers, k=arch["k_win"], h=h,
    ).to("cuda")
    N, L, n_lay, d = buf_multilayer.shape
    def gen(B: int) -> torch.Tensor:
        seq = torch.randint(0, N, (B,), device=buf_multilayer.device)
        off = torch.randint(0, L - 1, (B,), device=buf_multilayer.device)
        x_cur = buf_multilayer[seq, off].float()                 # (B, n_lay, d)
        x_next = buf_multilayer[seq, off + 1].float()
        return torch.stack([x_cur, x_next], dim=1)               # (B, 2, n_lay, d)
    alpha = arch.get("alpha", 1.0)
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=alpha, init_x_for_geom_median=init_x)
    log["h"] = h; log["alpha"] = alpha
    return model, log


def train_mlc_contrastive_multiscale(arch: dict, cfg: TrainCfg, buf_multilayer) -> tuple:
    """Row 6: MLCContrastiveMultiscale — multi-scale InfoNCE."""
    from src.architectures.mlc_contrastive_multiscale import MLCContrastiveMultiscale
    h = int(DEFAULT_D_SAE * 0.2)
    n_layers = arch.get("n_layers", len(MLC_LAYERS))
    model = MLCContrastiveMultiscale(
        DEFAULT_D_IN, DEFAULT_D_SAE, n_layers=n_layers, k=arch["k_win"],
        gamma=arch.get("gamma", 0.5), h=h,
    ).to("cuda")
    N, L, n_lay, d = buf_multilayer.shape
    def gen(B: int) -> torch.Tensor:
        seq = torch.randint(0, N, (B,), device=buf_multilayer.device)
        off = torch.randint(0, L - 1, (B,), device=buf_multilayer.device)
        x_cur = buf_multilayer[seq, off].float()
        x_next = buf_multilayer[seq, off + 1].float()
        return torch.stack([x_cur, x_next], dim=1)
    alpha = arch.get("alpha", 1.0)
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=alpha, init_x_for_geom_median=init_x)
    log["h"] = h; log["gamma"] = arch.get("gamma", 0.5)
    log["n_scales"] = arch.get("n_scales", 3); log["alpha"] = alpha
    return model, log


def train_tfa(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Row 7: TFA — TemporalSAE with input scaling + decoder unit-norm.

    TFA processes FULL sequences (B, T=128, d) through attention + an
    SAE on every token, so memory scales as B * T * d_sae. At
    cfg.batch_size=4096 the d_sae=18432 forward needs ~36 GB which
    OOMs on a 141 GB H200 already holding 84 GB of preloaded buffers
    + ~10 GB model. Override to a much smaller batch (TFA-specific).
    Phase 5's `train_primary_archs.py` defaulted to tfa_batch_size=64
    — same convention here.

    Other TFA-specific handling per src/architectures/tfa.py:
    - Input scaling sqrt(d)/mean_norm so internal lam=1/(4*dimin) sees
      sqrt(d)-norm inputs.
    - Decoder rows renormalised to unit norm after every step.
    - Skip NaN/inf loss + NaN grad batches; cosine LR with warmup +
      decay to 0.1*lr.
    """
    import math
    from src.architectures._tfa_module import TemporalSAE

    # TFA-specific batch size override (see docstring). Use 32 to be
    # safe given we still hold the multilayer buffer in GPU.
    TFA_BATCH = 32
    cfg = TrainCfg(seed=cfg.seed, max_steps=cfg.max_steps, lr=cfg.lr,
                    batch_size=TFA_BATCH, log_every=cfg.log_every,
                    grad_clip=cfg.grad_clip,
                    plateau_threshold=cfg.plateau_threshold,
                    min_steps=cfg.min_steps)

    # TFA at "novel head" k = full TopK budget. Use k_win as kval_topk.
    k = arch["k_win"]
    seq_len = buf_anchor.shape[1]
    model = TemporalSAE(
        dimin=DEFAULT_D_IN, width=DEFAULT_D_SAE, n_heads=4,
        sae_diff_type="topk", kval_topk=k, tied_weights=True,
        n_attn_layers=1, bottleneck_factor=4,
        use_pos_encoding=False, max_seq_len=seq_len,
    ).to("cuda")

    N, L, d = buf_anchor.shape
    # TFA processes full sequences; sample whole context.
    def gen(B: int) -> torch.Tensor:
        idx = torch.randint(0, N, (B,), device=buf_anchor.device)
        return buf_anchor[idx].float()                # (B, L, d)

    # Compute input scale on first batch.
    sample = gen(64)
    mean_norm = sample.norm(dim=-1).mean().item()
    scaling = math.sqrt(d) / max(mean_norm, 1e-8)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-4},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    min_lr = cfg.lr * 0.1
    warmup = min(500, cfg.max_steps // 10)
    losses, l0s, steps_logged = [], [], []
    skipped = 0
    converged, plateau_val = False, None
    t0 = time.time()
    model.train()
    for step in range(cfg.max_steps):
        if step < warmup:
            cur_lr = cfg.lr * step / max(1, warmup)
        else:
            r = (step - warmup) / max(1, cfg.max_steps - warmup)
            cur_lr = min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (cfg.lr - min_lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        x = gen(cfg.batch_size) * scaling
        recons, inter = model(x)
        n_tok = x.shape[0] * x.shape[1]
        loss = ((recons - x) ** 2).sum() / n_tok
        if loss.isnan().any() or loss.isinf().any():
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if any(p.isnan().any() for p in model.parameters()):
                print(f"  [TFA] step {step}: params NaN, halt"); break
            continue
        opt.zero_grad(); loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if gn.isnan() or gn.isinf():
            skipped += 1; opt.zero_grad(set_to_none=True); continue
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        # Decoder unit-norm renorm.
        with torch.no_grad():
            norms = model.D.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            model.D.data.div_(norms)
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                novel_l0 = (inter["novel_codes"] > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item()); l0s.append(novel_l0); steps_logged.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if plateau_val is not None and plateau_val < cfg.plateau_threshold and step >= cfg.min_steps:
                converged = True; break
    elapsed = time.time() - t0
    model.eval()
    log = {
        "loss": losses, "l0": l0s, "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "scaling_factor": float(scaling), "skipped_steps": int(skipped),
    }
    return model, log


# ────── Group 2: fixed-T TXC variants ─────────────────────────────


def train_agentic_txc_02(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Row 8: MatryoshkaTXCDRContrastiveMultiscale at T=5."""
    from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
        MatryoshkaTXCDRContrastiveMultiscale,
    )
    T = arch["T"]; k = arch["k_win"]
    n_scales = arch.get("n_scales", 3); gamma = arch.get("gamma", 0.5)
    model = MatryoshkaTXCDRContrastiveMultiscale(
        DEFAULT_D_IN, DEFAULT_D_SAE, T=T, k=k,
        n_contr_scales=n_scales, gamma=gamma,
    ).to("cuda")
    gen = make_pair_window_gen_gpu(buf_anchor, T)
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)
    log["n_scales"] = n_scales; log["gamma"] = gamma
    return model, log


def train_txc_bare_antidead(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Rows 9-11: TXCBareAntidead (Track 2) at T ∈ {5, 10, 20}."""
    from src.architectures.txc_bare_antidead import TXCBareAntidead
    T = arch["T"]; k = arch["k_win"]
    model = TXCBareAntidead(DEFAULT_D_IN, DEFAULT_D_SAE, T, k).to("cuda")
    gen = make_window_gen_gpu(buf_anchor, T)
    init_x = gen(cfg.batch_size)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=init_x)
    return model, log


def train_subseq_track2(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Row 12: SubseqTXCBareAntidead — Track 2 + subseq sampling (B2)."""
    from src.architectures.phase5b_subseq_sampling_txcdr import SubseqTXCBareAntidead
    T_max = arch["T_max"]; t_sample = arch["t_sample"]; k = arch["k_win"]
    model = SubseqTXCBareAntidead(
        DEFAULT_D_IN, DEFAULT_D_SAE, T_max=T_max, k=k, t_sample=t_sample,
        contiguous=False,
    ).to("cuda")
    gen = make_window_gen_gpu(buf_anchor, T_max)
    init_x = gen(cfg.batch_size)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=init_x)
    log["T_max"] = T_max; log["t_sample"] = t_sample
    return model, log


def train_subseq_h8(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Rows 13, 48, 49: SubseqH8 — H8 stack + subseq sampling at varying T_max."""
    from src.architectures.phase5b_subseq_sampling_txcdr import SubseqH8
    T_max = arch["T_max"]; t_sample = arch["t_sample"]; k = arch["k_win"]
    h = int(DEFAULT_D_SAE * 0.2)
    # Auto-scaled shifts at T_max (consistent with phase5b convention):
    raw_shifts = (1, max(1, T_max // 4), max(1, T_max // 2))
    shifts = tuple(sorted(set(s for s in raw_shifts if 1 <= s <= T_max - 1)))
    model = SubseqH8(
        DEFAULT_D_IN, DEFAULT_D_SAE, T_max=T_max, k=k, t_sample=t_sample,
        contiguous=False, shifts=shifts, weights=None,
        matryoshka_h_size=h, alpha=1.0,
    ).to("cuda")
    gen = make_multidistance_pair_gen_gpu(buf_anchor, T_max, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)
    log["T_max"] = T_max; log["t_sample"] = t_sample
    log["shifts"] = list(shifts); log["matryoshka_h_size"] = h
    return model, log


# ────── Group 3 & 5 (anchor cell): vanilla TXCDR ──────────────────


def train_txcdr(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Rows 14-29 + row 46: TemporalCrosscoder — vanilla TXCDR.

    Anchor cell row 46 just has k_win=2000 instead of 500; arch fields
    encode that already, no special handling needed here.
    """
    from src.architectures.crosscoder import TemporalCrosscoder
    T = arch["T"]; k = arch["k_win"]
    model = TemporalCrosscoder(DEFAULT_D_IN, DEFAULT_D_SAE, T, k).to("cuda")
    gen = make_window_gen_gpu(buf_anchor, T)
    init_x = gen(cfg.batch_size)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=init_x)
    return model, log


# ────── Group 4 & 5 (anchor cell): H8 multidistance ───────────────


def train_h8_multidistance(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Rows 30-45 + row 47: TXCBareMultiDistanceContrastiveAntidead.

    H8 = TXC + anti-dead + Matryoshka H/L + multi-distance contrastive.
    Shifts come from canonical_archs.json (already T-scaled). Anchor cell
    row 47 just has k_win=2000 instead of 500; arch fields encode that.
    """
    from src.architectures.txc_bare_multidistance_contrastive_antidead import (
        TXCBareMultiDistanceContrastiveAntidead,
    )
    T = arch["T"]; k = arch["k_win"]
    h = int(DEFAULT_D_SAE * 0.2)
    shifts = tuple(arch["shifts"]) if arch.get("shifts") else (1,)
    model = TXCBareMultiDistanceContrastiveAntidead(
        DEFAULT_D_IN, DEFAULT_D_SAE, T, k,
        shifts=shifts, weights=None,
        matryoshka_h_size=h, alpha=1.0,
    ).to("cuda")
    gen = make_multidistance_pair_gen_gpu(buf_anchor, T, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)
    log["shifts"] = list(shifts); log["matryoshka_h_size"] = h; log["alpha"] = 1.0
    return model, log


# ════════════════════════════════════════════════════════════════════
# Dispatcher
# ════════════════════════════════════════════════════════════════════


def dispatch(arch: dict, cfg: TrainCfg,
             buf_anchor=None, buf_multilayer=None) -> tuple:
    """Route an arch dict to its trainer fn. Returns (model, log)."""
    arch_id = arch["arch_id"]
    src_class = arch["src_class"]
    # Per-class routing — uses src_class for unambiguous mapping.
    if src_class == "TopKSAE":
        return train_topk_sae(arch, cfg, buf_anchor)
    if src_class == "TemporalMatryoshkaBatchTopKSAE":
        return train_tsae_paper(arch, cfg, buf_anchor)
    if src_class == "MultiLayerCrosscoder":
        return train_mlc(arch, cfg, buf_multilayer)
    if src_class == "MLCContrastive":
        return train_mlc_contrastive(arch, cfg, buf_multilayer)
    if src_class == "MLCContrastiveMultiscale":
        return train_mlc_contrastive_multiscale(arch, cfg, buf_multilayer)
    if src_class == "TemporalSAE":   # TFA
        return train_tfa(arch, cfg, buf_anchor)
    if src_class == "MatryoshkaTXCDRContrastiveMultiscale":
        return train_agentic_txc_02(arch, cfg, buf_anchor)
    if src_class == "TXCBareAntidead":
        return train_txc_bare_antidead(arch, cfg, buf_anchor)
    if src_class == "SubseqTXCBareAntidead":
        return train_subseq_track2(arch, cfg, buf_anchor)
    if src_class == "SubseqH8":
        return train_subseq_h8(arch, cfg, buf_anchor)
    if src_class == "TemporalCrosscoder":
        return train_txcdr(arch, cfg, buf_anchor)
    if src_class == "TXCBareMultiDistanceContrastiveAntidead":
        return train_h8_multidistance(arch, cfg, buf_anchor)
    raise ValueError(f"Unknown src_class={src_class} for arch_id={arch_id}")


# ════════════════════════════════════════════════════════════════════
# Save + index + HF push
# ════════════════════════════════════════════════════════════════════


def _save_run(model, log: dict, run_id: str, meta: dict) -> Path:
    """Save fp16 ckpt + per-run training_log JSON + append to index.
    Returns the ckpt Path. meta MUST include subject_model + anchor_layer
    for the HF uploader's verification.
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    state_fp16 = {
        k: v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v
        for k, v in model.state_dict().items()
    }
    torch.save(state_fp16, ckpt_path)
    log_path = LOGS_DIR / f"{run_id}.json"
    with open(log_path, "w") as f:
        json.dump({**log, **meta, "run_id": run_id}, f, indent=2)
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
    return ckpt_path


def _meta_from_arch(arch: dict, seed: int) -> dict:
    """Build the meta dict saved alongside the ckpt + index row.

    Must include `subject_model`, `anchor_layer`, `mlc_layers` so
    `scripts/hf_upload_phase7_ckpts.py` can verify before pushing to
    `txcdr-base`.
    """
    return {
        "row": arch["row"],
        "arch_id": arch["arch_id"],
        "arch": arch["arch_id"],          # back-compat alias used by some readers
        "group": arch["group"],
        "src_class": arch["src_class"],
        "src_module": arch["src_module"],
        "T": arch.get("T"),
        "T_max": arch.get("T_max"),
        "t_sample": arch.get("t_sample"),
        "n_layers": arch.get("n_layers"),
        "k_win": arch["k_win"],
        "k_pos": arch["k_pos"],
        "shifts": arch.get("shifts"),
        "alpha": arch.get("alpha"),
        "gamma": arch.get("gamma"),
        "n_scales": arch.get("n_scales"),
        "seed": seed,
        "d_in": DEFAULT_D_IN,
        "d_sae": DEFAULT_D_SAE,
        "subject_model": SUBJECT_MODEL,
        "anchor_layer": ANCHOR_LAYER,
        "mlc_layers": list(MLC_LAYERS),
        "phase": "phase7_unification",
    }


def _hf_push_ckpt(ckpt_path: Path, run_id: str,
                   delete_local_after: bool = False) -> bool:
    """Idempotent per-ckpt push to HF txcdr-base. Returns True if HF push
    succeeded. If `delete_local_after` is True AND the push succeeded,
    the local ckpt is removed.

    Default is False (keep local) since Agent A's pod has 2 TB volume —
    all 147 ckpts (~250 GB total) fit comfortably and keeping them
    on disk avoids HF re-download during the probing pass. Flip the
    flag to True if running on the legacy 1 TB volume where the
    HF-then-delete dance is needed to fit.

    Training logs are kept locally regardless — they're tiny (KB) and
    useful for quick analysis without HF round-trips.
    """
    pushed = False
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=f"ckpts/{run_id}.pt",
            repo_id=HF_CKPT_REPO,
            repo_type="model",
        )
        log_path = LOGS_DIR / f"{run_id}.json"
        if log_path.exists():
            api.upload_file(
                path_or_fileobj=str(log_path),
                path_in_repo=f"training_logs/{run_id}.json",
                repo_id=HF_CKPT_REPO,
                repo_type="model",
            )
        print(f"  [hf] pushed {run_id}")
        pushed = True
    except Exception as e:
        print(f"  [hf] push FAIL {run_id}: {type(e).__name__}: {e}")
        print(f"  [hf] keeping local ckpt at {ckpt_path} (re-push later)")
        return False
    if pushed and delete_local_after:
        try:
            sz_gb = ckpt_path.stat().st_size / 1e9
            ckpt_path.unlink()
            print(f"  [disk] removed local {ckpt_path.name} ({sz_gb:.1f} GB) — HF copy is canonical")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  [disk] couldn't remove {ckpt_path}: {type(e).__name__}: {e}")
    return pushed


def _push_seed_marker(seed: int, completed_arch_ids: list[str]) -> None:
    """After a seed-batch outer-loop iteration completes, post a
    `seed{N}_complete.json` listing the completed arch_ids. Local mirror
    + HF push so Agent B can poll a single file (per plan.md §"Per-seed
    signaling to Agent B").
    """
    SEED_MARKER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "phase": "phase7_unification",
        "n_archs": len(completed_arch_ids),
        "arch_ids": completed_arch_ids,
        "timestamp": time.time(),
    }
    marker = SEED_MARKER_DIR / f"seed{seed}_complete.json"
    marker.write_text(json.dumps(payload, indent=2))
    print(f"  [marker] wrote {marker}")
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(marker),
            path_in_repo=f"markers/seed{seed}_complete.json",
            repo_id=HF_CKPT_REPO,
            repo_type="model",
        )
        print(f"  [marker] pushed to HF")
    except Exception as e:
        print(f"  [marker] HF push FAIL: {type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════
# Top-level loops
# ════════════════════════════════════════════════════════════════════


def _train_one_with_bufs(arch: dict, seed: int, max_steps: int | None,
                          push_to_hf: bool,
                          buf_anchor=None, buf_multilayer=None) -> dict:
    """Inner helper: train one arch given pre-loaded buffers. The caller
    (run_one or run_canonical) handles buffer lifecycle so we don't pay
    the 30s+ preload cost per arch in batch runs.

    Always empties the CUDA cache on exit — even on exception — so an
    OOM during one arch (e.g. TFA at large batch) doesn't fragment
    memory and cascade into subsequent archs.
    """
    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    arch_id = arch["arch_id"]
    print(f"=== {arch_id} (row {arch['row']}, group {arch['group']}) seed={seed} ===",
          flush=True)
    model = None
    try:
        model, log = dispatch(arch, cfg, buf_anchor=buf_anchor, buf_multilayer=buf_multilayer)
        meta = _meta_from_arch(arch, seed)
        run_id = f"{arch_id}__seed{seed}"
        ckpt_path = _save_run(model, log, run_id, meta)
        if push_to_hf:
            _hf_push_ckpt(ckpt_path, run_id)
        return meta
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def run_one(arch_id: str, seed: int = 42, max_steps: int | None = None,
            push_to_hf: bool = True) -> dict:
    """Train a single (arch, seed). Loads its own buffers — for one-off
    use. For batch runs use `run_canonical` which preloads buffers once.
    """
    canonical = load_canonical()
    arch = find_arch(canonical, arch_id)
    needs_multilayer = arch["src_class"] in {
        "MultiLayerCrosscoder", "MLCContrastive", "MLCContrastiveMultiscale",
    }
    buf_anchor = preload_single() if not needs_multilayer else None
    buf_multilayer = preload_multilayer() if needs_multilayer else None
    if buf_anchor is not None:
        print(f"  [data] anchor preloaded: {tuple(buf_anchor.shape)}")
    else:
        print(f"  [data] multilayer preloaded: {tuple(buf_multilayer.shape)}")
    try:
        return _train_one_with_bufs(
            arch, seed, max_steps, push_to_hf,
            buf_anchor=buf_anchor, buf_multilayer=buf_multilayer,
        )
    finally:
        if buf_anchor is not None: del buf_anchor
        if buf_multilayer is not None: del buf_multilayer
        torch.cuda.empty_cache()


def run_canonical(seed: int, push_to_hf: bool = True,
                  arch_subset: list[str] | None = None,
                  max_steps: int | None = None) -> list[str]:
    """Run all canonical archs at one seed. **Preloads both anchor +
    multilayer buffers ONCE** at the top and keeps them in GPU RAM
    across all 49 trainings — saves ~30s × 49 = 25 min per seed batch
    of wasted preload I/O. H200 141 GB easily holds:
        anchor (14.2 GB) + multilayer (70.8 GB) + model (~10 GB) +
        activations (varies) ≈ 100 GB, ~40 GB free for headroom.
    """
    canonical = load_canonical()
    archs = canonical["archs"]
    if arch_subset:
        archs = [a for a in archs if a["arch_id"] in arch_subset]

    # Determine which buffers we need (most batches need both).
    need_anchor = any(
        a["src_class"] not in {"MultiLayerCrosscoder", "MLCContrastive",
                               "MLCContrastiveMultiscale"}
        for a in archs
    )
    need_multilayer = any(
        a["src_class"] in {"MultiLayerCrosscoder", "MLCContrastive",
                           "MLCContrastiveMultiscale"}
        for a in archs
    )
    print(f"=== seed={seed} batch: {len(archs)} archs ===")
    print(f"  preloading buffers: anchor={need_anchor} multilayer={need_multilayer}")
    t0 = time.time()
    buf_anchor = preload_single() if need_anchor else None
    buf_multilayer = preload_multilayer() if need_multilayer else None
    print(f"  preload done in {time.time()-t0:.1f}s")
    if buf_anchor is not None:
        print(f"    anchor:     {tuple(buf_anchor.shape)} "
              f"({buf_anchor.element_size()*buf_anchor.numel()/1e9:.1f} GB)")
    if buf_multilayer is not None:
        print(f"    multilayer: {tuple(buf_multilayer.shape)} "
              f"({buf_multilayer.element_size()*buf_multilayer.numel()/1e9:.1f} GB)")

    completed = []
    for arch in archs:
        arch_buf_anchor = buf_anchor if arch["src_class"] not in {
            "MultiLayerCrosscoder", "MLCContrastive", "MLCContrastiveMultiscale"
        } else None
        arch_buf_multilayer = buf_multilayer if arch["src_class"] in {
            "MultiLayerCrosscoder", "MLCContrastive", "MLCContrastiveMultiscale"
        } else None
        try:
            _train_one_with_bufs(
                arch, seed, max_steps, push_to_hf,
                buf_anchor=arch_buf_anchor, buf_multilayer=arch_buf_multilayer,
            )
            completed.append(arch["arch_id"])
        except Exception as e:
            import traceback as _tb
            print(f"  [skip] {arch['arch_id']} seed={seed}: {type(e).__name__}: {e}")
            _tb.print_exc()
    # Free shared buffers + write seed-complete marker.
    if buf_anchor is not None: del buf_anchor
    if buf_multilayer is not None: del buf_multilayer
    torch.cuda.empty_cache()
    _push_seed_marker(seed, completed)
    return completed


def run_all(seeds: tuple[int, ...] = SEEDS, push_to_hf: bool = True) -> None:
    """Outer-loop seed; inner-loop arch. Per plan.md §"Training loop ordering".
    """
    for seed in seeds:
        print(f"\n╔══════ seed={seed} batch ══════╗")
        run_canonical(seed, push_to_hf=push_to_hf)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default=None, help="single arch_id to train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--canonical", action="store_true",
                   help="run all canonical archs at --seed")
    p.add_argument("--all", action="store_true",
                   help="run all archs × all seeds (outer=seed)")
    p.add_argument("--no-hf-push", action="store_true",
                   help="don't push ckpts to HF")
    args = p.parse_args()
    banner(__file__)
    push = not args.no_hf_push

    if args.all:
        run_all(push_to_hf=push)
    elif args.canonical:
        run_canonical(args.seed, push_to_hf=push)
    elif args.arch:
        run_one(args.arch, seed=args.seed, max_steps=args.max_steps, push_to_hf=push)
    else:
        p.error("Specify one of --arch, --canonical, --all")


if __name__ == "__main__":
    main()
