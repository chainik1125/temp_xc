"""Train ONE TXCBareMDxMSContrastiveAntidead cell — Z R7 hill-climb.

H13 = multi-distance × multi-scale contrastive (Σ_s Σ_p w_s · γ^p ·
InfoNCE at multi-scale matryoshka prefixes for each shift). This
architecture exists in `src/architectures/txc_bare_md_ms_contrastive_antidead.py`
but has NEVER been trained on Phase 7 base.

Compared to H8 (current k=5 leader's family):
  H8  = multi-distance, single-scale at h prefix (1 InfoNCE per shift)
  H13 = multi-distance, multi-scale (n_contr_scales InfoNCE per shift)

The 6 InfoNCE terms (3 scales × 2 shifts default) could either add
information OR dominate the recon loss. If it works, it's a clean
architectural extension to the leader family.

Usage:
    .venv/bin/python -m experiments.phase7_unification.hill_climb._run_one_mdms 5 --shifts 1 2 --seed 42 --ctx 64
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.phase7_unification._paths import (
    banner, CKPT_DIR, DEFAULT_D_IN, DEFAULT_D_SAE,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, preload_single, make_multidistance_pair_gen_gpu, compute_plateau,
)
from experiments.phase7_unification.train_phase7 import (
    _save_run, _meta_from_arch, _hf_push_ckpt,
)


def make_arch(T: int, shifts: tuple[int, ...], n_scales: int, gamma: float) -> dict:
    shifts_str = "_".join(str(s) for s in shifts)
    arch_id = f"hill_z_mdms_t{T}_shifts{shifts_str}_ns{n_scales}"
    return {
        "row": 700 + T,
        "arch_id": arch_id,
        "group": 99,
        "T": T,
        "T_max": None,
        "t_sample": None,
        "n_layers": None,
        "k_win": 500,
        "k_pos": int(500 / T),
        "shifts": list(shifts),
        "alpha": 1.0,
        "gamma": gamma,
        "n_scales": n_scales,
        "src_module": "src.architectures.txc_bare_md_ms_contrastive_antidead",
        "src_class": "TXCBareMDxMSContrastiveAntidead",
        "recipe": (
            f"H13 multi-distance × multi-scale: T={T} shifts={list(shifts)} "
            f"n_scales={n_scales} gamma={gamma} matryoshka_h={int(DEFAULT_D_SAE * 0.2)}"
        ),
        "purpose": "Z R7 hill-climb: untried multi-distance × multi-scale contrastive",
    }


def train_verbose(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    from src.architectures.txc_bare_md_ms_contrastive_antidead import (
        TXCBareMDxMSContrastiveAntidead,
    )
    T = arch["T"]
    k = arch["k_win"]
    h = int(DEFAULT_D_SAE * 0.2)
    shifts = tuple(arch["shifts"])
    n_scales = int(arch["n_scales"])
    gamma = float(arch["gamma"])

    model = TXCBareMDxMSContrastiveAntidead(
        DEFAULT_D_IN, DEFAULT_D_SAE, T, k,
        shifts=shifts,
        n_contr_scales=n_scales,
        gamma=gamma,
        matryoshka_h_size=h,
        alpha=1.0,
    ).to("cuda")
    gen = make_multidistance_pair_gen_gpu(buf_anchor, T, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    if hasattr(model, "init_b_dec_geometric_median"):
        model.init_b_dec_geometric_median(init_x)
    del init_x
    torch.cuda.empty_cache()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses, l0s, steps_logged = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    print(f"  [train] start MDxMS T={T} shifts={shifts} n_scales={n_scales} "
          f"gamma={gamma} max_steps={cfg.max_steps} log_every={cfg.log_every} "
          f"min_steps={cfg.min_steps} plateau_thr={cfg.plateau_threshold}",
          flush=True)
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x, alpha=1.0)
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
            elapsed_min = (time.time() - t0) / 60.0
            print(f"  [train] step={step:5d} loss={loss.item():.2f} "
                  f"l0={l0:.2f} plateau={plateau_val} "
                  f"elapsed={elapsed_min:.1f}m",
                  flush=True)
            if (
                plateau_val is not None
                and plateau_val < cfg.plateau_threshold
                and step >= cfg.min_steps
            ):
                converged = True
                print(f"  [train] CONVERGED at step={step} "
                      f"plateau={plateau_val:.4f} < {cfg.plateau_threshold}",
                      flush=True)
                break
    elapsed = time.time() - t0
    model.eval()
    log = {
        "loss": losses, "l0": l0s, "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T": T, "shifts": list(shifts),
        "n_contr_scales": n_scales, "gamma": gamma,
        "matryoshka_h_size": h, "alpha": 1.0,
        "src_class": "TXCBareMDxMSContrastiveAntidead",
    }
    return model, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("T", type=int)
    ap.add_argument("--shifts", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n_scales", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--n_seqs", type=int, default=24_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=8_000)
    args = ap.parse_args()
    banner(__file__)

    arch = make_arch(T=args.T, shifts=tuple(args.shifts),
                     n_scales=args.n_scales, gamma=args.gamma)
    seed = args.seed
    cfg = TrainCfg(seed=seed, max_steps=args.max_steps)
    run_id = f"{arch['arch_id']}__seed{seed}"
    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    if ckpt_path.exists():
        print(f"=== {run_id}: ckpt already exists, skip ===", flush=True)
        return

    print(f"=== {run_id} (T={args.T}, shifts={args.shifts}, n_scales={args.n_scales}, "
          f"gamma={args.gamma}, n_seqs={args.n_seqs}, ctx={args.ctx}) ===",
          flush=True)
    print(f"Preloading anchor buffer to GPU (n_seqs={args.n_seqs}, ctx={args.ctx})...",
          flush=True)
    t0 = time.time()
    buf_anchor = preload_single(n_seqs=args.n_seqs)
    if args.ctx < buf_anchor.shape[1]:
        buf_anchor = buf_anchor[:, -args.ctx:, :].contiguous()
        torch.cuda.empty_cache()
    print(f"  shape={tuple(buf_anchor.shape)} dtype={buf_anchor.dtype} "
          f"device={buf_anchor.device} "
          f"size={buf_anchor.element_size() * buf_anchor.numel() / 1e9:.1f} GB "
          f"(load {time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    torch.manual_seed(seed)
    try:
        model, log = train_verbose(arch, cfg, buf_anchor)
    except Exception as e:
        print(f"  TRAIN FAIL {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f"  train done in {elapsed/60:.1f} min "
          f"(final_step={log.get('final_step')}, "
          f"converged={log.get('converged')}, "
          f"plateau={log.get('plateau_last')})",
          flush=True)

    log["n_seqs_used"] = args.n_seqs
    log["ctx_used"] = args.ctx
    log["ctx_slice_direction"] = "last" if args.ctx < 128 else "full"
    meta = _meta_from_arch(arch, seed)
    out_path = _save_run(model, log, run_id, meta)
    print(f"  ckpt: {out_path}", flush=True)
    ok = _hf_push_ckpt(out_path, run_id)
    print(f"  HF push: {'OK' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
