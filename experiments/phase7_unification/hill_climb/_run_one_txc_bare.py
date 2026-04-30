"""Train ONE TXCBareAntidead cell at a given T — spawn-per-cell pattern.

Z-initiated T-sweep gap-fill for the `txc_bare_antidead_t<T>` family.
Existing T values trained: {5, 10, 20}. This script fills T ∈ {4, 6, 7,
8, 12} (and any other T value passed via --T).

The k=20 PAPER leader is `txc_bare_antidead_t5` at 0.9131. Gap-fill
seeks: (a) confirm T=5 is the local optimum, or (b) find a non-5 T
that beats it (low expected probability but cheap to try).

Usage:
    .venv/bin/python -m experiments.phase7_unification.hill_climb._run_one_txc_bare 8 --ctx 64

Memory at T=12 b=4096 (worst case): ~28 GB on 5090 with L=64 cache.
Slicing direction is `[:, -ctx:, :]` (X-approved hill-climb deviation;
see agent_z_paper/2026-04-30-z-x-cache-slice-question.md).
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
    TrainCfg, preload_single, compute_plateau,
)
from experiments.phase5b_t_scaling_explore._train_utils import (
    make_window_gen_gpu,
)
from experiments.phase7_unification.train_phase7 import (
    _save_run, _meta_from_arch, _hf_push_ckpt,
)


def make_arch(T: int) -> dict:
    arch_id = f"txc_bare_antidead_t{T}"
    return {
        "row": 600 + T,  # Z hill-climb row range
        "arch_id": arch_id,
        "group": 99,
        "T": T,
        "T_max": None,
        "t_sample": None,
        "n_layers": None,
        "k_win": 500,
        "k_pos": int(500 / T),
        "shifts": None,
        "alpha": None,
        "gamma": None,
        "n_scales": None,
        "src_module": "src.architectures.txc_bare_antidead",
        "src_class": "TXCBareAntidead",
        "recipe": (
            f"Bare TXC at T={T} + tsae_paper anti-dead stack. "
            f"No matryoshka, no contrastive. Z hill-climb gap-fill."
        ),
        "purpose": "round5 hill-climb: TXCBareAntidead T-sweep gap-fill",
    }


def train_txc_bare_verbose(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Verbose loop, mirrors train_phase7's TXCBareAntidead path with
    flushed prints per log_every step.
    """
    from src.architectures.txc_bare_antidead import TXCBareAntidead

    T = arch["T"]
    k_win = arch["k_win"]
    model = TXCBareAntidead(DEFAULT_D_IN, DEFAULT_D_SAE, T=T, k=k_win).to("cuda")
    gen = make_window_gen_gpu(buf_anchor, T)
    init_x = gen(cfg.batch_size)
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
    print(f"  [train] start TXCBareAntidead T={T} k_win={k_win} "
          f"max_steps={cfg.max_steps} log_every={cfg.log_every} "
          f"min_steps={cfg.min_steps} plateau_thr={cfg.plateau_threshold}",
          flush=True)
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x)
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
        "T": T, "src_class": "TXCBareAntidead",
    }
    return model, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("T", type=int, help="window length (gap-fill: 4, 6, 7, 8, 12)")
    ap.add_argument("--ctx", type=int, default=64,
                    help="L axis crop, [:, -ctx:, :] direction. Default 64.")
    ap.add_argument("--n_seqs", type=int, default=24_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=8_000)
    args = ap.parse_args()
    banner(__file__)

    arch = make_arch(T=args.T)
    seed = args.seed
    cfg = TrainCfg(seed=seed, max_steps=args.max_steps)
    run_id = f"{arch['arch_id']}__seed{seed}"
    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    if ckpt_path.exists():
        print(f"=== {run_id}: ckpt already exists, skip ===", flush=True)
        return

    print(f"=== {run_id} (T={args.T}, n_seqs={args.n_seqs}, ctx={args.ctx}) ===",
          flush=True)
    print(f"Preloading anchor buffer to GPU (n_seqs={args.n_seqs}, "
          f"ctx={args.ctx})...", flush=True)
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
        model, log = train_txc_bare_verbose(arch, cfg, buf_anchor)
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
