"""Phase 7 Y — grow T from T=2 (best matched-sparsity cell) without losing performance.

Han's directive (2026-04-30): take the BEST TXC at T=2 (multi-seed
mean 1.200 at coh ≥ 1.5) and see if we can grow the window WITHOUT
making it worse. Warm-start a larger-T model from the T=2 ckpt,
tiling existing weights into the new T positions.

Approach:
  - Load txc_bare_antidead_t2_kpos20__seed{seed}.pt
  - Build TXCBareAntidead at T_new (3 or 4 or 5), k_pos=20, k_win=20*T_new
  - Initialize:
      W_enc[0]              = T2 W_enc[0]  (existing position 0)
      W_enc[1]              = T2 W_enc[1]  (existing position 1)
      W_enc[2..T_new-1]     = T2 W_enc[1] copied / per-position random
      W_dec[:, 0, :]        = T2 W_dec[:, 0, :]
      W_dec[:, 1, :]        = T2 W_dec[:, 1, :]
      W_dec[:, 2..T_new-1] = T2 W_dec[:, 1, :] copied / fresh
      Renormalize decoder unit-norm per atom.
      b_enc                 = T2 b_enc
      b_dec[0]              = T2 b_dec[0]
      b_dec[1]              = T2 b_dec[1]
      b_dec[2..T_new-1]     = T2 b_dec[1] (geometric median of typical pos 1 features)
  - Train at canonical TrainCfg.

Different from the original brief's warm-start (tile T-SAE k=20 across
T positions, divide decoder by T): this warm-starts from a TXC-trained
ckpt rather than per-token. T=2 has already learned multi-token
structure across 2 positions; we extend that structure to more positions.

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_grow \\
      --T-new 3 --src-T 2 --src-seed 42 --seed 42

  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_grow \\
      --T-new 5 --src-T 2 --src-seed 42 --seed 42  (skip T=4, jump direct)

  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_grow \\
      --T-new 4 --src-T 3 --seed 42  (chain after T=3 grown ckpt lands)

The grown arch_id encodes the chain: txc_bare_antidead_t<T_new>_kpos20_grownFrom<T_src>.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch
import numpy as np

from experiments.phase7_unification._paths import (
    DEFAULT_D_IN, DEFAULT_D_SAE, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, CKPT_DIR, banner,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, preload_single, make_window_gen_gpu,
)
from experiments.phase7_unification.train_phase7 import (
    _flat_train, _save_run, _hf_push_ckpt,
)
from src.architectures.txc_bare_antidead import TXCBareAntidead


K_POS = 20


def warm_start_from_t2(model_new: TXCBareAntidead, src_ckpt_path: Path, T_src: int, T_new: int) -> None:
    """In-place warm-start: copy src ckpt's W_enc/W_dec/b_enc/b_dec into
    model_new's first T_src positions, then duplicate the last src position
    into the new positions.
    """
    sd_src = torch.load(src_ckpt_path, map_location="cpu", weights_only=True)
    # Some keys we need:
    W_enc_src = sd_src["W_enc"].float()       # (T_src, d_in, d_sae)
    W_dec_src = sd_src["W_dec"].float()       # (d_sae, T_src, d_in)
    b_enc_src = sd_src["b_enc"].float()       # (d_sae,)
    b_dec_src = sd_src["b_dec"].float()       # (T_src, d_in)

    if W_enc_src.shape[0] != T_src or W_dec_src.shape[1] != T_src:
        raise RuntimeError(f"src ckpt has T={W_enc_src.shape[0]} (W_enc) / "
                          f"{W_dec_src.shape[1]} (W_dec); expected T_src={T_src}")

    print(f"  warm-start: copy src T_src={T_src} positions, duplicate last src pos for new positions {T_src}..{T_new-1}")

    with torch.no_grad():
        for t in range(T_new):
            t_src = min(t, T_src - 1)  # for t < T_src, copy directly; for t >= T_src, duplicate last src position
            model_new.W_enc.data[t] = W_enc_src[t_src].to(model_new.W_enc.device).to(model_new.W_enc.dtype)
            model_new.W_dec.data[:, t, :] = W_dec_src[:, t_src, :].to(model_new.W_dec.device).to(model_new.W_dec.dtype)
            model_new.b_dec.data[t] = b_dec_src[t_src].to(model_new.b_dec.device).to(model_new.b_dec.dtype)
        model_new.b_enc.data[:] = b_enc_src.to(model_new.b_enc.device).to(model_new.b_enc.dtype)

        # Renormalize decoder unit-norm per atom (over (T_new, d_in))
        model_new._normalize_decoder()
        # Mark b_dec as initialized so we don't re-do geom-median on a sample
        model_new.b_dec_initialized.fill_(True)


def build_arch_dict(T_new: int, T_src: int, src_seed: int, seed: int) -> dict:
    k_win = K_POS * T_new
    return {
        "row": -1,
        "arch_id": f"txc_bare_antidead_t{T_new}_kpos{K_POS}_grownFromT{T_src}sd{src_seed}",
        "group": "hailmary_phase2_grow",
        "T": T_new,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": K_POS,
        "shifts": None,
        "src_module": "src.architectures.txc_bare_antidead",
        "src_class": "TXCBareAntidead",
        "recipe": (f"TXC bare-antidead T={T_new}, k_pos={K_POS} (k_win={k_win}), "
                   f"warm-started from T={T_src} seed={src_seed} ckpt"),
        "purpose": (f"Han's grow-window directive — extend best matched-sparsity "
                    f"cell (T={T_src}) without losing performance"),
        "warm_start_src": f"txc_bare_antidead_t{T_src}_kpos{K_POS}__seed{src_seed}.pt",
    }


def build_meta(arch: dict, seed: int) -> dict:
    return {
        "row": arch["row"],
        "arch_id": arch["arch_id"],
        "arch": arch["arch_id"],
        "group": arch["group"],
        "src_class": arch["src_class"],
        "src_module": arch["src_module"],
        "T": arch["T"],
        "T_max": None,
        "t_sample": None,
        "n_layers": None,
        "k_win": arch["k_win"],
        "k_pos": arch["k_pos"],
        "shifts": None,
        "alpha": None,
        "gamma": None,
        "n_scales": None,
        "seed": seed,
        "d_in": DEFAULT_D_IN,
        "d_sae": DEFAULT_D_SAE,
        "subject_model": SUBJECT_MODEL,
        "anchor_layer": ANCHOR_LAYER,
        "mlc_layers": list(MLC_LAYERS),
        "phase": "phase7_unification",
        "purpose": arch["purpose"],
        "recipe": arch["recipe"],
        "warm_start_src": arch.get("warm_start_src"),
    }


def train_one(T_new: int, T_src: int, src_seed: int, seed: int,
              push_to_hf: bool, max_steps: int | None) -> None:
    arch = build_arch_dict(T_new, T_src, src_seed, seed)
    arch_id = arch["arch_id"]
    print(f"\n=== Grow cell: {arch_id} ===")
    print(f"  warm-start from T={T_src} seed={src_seed} → train T={T_new}, k_pos={K_POS}, k_win={arch['k_win']}, target seed={seed}")

    src_ckpt = CKPT_DIR / f"txc_bare_antidead_t{T_src}_kpos{K_POS}__seed{src_seed}.pt"
    if not src_ckpt.exists():
        raise SystemExit(f"missing src ckpt: {src_ckpt}")

    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps} "
          f"plateau={cfg.plateau_threshold} min_steps={cfg.min_steps}")

    t0 = time.time()
    print("  preloading L12 anchor cache to GPU...")
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)} preload took {time.time()-t0:.1f}s")

    # Build model and warm-start from src ckpt
    model = TXCBareAntidead(DEFAULT_D_IN, DEFAULT_D_SAE, T_new, arch["k_win"]).to("cuda")
    warm_start_from_t2(model, src_ckpt, T_src, T_new)
    model.to(torch.float32)
    print(f"  warm-start done; model has T={T_new}, k_win={arch['k_win']}")

    gen = make_window_gen_gpu(buf, T_new)
    # b_dec already warm-started; pass init_x=None so _flat_train doesn't redo geom-median
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=None)
    log["final_step_wall_s"] = time.time() - t0
    log["warm_start_src"] = arch["warm_start_src"]

    meta = build_meta(arch, seed)
    run_id = f"{arch_id}__seed{seed}"
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  trained in {log['final_step_wall_s']/60:.1f} min  "
          f"final_step={log.get('final_step')}  converged={log.get('converged')}  "
          f"plateau_last={log.get('plateau_last')}")

    if push_to_hf:
        _hf_push_ckpt(ckpt_path, run_id)

    del buf, model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T-new", type=int, required=True, help="target T (must be > src_T)")
    p.add_argument("--src-T", type=int, default=2, help="source ckpt T (default 2)")
    p.add_argument("--src-seed", type=int, default=42, help="source ckpt seed (default 42)")
    p.add_argument("--seed", type=int, default=42, help="target seed (random init for new positions)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    args = p.parse_args()
    banner(__file__)

    if args.T_new <= args.src_T:
        raise SystemExit(f"T_new ({args.T_new}) must be > src_T ({args.src_T})")

    push = not args.no_hf_push
    train_one(args.T_new, args.src_T, args.src_seed, args.seed, push, args.max_steps)


if __name__ == "__main__":
    main()
