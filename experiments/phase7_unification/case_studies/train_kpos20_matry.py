"""Paper-grade trainer for cell E (matryoshka multiscale TXC at k_pos=20).

Mirrors `train_phase7.train_agentic_txc_02` (the canonical recipe for
`agentic_txc_02` at k_pos=100) but holds k_pos=20 fixed instead.

Cell E spec (per W's plan.md):
  arch_id: agentic_txc_02_kpos20
  src_class: MatryoshkaTXCDRContrastiveMultiscale
  T: 5
  k_pos: 20
  k_win: 100  (= k_pos × T)
  gamma: 0.5
  n_scales: 3
  d_sae: 18432

Outputs (canonical paths):
  results/ckpts/<arch_id>__seed<seed>.pt
  results/training_logs/<arch_id>__seed<seed>.json

Run from repo root:
  TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_matry \
    --T 5 --k-pos 20 --seed 42

Requires the L12 activation cache:
  TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.build_act_cache_phase7 --layer 12
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    CACHE_DIR, CKPT_DIR, LOGS_DIR, ANCHOR_LAYER, ANCHOR_LAYER_KEY,
    SUBJECT_MODEL, DEFAULT_D_IN, DEFAULT_D_SAE, banner,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, preload_single, make_pair_window_gen_gpu,
)
from experiments.phase7_unification.train_phase7 import _contrastive_train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--k-pos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch-id", type=str, default=None,
                    help="default = agentic_txc_02_kpos<k_pos> (T baked in name only if non-5)")
    ap.add_argument("--n-scales", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=25_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-seqs", type=int, default=24_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    banner(__file__)

    T = args.T
    k_pos = args.k_pos
    k_win = k_pos * T
    n_scales = args.n_scales
    gamma = args.gamma

    if args.arch_id is not None:
        arch_id = args.arch_id
    elif T == 5:
        arch_id = f"agentic_txc_02_kpos{k_pos}"
    else:
        arch_id = f"agentic_txc_02_t{T}_kpos{k_pos}"

    print(f"\n[W cell E] arch_id={arch_id}  T={T}  k_pos={k_pos}  k_win={k_win}")
    print(f"  d_in={DEFAULT_D_IN}  d_sae={DEFAULT_D_SAE}  seed={args.seed}")
    print(f"  n_scales={n_scales}  gamma={gamma}")

    ckpt_path = CKPT_DIR / f"{arch_id}__seed{args.seed}.pt"
    log_path = LOGS_DIR / f"{arch_id}__seed{args.seed}.json"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists() and not args.force:
        print(f"  ckpt exists at {ckpt_path}; skipping. Use --force to retrain.")
        return

    cache_file = CACHE_DIR / f"{ANCHOR_LAYER_KEY}.npy"
    cache_present = cache_file.exists()
    if args.dry_run:
        print(f"  dry-run: ckpt would write to {ckpt_path}")
        print(f"  activation cache {'PRESENT' if cache_present else 'MISSING'} at {cache_file}")
        return
    if not cache_present:
        raise FileNotFoundError(
            f"activation cache missing at {cache_file}. Build with "
            f"`TQDM_DISABLE=1 .venv/bin/python -m experiments.phase7_unification.build_act_cache_phase7 "
            f"--layer {ANCHOR_LAYER}`"
        )

    # Build model
    from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
        MatryoshkaTXCDRContrastiveMultiscale,
    )
    print(f"  building MatryoshkaTXCDRContrastiveMultiscale(d_in={DEFAULT_D_IN}, "
          f"d_sae={DEFAULT_D_SAE}, T={T}, k={k_win}, n_contr_scales={n_scales}, gamma={gamma})")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = MatryoshkaTXCDRContrastiveMultiscale(
        DEFAULT_D_IN, DEFAULT_D_SAE, T=T, k=k_win,
        n_contr_scales=n_scales, gamma=gamma,
    ).to("cuda")

    # Preload cache
    print(f"  preloading L{ANCHOR_LAYER} cache: {args.n_seqs} seqs from {cache_file}")
    t0 = time.time()
    buf_anchor = preload_single(layer_key=ANCHOR_LAYER_KEY, n_seqs=args.n_seqs).to("cuda")
    print(f"    cache shape={tuple(buf_anchor.shape)}  ({(time.time()-t0):.1f}s)")
    print(f"    cache VRAM = {buf_anchor.numel() * buf_anchor.element_size() / 1e9:.2f} GB")

    # Sample generator: pair windows for InfoNCE
    gen = make_pair_window_gen_gpu(buf_anchor, T)
    init_x = gen(args.batch_size)[:, 0]

    # Train
    cfg = TrainCfg(
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(f"  TrainCfg: lr={cfg.lr}  batch={cfg.batch_size}  max_steps={cfg.max_steps}")
    print(f"  starting train…")
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)

    meta = {
        "arch_id": arch_id,
        "src_class": "MatryoshkaTXCDRContrastiveMultiscale",
        "src_module": "src.architectures.matryoshka_txcdr_contrastive_multiscale",
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": k_pos,
        "shifts": [1, 2, 3],          # multiscale contrastive head (paper recipe)
        "alpha": 1.0,
        "gamma": gamma,
        "n_scales": n_scales,
        "n_layers": None,
        "mlc_layers": None,
        "d_in": DEFAULT_D_IN,
        "d_sae": DEFAULT_D_SAE,
        "subject_model": SUBJECT_MODEL,
        "anchor_layer": ANCHOR_LAYER,
        "hook_name": None,
        "seed": args.seed,
        "phase": "phase7_unification",
        "group": "W_phase1_kpos20",
        "recipe": (f"agentic_txc_02 recipe (TXC + multi-scale matryoshka InfoNCE) "
                   f"at k_pos={k_pos}, T={T}"),
        "purpose": "W Phase 1 sweep cell E — matryoshka multiscale TXC at matched per-token sparsity",
        # training info
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_steps": cfg.max_steps,
        "elapsed_s": log.get("elapsed_s"),
        "final_step": log.get("final_step"),
        "converged": log.get("converged"),
        "plateau_last": log.get("plateau_last"),
        "loss": log.get("loss"),
        "l0": log.get("l0"),
        "steps_logged": log.get("steps_logged"),
        "n_train_seqs": int(buf_anchor.shape[0]),
    }

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, ckpt_path)
    log_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  saved ckpt -> {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  saved log  -> {log_path}")
    print(f"  elapsed: {meta['elapsed_s']:.0f}s ({meta['elapsed_s']/60:.1f} min); converged={meta['converged']}")


if __name__ == "__main__":
    main()
