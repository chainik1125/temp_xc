"""Galaxy 4 trainer — TXCHierarchicalMultiScale at matched k_pos=20.

Y's GIGABRAIN architectural proposal (2026-05-01): explicit decomposition
of latent into window-level features (multi-token concepts) and
per-position features (per-token concepts). See
`docs/han/research_logs/phase7_unification/agent_y_phase2/2026-05-01-y-galaxy-brainstorm.md`.

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_galaxy \\
      --T 2 --k-window 10 --k-pos 10 --d-sae-w 9216 --d-sae-p 9216 --seed 42

Matched-sparsity convention: K_window + K_pos = k_pos = 20 (active
features per token).

For T=2: 9216 + 9216 = 18432 (matches DEFAULT_D_SAE) total feature
parameters split equally.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch

from experiments.phase7_unification._paths import (
    DEFAULT_D_IN, DEFAULT_D_SAE, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, banner,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, preload_single, make_window_gen_gpu,
)
from experiments.phase7_unification.train_phase7 import (
    _flat_train, _save_run, _hf_push_ckpt,
)
from src.architectures.txc_hierarchical_multiscale import TXCHierarchicalMultiScale


def build_arch_dict(T: int, k_window: int, k_pos: int, d_sae_w: int, d_sae_p: int) -> dict:
    arch_id = f"txc_galaxy4_t{T}_kw{k_window}_kp{k_pos}"
    return {
        "row": -1,
        "arch_id": arch_id,
        "group": "galaxy_phase8",
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_window + T * k_pos,  # total active per window (for compat)
        "k_pos": k_window + k_pos,       # active per token = window_share + per_pos
        "d_sae_w": d_sae_w,
        "d_sae_p": d_sae_p,
        "k_window": k_window,
        "k_pos_only": k_pos,
        "shifts": None,
        "src_module": "src.architectures.txc_hierarchical_multiscale",
        "src_class": "TXCHierarchicalMultiScale",
        "recipe": (f"Galaxy 4 hierarchical multi-scale T={T}, "
                   f"d_sae_w={d_sae_w} K_w={k_window}, "
                   f"d_sae_p={d_sae_p} K_p={k_pos}"),
        "purpose": "Galaxy 4 prototype — hierarchical window/per-position decomposition",
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
        "k_window": arch["k_window"],
        "k_pos_only": arch["k_pos_only"],
        "d_sae_w": arch["d_sae_w"],
        "d_sae_p": arch["d_sae_p"],
        "shifts": None,
        "alpha": None,
        "gamma": None,
        "n_scales": None,
        "seed": seed,
        "d_in": DEFAULT_D_IN,
        "d_sae": arch["d_sae_w"] + arch["T"] * arch["d_sae_p"],
        "subject_model": SUBJECT_MODEL,
        "anchor_layer": ANCHOR_LAYER,
        "mlc_layers": list(MLC_LAYERS),
        "phase": "phase7_unification",
        "purpose": arch["purpose"],
        "recipe": arch["recipe"],
    }


def train_one(T: int, k_window: int, k_pos: int, d_sae_w: int, d_sae_p: int,
              seed: int, push_to_hf: bool, max_steps: int | None) -> None:
    arch = build_arch_dict(T, k_window, k_pos, d_sae_w, d_sae_p)
    arch_id = arch["arch_id"]
    print(f"\n=== Galaxy 4 cell: {arch_id} ===")
    print(f"  T={T}  K_window={k_window}  K_pos={k_pos}")
    print(f"  d_sae_w={d_sae_w}  d_sae_p={d_sae_p}  total d_sae = {d_sae_w + T * d_sae_p}")
    print(f"  matched k_pos = K_window + K_pos = {k_window + k_pos}")

    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps}")

    t0 = time.time()
    print("  preloading L12 anchor cache to GPU...")
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)} preload took {time.time()-t0:.1f}s")

    model = TXCHierarchicalMultiScale(
        d_in=DEFAULT_D_IN,
        d_sae_w=d_sae_w, d_sae_p=d_sae_p,
        T=T, k_window=k_window, k_pos=k_pos,
    ).to("cuda").to(torch.float32)
    print(f"  param count: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    gen = make_window_gen_gpu(buf, T)
    init_x = gen(cfg.batch_size)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=init_x)
    log["final_step_wall_s"] = time.time() - t0

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
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--k-window", type=int, default=10, help="TopK budget for window-level features")
    p.add_argument("--k-pos", type=int, default=10, help="TopK budget per position for per-pos features")
    p.add_argument("--d-sae-w", type=int, default=9216)
    p.add_argument("--d-sae-p", type=int, default=9216)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    args = p.parse_args()
    banner(__file__)

    push = not args.no_hf_push
    train_one(args.T, args.k_window, args.k_pos, args.d_sae_w, args.d_sae_p,
              args.seed, push, args.max_steps)


if __name__ == "__main__":
    main()
