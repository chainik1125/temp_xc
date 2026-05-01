"""Galaxy 6 trainer — TXCMaxPool at matched k_pos=20.

Y's GIGABRAIN architectural proposal (2026-05-01): max-pool over T
positions instead of sum. See `2026-05-01-y-galaxy-brainstorm.md`.

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_galaxy6 \\
      --T 2 --seed 42

For matched per-token sparsity k_pos=20, k_win=k_pos*T=40.
"""
from __future__ import annotations

import argparse
import os
import time

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
from src.architectures.txc_maxpool import TXCMaxPool


K_POS = 20


def build_arch_dict(T: int) -> dict:
    k_win = K_POS * T
    return {
        "row": -1,
        "arch_id": f"txc_maxpool_t{T}_kpos{K_POS}",
        "group": "galaxy_phase8",
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": K_POS,
        "shifts": None,
        "src_module": "src.architectures.txc_maxpool",
        "src_class": "TXCMaxPool",
        "recipe": f"Galaxy 6 max-pool TXC T={T}, k_pos={K_POS} (k_win={k_win})",
        "purpose": "Galaxy 6 prototype — max-pool encoder (vs additive sum)",
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
    }


def train_one(T: int, seed: int, push_to_hf: bool, max_steps: int | None) -> None:
    arch = build_arch_dict(T)
    arch_id = arch["arch_id"]
    print(f"\n=== Galaxy 6 cell: {arch_id} ===")
    print(f"  T={T}, k_pos={K_POS}, k_win={arch['k_win']}, seed={seed}")

    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps}")

    t0 = time.time()
    print("  preloading L12 anchor cache to GPU...")
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)}  preload took {time.time()-t0:.1f}s")

    model = TXCMaxPool(DEFAULT_D_IN, DEFAULT_D_SAE, T, arch["k_win"]).to("cuda").to(torch.float32)
    print(f"  param count: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    gen = make_window_gen_gpu(buf, T)
    init_x = gen(cfg.batch_size)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=init_x)
    log["final_step_wall_s"] = time.time() - t0

    meta = build_meta(arch, seed)
    run_id = f"{arch_id}__seed{seed}"
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  trained in {log['final_step_wall_s']/60:.1f} min  "
          f"final_step={log.get('final_step')}  converged={log.get('converged')}")

    if push_to_hf:
        _hf_push_ckpt(ckpt_path, run_id)

    del buf, model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    args = p.parse_args()
    banner(__file__)

    push = not args.no_hf_push
    train_one(args.T, args.seed, push, args.max_steps)


if __name__ == "__main__":
    main()
