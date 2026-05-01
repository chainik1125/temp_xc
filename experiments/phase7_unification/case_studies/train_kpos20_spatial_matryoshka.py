"""Train SpatialMatryoshkaH8 — Han's deadzone-escape via random-subset
Matryoshka decoder loss.

Each Matryoshka feature-prefix level reconstructs a different RANDOM subset
of positions per training step. The H prefix (small features count) only
sees small position subsets → forced to learn position-flexible local
features. The full-feature prefix reconstructs full T-window.

CLI knobs:
  --T              window length (e.g. 10)
  --shifts         multi-distance contrastive shifts (e.g. --shifts 2 5)
  --level-prefix-sizes / --level-subset-sizes
                   matched lists. Defaults: prefix=(H, d_sae/2, d_sae),
                   subset=(1, T/2, T)
  --nested         nested vs independent subsets (default: independent)
  --subset-mode    uniform | gaussian
  --no-contrastive disable H8 InfoNCE (test if spatial-mat replaces it)

Run examples:
  # 4 main combinatorial variants (push all to git):
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \\
      --T 10 --shifts 2 --subset-mode uniform --seed 42

  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \\
      --T 10 --shifts 2 --subset-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 \\
      --n-gaussians 2 --nested --seed 42
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import sys
sys.path.insert(0, "/workspace/temp_xc")

from experiments.phase7_unification._paths import (
    ANCHOR_LAYER, MLC_LAYERS, OUT_DIR, SUBJECT_MODEL, banner,
)
from experiments.phase7_unification._train_utils import TrainCfg, preload_single
from experiments.phase7_unification.train_phase7 import (
    DEFAULT_D_IN, DEFAULT_D_SAE,
    make_multidistance_pair_gen_gpu, _contrastive_train,
    _save_run, _hf_push_ckpt,
)


K_POS = 20


def build_arch(
    T: int,
    shifts: tuple,
    level_prefix_sizes: tuple[int, ...],
    level_subset_sizes: tuple[int, ...],
    nested: bool,
    subset_mode: str,
    sigma_range: tuple[float, float] | None,
    n_gaussians: int,
    enable_contrastive: bool,
) -> dict:
    k_win = K_POS * T
    shifts_str = "_".join(map(str, shifts))
    prefix_str = "_".join(map(str, level_prefix_sizes))
    subset_str = "_".join(map(str, level_subset_sizes))
    nested_tag = "nested" if nested else "indep"
    contr_tag = "contr" if enable_contrastive else "nocontr"
    mode_tag = subset_mode
    if subset_mode == "gaussian" and sigma_range is not None:
        mode_tag = f"gauss_s{sigma_range[0]:.1f}_{sigma_range[1]:.1f}_g{n_gaussians}"
    arch_id = (
        f"spatial_matry_h8_t{T}_kpos{K_POS}_shifts{shifts_str}"
        f"_pref{prefix_str}_sub{subset_str}_{nested_tag}_{mode_tag}_{contr_tag}"
    )
    return {
        "row": -1,
        "arch_id": arch_id,
        "group": "deadzone_escape_phase2",
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": K_POS,
        "shifts": list(shifts),
        "level_prefix_sizes": list(level_prefix_sizes),
        "level_subset_sizes": list(level_subset_sizes),
        "nested": nested,
        "subset_mode": subset_mode,
        "sigma_range": list(sigma_range) if sigma_range else None,
        "n_gaussians": n_gaussians,
        "enable_contrastive": enable_contrastive,
        "src_module": "src.architectures.spatial_matryoshka_h8",
        "src_class": "SpatialMatryoshkaH8",
        "recipe": (
            f"SpatialMatryoshkaH8 — H8 stack at T={T} k_pos={K_POS} "
            f"with random-subset Matryoshka decoder loss "
            f"(prefixes={level_prefix_sizes}, subsets={level_subset_sizes}, "
            f"{nested_tag}, sample={mode_tag}, contrastive={enable_contrastive})"
        ),
        "purpose": (
            "Han's deadzone-escape: low-rank features must reconstruct any "
            "single position (position-flexible); deeper features add "
            "compositional cross-position info. Tests whether T=10/20 "
            "with this loss escapes the T=2-5 deadzone."
        ),
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
        "shifts": arch["shifts"],
        "level_prefix_sizes": arch["level_prefix_sizes"],
        "level_subset_sizes": arch["level_subset_sizes"],
        "nested": arch["nested"],
        "subset_mode": arch["subset_mode"],
        "sigma_range": arch.get("sigma_range"),
        "n_gaussians": arch.get("n_gaussians", 1),
        "enable_contrastive": arch["enable_contrastive"],
        "alpha": 1.0,
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


def train_one(arch: dict, seed: int, push_to_hf: bool, max_steps: int | None) -> None:
    from src.architectures.spatial_matryoshka_h8 import SpatialMatryoshkaH8
    arch_id = arch["arch_id"]
    print(f"\n=== Spatial-Matryoshka: {arch_id} ===", flush=True)
    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps}")

    t0 = time.time()
    print("  preloading L12 anchor cache to GPU...")
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)} dtype={buf.dtype}")

    T = arch["T"]
    k = arch["k_win"]
    shifts = tuple(arch["shifts"])
    h = int(DEFAULT_D_SAE * 0.2)
    sigma_range = arch.get("sigma_range")
    if sigma_range is not None:
        sigma_range = tuple(sigma_range)

    model = SpatialMatryoshkaH8(
        DEFAULT_D_IN, DEFAULT_D_SAE, T=T, k=k,
        shifts=shifts, weights=None,
        level_prefix_sizes=tuple(arch["level_prefix_sizes"]),
        level_subset_sizes=tuple(arch["level_subset_sizes"]),
        nested=arch["nested"],
        subset_sampling_mode=arch["subset_mode"],
        sigma_range=sigma_range,
        n_gaussians=arch.get("n_gaussians", 1),
        enable_contrastive=arch["enable_contrastive"],
        matryoshka_h_size=h, alpha=1.0,
    ).to("cuda")
    gen = make_multidistance_pair_gen_gpu(buf, T, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)
    log["shifts"] = list(shifts)
    log["level_prefix_sizes"] = list(arch["level_prefix_sizes"])
    log["level_subset_sizes"] = list(arch["level_subset_sizes"])
    log["nested"] = arch["nested"]
    log["subset_mode"] = arch["subset_mode"]
    log["enable_contrastive"] = arch["enable_contrastive"]
    log["matryoshka_h_size"] = h
    log["final_step_wall_s"] = time.time() - t0

    meta = build_meta(arch, seed)
    run_id = f"{arch_id}__seed{seed}"
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  trained in {log['final_step_wall_s']/60:.1f} min wall  "
          f"(final_step={log.get('final_step')}, converged={log.get('converged')})")
    if push_to_hf:
        _hf_push_ckpt(ckpt_path, run_id)

    import torch
    del buf, model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--shifts", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    p.add_argument("--level-prefix-sizes", type=int, nargs="+", default=None,
                   help="defaults to (H, d_sae/2, d_sae)")
    p.add_argument("--level-subset-sizes", type=int, nargs="+", default=None,
                   help="defaults to (1, T/2, T)")
    p.add_argument("--nested", action="store_true",
                   help="level (i+1) subset is a superset of level i (default: independent)")
    p.add_argument("--subset-mode", default="uniform", choices=["uniform", "gaussian"])
    p.add_argument("--sigma-lo", type=float, default=None)
    p.add_argument("--sigma-hi", type=float, default=None)
    p.add_argument("--n-gaussians", type=int, default=1)
    p.add_argument("--no-contrastive", action="store_true",
                   help="disable H8 InfoNCE — test if spatial-mat replaces it")
    args = p.parse_args()
    banner(__file__)

    h = int(DEFAULT_D_SAE * 0.2)
    if args.level_prefix_sizes:
        level_prefix_sizes = tuple(args.level_prefix_sizes)
    else:
        level_prefix_sizes = (h, DEFAULT_D_SAE // 2, DEFAULT_D_SAE)
    if args.level_subset_sizes:
        level_subset_sizes = tuple(args.level_subset_sizes)
    else:
        level_subset_sizes = (1, max(1, args.T // 2), args.T)

    sigma_range = None
    if args.sigma_lo is not None and args.sigma_hi is not None:
        sigma_range = (args.sigma_lo, args.sigma_hi)

    arch = build_arch(
        T=args.T,
        shifts=tuple(args.shifts),
        level_prefix_sizes=level_prefix_sizes,
        level_subset_sizes=level_subset_sizes,
        nested=args.nested,
        subset_mode=args.subset_mode,
        sigma_range=sigma_range,
        n_gaussians=args.n_gaussians,
        enable_contrastive=not args.no_contrastive,
    )

    train_one(arch, args.seed, push_to_hf=not args.no_hf_push,
              max_steps=args.max_steps)


if __name__ == "__main__":
    main()
