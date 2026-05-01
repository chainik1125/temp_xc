"""Train SubseqH8 at k_pos=20 — Han's "T_max + subseq sample" deadzone-escape variant.

SubseqH8: H8 stack (anti-dead + matryoshka H/L + multi-distance InfoNCE)
operating on T_max-position windows, but during training each step samples
t_sample positions from the window via one of three strategies:

- contiguous (B1): random t_sample-window inside T_max
- random (B2): random non-contiguous subset of size t_sample
- gaussian: mixture-of-Gaussians spatial prior (Han's idea — features cluster
            locally; sample positions weighted by Gaussian centred at random
            offset with σ ~ Uniform(sigma_range))

Hypothesis: T=10/20 with t_sample=5 lets the encoder learn sequence-level
features (long context) while remaining flexible at inference. The Gaussian
mode tests Han's theory that real linguistic features have spatial locality
(active in 1-2 contiguous positions, not consistently across full window).

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_subseq_h8 \\
      --T-max 10 --t-sample 5 --shifts 5 --seed 42 \\
      --sampling-mode contiguous

  # Gaussian-splat variant (Han's idea):
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_subseq_h8 \\
      --T-max 10 --t-sample 5 --shifts 2 --seed 42 \\
      --sampling-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import sys
sys.path.insert(0, "/workspace/temp_xc")

from experiments.phase7_unification._paths import (
    OUT_DIR, banner, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL,
)
from experiments.phase7_unification._train_utils import TrainCfg, preload_single
from experiments.phase7_unification.train_phase7 import (
    DEFAULT_D_IN, DEFAULT_D_SAE,
    make_multidistance_pair_gen_gpu, _contrastive_train,
    _save_run, _hf_push_ckpt,
)


K_POS = 20


def build_arch(T_max: int, t_sample: int, shifts: tuple,
               sampling_mode: str = "contiguous",
               sigma_range: tuple[float, float] | None = None,
               n_gaussians: int = 1) -> dict:
    k_win = K_POS * t_sample  # at inference, window encodes t_sample positions max
    shifts_str = "_".join(map(str, shifts))
    mode_tag = sampling_mode if sampling_mode != "contiguous" else "ctg"
    if sampling_mode == "gaussian":
        slo, shi = sigma_range or (1.0, max(2.0, T_max / 4))
        mode_tag = f"gauss_s{slo:.1f}_{shi:.1f}_g{n_gaussians}"
    arch_id = f"subseq_h8_tmax{T_max}_tsamp{t_sample}_kpos{K_POS}_shifts{shifts_str}_{mode_tag}"
    return {
        "row": -1,
        "arch_id": arch_id,
        "group": "deadzone_escape_phase2",
        "T": t_sample,           # at inference, window seen is t_sample (effective)
        "T_max": T_max,          # but the encoder has T_max position slabs
        "t_sample": t_sample,
        "k_win": k_win,
        "k_pos": K_POS,
        "shifts": list(shifts),
        "sampling_mode": sampling_mode,
        "sigma_range": list(sigma_range) if sigma_range else None,
        "n_gaussians": n_gaussians,
        "src_module": "src.architectures.phase5b_subseq_sampling_txcdr",
        "src_class": "SubseqH8",
        "recipe": (f"SubseqH8 — H8 stack at T_max={T_max} with t_sample={t_sample} "
                   f"({sampling_mode}) subseq sampling, "
                   f"k_pos={K_POS} (k_win={k_win}), shifts={shifts}"
                   + (f", σ ∈ {sigma_range}, {n_gaussians} Gaussians/row"
                      if sampling_mode == "gaussian" else "")),
        "purpose": (f"Han's deadzone-escape: high T_max context with subseq-sampled "
                    f"training. Sampling mode '{sampling_mode}' tests whether linguistic "
                    f"feature locality (Han's hypothesis) is captured better by "
                    f"contiguous chunks vs Gaussian-splatted spatial priors."),
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
        "T_max": arch["T_max"],
        "t_sample": arch["t_sample"],
        "n_layers": None,
        "k_win": arch["k_win"],
        "k_pos": arch["k_pos"],
        "shifts": arch["shifts"],
        "sampling_mode": arch["sampling_mode"],
        "sigma_range": arch.get("sigma_range"),
        "n_gaussians": arch.get("n_gaussians", 1),
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
    """Self-contained train function for SubseqH8 with sampling_mode."""
    from src.architectures.phase5b_subseq_sampling_txcdr import SubseqH8
    arch_id = arch["arch_id"]
    print(f"\n=== Deadzone-escape: {arch_id} ===", flush=True)
    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps}")

    t0 = time.time()
    print("  preloading L12 anchor cache to GPU...")
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)} dtype={buf.dtype}  "
          f"(preload took {time.time()-t0:.1f}s)")

    T_max = arch["T_max"]
    t_sample = arch["t_sample"]
    k = arch["k_win"]
    shifts = tuple(arch["shifts"])
    h = int(DEFAULT_D_SAE * 0.2)
    sigma_range = arch.get("sigma_range")
    if sigma_range is not None:
        sigma_range = tuple(sigma_range)

    model = SubseqH8(
        DEFAULT_D_IN, DEFAULT_D_SAE,
        T_max=T_max, k=k, t_sample=t_sample,
        shifts=shifts, weights=None,
        sampling_mode=arch["sampling_mode"],
        sigma_range=sigma_range,
        n_gaussians=arch.get("n_gaussians", 1),
        matryoshka_h_size=h, alpha=1.0,
    ).to("cuda")
    gen = make_multidistance_pair_gen_gpu(buf, T_max, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)
    log["T_max"] = T_max
    log["t_sample"] = t_sample
    log["shifts"] = list(shifts)
    log["matryoshka_h_size"] = h
    log["sampling_mode"] = arch["sampling_mode"]
    log["sigma_range"] = list(sigma_range) if sigma_range else None
    log["n_gaussians"] = arch.get("n_gaussians", 1)
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
    p.add_argument("--T-max", type=int, required=True, dest="T_max")
    p.add_argument("--t-sample", type=int, required=True, dest="t_sample")
    p.add_argument("--shifts", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    p.add_argument("--sampling-mode", default="contiguous",
                   choices=["contiguous", "random", "gaussian"],
                   help="contiguous (B1, default), random (B2), or gaussian "
                        "(mixture-of-Gaussians spatial prior — Han's idea)")
    p.add_argument("--sigma-lo", type=float, default=None,
                   help="gaussian mode: lower σ for spatial spread")
    p.add_argument("--sigma-hi", type=float, default=None,
                   help="gaussian mode: upper σ for spatial spread")
    p.add_argument("--n-gaussians", type=int, default=1,
                   help="gaussian mode: number of Gaussian components per row")
    args = p.parse_args()
    banner(__file__)

    sigma_range = None
    if args.sigma_lo is not None and args.sigma_hi is not None:
        sigma_range = (args.sigma_lo, args.sigma_hi)

    arch = build_arch(
        args.T_max, args.t_sample, tuple(args.shifts),
        sampling_mode=args.sampling_mode,
        sigma_range=sigma_range,
        n_gaussians=args.n_gaussians,
    )
    arch_id = arch["arch_id"]
    print(f"\n=== {arch_id} (T_max={args.T_max}, t_sample={args.t_sample}, "
          f"k_pos={K_POS}, shifts={args.shifts}, mode={args.sampling_mode}"
          + (f", σ ∈ {sigma_range}, {args.n_gaussians} gaussians"
             if args.sampling_mode == "gaussian" else "")
          + f") seed={args.seed} ===", flush=True)

    train_one(arch, args.seed, push_to_hf=not args.no_hf_push,
              max_steps=args.max_steps)


if __name__ == "__main__":
    main()
