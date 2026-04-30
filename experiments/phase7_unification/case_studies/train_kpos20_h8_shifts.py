"""Phase 7 Y creative cell — TXCBareMultiDistanceContrastiveAntidead at k_pos=20 with custom shifts.

Han's suggestion (2026-04-30): set the multi-distance contrastive shift
to T (single shift = window length) instead of canonical auto-scaled
(1, T//4, T//2). Tests whether explicit T-window temporal consistency
lifts the matched-sparsity result.

Apples-to-apples with prior cells:
  - Same activation cache (data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy)
  - Same TrainCfg (b=4096, lr=3e-4, max_steps=25k, plateau=0.02)
  - Same seed=42 primary
  - Random-init (matches T-SAE k=20 anchor + Y's earlier cells)

Key difference vs W's cell E (MatryoshkaTXCDRContrastiveMultiscale):
  - Cell E is multiscale matryoshka (n_scales=3, gamma=0.5) with auto
    shifts. Different src_class.
  - This cell is TXCBareMultiDistanceContrastiveAntidead — H8 stack
    (anti-dead + matryoshka H/L + multi-distance InfoNCE) with explicit
    custom shifts. Same as canonical phase57_partB_h8_bare_multidistance_*
    family but at k_pos=20 instead of k_pos=k_win/T.

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_h8_shifts \\
      --T 5 --shifts 5 --seed 42

The --shifts arg can take multiple ints (e.g. --shifts 5 1) for multi-distance.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    DEFAULT_D_IN, DEFAULT_D_SAE, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, banner,
)
from experiments.phase7_unification._train_utils import TrainCfg, preload_single
from experiments.phase7_unification.train_phase7 import (
    train_h8_multidistance, _save_run, _hf_push_ckpt,
)


K_POS = 20


def build_arch(T: int, shifts: tuple) -> dict:
    """Construct in-process arch dict for TXCBareMultiDistanceContrastiveAntidead."""
    k_win = K_POS * T
    shifts_str = "_".join(map(str, shifts))
    return {
        "row": -1,
        "arch_id": f"txc_h8_t{T}_kpos{K_POS}_shifts{shifts_str}",
        "group": "hailmary_phase2_creative",
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": K_POS,
        "shifts": list(shifts),
        "src_module": "src.architectures.txc_bare_multidistance_contrastive_antidead",
        "src_class": "TXCBareMultiDistanceContrastiveAntidead",
        "recipe": (f"H8 stack (anti-dead + matryoshka H/L + multi-distance InfoNCE) "
                   f"at T={T}, k_pos={K_POS} (k_win={k_win}), shifts={shifts}"),
        "purpose": (f"Hail Mary creative cell — Han's shifts=(T,) suggestion at "
                    f"matched per-token sparsity"),
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


def train_one(T: int, shifts: tuple, seed: int, push_to_hf: bool, max_steps: int | None) -> None:
    arch = build_arch(T, shifts)
    arch_id = arch["arch_id"]
    print(f"\n=== Hail Mary creative: {arch_id} (T={T}, k_pos={K_POS}, k_win={arch['k_win']}, shifts={shifts}) seed={seed} ===",
          flush=True)
    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps} "
          f"plateau={cfg.plateau_threshold} min_steps={cfg.min_steps}")

    t0 = time.time()
    print("  preloading L12 anchor cache to GPU...")
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)} dtype={buf.dtype} device={buf.device} "
          f"size={buf.element_size()*buf.numel()/1e9:.1f} GB  "
          f"(preload took {time.time()-t0:.1f}s)")

    model, log = train_h8_multidistance(arch, cfg, buf)
    log["final_step_wall_s"] = time.time() - t0

    meta = build_meta(arch, seed)
    run_id = f"{arch_id}__seed{seed}"
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  trained in {log['final_step_wall_s']/60:.1f} min total wall "
          f"(final_step={log.get('final_step')}, converged={log.get('converged')}, "
          f"plateau_last={log.get('plateau_last')})")

    if push_to_hf:
        _hf_push_ckpt(ckpt_path, run_id)

    import torch
    del buf, model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--shifts", type=int, nargs="+", required=True,
                   help="multi-distance contrastive shifts (e.g. --shifts 5 for distance=T)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    args = p.parse_args()
    banner(__file__)

    push = not args.no_hf_push
    train_one(args.T, tuple(args.shifts), args.seed, push, args.max_steps)


if __name__ == "__main__":
    main()
