"""Hill-climbing round 2: long-distance contrastive shifts.

Phase 5 (-it model, 2026-04-24-groundbreaking-handover.md) found a
counter-intuitive U-shape in single-shift contrastive at T=5:
  - shift=1, 2 (60-80% overlap): "local invariance" works
  - shift=4-10: dead zone (worst results)
  - shift=20+ (large gap, same-sequence): "semantic invariance" works again

Single shift={20} alone TIED H8 at lp and TOPPED H8 at mp. Test whether
this transfers to BASE Gemma in Phase 7.

Variants (each ~25-40 min; total ~2 hr):
  L1: H8 T=5 shifts=[1, 32]    — local + long combo
  L2: H8 T=5 shifts=[32]       — long only
  L3: H8 T=8 shifts=[1, 2, 4, 32] — extend current winner with long shift
  L4: H8 T=8 shifts=[1, 2, 32] — drop dead-zone shift=4, add long

Class: TXCBareMultiDistanceContrastiveAntidead (same as h8_md_t8 winner).
Cache anchor is L=128 → max shift = 128 - T = 120; shift=32 well within.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.hill_climb.round2_long_shifts
"""
from __future__ import annotations

import time

import torch

from experiments.phase7_unification._paths import banner
from experiments.phase7_unification._train_utils import preload_single
from experiments.phase7_unification.train_phase7 import (
    TrainCfg, train_h8_multidistance, _save_run, _meta_from_arch, _hf_push_ckpt,
)


def make_arch(label: str, T: int, shifts: list[int], row: int) -> dict:
    arch_id = f"hill_h8_T{T}_shifts{label}"
    return {
        "row": row,
        "arch_id": arch_id,
        "group": 99,
        "T": T,
        "T_max": None,
        "t_sample": None,
        "n_layers": None,
        "k_win": 500,
        "k_pos": int(500 / T),
        "shifts": shifts,
        "alpha": 1.0,
        "gamma": None,
        "n_scales": None,
        "src_module": "src.architectures.txc_bare_multidistance_contrastive_antidead",
        "src_class": "TXCBareMultiDistanceContrastiveAntidead",
        "recipe": f"hill-climb H8 T={T} with long shift {shifts}",
        "purpose": "round2 hill-climb: test Phase 5 long-distance shift U-shape on BASE",
    }


def main() -> None:
    banner(__file__)
    seed = 42
    cfg = TrainCfg(seed=seed, max_steps=8000)

    print("Preloading anchor buffer...")
    buf_anchor = preload_single()

    archs = [
        make_arch("1and32",   T=5, shifts=[1, 32],     row=200),
        make_arch("32only",   T=5, shifts=[32],        row=201),
        make_arch("1_2_4_32", T=8, shifts=[1, 2, 4, 32], row=202),
        make_arch("1_2_32",   T=8, shifts=[1, 2, 32],  row=203),
    ]

    for arch in archs:
        run_id = f"{arch['arch_id']}__seed{seed}"
        print(f"\n=== {run_id} (T={arch['T']}, shifts={arch['shifts']}) ===")
        t0 = time.time()
        torch.manual_seed(seed)
        try:
            model, log = train_h8_multidistance(arch, cfg, buf_anchor)
        except Exception as e:
            print(f"  TRAIN FAIL {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue
        elapsed = time.time() - t0
        print(f"  train done in {elapsed/60:.1f} min")

        meta = _meta_from_arch(arch, seed)
        ckpt_path = _save_run(model, log, run_id, meta)
        ok = _hf_push_ckpt(ckpt_path, run_id)
        print(f"  HF push: {'OK' if ok else 'FAIL'}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Round 2 training DONE. Probing kicked off after this.")


if __name__ == "__main__":
    main()
