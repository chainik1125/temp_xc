"""Hill-climbing round 1: SubseqH8 T_max sweep.

Goal: beat 0.8989 (current k_feat=5 winner = phase57_partB_h8_bare_multidistance_t8)
on Phase 7 seed=42 sparse-probing using subseq sampling at larger T_max.

Variants (each ~25 min train on H200):
  V1: hill_subseq_h8_T12   (T_max=12, t_sample=5, shifts=[1, 3, 6])
  V2: hill_subseq_h8_T16   (T_max=16, t_sample=5, shifts=[1, 4, 8])
  V3: hill_subseq_h8_T20   (T_max=20, t_sample=5, shifts=[1, 5, 10])

Distinct arch_id prefix `hill_` so leaderboard generation can filter
these out of the canonical 38-arch headline.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.hill_climb.round1_subseq_t_sweep
"""
from __future__ import annotations

import time

import torch

from experiments.phase7_unification._paths import banner
from experiments.phase7_unification._train_utils import preload_single
from experiments.phase7_unification.train_phase7 import (
    TrainCfg, train_subseq_h8, _save_run, _meta_from_arch, _hf_push_ckpt,
)


def make_arch(T_max: int, row: int) -> dict:
    arch_id = f"hill_subseq_h8_T{T_max}_s5"
    return {
        "row": row,
        "arch_id": arch_id,
        "group": 99,             # 99 = hill-climb (not canonical)
        "T": None,
        "T_max": T_max,
        "t_sample": 5,
        "n_layers": None,
        "k_win": 500,
        "k_pos": int(500 / T_max),
        "shifts": None,          # auto-scaled in train_subseq_h8
        "alpha": 1.0,
        "gamma": None,
        "n_scales": None,
        "src_module": "src.architectures.phase5b_subseq_sampling_txcdr",
        "src_class": "SubseqH8",
        "recipe": f"hill-climb subseq H8 at T_max={T_max}",
        "purpose": "round1 hill-climb: subseq T_max sweep beyond Phase 5b's T_max=10",
    }


def main() -> None:
    banner(__file__)
    seed = 42
    cfg = TrainCfg(seed=seed, max_steps=8000)

    print("Preloading anchor buffer...")
    buf_anchor = preload_single()

    archs = [make_arch(T_max=t, row=100 + i) for i, t in enumerate((12, 16, 20))]

    for arch in archs:
        run_id = f"{arch['arch_id']}__seed{seed}"
        print(f"\n=== {run_id} (T_max={arch['T_max']}, t_sample={arch['t_sample']}) ===")
        t0 = time.time()
        torch.manual_seed(seed)
        try:
            model, log = train_subseq_h8(arch, cfg, buf_anchor)
        except Exception as e:
            print(f"  TRAIN FAIL {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue
        elapsed = time.time() - t0
        print(f"  train done in {elapsed/60:.1f} min")

        meta = _meta_from_arch(arch, seed)
        ckpt_path = _save_run(model, log, run_id, meta)
        print(f"  ckpt: {ckpt_path}")
        ok = _hf_push_ckpt(ckpt_path, run_id)
        print(f"  HF push: {'OK' if ok else 'FAIL'}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Round 1 training DONE. Run probing next:")
    print("  .venv/bin/python -m experiments.phase7_unification.run_probing_phase7 \\")
    print("    --run_ids hill_subseq_h8_T12_s5__seed42 \\")
    print("              hill_subseq_h8_T16_s5__seed42 \\")
    print("              hill_subseq_h8_T20_s5__seed42 \\")
    print("    --S 32 --k_feat 5 20")


if __name__ == "__main__":
    main()
