"""Train hill_subseq_h8_T20_s8 — directly answers Han's 2026-04-29 concern
on whether higher resampling at higher T_max could win the leaderboard.

T_max=20, t_sample=8, k_win=500. Auto-scaled shifts (1, 5, 10).

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.hill_climb.train_t20_s8
"""
from __future__ import annotations

import time

import torch

from experiments.phase7_unification._paths import banner
from experiments.phase7_unification._train_utils import preload_single
from experiments.phase7_unification.train_phase7 import (
    TrainCfg, train_subseq_h8, _save_run, _meta_from_arch, _hf_push_ckpt,
)


def make_arch() -> dict:
    return {
        "row": 200,
        "arch_id": "hill_subseq_h8_T20_s8",
        "group": 99,
        "T": None,
        "T_max": 20,
        "t_sample": 8,
        "n_layers": None,
        "k_win": 500,
        "k_pos": int(500 / 20),       # 25
        "shifts": None,                # auto-scaled in train_subseq_h8
        "alpha": 1.0,
        "gamma": None,
        "n_scales": None,
        "src_module": "src.architectures.phase5b_subseq_sampling_txcdr",
        "src_class": "SubseqH8",
        "recipe": "hill-climb subseq H8 at T_max=20 t_sample=8",
        "purpose": "answer Han's 2026-04-29 question on T=10/8-resampling/T=20",
    }


def main() -> None:
    banner(__file__)
    seed = 42
    # b=1024 instead of paper-wide b=4096: A40 OOM at b=4096 due to
    # T_max=20 × t_sample=8 backward-pass intermediates.
    # This deviates from paper batch_size constant — flagged in writeup.
    cfg = TrainCfg(seed=seed, max_steps=8000, batch_size=1024)

    print("Preloading anchor buffer (n_seqs=6000 for A40)...")
    # Default PRELOAD_SEQS=24000 was for H200 188 GB; A40 OOMs.
    # 6000 was Phase 5B's 5090 setting; trades preloaded coverage for VRAM.
    buf_anchor = preload_single(n_seqs=6000)

    arch = make_arch()
    run_id = f"{arch['arch_id']}__seed{seed}"
    print(f"\n=== {run_id} (T_max={arch['T_max']}, t_sample={arch['t_sample']}) ===")
    t0 = time.time()
    torch.manual_seed(seed)
    try:
        model, log = train_subseq_h8(arch, cfg, buf_anchor)
    except Exception as e:
        print(f"  TRAIN FAIL {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f"  train done in {elapsed/60:.1f} min")

    meta = _meta_from_arch(arch, seed)
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  ckpt: {ckpt_path}")
    ok = _hf_push_ckpt(ckpt_path, run_id)
    print(f"  HF push: {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
