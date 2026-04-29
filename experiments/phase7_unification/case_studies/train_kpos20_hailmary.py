"""Phase 7 Hail Mary — train TXCBareAntidead at k_pos=20 (genuine gap cells).

Atomic single-axis ladder from T-SAE k=20 → full TXCDR at fixed k_pos=20.
Trains the cells canonical_archs.json never reached (canonical convention
is k_win=500 ⇒ k_pos drops as T grows; we hold k_pos=20 fixed instead).

Apples-to-apples with canonical T-SAE k=20:
  - Same activation source: experiments/phase7_unification/data/cached_activations/...
  - Same TrainCfg (b=4096, lr=3e-4, max_steps=25k, plateau early-stop @ 0.02).
  - Same seed=42 primary.
  - Random-init (no warm-start) — matches T-SAE k=20's init protocol.

Cells covered:
  - Step 1: T=2, k_pos=20, k_win=40   →  arch_id  txc_bare_antidead_t2_kpos20
  - Step 2: T=5, k_pos=20, k_win=100  →  arch_id  txc_bare_antidead_t5_kpos20  (meeting cell with W)
  - Step 3+: T=5, per-position decoder etc — handled separately if Step 2 motivates it.

Outputs (canonical paths):
  results/ckpts/<arch_id>__seed{seed}.pt
  results/training_logs/<arch_id>__seed{seed}.json
  results/training_index.jsonl  (appended)

Run from repo root:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_hailmary \\
      --T 5 --seed 42

Multiple cells in one invocation:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_hailmary \\
      --T 5 --T 2 --seed 42
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
    train_txc_bare_antidead, _save_run, _hf_push_ckpt,
)


K_POS = 20  # the matched-sparsity target (= T-SAE k=20 per-token sparsity)


def build_arch(T: int) -> dict:
    """Construct an in-process arch dict matching canonical_archs.json schema
    for TXCBareAntidead. Not added to canonical_archs.json — kept in-process
    so the source-of-truth file stays pristine.
    """
    k_win = K_POS * T
    return {
        "row": -1,                   # synthetic; not in canonical
        "arch_id": f"txc_bare_antidead_t{T}_kpos{K_POS}",
        "group": "hailmary_phase2",  # synthetic group
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": K_POS,
        "shifts": None,
        "src_module": "src.architectures.txc_bare_antidead",
        "src_class": "TXCBareAntidead",
        "recipe": f"TXC barebones + tsae anti-dead at T={T}, k_pos={K_POS} (k_win={k_win})",
        "purpose": (f"Phase 7 Hail Mary atomic-axis ladder cell — TXC at matched "
                    f"per-token sparsity k_pos={K_POS} (T-SAE k=20 anchor)"),
    }


def build_meta(arch: dict, seed: int) -> dict:
    """Build the per-run meta dict (saved alongside ckpt, used by loaders).

    Critical fields the loader expects (per run_probing_phase7._load_phase7_model
    + _arch_utils.load_phase7_model_safe): src_class, T, k_win, k_pos, d_in,
    d_sae. Plus the phase/anchor_layer/subject_model plumbing.
    """
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
    arch = build_arch(T)
    arch_id = arch["arch_id"]
    print(f"\n=== Hail Mary cell: {arch_id} (T={T}, k_pos={K_POS}, k_win={arch['k_win']}) seed={seed} ===",
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

    model, log = train_txc_bare_antidead(arch, cfg, buf)
    log["final_step_wall_s"] = time.time() - t0

    meta = build_meta(arch, seed)
    run_id = f"{arch_id}__seed{seed}"
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  trained in {log['final_step_wall_s']/60:.1f} min total wall "
          f"(final_step={log.get('final_step')}, converged={log.get('converged')}, "
          f"plateau_last={log.get('plateau_last')})")

    if push_to_hf:
        _hf_push_ckpt(ckpt_path, run_id)

    # Free GPU buffer between cells.
    import torch
    del buf, model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, action="append", required=True,
                   help="window length T (use multiple times for several cells in one invocation)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap training steps (default: TrainCfg.max_steps=25000)")
    p.add_argument("--no-hf-push", action="store_true",
                   help="don't push ckpts to HF txcdr-base")
    args = p.parse_args()
    banner(__file__)

    push = not args.no_hf_push
    for T in args.T:
        train_one(T, args.seed, push, args.max_steps)


if __name__ == "__main__":
    main()
