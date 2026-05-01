"""W's MYSTERY arch — train TXCContrastiveMergeH8 at k_pos=20 shifts=(T,).

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_contrastive_merge_h8 \
    --T 2 --shifts 2 --seed 42
"""
from __future__ import annotations
import argparse
import os
import time
import json

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    DEFAULT_D_IN, DEFAULT_D_SAE, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, banner, OUT_DIR,
)
from experiments.phase7_unification._train_utils import TrainCfg, preload_single
from experiments.phase7_unification.train_phase7 import _hf_push_ckpt, _contrastive_train
from src.architectures.txc_bare_multidistance_contrastive_antidead import make_multidistance_pair_gen_gpu
from src.architectures.txc_contrastive_merge_h8 import TXCContrastiveMergeH8

K_POS = 20


def train_one(T, shifts, seed, push_to_hf, max_steps):
    arch_id = f"txc_contrastive_h8_t{T}_kpos{K_POS}_shifts{'_'.join(str(s) for s in shifts)}"
    print(f"\n=== MYSTERY (contrastive-merge): {arch_id} (T={T}, k_pos={K_POS}, shifts={shifts}) seed={seed} ===", flush=True)
    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)

    print("  preloading L12 anchor cache to GPU...")
    t0 = time.time()
    buf = preload_single()
    print(f"    {tuple(buf.shape)} dtype={buf.dtype} (preload took {time.time()-t0:.1f}s)")

    h = int(DEFAULT_D_SAE * 0.2)
    model = TXCContrastiveMergeH8(
        DEFAULT_D_IN, DEFAULT_D_SAE, T, K_POS * T,
        shifts=shifts, weights=None,
        matryoshka_h_size=h, alpha=1.0,
    ).to("cuda")

    gen = make_multidistance_pair_gen_gpu(buf, T, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)
    log.update({
        "shifts": list(shifts), "matryoshka_h_size": h, "alpha": 1.0,
        "arch_id": arch_id, "src_class": "TXCContrastiveMergeH8",
        "src_module": "src.architectures.txc_contrastive_merge_h8",
        "T": T, "k_pos": K_POS, "k_win": K_POS * T,
        "d_sae": DEFAULT_D_SAE, "d_in": DEFAULT_D_IN,
        "subject_model": SUBJECT_MODEL, "anchor_layer": ANCHOR_LAYER, "seed": seed,
    })

    print(f"  trained in {log.get('final_step_wall_s', 0)/60:.1f} min "
          f"(final_step={log.get('final_step')}, converged={log.get('converged')})", flush=True)

    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed{seed}.pt"
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed{seed}.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), ckpt_path)
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"  [save] {arch_id} → {ckpt_path}")

    row = {
        "run_id": f"{arch_id}__seed{seed}", "row": -1, "arch_id": arch_id, "arch": arch_id,
        "group": "mystery_contrastive", "src_class": "TXCContrastiveMergeH8",
        "src_module": "src.architectures.txc_contrastive_merge_h8",
        "T": T, "T_max": None, "t_sample": None, "n_layers": None,
        "k_win": K_POS * T, "k_pos": K_POS, "shifts": list(shifts),
        "alpha": 1.0, "gamma": None, "n_scales": None, "seed": seed,
        "d_in": DEFAULT_D_IN, "d_sae": DEFAULT_D_SAE,
        "subject_model": SUBJECT_MODEL, "anchor_layer": ANCHOR_LAYER,
        "mlc_layers": list(MLC_LAYERS), "phase": "phase7_unification",
        "purpose": "W's MYSTERY arch — contrastive (end-minus-start) merge encoder",
        "recipe": "TXCContrastiveMergeH8: z = enc(x[T-1]) - enc(x[0]). Captures CHANGE.",
        "final_step": log.get("final_step"), "converged": log.get("converged"),
        "elapsed_s": log.get("final_step_wall_s"), "ckpt": str(ckpt_path),
    }
    with open(OUT_DIR / "training_index.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    if push_to_hf: _hf_push_ckpt(ckpt_path, f"{arch_id}__seed{seed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--shifts", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    args = p.parse_args()
    banner(__file__)
    train_one(args.T, tuple(args.shifts), args.seed, not args.no_hf_push, args.max_steps)


if __name__ == "__main__":
    main()
