"""W's MYSTERY arch — train TXCMultiplicativeMergeH8 at k_pos=20 shifts=(T,).

Same training setup as train_kpos20_h8_shifts.py (Y's OBLITERATION) but
with the multiplicative-merge encoder. The arch class subclasses
TXCBareMultiDistanceContrastiveAntidead and overrides only _pre_activation
to use multiplicative merge.

Hypothesis: multiplicative-merge features are sparser, higher-confidence,
representing "concept active across the entire window". Steering them
should be more coherent at high strength than canonical additive-merge.

Run:
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_multiplicative_h8 \\
      --T 2 --shifts 2 --seed 42
"""
from __future__ import annotations

import argparse
import os
import time
import json

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    DEFAULT_D_IN, DEFAULT_D_SAE, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, banner,
    OUT_DIR,
)
from experiments.phase7_unification._train_utils import TrainCfg, preload_single
from experiments.phase7_unification.train_phase7 import (
    _save_run, _hf_push_ckpt, _contrastive_train,
)
from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    make_multidistance_pair_gen_gpu,
)
from src.architectures.txc_multiplicative_h8 import TXCMultiplicativeMergeH8


K_POS = 20


def train_one(T: int, shifts: tuple, seed: int, push_to_hf: bool, max_steps: int | None) -> None:
    arch_id = f"txc_mult_h8_t{T}_kpos{K_POS}_shifts{'_'.join(str(s) for s in shifts)}"
    print(f"\n=== MYSTERY (multiplicative): {arch_id} (T={T}, k_pos={K_POS}, k_win={K_POS*T}, shifts={shifts}) seed={seed} ===",
          flush=True)
    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps} "
          f"plateau={cfg.plateau_threshold} min_steps={cfg.min_steps}")

    print("  preloading L12 anchor cache to GPU...")
    t0 = time.time()
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)} dtype={buf.dtype} device={buf.device} "
          f"size={buf.element_size() * buf.nelement() / 1e9:.1f} GB  (preload took {time.time()-t0:.1f}s)")

    h = int(DEFAULT_D_SAE * 0.2)
    model = TXCMultiplicativeMergeH8(
        DEFAULT_D_IN, DEFAULT_D_SAE, T, K_POS * T,  # k = k_pos * T = k_win
        shifts=shifts, weights=None,
        matryoshka_h_size=h, alpha=1.0,
    ).to("cuda")

    gen = make_multidistance_pair_gen_gpu(buf, T, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    log = _contrastive_train(model, gen, cfg, alpha=1.0, init_x_for_geom_median=init_x)
    log["shifts"] = list(shifts); log["matryoshka_h_size"] = h; log["alpha"] = 1.0
    log["arch_id"] = arch_id
    log["src_class"] = "TXCMultiplicativeMergeH8"
    log["src_module"] = "src.architectures.txc_multiplicative_h8"
    log["T"] = T
    log["k_pos"] = K_POS
    log["k_win"] = K_POS * T
    log["d_sae"] = DEFAULT_D_SAE
    log["d_in"] = DEFAULT_D_IN
    log["subject_model"] = SUBJECT_MODEL
    log["anchor_layer"] = ANCHOR_LAYER
    log["seed"] = seed

    print(f"  trained in {log.get('final_step_wall_s', 0)/60:.1f} min "
          f"(final_step={log.get('final_step')}, converged={log.get('converged')})",
          flush=True)

    # Save
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed{seed}.pt"
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed{seed}.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), ckpt_path)
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"  [save] {arch_id} → {ckpt_path}")

    # Append to training_index
    _append_index(arch_id, log, seed, ckpt_path)

    if push_to_hf:
        _hf_push_ckpt(ckpt_path, f"{arch_id}__seed{seed}")


def _append_index(arch_id: str, log: dict, seed: int, ckpt_path) -> None:
    """Append a row to training_index.jsonl matching Y's format."""
    row = {
        "run_id": f"{arch_id}__seed{seed}",
        "row": -1,
        "arch_id": arch_id,
        "arch": arch_id,
        "group": "mystery_multiplicative",
        "src_class": "TXCMultiplicativeMergeH8",
        "src_module": "src.architectures.txc_multiplicative_h8",
        "T": log["T"],
        "T_max": None,
        "t_sample": None,
        "n_layers": None,
        "k_win": log["k_win"],
        "k_pos": log["k_pos"],
        "shifts": list(log["shifts"]),
        "alpha": float(log.get("alpha", 1.0)),
        "gamma": None,
        "n_scales": None,
        "seed": seed,
        "d_in": log["d_in"],
        "d_sae": log["d_sae"],
        "subject_model": log["subject_model"],
        "anchor_layer": log["anchor_layer"],
        "mlc_layers": list(MLC_LAYERS),
        "phase": "phase7_unification",
        "purpose": "W's MYSTERY arch — multiplicative-merge encoder for coherent steering",
        "recipe": "TXCMultiplicativeMergeH8: per-pos softplus + product across T positions. H8 stack everywhere else.",
        "final_step": log.get("final_step"),
        "converged": log.get("converged"),
        "elapsed_s": log.get("final_step_wall_s"),
        "ckpt": str(ckpt_path),
    }
    index_path = OUT_DIR / "training_index.jsonl"
    with open(index_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--shifts", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    args = p.parse_args()
    banner(__file__)
    push = not args.no_hf_push
    train_one(args.T, tuple(args.shifts), args.seed, push, args.max_steps)


if __name__ == "__main__":
    main()
