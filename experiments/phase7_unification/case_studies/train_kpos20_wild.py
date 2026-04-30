"""Phase 7 Y — wild-trick trainer for matched-sparsity TXC variants.

Beyond the brief's atomic axis ladder; explores creative axes flagged in
W's plan + Han's recent suggestions:

  1. T-SAE k=20 encoder warm-start (brief's suggested trick — never tried!).
     Initialize TXC encoder by broadcasting T-SAE k=20's per-token encoder
     across T positions; decoder fresh random.
  2. Custom k_pos (sparser than 20, e.g. 10 or 5).
  3. Custom k_win (anchor regime: k_win > T × k_pos).

Run examples:

  # Wild #1: T-SAE encoder warm-start at T=2, k_pos=20
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_wild \\
      --T 2 --k-pos 20 --warm-start tsae_encoder --seed 42

  # Wild #2: super-sparse k_pos=10 at T=2
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_wild \\
      --T 2 --k-pos 10 --seed 42

  # Wild #3: anchor regime (k_win=200 at T=5, k_pos=20)
  TQDM_DISABLE=1 .venv/bin/python -m \\
      experiments.phase7_unification.case_studies.train_kpos20_wild \\
      --T 5 --k-pos 20 --k-win 200 --seed 42
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch
import numpy as np

from experiments.phase7_unification._paths import (
    DEFAULT_D_IN, DEFAULT_D_SAE, ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL, CKPT_DIR, banner,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, preload_single, make_window_gen_gpu,
)
from experiments.phase7_unification.train_phase7 import (
    _flat_train, _save_run, _hf_push_ckpt,
)
from src.architectures.txc_bare_antidead import TXCBareAntidead


def warm_start_from_tsae(model: TXCBareAntidead, tsae_ckpt: Path, T: int) -> None:
    """Initialize TXC encoder by broadcasting T-SAE k=20's per-token encoder.

    T-SAE shapes:
        W_enc: (d_in, d_sae)
        W_dec: (d_sae, d_in)
        b_enc: (d_sae,)
        b_dec: (d_in,)

    TXC shapes:
        W_enc: (T, d_in, d_sae)  -- broadcast tsae W_enc to all T positions
        W_dec: (d_sae, T, d_in)  -- random init (encoder-only warm-start)
        b_enc: (d_sae,)
        b_dec: (T, d_in)         -- broadcast tsae b_dec
    """
    sd = torch.load(tsae_ckpt, map_location="cpu", weights_only=True)
    W_enc_t = sd["W_enc"].float()  # (d_in, d_sae)
    b_enc_t = sd["b_enc"].float()  # (d_sae,)
    b_dec_t = sd["b_dec"].float()  # (d_in,)

    print(f"  warm-start: T-SAE k=20 encoder broadcast across T={T} positions; decoder = random")
    with torch.no_grad():
        for t in range(T):
            model.W_enc.data[t] = W_enc_t.to(model.W_enc.device).to(model.W_enc.dtype)
            model.b_dec.data[t] = b_dec_t.to(model.b_dec.device).to(model.b_dec.dtype)
        model.b_enc.data[:] = b_enc_t.to(model.b_enc.device).to(model.b_enc.dtype)
        # Decoder stays fresh random (was kaiming init in __init__, then unit-norm).
        # Renormalize decoder so subsequent training starts from unit-norm atoms.
        model._normalize_decoder()
        model.b_dec_initialized.fill_(True)


def build_arch_dict(T: int, k_pos: int, k_win: int, warm_start: str | None) -> dict:
    suffix_parts = []
    if warm_start:
        suffix_parts.append(f"ws_{warm_start}")
    if k_win != T * k_pos:
        suffix_parts.append(f"kwin{k_win}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    arch_id = f"txc_bare_antidead_t{T}_kpos{k_pos}{suffix}"
    return {
        "row": -1,
        "arch_id": arch_id,
        "group": "hailmary_phase2_wild",
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": k_pos,
        "shifts": None,
        "src_module": "src.architectures.txc_bare_antidead",
        "src_class": "TXCBareAntidead",
        "recipe": (f"TXC bare-antidead T={T}, k_pos={k_pos} (k_win={k_win}), "
                   f"warm_start={warm_start or 'random'}"),
        "purpose": (f"Hail Mary wild trick — extend matched-sparsity exploration"),
        "warm_start": warm_start,
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
        "warm_start": arch.get("warm_start"),
    }


def train_one(T: int, k_pos: int, k_win: int, warm_start: str | None, seed: int,
              push_to_hf: bool, max_steps: int | None) -> None:
    arch = build_arch_dict(T, k_pos, k_win, warm_start)
    arch_id = arch["arch_id"]
    print(f"\n=== Wild cell: {arch_id} ===")
    print(f"  T={T}, k_pos={k_pos}, k_win={k_win}, warm_start={warm_start}, seed={seed}")

    cfg = TrainCfg(seed=seed) if max_steps is None else TrainCfg(seed=seed, max_steps=max_steps)
    print(f"  TrainCfg: batch={cfg.batch_size} lr={cfg.lr} max_steps={cfg.max_steps} plateau={cfg.plateau_threshold}")

    t0 = time.time()
    print("  preloading L12 anchor cache to GPU...")
    buf = preload_single()
    print(f"    shape={tuple(buf.shape)} preload took {time.time()-t0:.1f}s")

    model = TXCBareAntidead(DEFAULT_D_IN, DEFAULT_D_SAE, T, k_win).to("cuda")

    init_x = None
    if warm_start == "tsae_encoder":
        tsae_ckpt = CKPT_DIR / "tsae_paper_k20__seed42.pt"
        if not tsae_ckpt.exists():
            raise SystemExit(f"missing T-SAE k=20 ckpt: {tsae_ckpt}")
        warm_start_from_tsae(model, tsae_ckpt, T)
        # b_dec already initialized; pass init_x=None so _flat_train doesn't redo geom-median
    else:
        # Random init — let _flat_train do geom-median b_dec init from a sample
        gen0 = make_window_gen_gpu(buf, T)
        init_x = gen0(cfg.batch_size)

    model.to(torch.float32)

    gen = make_window_gen_gpu(buf, T)
    log = _flat_train(model, gen, cfg, init_x_for_geom_median=init_x)
    log["final_step_wall_s"] = time.time() - t0
    log["warm_start"] = warm_start

    meta = build_meta(arch, seed)
    run_id = f"{arch_id}__seed{seed}"
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  trained in {log['final_step_wall_s']/60:.1f} min  "
          f"final_step={log.get('final_step')}  converged={log.get('converged')}  "
          f"plateau_last={log.get('plateau_last')}")

    if push_to_hf:
        _hf_push_ckpt(ckpt_path, run_id)

    del buf, model
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--k-pos", type=int, required=True, help="per-position sparsity cap")
    p.add_argument("--k-win", type=int, default=None,
                   help="window-level TopK budget (default = T × k_pos for canonical convention)")
    p.add_argument("--warm-start", choices=[None, "tsae_encoder"], default=None,
                   help="warm-start scheme; None = random init")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-hf-push", action="store_true")
    args = p.parse_args()
    banner(__file__)

    k_win = args.k_win if args.k_win is not None else args.T * args.k_pos
    push = not args.no_hf_push
    train_one(args.T, args.k_pos, k_win, args.warm_start, args.seed, push, args.max_steps)


if __name__ == "__main__":
    main()
