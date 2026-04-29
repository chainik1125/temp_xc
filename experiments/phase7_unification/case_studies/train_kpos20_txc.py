"""Paper-grade trainer for k_pos=20 TXCBareAntidead variants (W cells C/D/F).

Mirrors `train_phase7.train_txc_bare_antidead` but exposes:
  - `--T`, `--k-pos`  (cells C: T=3 / D: T=5 / F: T=10, all k_pos=20)
  - `--warm-start auto`  → init from `tsae_paper_k20__seed42.pt` by tiling
    the per-token encoder across T positions and dividing the decoder by T.
  - `--arch-id`  → custom arch_id; default is `txc_bare_antidead_t<T>_kpos<k_pos>`.

Saves to canonical paths so the existing case-study pipeline can pick up
the new ckpt without modification:
  results/ckpts/<arch_id>__seed<seed>.pt
  results/training_logs/<arch_id>__seed<seed>.json

The activation cache at `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy`
is REQUIRED — build it once with:
  TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.build_act_cache_phase7 --layer 12

Run example:
  TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_txc \
    --T 5 --k-pos 20 --warm-start auto --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    CACHE_DIR, CKPT_DIR, LOGS_DIR, ANCHOR_LAYER, ANCHOR_LAYER_KEY,
    SUBJECT_MODEL, DEFAULT_D_IN, DEFAULT_D_SAE, banner,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, iterate_train, make_window_gen_gpu, preload_single,
)
from src.architectures.txc_bare_antidead import TXCBareAntidead


def warm_start_from_tsae(model: TXCBareAntidead, tsae_ckpt: Path, T: int) -> dict:
    """In-place warm-start: tile T-SAE k=20's per-token encoder across T positions.

    T-SAE state dict shapes (TemporalMatryoshkaBatchTopKSAE):
      W_enc: (d_in, d_sae)
      b_enc: (d_sae,)
      W_dec: (d_sae, d_in)
      b_dec: (d_in,)

    TXCBareAntidead state dict shapes (T windows):
      W_enc: (T, d_in, d_sae)
      b_enc: (d_sae,)
      W_dec: (d_sae, T, d_in)
      b_dec: (T, d_in)

    Mapping (preserves T-SAE's per-token output as the SUM across positions):
      W_enc[t, :, :]   = W_enc_tsae           (broadcast)
      b_enc            = b_enc_tsae
      W_dec[:, t, :]   = W_dec_tsae / T       (broadcast / T)
      b_dec[t, :]      = b_dec_tsae / T

    Then `decode(z)` summed over t == T-SAE's output. This makes the warm-
    started TXC behave like T-SAE k=20 at the per-position-write protocol
    (Q2.C); under the right-edge protocol the intervention amplitude is
    1/T of T-SAE's at init, but training adapts.
    """
    sd_tsae = torch.load(tsae_ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd_tsae, dict) and "state_dict" in sd_tsae:
        sd_tsae = sd_tsae["state_dict"]
    # Sanity check shapes
    W_enc_tsae = sd_tsae["W_enc"]   # (d_in, d_sae)
    b_enc_tsae = sd_tsae["b_enc"]
    W_dec_tsae = sd_tsae["W_dec"]   # (d_sae, d_in)
    b_dec_tsae = sd_tsae["b_dec"]
    d_in_tsae, d_sae_tsae = W_enc_tsae.shape
    assert d_in_tsae == model.d_in == DEFAULT_D_IN, f"d_in mismatch: tsae={d_in_tsae} model={model.d_in}"
    assert d_sae_tsae == model.d_sae == DEFAULT_D_SAE, f"d_sae mismatch: tsae={d_sae_tsae} model={model.d_sae}"
    with torch.no_grad():
        for t in range(T):
            model.W_enc.data[t].copy_(W_enc_tsae.to(model.W_enc.dtype))
            model.W_dec.data[:, t, :].copy_((W_dec_tsae / T).to(model.W_dec.dtype))
            model.b_dec.data[t].copy_((b_dec_tsae / T).to(model.b_dec.dtype))
        model.b_enc.data.copy_(b_enc_tsae.to(model.b_enc.dtype))
        # Re-normalise decoder to unit-norm per-feature (model invariant).
        model._normalize_decoder()
        # Mark b_dec as initialised so iterate_train skips the geom-median init.
        model.b_dec_initialized.fill_(True)
    return {
        "warm_start_source": str(tsae_ckpt),
        "warm_start_pattern": "tile_W_enc + W_dec/T + b_dec/T (right-edge sum reproduces T-SAE per-token output)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True, help="window length")
    ap.add_argument("--k-pos", type=int, required=True, help="per-position TopK budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch-id", type=str, default=None,
                    help="default = txc_bare_antidead_t<T>_kpos<k_pos>")
    ap.add_argument("--warm-start", type=str, default="auto",
                    help="'auto' = tsae_paper_k20__seed42.pt; '' = cold start; or a .pt path")
    ap.add_argument("--max-steps", type=int, default=25_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-seqs", type=int, default=24_000,
                    help="how many seqs from the cache to preload")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing ckpt + log")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse args + check artefacts, do nothing else")
    args = ap.parse_args()

    banner(__file__)

    T = args.T
    k_pos = args.k_pos
    k_win = k_pos * T
    arch_id = args.arch_id or f"txc_bare_antidead_t{T}_kpos{k_pos}"

    print(f"\n[W cell] arch_id={arch_id}  T={T}  k_pos={k_pos}  k_win={k_win}")
    print(f"  d_in={DEFAULT_D_IN}  d_sae={DEFAULT_D_SAE}  seed={args.seed}")

    ckpt_path = CKPT_DIR / f"{arch_id}__seed{args.seed}.pt"
    log_path = LOGS_DIR / f"{arch_id}__seed{args.seed}.json"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists() and not args.force:
        print(f"  ckpt already exists at {ckpt_path}; skipping. (Use --force to retrain.)")
        return

    # Resolve warm-start path
    warm_path: Path | None = None
    if args.warm_start == "auto":
        warm_path = CKPT_DIR / f"tsae_paper_k20__seed{args.seed}.pt"
    elif args.warm_start:
        warm_path = Path(args.warm_start)
    if warm_path is not None and not warm_path.exists():
        raise FileNotFoundError(
            f"warm-start ckpt not found at {warm_path}. Pull with "
            f"`.venv/bin/python -m experiments.phase7_unification.case_studies._download_ckpts "
            f"--seed {args.seed} tsae_paper_k20`"
        )

    # Cache check
    cache_file = CACHE_DIR / f"{ANCHOR_LAYER_KEY}.npy"
    cache_present = cache_file.exists()
    if args.dry_run:
        print(f"  dry-run: ckpt would write to {ckpt_path}")
        if warm_path:
            print(f"  warm-start would load from {warm_path}")
        print(f"  activation cache {'PRESENT' if cache_present else 'MISSING'} at {cache_file}")
        return
    if not cache_present:
        raise FileNotFoundError(
            f"activation cache missing at {cache_file}. Build with "
            f"`TQDM_DISABLE=1 .venv/bin/python -m experiments.phase7_unification.build_act_cache_phase7 "
            f"--layer {ANCHOR_LAYER}`"
        )

    # Build model
    print(f"  building TXCBareAntidead(d_in={DEFAULT_D_IN}, d_sae={DEFAULT_D_SAE}, T={T}, k={k_win})")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = TXCBareAntidead(DEFAULT_D_IN, DEFAULT_D_SAE, T=T, k=k_win).to("cuda")

    warm_meta: dict = {}
    if warm_path is not None:
        print(f"  warm-start from {warm_path}")
        warm_meta = warm_start_from_tsae(model, warm_path, T=T)
        print(f"    {warm_meta['warm_start_pattern']}")

    # Preload activation cache
    print(f"  preloading L{ANCHOR_LAYER} cache: {args.n_seqs} seqs from {cache_file}")
    t0 = time.time()
    buf_anchor = preload_single(layer_key=ANCHOR_LAYER_KEY, n_seqs=args.n_seqs).to("cuda")
    print(f"    cache shape={tuple(buf_anchor.shape)}  ({(time.time()-t0):.1f}s)")
    print(f"    cache VRAM = {buf_anchor.numel() * buf_anchor.element_size() / 1e9:.2f} GB")

    # Sample generator
    gen = make_window_gen_gpu(buf_anchor, T)

    # Geom-median init for b_dec — only if NOT warm-started (warm-start sets it).
    if not bool(model.b_dec_initialized):
        init_x = gen(args.batch_size)
        print(f"  geom-median b_dec init from batch shape={tuple(init_x.shape)}")
        model.init_b_dec_geometric_median(init_x)

    # Train
    cfg = TrainCfg(
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(f"  TrainCfg: lr={cfg.lr}  batch={cfg.batch_size}  max_steps={cfg.max_steps}")
    print(f"  starting train…")
    log = iterate_train(
        model, gen, cfg, torch.device("cuda"),
        normalize_decoder=model._normalize_decoder,
        grad_post_hook=model.remove_gradient_parallel_to_decoder,
    )

    # Build meta dict (matches the schema expected by case-study loaders)
    meta = {
        "arch_id": arch_id,
        "src_class": "TXCBareAntidead",
        "src_module": "src.architectures.txc_bare_antidead",
        "T": T,
        "T_max": None,
        "t_sample": None,
        "k_win": k_win,
        "k_pos": k_pos,
        "shifts": None,
        "alpha": None,
        "gamma": None,
        "n_scales": None,
        "n_layers": None,
        "mlc_layers": None,
        "d_in": DEFAULT_D_IN,
        "d_sae": DEFAULT_D_SAE,
        "subject_model": SUBJECT_MODEL,
        "anchor_layer": ANCHOR_LAYER,
        "hook_name": None,  # default → resid_post (the layer-output residual)
        "seed": args.seed,
        "phase": "phase7_unification",
        "group": 2,
        "recipe": "TXCBareAntidead at k_pos=20 (W Phase 1 sweep)",
        "purpose": "W Phase 1 sweep cell — sparsity-matched TXC",
        # training info
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "max_steps": cfg.max_steps,
        "elapsed_s": log.get("elapsed_s"),
        "final_step": log.get("final_step"),
        "converged": log.get("converged"),
        "plateau_last": log.get("plateau_last"),
        "loss": log.get("loss"),
        "l0": log.get("l0"),
        "steps_logged": log.get("steps_logged"),
        "n_train_seqs": int(buf_anchor.shape[0]),
    }
    meta.update(warm_meta)

    # Save
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, ckpt_path)
    log_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  saved ckpt -> {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  saved log  -> {log_path}")
    print(f"  elapsed: {meta['elapsed_s']:.0f}s ({meta['elapsed_s']/60:.1f} min); converged={meta['converged']}")
    final_l0 = meta["l0"][-1] if meta["l0"] else None
    print(f"  final l0={final_l0}  (target ≈ {k_win})")


if __name__ == "__main__":
    main()
