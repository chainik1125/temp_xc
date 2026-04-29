"""Retrain phase63_track2_t20 in the original Phase 6.3 config.

Reproduces the ckpt that won Phase 6.3's "T=20 Pareto-dominates T-SAE"
finding:
  - subject model:  google/gemma-2-2b-it (NOT base)
  - layer:          L13 (NOT L12)
  - architecture:   TXCBareAntidead (Track 2 recipe)
  - T:              20
  - k_pos:          100  (k_win = 2000)
  - aux_k:          512
  - dead_threshold: 10M tokens
  - auxk_alpha:     1/32
  - data:           FineWeb activations cached at L13 of gemma-2-2b-it

Differs from the HF txc_bare_antidead_t20 retrain (Phase 7 conventions:
gemma-2-2b base, L12, k_win=500). The hypothesis being tested: does the
original L13-IT setup recover steerable concept directions, or is the
T=20 steering failure structural?

Prerequisite: cache must already be built at
    data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy
(via `src.data.nlp.cache_activations`).

Usage:
    PHASE7_REPO=/root/temp_xc TQDM_DISABLE=1 \
    .venv/bin/python -m experiments.phase7_unification.case_studies.steering.retrain_phase63_t20
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

from src.architectures.txc_bare_antidead import TXCBareAntidead
from experiments.phase5b_t_scaling_explore._train_utils import make_window_gen_gpu
from experiments.phase7_unification._train_utils import TrainCfg, iterate_train


REPO = Path(os.environ.get("PHASE7_REPO", os.getcwd())).resolve()
CACHE_DIR = REPO / "data/cached_activations/gemma-2-2b-it/fineweb"
ANCHOR_LAYER_KEY = "resid_L13"
PRELOAD_SEQS = 6000

OUT_DIR = REPO / "experiments/phase7_unification/results"
CKPT_DIR = OUT_DIR / "ckpts"
LOGS_DIR = OUT_DIR / "training_logs"
ARCH_ID = "phase63_track2_t20_retrain"
SEED = 42

# Phase 6.3 / Track 2 hyperparameters
T = 20
K_WIN = 2000           # k_pos=100 × T=20
AUX_K = 512
DEAD_THRESHOLD = 10_000_000
AUXK_ALPHA = 1.0 / 32.0
D_IN = 2304
D_SAE = 18432


def main():
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cache_path = CACHE_DIR / f"{ANCHOR_LAYER_KEY}.npy"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"missing cache: {cache_path}. Build it first via "
            f"`python -m src.data.nlp.cache_activations --model gemma-2-2b-it "
            f"--dataset fineweb --num-sequences {PRELOAD_SEQS} --seq-length 128 "
            f"--layer_indices 13`"
        )

    print(f"loading cache from {cache_path} ...")
    arr = np.load(cache_path, mmap_mode="r")
    n_have = min(arr.shape[0], PRELOAD_SEQS)
    sub = np.asarray(arr[:n_have], dtype=np.float16)
    buf_anchor = torch.from_numpy(sub).to(device)
    print(f"  buf_anchor shape: {tuple(buf_anchor.shape)}")

    print(f"building TXCBareAntidead T={T} k_win={K_WIN}")
    model = TXCBareAntidead(
        D_IN, D_SAE, T, K_WIN,
        aux_k=AUX_K,
        dead_threshold_tokens=DEAD_THRESHOLD,
        auxk_alpha=AUXK_ALPHA,
    ).to(device)

    gen_fn = make_window_gen_gpu(buf_anchor, T)
    init_x = gen_fn(1024)
    if hasattr(model, "init_b_dec_geometric_median"):
        print("  init b_dec via geometric median")
        model.init_b_dec_geometric_median(init_x)

    cfg = TrainCfg(
        lr=3e-4,
        batch_size=1024,    # Phase 6.3 used 1024 on A40
        max_steps=25_000,
        log_every=200,
        grad_clip=1.0,
        plateau_threshold=0.02,
        min_steps=3_000,
    )
    print(f"training: lr={cfg.lr} batch={cfg.batch_size} max_steps={cfg.max_steps} "
          f"plateau_threshold={cfg.plateau_threshold} min_steps={cfg.min_steps}")

    norm_dec = model._normalize_decoder if hasattr(model, "_normalize_decoder") else None
    grad_post = (
        model.remove_gradient_parallel_to_decoder
        if hasattr(model, "remove_gradient_parallel_to_decoder") else None
    )

    t0 = time.time()
    log = iterate_train(
        model, gen_fn, cfg, device,
        normalize_decoder=norm_dec, grad_post_hook=grad_post,
    )
    elapsed = time.time() - t0
    print(f"trained in {elapsed:.0f}s")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{ARCH_ID}__seed{SEED}.pt"
    log_path = LOGS_DIR / f"{ARCH_ID}__seed{SEED}.json"
    torch.save(model.state_dict(), ckpt_path)
    log_payload = {
        "arch": ARCH_ID,
        "src_class": "TXCBareAntidead",
        "seed": SEED,
        "T": T,
        "k_pos": K_WIN // T,
        "k_win": K_WIN,
        "aux_k": AUX_K,
        "auxk_alpha": AUXK_ALPHA,
        "dead_threshold_tokens": DEAD_THRESHOLD,
        "d_in": D_IN,
        "d_sae": D_SAE,
        "subject_model": "google/gemma-2-2b-it",
        "anchor_layer": 13,
        "training_time_s": elapsed,
        **log,
    }
    log_path.write_text(json.dumps(log_payload, indent=2))
    print(f"saved {ckpt_path} ({ckpt_path.stat().st_size / 1e9:.2f} GB)")
    print(f"saved {log_path}")


if __name__ == "__main__":
    main()
