"""Compute alive-feature fraction per arch across the T-sweep.

For each arch: load ckpt, forward N fineweb batches, record which
feature indices ever fired. Write per-arch alive count + fraction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/workspace/temp_xc")
sys.path.insert(0, str(REPO))

from experiments.phase5_downstream_utility.probing.run_probing import (  # noqa: E402
    _load_model_for_run,
)

CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
BUF_PATH = REPO / "data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy"
ML_LAYERS = (11, 12, 13, 14, 15)
OUT_JSON = REPO / "experiments/phase5_downstream_utility/results/alive_fraction.json"

N_BATCHES = 40
BATCH_SIZE = 128


def make_window_gen(buf: np.ndarray, T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    N, L, d = buf.shape
    while True:
        seq_idx = rng.integers(0, N, size=BATCH_SIZE)
        start = rng.integers(0, L - T + 1, size=BATCH_SIZE)
        wins = np.empty((BATCH_SIZE, T, d), dtype=np.float32)
        for i in range(BATCH_SIZE):
            wins[i] = buf[seq_idx[i], start[i]:start[i] + T]
        yield torch.from_numpy(wins)


def make_multilayer_gen(seed: int = 0):
    """Yield (BATCH_SIZE, n_layers, d) at random tokens across L11..L15."""
    layers = []
    for L in ML_LAYERS:
        layers.append(np.load(REPO / f"data/cached_activations/gemma-2-2b-it/fineweb/resid_L{L}.npy",
                               mmap_mode="r"))
    rng = np.random.default_rng(seed)
    N, L_seq, d = layers[0].shape
    while True:
        seq_idx = rng.integers(0, N, size=BATCH_SIZE)
        tok_idx = rng.integers(0, L_seq, size=BATCH_SIZE)
        out = np.empty((BATCH_SIZE, len(ML_LAYERS), d), dtype=np.float32)
        for li, lyr in enumerate(layers):
            for i in range(BATCH_SIZE):
                out[i, li] = lyr[seq_idx[i], tok_idx[i]]
        yield torch.from_numpy(out)


def alive_for(arch: str, device: str = "cuda") -> dict:
    rid = f"{arch}__seed42"
    ckpt_path = CKPT_DIR / f"{rid}.pt"
    if not ckpt_path.exists():
        return {"error": "ckpt missing"}
    model, arch_name, meta = _load_model_for_run(rid, ckpt_path, torch.device(device))
    model.eval()
    T = meta.get("T", 5)
    d_sae = 18432

    if arch.startswith("mlc_"):
        gen = make_multilayer_gen()
    else:
        buf = np.load(BUF_PATH, mmap_mode="r")
        gen = make_window_gen(buf, T)

    ever_fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    with torch.no_grad():
        for i, X in enumerate(gen):
            if i >= N_BATCHES:
                break
            X = X.to(device)
            z = model.encode(X)
            # z shape: (B, d_sae). Count positive activations.
            fired = (z > 0).any(dim=0)
            ever_fired |= fired

    n_alive = int(ever_fired.sum().item())
    n_total = d_sae
    return {
        "T": int(T),
        "n_total": n_total,
        "n_alive": n_alive,
        "alive_fraction": n_alive / n_total,
        "dead_fraction": 1 - n_alive / n_total,
        "N_batches": N_BATCHES,
        "BATCH_SIZE": BATCH_SIZE,
    }


def main():
    archs = [
        "txcdr_t2", "txcdr_t3", "txcdr_t5", "txcdr_t6", "txcdr_t7",
        "txcdr_t8", "txcdr_t10", "txcdr_t15", "txcdr_t20",
        "txcdr_t2_batchtopk", "txcdr_t3_batchtopk", "txcdr_t5_batchtopk",
        "txcdr_t6_batchtopk", "txcdr_t7_batchtopk", "txcdr_t8_batchtopk",
        "txcdr_t10_batchtopk", "txcdr_t15_batchtopk", "txcdr_t20_batchtopk",
        "agentic_txc_02", "agentic_txc_02_t2", "agentic_txc_02_t3",
        "agentic_txc_02_t6", "agentic_txc_02_t7", "agentic_txc_02_t8",
        "agentic_txc_02_batchtopk",
        "phase57_partB_h7_bare_multiscale",
        "phase57_partB_h8_bare_multidistance",
        # H8 T-sweep
        "phase57_partB_h8_bare_multidistance_t10",
        "phase57_partB_h8_bare_multidistance_t15",
        "phase57_partB_h8_bare_multidistance_t20",
        "phase57_partB_h8_bare_multidistance_t30",
        # H8a shift ablation
        "phase57_partB_h8a_shifts1",
        "phase57_partB_h8a_shifts123",
        "phase57_partB_h8a_shifts1234",
        "phase57_partB_h8a_shifts124",
        "phase57_partB_h8a_shifts2",
        "phase57_partB_h8a_shifts4",
        "phase57_partB_h8a_shifts123_uniform",
        # H3 log-matryoshka T-sweep
        "log_matryoshka_t5", "log_matryoshka_t10", "log_matryoshka_t15",
        "log_matryoshka_t20", "log_matryoshka_t30",
        # P0c MLC + anti-dead fairness counterparts
        "mlc_bare_antidead",
        "mlc_bare_matryoshka_contrastive_antidead",
        "mlc_bare_multiscale_antidead",
        "feature_nested_matryoshka_t5",
        "matryoshka_t5",
        "matryoshka_txcdr_contrastive_t5_alpha100",
    ]
    results = {}
    for arch in archs:
        try:
            r = alive_for(arch)
            results[arch] = r
            if "error" not in r:
                print(f"{arch:<55s} T={r['T']:2d}  alive={r['n_alive']:5d}/{r['n_total']}  "
                      f"({100*r['alive_fraction']:5.1f}%)  dead={100*r['dead_fraction']:5.1f}%")
            else:
                print(f"{arch:<55s} {r['error']}")
        except Exception as e:
            print(f"{arch:<55s} ERROR: {e}")
            results[arch] = {"error": str(e)}
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
