"""TXCDR sweep over window size T. Run with T=<value> as argument.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_txcdr_sweep_T.py 5
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_txcdr_sweep_T.py 64
"""

import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.temporal_crosscoder import (
    TemporalCrosscoder, CrosscoderTrainingConfig, train_crosscoder,
)
from src.v2_temporal_schemeC.feature_recovery import feature_recovery_score

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4
DICT_WIDTH = 40

TOPK_K_VALUES = [1, 3, 5, 8, 10, 15, 20]
L1_COEFFS = np.logspace(-1.5, 1.5, 12).tolist()

TXCDR_STEPS = 80_000
TXCDR_BATCH_SIZE = 2048
TXCDR_LR = 3e-4

EVAL_N_SEQ = 2000
SEED = 42


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(HIDDEN_DIM) / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()


def make_window_gen(model, pi_t, rho_t, device, sf, T):
    def gen(batch_size):
        n_seq = max(1, batch_size // (SEQ_LEN - T + 1)) + 1
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts) * sf
        windows = []
        for t in range(SEQ_LEN - T + 1):
            windows.append(hidden[:, t:t+T, :])
        all_w = torch.cat(windows, dim=0)
        idx = torch.randperm(all_w.shape[0], device=device)[:batch_size]
        return all_w[idx]
    return gen


def eval_crosscoder(txcdr, eval_hidden, device, T):
    txcdr.eval()
    total_se = total_signal = total_l0 = 0.0
    n_windows = 0
    with torch.no_grad():
        for s in range(0, eval_hidden.shape[0], 256):
            seqs = eval_hidden[s:min(s+256, eval_hidden.shape[0])].to(device)
            for t in range(SEQ_LEN - T + 1):
                w = seqs[:, t:t+T, :]
                loss, x_hat, z = txcdr(w)
                total_se += (x_hat - w).pow(2).sum().item()
                total_signal += w.pow(2).sum().item()
                total_l0 += (z > 0).float().sum(dim=-1).sum().item()
                n_windows += w.shape[0]
    return {"nmse": total_se / total_signal, "l0": total_l0 / n_windows}


def txcdr_avg_auc(txcdr, true_features, device, T):
    aucs = []
    tf = true_features.T.to(device)
    for pos in range(T):
        dd = txcdr.decoder_directions(pos).to(device)
        a = feature_recovery_score(dd, tf)
        aucs.append(a["auc"])
    return np.mean(aucs)


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    results_dir = os.path.join(
        os.path.dirname(__file__), "results", f"txcdr_T{T}"
    )
    os.makedirs(results_dir, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}, T={T}", flush=True)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    true_features = model.feature_directions

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    gen_windows = make_window_gen(model, pi_t, rho_t, device, sf, T)

    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts_eval) * sf

    # ── TopK sweep ───────────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print(f"TopK sweep (TXCDR T={T}, {TXCDR_STEPS//1000}K steps)", flush=True)
    print(f"{'='*60}", flush=True)

    topk_cfg = CrosscoderTrainingConfig(
        total_steps=TXCDR_STEPS, batch_size=TXCDR_BATCH_SIZE,
        lr=TXCDR_LR, log_every=TXCDR_STEPS,
    )

    topk_results = []
    for k in TOPK_K_VALUES:
        if k > DICT_WIDTH:
            print(f"  k={k}: SKIPPED (k > dict_width={DICT_WIDTH})", flush=True)
            continue
        set_seed(SEED); t0 = time.time()
        txcdr = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, T, k).to(device)
        txcdr, _ = train_crosscoder(txcdr, gen_windows, topk_cfg, device)
        r = eval_crosscoder(txcdr, eval_hidden, device, T)
        r["auc"] = txcdr_avg_auc(txcdr, true_features, device, T)
        r["k"] = k
        topk_results.append(r)
        print(f"  k={k}: NMSE={r['nmse']:.6f} L0={r['l0']:.2f} AUC={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del txcdr; torch.cuda.empty_cache()

    # ── ReLU+L1 sweep ────────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print(f"ReLU+L1 sweep (TXCDR T={T})", flush=True)
    print(f"{'='*60}", flush=True)

    l1_results = []
    for l1c in L1_COEFFS:
        set_seed(SEED); t0 = time.time()
        txcdr = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, T, k=None).to(device)
        l1_cfg = CrosscoderTrainingConfig(
            total_steps=TXCDR_STEPS, batch_size=TXCDR_BATCH_SIZE,
            lr=TXCDR_LR, l1_coeff=l1c, log_every=TXCDR_STEPS,
        )
        txcdr, _ = train_crosscoder(txcdr, gen_windows, l1_cfg, device)
        r = eval_crosscoder(txcdr, eval_hidden, device, T)
        r["auc"] = txcdr_avg_auc(txcdr, true_features, device, T)
        r["l1"] = l1c
        l1_results.append(r)
        print(f"  l1={l1c:.4f}: NMSE={r['nmse']:.6f} L0={r['l0']:.2f} AUC={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del txcdr; torch.cuda.empty_cache()

    # ── Save ─────────────────────────────────────────────────────────

    save_data = {
        "config": {"T": T, "steps": TXCDR_STEPS, "seed": SEED,
                    "k_values": TOPK_K_VALUES, "dict_width": DICT_WIDTH},
        "topk": topk_results,
        "l1": l1_results,
    }
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\nResults saved to {results_dir}", flush=True)


if __name__ == "__main__":
    main()
