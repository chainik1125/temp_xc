"""Run a single TXCDR TopK experiment at low-pi regime. For parallel execution.

Usage:
  python src/v2_temporal_schemeC/run_txcdr_single_k_lowpi.py T k output_path
"""

import json
import math
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

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

NUM_FEATURES = 100
HIDDEN_DIM = 100
SEQ_LEN = 64
PI = [0.05] * NUM_FEATURES
RHO = [0.0] * 20 + [0.3] * 20 + [0.5] * 20 + [0.7] * 20 + [0.9] * 20
DICT_WIDTH = 100
STEPS = 80_000
BATCH_SIZE = 2048
LR = 3e-4
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
    T = int(sys.argv[1])
    k = int(sys.argv[2])
    out_path = sys.argv[3]
    device = DEFAULT_DEVICE
    print(f"TXCDR low-pi T={T} k={k} on {device}", flush=True)

    pi_t, rho_t = torch.tensor(PI), torch.tensor(RHO)
    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    true_features = model.feature_directions

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    gen_windows = make_window_gen(model, pi_t, rho_t, device, sf, T)

    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts_eval) * sf

    cfg = CrosscoderTrainingConfig(total_steps=STEPS, batch_size=BATCH_SIZE, lr=LR, log_every=STEPS)
    set_seed(SEED)
    t0 = time.time()
    txcdr = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, T, k).to(device)
    txcdr, _ = train_crosscoder(txcdr, gen_windows, cfg, device)
    r = eval_crosscoder(txcdr, eval_hidden, device, T)
    r["auc"] = txcdr_avg_auc(txcdr, true_features, device, T)
    r["k"] = k
    elapsed = time.time() - t0
    print(f"  k={k}: NMSE={r['nmse']:.6f} L0={r['l0']:.2f} AUC={r['auc']:.4f} ({elapsed:.0f}s)", flush=True)

    with open(out_path, "w") as f:
        json.dump(r, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
