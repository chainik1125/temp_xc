"""Re-run TFA and TFA-pos L1 sweeps tracking both novel_l0, pred_l0, total_l0, and AUC.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_tfa_l1_with_total_l0.py
"""

import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig, create_tfa, train_tfa,
)
from src.v2_temporal_schemeC.feature_recovery import (
    feature_recovery_score, tfa_decoder_directions,
)

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4
DICT_WIDTH = 40
L1_COEFFS = np.logspace(-0.8, 1.8, 12).tolist()
TFA_STEPS = 30_000
TFA_BATCH = 64
TFA_LR = 1e-3
EVAL_N_SEQ = 2000
SEED = 42


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(HIDDEN_DIM) / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()


def make_seq_gen(model, pi_t, rho_t, device, sf):
    def gen(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        return model(acts) * sf
    return gen


def eval_tfa_full(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    total_se = total_signal = total_novel_l0 = total_pred_l0 = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, 256):
            x = eval_hidden[s:min(s+256, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf, rf = x.reshape(-1, D), recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            total_novel_l0 += (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
            total_pred_l0 += (inter["pred_codes"].abs() > 1e-8).float().sum(dim=-1).sum().item()
            n_tokens += B * T
    novel_l0 = total_novel_l0 / n_tokens
    pred_l0 = total_pred_l0 / n_tokens
    return {
        "nmse": total_se / total_signal,
        "novel_l0": novel_l0,
        "pred_l0": pred_l0,
        "total_l0": novel_l0 + pred_l0,
    }


def main():
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    pi_t, rho_t = torch.tensor(PI), torch.tensor(RHO)
    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    true_features = model.feature_directions
    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    gen_seq = make_seq_gen(model, pi_t, rho_t, device, sf)

    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts_eval) * sf

    results = {"tfa": [], "tfa_pos": []}

    for use_pos, label in [(False, "tfa"), (True, "tfa_pos")]:
        print(f"\n{'='*60}", flush=True)
        print(f"ReLU+L1 sweep: {label.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        for l1c in L1_COEFFS:
            set_seed(SEED)
            t0 = time.time()
            tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=None, n_heads=4,
                             n_attn_layers=1, bottleneck_factor=1,
                             use_pos_encoding=use_pos, device=device)
            tfa.sae_diff_type = "relu"
            cfg = TFATrainingConfig(total_steps=TFA_STEPS, batch_size=TFA_BATCH,
                                     lr=TFA_LR, l1_coeff=l1c, log_every=TFA_STEPS)
            tfa, _ = train_tfa(tfa, gen_seq, cfg, device)
            r = eval_tfa_full(tfa, eval_hidden, device)
            dd = tfa_decoder_directions(tfa).to(device)
            tf = true_features.T.to(device)
            auc_r = feature_recovery_score(dd, tf)
            r["auc"] = auc_r["auc"]
            r["l1"] = l1c
            results[label].append(r)
            print(f"  l1={l1c:.4f}: NMSE={r['nmse']:.6f} nL0={r['novel_l0']:.2f} "
                  f"tL0={r['total_l0']:.2f} AUC={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            del tfa; torch.cuda.empty_cache()

    out_dir = os.path.join(os.path.dirname(__file__), "results", "tfa_l1_total_l0")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nSaved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
