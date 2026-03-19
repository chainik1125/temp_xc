"""TFA-pos experiments: TFA with sinusoidal positional encoding in attention.

Runs TFA-pos and TFA-pos-shuffled across Experiments 1 and 2, loading
SAE/TFA/TFA-shuf/TXCDR baselines from existing results.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_tfa_pos_experiments.py
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
from src.utils.plot import save_figure
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig, create_tfa, train_tfa,
)
from src.v2_temporal_schemeC.feature_recovery import (
    feature_recovery_score, tfa_decoder_directions,
)

# ── Configuration (same as main experiments) ──────────────────────────

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4
DICT_WIDTH = 40

TOPK_K_VALUES = [1, 3, 5, 8, 10, 15, 20]
L1_COEFFS_TFA = np.logspace(-0.8, 1.8, 12).tolist()

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 2000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "tfa_pos")


# ── Helpers ───────────────────────────────────────────────────────────

def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(HIDDEN_DIM) / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()


def make_seq_gen(model, pi_t, rho_t, device, sf, shuffle=False):
    def gen(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                acts[i] = acts[i, torch.randperm(SEQ_LEN, device=device)]
        return model(acts) * sf
    return gen


def eval_tfa(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    total_se = total_signal = total_novel_l0 = total_pred_l0 = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, 256):
            x = eval_hidden[s:min(s + 256, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf, rf = x.reshape(-1, D), recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            total_novel_l0 += (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
            total_pred_l0 += (inter["pred_codes"] > 0).float().sum(dim=-1).sum().item()
            n_tokens += B * T
    return {
        "nmse": total_se / total_signal,
        "novel_l0": total_novel_l0 / n_tokens,
        "pred_l0": total_pred_l0 / n_tokens,
    }


def compute_auc(tfa, true_features, device):
    dd = tfa_decoder_directions(tfa).to(device)
    tf = true_features.T.to(device)
    return feature_recovery_score(dd, tf)


def load_baselines():
    """Load existing results for SAE, TFA, TFA-shuf, TXCDR."""
    base = os.path.join(os.path.dirname(__file__), "results", "auc_and_crosscoder", "results.json")
    if not os.path.exists(base):
        print(f"WARNING: baseline results not found at {base}", flush=True)
        return None
    with open(base) as f:
        return json.load(f)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    # Build toy model and data generators
    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    true_features = model.feature_directions

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    gen_seq = make_seq_gen(model, pi_t, rho_t, device, sf, shuffle=False)
    gen_seq_shuf = make_seq_gen(model, pi_t, rho_t, device, sf, shuffle=True)

    # Eval data
    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts_eval) * sf

    # Load baselines
    baselines = load_baselines()

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: TopK sweep — TFA-pos and TFA-pos-shuffled
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 1: TopK sweep (TFA-pos + TFA-pos-shuffled)", flush=True)
    print(f"{'='*70}", flush=True)

    tfa_cfg = TFATrainingConfig(
        total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
        lr=TFA_LR, log_every=TFA_TOTAL_STEPS,
    )

    exp1_pos = []
    exp1_pos_shuf = []

    for k in TOPK_K_VALUES:
        print(f"\n  k={k}:", flush=True)

        # TFA-pos (temporal data)
        set_seed(SEED)
        t0 = time.time()
        tfa_pos = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
                             n_attn_layers=1, bottleneck_factor=1,
                             use_pos_encoding=True, device=device)
        tfa_pos, _ = train_tfa(tfa_pos, gen_seq, tfa_cfg, device)
        r = eval_tfa(tfa_pos, eval_hidden, device)
        auc = compute_auc(tfa_pos, true_features, device)
        r["auc"] = auc["auc"]
        r["r90"] = auc["frac_recovered_90"]
        r["k"] = k
        exp1_pos.append(r)
        print(f"    TFA-pos:      NMSE={r['nmse']:.6f} nL0={r['novel_l0']:.1f} AUC={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del tfa_pos; torch.cuda.empty_cache()

        # TFA-pos-shuffled (shuffled data, eval on temporal)
        set_seed(SEED)
        t0 = time.time()
        tfa_pos_shuf = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
                                   n_attn_layers=1, bottleneck_factor=1,
                                   use_pos_encoding=True, device=device)
        tfa_pos_shuf, _ = train_tfa(tfa_pos_shuf, gen_seq_shuf, tfa_cfg, device)
        r = eval_tfa(tfa_pos_shuf, eval_hidden, device)
        auc = compute_auc(tfa_pos_shuf, true_features, device)
        r["auc"] = auc["auc"]
        r["r90"] = auc["frac_recovered_90"]
        r["k"] = k
        exp1_pos_shuf.append(r)
        print(f"    TFA-pos-shuf: NMSE={r['nmse']:.6f} nL0={r['novel_l0']:.1f} AUC={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del tfa_pos_shuf; torch.cuda.empty_cache()

    # Print summary with baselines
    print(f"\n{'='*70}", flush=True)
    print("Experiment 1 Summary (NMSE):", flush=True)
    print(f"{'k':>3} | {'SAE':>9} | {'TFA':>9} | {'TFA-shuf':>9} | {'TFA-pos':>9} | {'pos-shuf':>9}", flush=True)
    print("-" * 65, flush=True)
    for i, k in enumerate(TOPK_K_VALUES):
        sae_nmse = baselines["exp1"]["sae"][i]["nmse"] if baselines else "---"
        tfa_nmse = baselines["exp1"]["tfa"][i]["nmse"] if baselines else "---"
        shuf_nmse = baselines["exp1"]["tfa_shuf"][i]["nmse"] if baselines else "---"
        pos_nmse = exp1_pos[i]["nmse"]
        pos_shuf_nmse = exp1_pos_shuf[i]["nmse"]
        print(f"{k:>3} | {sae_nmse:>9.4f} | {tfa_nmse:>9.4f} | {shuf_nmse:>9.4f} | {pos_nmse:>9.4f} | {pos_shuf_nmse:>9.4f}", flush=True)

    print(f"\nExperiment 1 Summary (AUC):", flush=True)
    print(f"{'k':>3} | {'SAE':>9} | {'TFA':>9} | {'TFA-shuf':>9} | {'TFA-pos':>9} | {'pos-shuf':>9}", flush=True)
    print("-" * 65, flush=True)
    for i, k in enumerate(TOPK_K_VALUES):
        sae_auc = baselines["exp1"]["sae"][i]["auc"] if baselines else "---"
        tfa_auc = baselines["exp1"]["tfa"][i]["auc"] if baselines else "---"
        shuf_auc = baselines["exp1"]["tfa_shuf"][i]["auc"] if baselines else "---"
        pos_auc = exp1_pos[i]["auc"]
        pos_shuf_auc = exp1_pos_shuf[i]["auc"]
        print(f"{k:>3} | {sae_auc:>9.4f} | {tfa_auc:>9.4f} | {shuf_auc:>9.4f} | {pos_auc:>9.4f} | {pos_shuf_auc:>9.4f}", flush=True)

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: ReLU+L1 Pareto — TFA-pos only
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print("EXPERIMENT 2: ReLU+L1 Pareto (TFA-pos)", flush=True)
    print(f"{'='*70}", flush=True)

    exp2_pos = []
    for l1c in L1_COEFFS_TFA:
        set_seed(SEED)
        t0 = time.time()
        tfa_pos = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=None, n_heads=4,
                             n_attn_layers=1, bottleneck_factor=1,
                             use_pos_encoding=True, device=device)
        tfa_pos.sae_diff_type = "relu"
        l1_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR, l1_coeff=l1c, log_every=TFA_TOTAL_STEPS,
        )
        tfa_pos, _ = train_tfa(tfa_pos, gen_seq, l1_cfg, device)
        r = eval_tfa(tfa_pos, eval_hidden, device)
        auc = compute_auc(tfa_pos, true_features, device)
        r["auc"] = auc["auc"]
        r["l1"] = l1c
        exp2_pos.append(r)
        print(f"  l1={l1c:.4f}: NMSE={r['nmse']:.6f} nL0={r['novel_l0']:.2f} AUC={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del tfa_pos; torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────

    def ser(d):
        if isinstance(d, np.floating):
            return float(d)
        if isinstance(d, np.integer):
            return int(d)
        if isinstance(d, np.ndarray):
            return d.tolist()
        return d

    save_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "tfa_steps": TFA_TOTAL_STEPS,
            "seed": SEED,
        },
        "exp1_tfa_pos": exp1_pos,
        "exp1_tfa_pos_shuf": exp1_pos_shuf,
        "exp2_tfa_pos": exp2_pos,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=ser)
    print(f"\nResults saved to {RESULTS_DIR}", flush=True)

    # ── Compute temporal decomposition ────────────────────────────

    print(f"\n{'='*70}", flush=True)
    print("Temporal decomposition (TFA-pos):", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'k':>3} | {'gap':>9} | {'arch':>6} | {'temp':>6}", flush=True)
    print("-" * 35, flush=True)
    for i, k in enumerate(TOPK_K_VALUES):
        if not baselines:
            break
        sae = baselines["exp1"]["sae"][i]["nmse"]
        pos = exp1_pos[i]["nmse"]
        pos_shuf = exp1_pos_shuf[i]["nmse"]
        gap = sae - pos
        if gap > 0.001:
            arch_frac = (sae - pos_shuf) / gap
            temp_frac = 1 - arch_frac
            print(f"{k:>3} | {gap:>9.4f} | {arch_frac:>5.0%} | {temp_frac:>5.0%}", flush=True)
        else:
            print(f"{k:>3} | {gap:>9.4f} |   ---  |   ---", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
