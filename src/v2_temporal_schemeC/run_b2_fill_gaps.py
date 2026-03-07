"""Fill gaps in the ReLU+L1 Pareto frontier.

The initial sweep produced sparse coverage in the binding regime.
This script runs additional L1 values targeted at the gaps:
- SAE: need more points in L0 = 2-6 and 8-12 (L1 between 1.0 and 5.0)
- TFA: need more points in nL0 = 3-12 (L1 between 10 and 30)
"""

import json
import math
import os
import time

import numpy as np
import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig,
    create_tfa,
    train_tfa,
)
from src.v2_temporal_schemeC.relu_sae import (
    ReLUSAE,
    ReLUSAETrainingConfig,
    train_relu_sae,
)

# ── Same config as run_b1_b2_pareto.py ───────────────────────────────

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4
DICT_WIDTH = 40
TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
TFA_SEQ_BATCH = 64
TFA_N_HEADS = 4
TFA_N_ATTN_LAYERS = 1
TFA_BOTTLENECK = 1
EVAL_N_SEQ = 2000
SEED = 42
LOG_EVERY = 10000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "b1_b2_pareto")

# ── Gap-filling L1 values ────────────────────────────────────────────

# SAE: from pilots, L0=7 at l1=1.87, L0=1.3 at l1=3.38, L0=12.3 at l1=1.03
# Need denser sampling between l1=0.8 and l1=5.0
SAE_EXTRA_L1 = [0.7, 0.85, 1.2, 1.4, 1.6, 2.2, 2.5, 2.8, 3.5, 4.0, 5.0, 7.0]

# TFA: from pilots, nL0=12.4 at l1=10.8, nL0=7.3 at l1=16.6, nL0=3.6 at l1=25.5
# Need denser sampling between l1=8 and l1=35
TFA_EXTRA_L1 = [5.0, 8.0, 9.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 28.0, 35.0, 45.0]


# ── Helpers (same as main script) ────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device, n_samples=10000):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            n_samples, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
        mean_norm = norms.mean().item()
    return math.sqrt(HIDDEN_DIM) / mean_norm if mean_norm > 0 else 1.0


def make_generators(model, pi_t, rho_t, device, sf):
    def gen_flat(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]

    def gen_seq(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        return model(acts) * sf

    return gen_flat, gen_seq


def eval_sae(sae, eval_hidden, device):
    sae.eval()
    flat = eval_hidden.reshape(-1, HIDDEN_DIM)
    n = flat.shape[0]
    total_se = total_sig = total_l0 = 0.0
    bs = 4096
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            x = flat[s:e].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_sig += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()
    return {"nmse": total_se / total_sig, "mse": total_se / n, "l0": total_l0 / n}


def eval_tfa(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    bs = 256
    total_se = total_sig = total_nl0 = total_tl0 = 0.0
    total_pe = total_ne = n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, bs):
            e = min(s + bs, n_seq)
            x = eval_hidden[s:e].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf = x.reshape(-1, D)
            rf = recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_sig += xf.pow(2).sum().item()
            nc = inter["novel_codes"]
            pc = inter["pred_codes"]
            tc = nc + pc
            total_nl0 += (nc > 0).float().sum(dim=-1).sum().item()
            total_tl0 += (tc.abs() > 1e-8).float().sum(dim=-1).sum().item()
            total_pe += inter["pred_recons"].norm(dim=-1).pow(2).sum().item()
            total_ne += inter["novel_recons"].norm(dim=-1).pow(2).sum().item()
            n_tokens += B * T
    te = total_pe + total_ne + 1e-12
    return {
        "nmse": total_se / total_sig, "mse": total_se / n_tokens,
        "novel_l0": total_nl0 / n_tokens, "total_l0": total_tl0 / n_tokens,
        "pred_energy_frac": total_pe / te,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    device = DEFAULT_DEVICE
    print(f"Device: {device}")
    print("Gap-filling for ReLU+L1 Pareto frontier")

    set_seed(SEED)
    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    gen_flat, gen_seq = make_generators(model, pi_t, rho_t, device, sf)

    set_seed(SEED + 100)
    acts, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts) * sf

    # Load existing results
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path) as f:
        data = json.load(f)

    existing_sae_l1 = {r["l1_coeff"] for r in data["sae_results"]}
    existing_tfa_l1 = {r["l1_coeff"] for r in data["tfa_results"]}

    # ── SAE gap-fill ──
    print(f"\n{'='*60}")
    print(f"SAE GAP-FILL ({len(SAE_EXTRA_L1)} L1 values)")
    print(f"{'='*60}")

    new_sae = []
    for i, l1c in enumerate(SAE_EXTRA_L1):
        if l1c in existing_sae_l1:
            print(f"  Skipping l1={l1c:.2e} (already exists)")
            continue
        set_seed(SEED)
        print(f"\n--- SAE [{i+1}/{len(SAE_EXTRA_L1)}] l1={l1c:.2e} ---")
        t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH).to(device)
        cfg = ReLUSAETrainingConfig(
            total_steps=TOTAL_STEPS, batch_size=SAE_BATCH_SIZE,
            l1_coeff=l1c, log_every=LOG_EVERY,
        )
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        result = eval_sae(sae, eval_hidden, device)
        dt = time.time() - t0
        print(f"  EVAL: NMSE={result['nmse']:.6f}, L0={result['l0']:.2f} ({dt:.1f}s)")
        new_sae.append({"l1_coeff": l1c, **result})
        del sae
        torch.cuda.empty_cache()

    # ── TFA gap-fill ──
    print(f"\n{'='*60}")
    print(f"TFA GAP-FILL ({len(TFA_EXTRA_L1)} L1 values)")
    print(f"{'='*60}")

    new_tfa = []
    for i, l1c in enumerate(TFA_EXTRA_L1):
        if l1c in existing_tfa_l1:
            print(f"  Skipping l1={l1c:.2e} (already exists)")
            continue
        set_seed(SEED)
        print(f"\n--- TFA [{i+1}/{len(TFA_EXTRA_L1)}] l1={l1c:.2e} ---")
        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=999,
            n_heads=TFA_N_HEADS, n_attn_layers=TFA_N_ATTN_LAYERS,
            bottleneck_factor=TFA_BOTTLENECK, device=device,
        )
        tfa.sae_diff_type = 'relu'
        tfa_cfg = TFATrainingConfig(
            total_steps=TOTAL_STEPS, batch_size=TFA_SEQ_BATCH,
            l1_coeff=l1c, log_every=LOG_EVERY,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        result = eval_tfa(tfa, eval_hidden, device)
        dt = time.time() - t0
        print(f"  EVAL: NMSE={result['nmse']:.6f}, nL0={result['novel_l0']:.2f}, "
              f"tL0={result['total_l0']:.2f} ({dt:.1f}s)")
        new_tfa.append({"l1_coeff": l1c, **result})
        del tfa
        torch.cuda.empty_cache()

    # ── Merge and save ──
    data["sae_results"].extend(new_sae)
    data["tfa_results"].extend(new_tfa)
    data["sae_l1_values"] = sorted(set(data.get("sae_l1_values", []) +
                                        SAE_EXTRA_L1))
    data["tfa_l1_values"] = sorted(set(data.get("tfa_l1_values", []) +
                                        TFA_EXTRA_L1))

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nMerged results saved to {json_path}")
    print(f"  SAE: {len(data['sae_results'])} total points")
    print(f"  TFA: {len(data['tfa_results'])} total points")

    # Summary
    print("\n--- New SAE points ---")
    for r in sorted(new_sae, key=lambda x: x["l0"]):
        print(f"  l1={r['l1_coeff']:.2e}  L0={r['l0']:.2f}  NMSE={r['nmse']:.6f}")
    print("\n--- New TFA points ---")
    for r in sorted(new_tfa, key=lambda x: x["novel_l0"]):
        print(f"  l1={r['l1_coeff']:.2e}  nL0={r['novel_l0']:.2f}  NMSE={r['nmse']:.6f}")


if __name__ == "__main__":
    main()
