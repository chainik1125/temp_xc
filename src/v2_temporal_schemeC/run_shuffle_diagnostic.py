"""Diagnostic: Does TFA's advantage come from temporal structure or extra capacity?

Test: Train TFA on shuffled sequences (destroying temporal correlations while
preserving marginal distribution) and compare to TFA on unshuffled sequences
and to a standard SAE. If TFA-shuffled matches TFA-unshuffled, the advantage
is purely capacity. If TFA-shuffled matches SAE, the advantage is temporal.

Also tests a wider SAE (matched parameter count) as a capacity control.
"""

import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.tfa import TemporalSAE
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

# ── Configuration ────────────────────────────────────────────────────

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4

DICT_WIDTH = 40
# Wider SAE to match TFA parameter count (~8200 params)
# SAE params = d_in * d_sae + d_sae + d_sae * d_in + d_in = 2*d_in*d_sae + d_in + d_sae
# For d_in=40: 80*d_sae + 40 + d_sae = 81*d_sae + 40 ≈ 8200 => d_sae ≈ 100
WIDE_DICT_WIDTH = 100

K_VALUES = [3, 5, 8, 10, 15]

SAE_TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 2000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "shuffle_diagnostic")


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
    return math.sqrt(HIDDEN_DIM) / norms.mean().item()


def make_generators(model, pi_t, rho_t, device, sf, shuffle=False):
    """Create flat (SAE) and seq (TFA) data generators.

    If shuffle=True, randomly permute sequence positions before returning,
    destroying temporal correlations while preserving marginals.
    """
    def gen_flat(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                perm = torch.randperm(SEQ_LEN, device=device)
                acts[i] = acts[i, perm]
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]

    def gen_seq(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                perm = torch.randperm(SEQ_LEN, device=device)
                acts[i] = acts[i, perm]
        return model(acts) * sf

    return gen_flat, gen_seq


def eval_sae(sae, eval_hidden, device):
    sae.eval()
    flat = eval_hidden.reshape(-1, HIDDEN_DIM)
    n = flat.shape[0]
    total_se = total_signal = total_l0 = 0.0
    bs = 4096
    with torch.no_grad():
        for s in range(0, n, bs):
            x = flat[s:min(s+bs, n)].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_signal += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()
    return {"nmse": total_se / total_signal, "l0": total_l0 / n}


def eval_tfa(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    bs = 256
    total_se = total_signal = total_novel_l0 = total_total_l0 = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, bs):
            x = eval_hidden[s:min(s+bs, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf = x.reshape(-1, D)
            rf = recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            nc = inter["novel_codes"]
            pc = inter["pred_codes"]
            total_novel_l0 += (nc > 0).float().sum(dim=-1).sum().item()
            total_total_l0 += ((nc + pc).abs() > 1e-8).float().sum(dim=-1).sum().item()
            n_tokens += B * T
    return {
        "nmse": total_se / total_signal,
        "novel_l0": total_novel_l0 / n_tokens,
        "total_l0": total_total_l0 / n_tokens,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")

    set_seed(SEED)
    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling factor: {sf:.4f}")

    # Generators: normal and shuffled
    gen_flat, gen_seq = make_generators(model, pi_t, rho_t, device, sf, shuffle=False)
    gen_flat_shuf, gen_seq_shuf = make_generators(model, pi_t, rho_t, device, sf, shuffle=True)

    # Eval data (unshuffled — we always evaluate on proper temporal data)
    set_seed(SEED + 100)
    acts, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts) * sf

    # Also create shuffled eval data
    set_seed(SEED + 100)
    acts_shuf, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    for i in range(EVAL_N_SEQ):
        perm = torch.randperm(SEQ_LEN, device=device)
        acts_shuf[i] = acts_shuf[i, perm]
    eval_hidden_shuf = model(acts_shuf) * sf

    print(f"Eval: {eval_hidden.shape[0]}x{eval_hidden.shape[1]} tokens")

    # Parameter counts
    sae_test = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=10)
    wide_sae_test = ReLUSAE(HIDDEN_DIM, WIDE_DICT_WIDTH, k=10)
    tfa_test = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=10,
                           n_heads=4, n_attn_layers=1, bottleneck_factor=1, device="cpu")
    print(f"\nParameter counts:")
    print(f"  SAE (width={DICT_WIDTH}): {sum(p.numel() for p in sae_test.parameters())}")
    print(f"  Wide SAE (width={WIDE_DICT_WIDTH}): {sum(p.numel() for p in wide_sae_test.parameters())}")
    print(f"  TFA (width={DICT_WIDTH}): {sum(p.numel() for p in tfa_test.parameters())}")
    del sae_test, wide_sae_test, tfa_test

    results = {"sae": [], "wide_sae": [], "tfa": [], "tfa_shuffled": []}

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"k = {k}")
        print(f"{'='*60}")

        # 1. Standard SAE
        set_seed(SEED)
        t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(total_steps=SAE_TOTAL_STEPS, batch_size=SAE_BATCH_SIZE,
                                      lr=SAE_LR, l1_coeff=0.0, log_every=30000)
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r = eval_sae(sae, eval_hidden, device)
        print(f"  SAE:          NMSE={r['nmse']:.6f}, L0={r['l0']:.2f} ({time.time()-t0:.1f}s)")
        results["sae"].append({"k": k, **r})
        del sae; torch.cuda.empty_cache()

        # 2. Wide SAE (capacity-matched)
        set_seed(SEED)
        t0 = time.time()
        wide_sae = ReLUSAE(HIDDEN_DIM, WIDE_DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(total_steps=SAE_TOTAL_STEPS, batch_size=SAE_BATCH_SIZE,
                                      lr=SAE_LR, l1_coeff=0.0, log_every=30000)
        wide_sae, _ = train_relu_sae(wide_sae, gen_flat, cfg, device)
        r = eval_sae(wide_sae, eval_hidden, device)
        print(f"  Wide SAE:     NMSE={r['nmse']:.6f}, L0={r['l0']:.2f} ({time.time()-t0:.1f}s)")
        results["wide_sae"].append({"k": k, **r})
        del wide_sae; torch.cuda.empty_cache()

        # 3. TFA (normal — unshuffled training data)
        set_seed(SEED)
        t0 = time.time()
        tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
                          n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa_cfg = TFATrainingConfig(total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
                                     lr=TFA_LR, log_every=30000)
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        r = eval_tfa(tfa, eval_hidden, device)
        print(f"  TFA:          NMSE={r['nmse']:.6f}, novel_L0={r['novel_l0']:.2f}, "
              f"total_L0={r['total_l0']:.2f} ({time.time()-t0:.1f}s)")
        results["tfa"].append({"k": k, **r})
        del tfa; torch.cuda.empty_cache()

        # 4. TFA on shuffled data (trained on shuffled, evaluated on UNSHUFFLED)
        set_seed(SEED)
        t0 = time.time()
        tfa_shuf = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
                               n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa_cfg = TFATrainingConfig(total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
                                     lr=TFA_LR, log_every=30000)
        tfa_shuf, _ = train_tfa(tfa_shuf, gen_seq_shuf, tfa_cfg, device)
        # Evaluate on UNSHUFFLED data (same eval as others)
        r = eval_tfa(tfa_shuf, eval_hidden, device)
        print(f"  TFA-shuffled: NMSE={r['nmse']:.6f}, novel_L0={r['novel_l0']:.2f}, "
              f"total_L0={r['total_l0']:.2f} ({time.time()-t0:.1f}s)")
        results["tfa_shuffled"].append({"k": k, **r})

        # Also evaluate TFA-shuffled on shuffled eval data
        r2 = eval_tfa(tfa_shuf, eval_hidden_shuf, device)
        print(f"  TFA-shuf→shuf: NMSE={r2['nmse']:.6f} (eval on shuffled)")

        del tfa_shuf; torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'k':>3} | {'SAE':>10} | {'Wide SAE':>10} | {'TFA':>10} | {'TFA-shuf':>10} | "
          f"{'TFA/SAE':>8} | {'shuf/SAE':>8} | {'shuf/TFA':>8}")
    print("-" * 90)
    for i, k in enumerate(K_VALUES):
        sae_n = results["sae"][i]["nmse"]
        wide_n = results["wide_sae"][i]["nmse"]
        tfa_n = results["tfa"][i]["nmse"]
        shuf_n = results["tfa_shuffled"][i]["nmse"]
        print(f"{k:>3} | {sae_n:>10.6f} | {wide_n:>10.6f} | {tfa_n:>10.6f} | {shuf_n:>10.6f} | "
              f"{sae_n/tfa_n:>7.1f}x | {sae_n/shuf_n:>7.1f}x | {shuf_n/tfa_n:>7.1f}x")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(10, 7))
    sae_nmse = [r["nmse"] for r in results["sae"]]
    wide_nmse = [r["nmse"] for r in results["wide_sae"]]
    tfa_nmse = [r["nmse"] for r in results["tfa"]]
    shuf_nmse = [r["nmse"] for r in results["tfa_shuffled"]]

    ax.plot(K_VALUES, sae_nmse, "o-", color="tab:blue", linewidth=2, markersize=7,
            label=f"SAE (width={DICT_WIDTH}, 3280 params)")
    ax.plot(K_VALUES, wide_nmse, "d-", color="tab:purple", linewidth=2, markersize=7,
            label=f"Wide SAE (width={WIDE_DICT_WIDTH}, ~8200 params)")
    ax.plot(K_VALUES, tfa_nmse, "s-", color="tab:orange", linewidth=2, markersize=7,
            label="TFA (8200 params, temporal)")
    ax.plot(K_VALUES, shuf_nmse, "^--", color="tab:red", linewidth=2, markersize=7,
            label="TFA-shuffled (8200 params, no temporal)")
    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="E[L0] = 10")

    ax.set_xlabel("k (TopK sparsity budget)", fontsize=13)
    ax.set_ylabel("NMSE", fontsize=13)
    ax.set_title("Capacity vs Temporal Structure Diagnostic\n"
                 "Does TFA's advantage survive when temporal correlations are destroyed?",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"shuffle_diagnostic.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save
    save_data = {
        "config": {
            "num_features": NUM_FEATURES, "hidden_dim": HIDDEN_DIM, "seq_len": SEQ_LEN,
            "pi": PI, "rho": RHO, "dict_width": DICT_WIDTH, "wide_dict_width": WIDE_DICT_WIDTH,
            "k_values": K_VALUES, "seed": SEED,
        },
        "results": results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
