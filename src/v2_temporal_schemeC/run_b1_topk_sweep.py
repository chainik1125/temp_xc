"""Part B1 supplementary: TopK sweep with scaled-up data.

Same B1 data (n=20, pi=0.5, E[L0]=10) but using TopK sparsity
for both SAE and TFA novel component. This gives clean, artifact-free
NMSE-vs-k curves without L1 training issues.

Complements the ReLU+L1 Pareto frontier in run_b1_b2_pareto.py.
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
import torch.nn.functional as F

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
EXPECTED_L0 = sum(PI)  # 10.0

DICT_WIDTH = 40
K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]

# SAE training (matched with TFA: 30K steps, 4096 tokens/step)
SAE_TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

# TFA training
TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3
TFA_N_HEADS = 4
TFA_N_ATTN_LAYERS = 1
TFA_BOTTLENECK = 1
TFA_LOG_EVERY = 10000

EVAL_N_SEQ = 2000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "b1_topk_sweep")


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device, n_samples=10000):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            n_samples, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
        mean_norm = norms.mean().item()
    target_norm = math.sqrt(HIDDEN_DIM)
    return target_norm / mean_norm if mean_norm > 0 else 1.0


def make_data_generators(model, pi_t, rho_t, device, sf):
    def gen_flat(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]

    def gen_seq(n_seq: int) -> torch.Tensor:
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return model(acts) * sf

    return gen_flat, gen_seq


# ── Evaluation ───────────────────────────────────────────────────────


def eval_sae_topk(sae, eval_hidden, device):
    """Evaluate custom TopK SAE."""
    sae.eval()
    flat = eval_hidden.reshape(-1, HIDDEN_DIM)
    n = flat.shape[0]
    total_se = 0.0
    total_signal = 0.0
    total_l0 = 0.0
    bs = 4096
    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            x = flat[start:end].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_signal += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()
    return {
        "nmse": total_se / total_signal,
        "mse": total_se / n,
        "l0": total_l0 / n,
    }


def eval_tfa_topk(tfa, eval_hidden, device):
    """Evaluate TFA with TopK novel component."""
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    bs = 256
    total_se = 0.0
    total_signal = 0.0
    total_novel_l0 = 0.0
    total_total_l0 = 0.0
    total_pred_e = 0.0
    total_novel_e = 0.0
    n_tokens = 0

    with torch.no_grad():
        for start in range(0, n_seq, bs):
            end = min(start + bs, n_seq)
            x = eval_hidden[start:end].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            n_tok = B * T
            x_f = x.reshape(-1, D)
            r_f = recons.reshape(-1, D)
            total_se += (x_f - r_f).pow(2).sum().item()
            total_signal += x_f.pow(2).sum().item()

            nc = inter["novel_codes"]
            pc = inter["pred_codes"]
            tc = nc + pc
            total_novel_l0 += (nc > 0).float().sum(dim=-1).sum().item()
            total_total_l0 += (tc.abs() > 1e-8).float().sum(dim=-1).sum().item()

            total_pred_e += inter["pred_recons"].norm(dim=-1).pow(2).sum().item()
            total_novel_e += inter["novel_recons"].norm(dim=-1).pow(2).sum().item()
            n_tokens += n_tok

    te = total_pred_e + total_novel_e + 1e-12
    return {
        "nmse": total_se / total_signal,
        "mse": total_se / n_tokens,
        "novel_l0": total_novel_l0 / n_tokens,
        "total_l0": total_total_l0 / n_tokens,
        "pred_energy_frac": total_pred_e / te,
    }


# ── Plotting ─────────────────────────────────────────────────────────


def plot_results(sae_results, tfa_results, results_dir):
    """NMSE vs k and three-panel comparison."""

    # Main: NMSE vs k
    fig, ax = plt.subplots(figsize=(10, 7))
    ks = [r["k"] for r in sae_results]
    sae_nmse = [r["nmse"] for r in sae_results]
    tfa_nmse = [r["nmse"] for r in tfa_results]

    ax.plot(ks, sae_nmse, "o-", color="tab:blue", linewidth=2, markersize=7,
            label="Standard SAE (L0 = k)")
    ax.plot(ks, tfa_nmse, "s-", color="tab:orange", linewidth=2, markersize=7,
            label="TFA (novel L0 = k)")

    # TFA total L0 (secondary x-axis would be complex, just annotate)
    tfa_total_l0 = [r["total_l0"] for r in tfa_results]

    ax.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5,
               label=f"$E[L_0]$ = {EXPECTED_L0:.0f}")

    ax.set_xlabel("k (TopK sparsity budget)", fontsize=13)
    ax.set_ylabel("NMSE", fontsize=13)
    ax.set_title("NMSE vs TopK Sparsity Budget\n"
                 f"(n={NUM_FEATURES}, d={HIDDEN_DIM}, $\\pi$={PI[0]}, "
                 f"$E[L_0]$={EXPECTED_L0:.0f})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"nmse_vs_k.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Two-panel: NMSE vs novel_L0 and NMSE vs total_L0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: NMSE vs novel L0 (= k for both)
    ax1.plot(ks, sae_nmse, "o-", color="tab:blue", linewidth=2, markersize=7,
             label="Standard SAE")
    ax1.plot(ks, tfa_nmse, "s-", color="tab:orange", linewidth=2, markersize=7,
             label="TFA (novel L0 = k)")
    ax1.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5,
                label=f"$E[L_0]$ = {EXPECTED_L0:.0f}")
    ax1.set_xlabel("Novel L0 (= k)", fontsize=12)
    ax1.set_ylabel("NMSE", fontsize=12)
    ax1.set_title("NMSE vs Novel L0", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Right: NMSE vs total L0 for TFA, vs L0 for SAE
    ax2.plot(ks, sae_nmse, "o-", color="tab:blue", linewidth=2, markersize=7,
             label="Standard SAE (L0 = k)")
    ax2.plot(tfa_total_l0, tfa_nmse, "^-", color="tab:red", linewidth=2,
             markersize=7, label="TFA (total L0)")
    ax2.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5,
                label=f"$E[L_0]$ = {EXPECTED_L0:.0f}")
    ax2.set_xlabel("L0 (total active features)", fontsize=12)
    ax2.set_ylabel("NMSE", fontsize=12)
    ax2.set_title("NMSE vs Total L0", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"nmse_comparison.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # TFA decomposition
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    pred_frac = [r["pred_energy_frac"] for r in tfa_results]

    ax1.plot(ks, pred_frac, "o-", color="tab:green", linewidth=2, markersize=7)
    ax1.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("k (TopK)", fontsize=12)
    ax1.set_ylabel("Predictable energy fraction", fontsize=12)
    ax1.set_title("TFA Predictable Energy vs k", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, tfa_total_l0, "o-", color="tab:red", linewidth=2, markersize=7)
    ax2.plot(ks, ks, "--", color="gray", alpha=0.5, label="L0 = k")
    ax2.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("k (TopK)", fontsize=12)
    ax2.set_ylabel("Total L0", fontsize=12)
    ax2.set_title("TFA Total L0 vs k", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"tfa_decomposition.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Plots saved.")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")
    print(f"Config: n={NUM_FEATURES}, d={HIDDEN_DIM}, T={SEQ_LEN}, "
          f"E[L0]={EXPECTED_L0}")
    print(f"k values: {K_VALUES}")

    set_seed(SEED)
    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling factor: {sf:.4f}")

    gen_flat, gen_seq = make_data_generators(model, pi_t, rho_t, device, sf)

    # Fixed eval data
    set_seed(SEED + 100)
    acts, support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts) * sf
    print(f"Eval: {eval_hidden.shape[0]}x{eval_hidden.shape[1]} = "
          f"{eval_hidden.shape[0] * eval_hidden.shape[1]} tokens\n")

    # ── SAE sweep ──
    print("=" * 60)
    print("STANDARD SAE (per-token TopK) SWEEP")
    print("=" * 60)

    sae_results = []
    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n--- SAE k={k} ---")
        t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(
            total_steps=SAE_TOTAL_STEPS,
            batch_size=SAE_BATCH_SIZE,
            lr=SAE_LR,
            l1_coeff=0.0,  # no L1, sparsity from TopK
            log_every=TFA_LOG_EVERY,
        )
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        result = eval_sae_topk(sae, eval_hidden, device)
        dt = time.time() - t0
        print(f"  NMSE={result['nmse']:.6f}, L0={result['l0']:.2f} ({dt:.1f}s)")
        sae_results.append({"k": k, **result})
        del sae
        torch.cuda.empty_cache()

    # ── TFA sweep ──
    print("\n" + "=" * 60)
    print("TFA (TopK) SWEEP")
    print("=" * 60)

    tfa_results = []
    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n--- TFA k={k} ---")
        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=TFA_N_HEADS, n_attn_layers=TFA_N_ATTN_LAYERS,
            bottleneck_factor=TFA_BOTTLENECK, device=device,
        )
        tfa_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS,
            batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR,
            log_every=TFA_LOG_EVERY,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        result = eval_tfa_topk(tfa, eval_hidden, device)
        dt = time.time() - t0
        print(f"  NMSE={result['nmse']:.6f}, novel_L0={result['novel_l0']:.2f}, "
              f"total_L0={result['total_l0']:.2f}, pred_E={result['pred_energy_frac']:.3f} "
              f"({dt:.1f}s)")
        tfa_results.append({"k": k, **result})
        del tfa
        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'k':>3} | {'SAE NMSE':>10} {'SAE L0':>7} | "
          f"{'TFA NMSE':>10} {'novel_L0':>9} {'total_L0':>9} {'pred_E':>7} | "
          f"{'Winner':>6} {'Ratio':>8}")
    print("-" * 85)
    for sr, tr in zip(sae_results, tfa_results):
        winner = "TFA" if tr["nmse"] < sr["nmse"] else "SAE"
        if sr["nmse"] > 0 and tr["nmse"] > 0:
            ratio = sr["nmse"] / tr["nmse"] if winner == "TFA" else tr["nmse"] / sr["nmse"]
        else:
            ratio = float("inf")
        print(f"{sr['k']:>3} | {sr['nmse']:>10.6f} {sr['l0']:>7.2f} | "
              f"{tr['nmse']:>10.6f} {tr['novel_l0']:>9.2f} {tr['total_l0']:>9.2f} "
              f"{tr['pred_energy_frac']:>7.3f} | {winner:>6} {ratio:>7.1f}x")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_results(sae_results, tfa_results, RESULTS_DIR)

    # ── Save ──
    results_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "pi": PI,
            "rho": RHO,
            "expected_l0": EXPECTED_L0,
            "dict_width": DICT_WIDTH,
            "k_values": K_VALUES,
            "tfa_total_steps": TFA_TOTAL_STEPS,
            "sae_total_steps": SAE_TOTAL_STEPS,
            "seed": SEED,
        },
        "sae_results": sae_results,
        "tfa_results": tfa_results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
