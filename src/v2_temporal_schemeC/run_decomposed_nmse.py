"""Decomposed NMSE analysis: separate TFA into pred-only and novel-only.

For each k, computes:
1. SAE NMSE (L0 = k, full signal)
2. TFA full NMSE (pred + novel)
3. TFA pred-only NMSE (just the predictable component, no sparse features)
4. TFA novel-only NMSE (just the novel component's k sparse features)

This enables an apples-to-apples comparison: SAE at L0=k vs TFA-novel at L0=k,
both reconstructing the full signal with k sparse features.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.shared.configs import TrainingConfig
from src.shared.train_sae import create_sae, train_sae
from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig,
    create_tfa,
    train_tfa,
)

# ── Configuration (same as pareto_benchmark) ─────────────────────────

NUM_FEATURES = 10
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.2] * 10
RHO = [0.0, 0.0, 0.0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.95]
K_VALUES = [1, 2, 3, 4, 6]

SAE_TOTAL_SAMPLES = 10_000_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4
SAE_D_SAE = 40

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3
TFA_WIDTH = 40
TFA_LOG_EVERY = 10000

EVAL_N_TOKENS = 200_000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "decomposed_nmse")


def compute_scaling_factor(model, pi_t, rho_t, device, n_samples=10000):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            n_samples, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
        mean_norm = norms.mean().item()
    target_norm = math.sqrt(HIDDEN_DIM)
    sf = target_norm / mean_norm if mean_norm > 0 else 1.0
    return sf


def make_data_generators(model, pi_t, rho_t, device, scaling_factor):
    def generate_flattened(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return hidden.reshape(-1, HIDDEN_DIM)[:batch_size] * scaling_factor

    def generate_sequences(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        return model(acts) * scaling_factor

    return generate_flattened, generate_sequences


def eval_sae_nmse(sae, gen_fn, device, mean_x_sq, n_tokens=EVAL_N_TOKENS):
    """Evaluate standard SAE, return NMSE."""
    sae.eval()
    total_se = 0.0
    n = 0
    with torch.no_grad():
        while n < n_tokens:
            bs = min(4096, n_tokens - n)
            x = gen_fn(bs).to(device)
            x_hat = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            n += x.shape[0]
    return total_se / n / mean_x_sq


def eval_tfa_decomposed(tfa, gen_fn, device, mean_x_sq, n_tokens=EVAL_N_TOKENS):
    """Evaluate TFA, returning full/pred-only/novel-only NMSE."""
    tfa.eval()
    total_se_full = 0.0
    total_se_pred = 0.0
    total_se_novel = 0.0
    total_x_sq = 0.0
    n = 0

    with torch.no_grad():
        while n < n_tokens:
            n_seq = min(256, max(1, (n_tokens - n) // SEQ_LEN))
            x = gen_fn(n_seq).to(device)  # (B, T, d)
            recons_full, intermediates = tfa(x)

            B, T, D = x.shape
            n_tok = B * T
            b = tfa.b  # (1, d)

            # Full reconstruction: D(z_p + z_n) + b = recons_full
            # Pred-only: D z_p + b
            pred_recons = intermediates["pred_recons"]  # (B, T, D), this is D z_p (no bias)
            pred_with_bias = pred_recons + b

            # Novel-only: D z_n + b
            novel_recons = intermediates["novel_recons"]  # (B, T, D), this is D z_n (no bias)
            novel_with_bias = novel_recons + b

            # Squared errors (full signal reconstruction)
            total_se_full += (x - recons_full).pow(2).sum().item()
            total_se_pred += (x - pred_with_bias).pow(2).sum().item()
            total_se_novel += (x - novel_with_bias).pow(2).sum().item()
            total_x_sq += x.pow(2).sum().item()

            n += n_tok

    empirical_mean_x_sq = total_x_sq / n
    return {
        "nmse_full": total_se_full / n / mean_x_sq,
        "nmse_pred_only": total_se_pred / n / mean_x_sq,
        "nmse_novel_only": total_se_novel / n / mean_x_sq,
        "empirical_mean_x_sq": empirical_mean_x_sq,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    set_seed(SEED)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    gen_flat, gen_seq = make_data_generators(model, pi_t, rho_t, device, sf)

    # Compute E[||x||^2] empirically for NMSE denominator
    set_seed(SEED + 200)
    with torch.no_grad():
        cal_x = gen_seq(2000)  # (2000, T, d)
        mean_x_sq = cal_x.pow(2).sum(dim=-1).mean().item()
    print(f"E[||x||^2] = {mean_x_sq:.4f} (per token)")
    print(f"Scaling factor: {sf:.4f}")
    print(f"k values: {K_VALUES}\n")

    # ── Train and evaluate ──
    sae_results = []
    tfa_results = []

    for k in K_VALUES:
        print(f"{'='*60}")
        print(f"  k = {k}")
        print(f"{'='*60}")

        # Standard SAE
        set_seed(SEED)
        sae = create_sae(HIDDEN_DIM, SAE_D_SAE, k=float(k), device=device)
        cfg = TrainingConfig(
            k=float(k), d_sae=SAE_D_SAE,
            total_training_samples=SAE_TOTAL_SAMPLES,
            batch_size=SAE_BATCH_SIZE, lr=SAE_LR, seed=SEED,
        )
        sae = train_sae(sae, gen_flat, cfg, device)
        sae_nmse = eval_sae_nmse(sae, gen_flat, device, mean_x_sq)
        sae_results.append({"k": k, "nmse": sae_nmse})
        print(f"  SAE NMSE:        {sae_nmse:.6f}")

        # TFA
        set_seed(SEED)
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=TFA_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR, log_every=TFA_LOG_EVERY,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        tfa_dec = eval_tfa_decomposed(tfa, gen_seq, device, mean_x_sq)
        tfa_results.append({"k": k, **tfa_dec})
        print(f"  TFA full NMSE:   {tfa_dec['nmse_full']:.6f}")
        print(f"  TFA pred-only:   {tfa_dec['nmse_pred_only']:.6f}")
        print(f"  TFA novel-only:  {tfa_dec['nmse_novel_only']:.6f}")
        print()

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("DECOMPOSED NMSE SUMMARY")
    print(f"{'='*80}")
    print(f"{'k':>3} | {'SAE':>10} | {'TFA full':>10} {'TFA pred':>10} {'TFA novel':>10} | {'SAE vs novel':>12}")
    print("-" * 80)
    for sr, tr in zip(sae_results, tfa_results):
        delta = sr["nmse"] - tr["nmse_novel_only"]
        sign = "+" if delta >= 0 else ""
        print(f"{sr['k']:>3} | {sr['nmse']:>10.6f} | "
              f"{tr['nmse_full']:>10.6f} {tr['nmse_pred_only']:>10.6f} {tr['nmse_novel_only']:>10.6f} | "
              f"{sign}{delta:>11.6f}")

    # ── Plot ──
    ks = [r["k"] for r in sae_results]
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ks, [r["nmse"] for r in sae_results],
            "o-", label="Standard SAE (L0=k)", color="tab:blue", linewidth=2, markersize=8)
    ax.plot(ks, [r["nmse_full"] for r in tfa_results],
            "s-", label="TFA full (pred+novel)", color="tab:orange", linewidth=2, markersize=8)
    ax.plot(ks, [r["nmse_pred_only"] for r in tfa_results],
            "^--", label="TFA pred-only (no sparse features)", color="tab:green",
            linewidth=2, markersize=8)
    ax.plot(ks, [r["nmse_novel_only"] for r in tfa_results],
            "v--", label="TFA novel-only (L0=k, no prediction)", color="tab:red",
            linewidth=2, markersize=8)

    ax.axvline(x=sum(PI), color="gray", linestyle="--", alpha=0.5,
               label=f"$E[L_0]$ = {sum(PI)}")
    ax.set_xlabel("k (TopK sparsity)", fontsize=12)
    ax.set_ylabel("NMSE", fontsize=12)
    ax.set_title("Decomposed NMSE: What drives TFA's advantage?", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"decomposed_nmse.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save
    results = {
        "config": {
            "num_features": NUM_FEATURES, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "pi": PI, "rho": RHO,
            "k_values": K_VALUES, "mean_x_sq": mean_x_sq,
            "scaling_factor": sf, "seed": SEED,
        },
        "sae_results": sae_results,
        "tfa_results": tfa_results,
    }
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
