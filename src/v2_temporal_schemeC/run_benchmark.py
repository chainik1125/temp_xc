"""Benchmark: Standard SAE vs TFA on Scheme C temporal toy data.

Sweeps sparsity (k) for both models and compares reconstruction quality.
Both models train and evaluate on identically scaled data.
"""

import json
import math
import os
import time

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
from src.v2_temporal_schemeC.tfa import TemporalSAE
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig,
    create_tfa,
    train_tfa,
)

# ── Configuration ────────────────────────────────────────────────────

NUM_FEATURES = 10
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.2] * 10
RHO = [0.0, 0.0, 0.0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.95]
K_VALUES = [1, 2, 4, 6, 8, 10]

# Standard SAE training
SAE_TOTAL_SAMPLES = 10_000_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

# TFA training
TFA_TOTAL_STEPS = 5000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3
TFA_WIDTH = 40  # exp_factor=1

SAE_D_SAE = 40  # match TFA width for fair comparison

EVAL_N_SAMPLES = 100_000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "benchmark")


# ── Scaling ─────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device, n_samples=10000):
    """Compute scaling_factor = sqrt(dimin) / mean(||h||).

    Both SAE and TFA operate on data scaled so that mean(||x||) = sqrt(dimin),
    matching the assumption in TFA's internal lam = 1/(4*dimin).
    """
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            n_samples, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
        mean_norm = norms.mean().item()

    target_norm = math.sqrt(HIDDEN_DIM)
    sf = target_norm / mean_norm if mean_norm > 0 else 1.0
    return sf, mean_norm


# ── Data Generation ──────────────────────────────────────────────────


def make_data_generators(model, pi_t, rho_t, device, scaling_factor):
    """Create scaled data generators for SAE (flattened) and TFA (sequences)."""

    def generate_flattened(batch_size: int) -> torch.Tensor:
        """For standard SAE: (batch_size, hidden_dim), scaled."""
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        return hidden.reshape(-1, HIDDEN_DIM)[:batch_size] * scaling_factor

    def generate_sequences(n_seq: int) -> torch.Tensor:
        """For TFA: (n_seq, seq_len, hidden_dim), scaled."""
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return model(acts) * scaling_factor

    return generate_flattened, generate_sequences


# ── Evaluation ───────────────────────────────────────────────────────


def eval_standard_sae(sae, gen_fn, k, device, n_samples=EVAL_N_SAMPLES):
    """Evaluate standard SAE. Returns dict with mse."""
    sae.eval()
    total_mse = 0.0
    n_processed = 0
    batch_size = 4096

    with torch.no_grad():
        while n_processed < n_samples:
            bs = min(batch_size, n_samples - n_processed)
            x = gen_fn(bs).to(device)
            x_hat = sae(x)
            total_mse += (x - x_hat).pow(2).sum().item()
            n_processed += x.shape[0]

    mse = total_mse / n_processed
    return {"model": "SAE", "k": k, "mse": mse}


def eval_tfa_model(tfa, gen_fn, k, device, n_samples=EVAL_N_SAMPLES):
    """Evaluate TFA. Returns dict with mse and decomposition."""
    tfa.eval()
    total_mse = 0.0
    pred_energy_sum = 0.0
    novel_energy_sum = 0.0
    novel_l0_sum = 0.0
    n_tokens = 0
    batch_seqs = 256

    with torch.no_grad():
        while n_tokens < n_samples:
            n_seq = min(batch_seqs, max(1, (n_samples - n_tokens) // SEQ_LEN))
            x = gen_fn(n_seq).to(device)  # (n_seq, T, d)
            recons, intermediates = tfa(x)

            B, T, D = x.shape
            n_tok = B * T

            # Overall reconstruction
            x_flat = x.reshape(-1, D)
            r_flat = recons.reshape(-1, D)
            total_mse += (x_flat - r_flat).pow(2).sum().item()

            # Pred-only: z_pred @ D (without bias; compare to x - b)
            pred_r = intermediates["pred_recons"]  # (B, T, D), without bias
            novel_r = intermediates["novel_recons"]  # (B, T, D), without bias

            # Energy decomposition
            pred_energy_sum += pred_r.norm(dim=-1).pow(2).sum().item()
            novel_energy_sum += novel_r.norm(dim=-1).pow(2).sum().item()

            # Novel L0
            novel_codes = intermediates["novel_codes"]
            novel_l0_sum += (novel_codes > 0).float().sum(dim=-1).mean().item() * n_tok

            n_tokens += n_tok

    mse = total_mse / n_tokens
    total_energy = pred_energy_sum + novel_energy_sum + 1e-12
    rel_energy_pred = pred_energy_sum / total_energy
    novel_l0 = novel_l0_sum / n_tokens

    return {
        "model": "TFA",
        "k": k,
        "mse": mse,
        "rel_energy_pred": rel_energy_pred,
        "novel_l0": novel_l0,
    }


# ── Plotting ─────────────────────────────────────────────────────────


def plot_mse_comparison(sae_results, tfa_results, results_dir):
    """Plot MSE vs k for SAE and TFA."""
    ks_sae = [r["k"] for r in sae_results]
    ks_tfa = [r["k"] for r in tfa_results]
    mse_sae = [r["mse"] for r in sae_results]
    mse_tfa = [r["mse"] for r in tfa_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks_sae, mse_sae, "o-", label="Standard SAE", color="tab:blue", linewidth=2)
    ax.plot(ks_tfa, mse_tfa, "s-", label="TFA", color="tab:orange", linewidth=2)
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="True L0=2.0")
    ax.set_xlabel("k (TopK sparsity)")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE vs Sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "mse_vs_k.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_tfa_decomposition(tfa_results, results_dir):
    """Plot TFA predictable vs novel energy decomposition."""
    ks = [r["k"] for r in tfa_results]
    rel_pred = [r["rel_energy_pred"] for r in tfa_results]
    novel_l0 = [r["novel_l0"] for r in tfa_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy decomposition
    ax1.bar(
        [str(k) for k in ks],
        rel_pred,
        color="tab:green",
        alpha=0.8,
        label="Predictable",
    )
    ax1.bar(
        [str(k) for k in ks],
        [1 - r for r in rel_pred],
        bottom=rel_pred,
        color="tab:red",
        alpha=0.8,
        label="Novel",
    )
    ax1.set_xlabel("k (TopK sparsity)")
    ax1.set_ylabel("Relative energy")
    ax1.set_title("TFA Energy Decomposition")
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Novel L0
    ax2.plot(ks, novel_l0, "o-", color="tab:red", linewidth=2)
    ax2.set_xlabel("k (TopK sparsity)")
    ax2.set_ylabel("Novel L0 (actual)")
    ax2.set_title("TFA Novel Code Sparsity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "tfa_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")

    set_seed(SEED)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)
    true_l0 = sum(PI)

    print(f"\nConfig: {NUM_FEATURES} features, hidden_dim={HIDDEN_DIM}, "
          f"seq_len={SEQ_LEN}")
    print(f"rho per feature: {RHO}")
    print(f"True L0 per position: {true_l0}")
    print(f"k values to sweep: {K_VALUES}\n")

    # ── Build toy model ──
    model = ToyModel(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    model.eval()

    # Compute scaling factor — both models see identically scaled data
    scaling_factor, mean_norm = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"  mean_norm={mean_norm:.4f}, scaling_factor={scaling_factor:.4f}")

    gen_flat, gen_seq = make_data_generators(
        model, pi_t, rho_t, device, scaling_factor
    )

    # ── Standard SAE sweep ──
    print("=" * 60)
    print("STANDARD SAE SWEEP")
    print("=" * 60)

    sae_results = []
    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n--- Standard SAE, k={k} ---")
        t0 = time.time()

        sae = create_sae(HIDDEN_DIM, SAE_D_SAE, k=float(k), device=device)
        cfg = TrainingConfig(
            k=float(k),
            d_sae=SAE_D_SAE,
            total_training_samples=SAE_TOTAL_SAMPLES,
            batch_size=SAE_BATCH_SIZE,
            lr=SAE_LR,
            seed=SEED,
        )
        sae = train_sae(sae, gen_flat, cfg, device)
        result = eval_standard_sae(sae, gen_flat, k, device)
        sae_results.append(result)

        dt = time.time() - t0
        print(f"  k={k}: MSE={result['mse']:.6f} ({dt:.1f}s)")

    # ── TFA sweep ──
    print("\n" + "=" * 60)
    print("TFA SWEEP")
    print("=" * 60)

    tfa_results = []
    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n--- TFA, k={k} ---")
        t0 = time.time()

        tfa = create_tfa(
            dimin=HIDDEN_DIM,
            width=TFA_WIDTH,
            k=k,
            n_heads=4,
            n_attn_layers=1,
            bottleneck_factor=1,
            device=device,
        )

        tfa_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS,
            batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR,
        )
        tfa, train_log = train_tfa(tfa, gen_seq, tfa_cfg, device)
        result = eval_tfa_model(tfa, gen_seq, k, device)
        tfa_results.append(result)

        dt = time.time() - t0
        print(f"  k={k}: MSE={result['mse']:.6f}")
        print(f"         pred_energy={result['rel_energy_pred']:.3f}, "
              f"novel_L0={result['novel_l0']:.2f} ({dt:.1f}s)")

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'k':>3} | {'SAE MSE':>10} | "
          f"{'TFA MSE':>10} {'pred_E':>7} | {'delta_MSE':>10}")
    print("-" * 55)
    for sr, tr in zip(sae_results, tfa_results):
        delta = sr["mse"] - tr["mse"]  # positive = TFA wins
        sign = "+" if delta >= 0 else ""
        print(f"{sr['k']:>3} | {sr['mse']:>10.6f} | "
              f"{tr['mse']:>10.6f} {tr['rel_energy_pred']:>7.3f} | "
              f"{sign}{delta:>9.6f}")

    # ── Plots ──
    plot_mse_comparison(sae_results, tfa_results, RESULTS_DIR)
    plot_tfa_decomposition(tfa_results, RESULTS_DIR)

    # ── Save results ──
    results_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "sae_d_sae": SAE_D_SAE,
            "tfa_width": TFA_WIDTH,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "pi": PI,
            "rho": RHO,
            "k_values": K_VALUES,
            "scaling_factor": scaling_factor,
            "seed": SEED,
        },
        "sae_results": sae_results,
        "tfa_results": tfa_results,
    }
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
