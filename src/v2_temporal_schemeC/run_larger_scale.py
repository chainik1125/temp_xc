"""Larger-scale benchmark: 30 features, hidden_dim=120.

Tests whether TFA's advantage widens at larger scale where
k << true L0 is the typical operating regime.
"""

import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

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

# ── Configuration ────────────────────────────────────────────────────

NUM_FEATURES = 30
HIDDEN_DIM = 120
SEQ_LEN = 64

# 30 features with varied rho: 10 iid, 5 weak, 5 moderate, 5 strong, 5 very strong
PI = [0.2] * 30  # true L0 = 6.0
RHO = (
    [0.0] * 10     # f0-f9: i.i.d.
    + [0.3] * 5    # f10-f14: weak persistence
    + [0.6] * 5    # f15-f19: moderate persistence
    + [0.9] * 5    # f20-f24: strong persistence
    + [0.95] * 5   # f25-f29: very strong persistence
)

K_VALUES = [1, 2, 4, 6, 10, 15]  # true L0 = 6.0, so k=6 matches

# SAE training
SAE_TOTAL_SAMPLES = 15_000_000  # more samples for larger model
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4
SAE_D_SAE = 120  # match hidden_dim

# TFA training
TFA_TOTAL_STEPS = 15000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3
TFA_WIDTH = 120  # match hidden_dim

EVAL_N_SAMPLES = 100_000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "larger_scale")


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
    sf = target_norm / mean_norm if mean_norm > 0 else 1.0
    return sf, mean_norm


def make_data_generators(model, pi_t, rho_t, device, scaling_factor):
    def generate_flattened(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        return hidden.reshape(-1, HIDDEN_DIM)[:batch_size] * scaling_factor

    def generate_sequences(n_seq: int) -> torch.Tensor:
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return model(acts) * scaling_factor

    return generate_flattened, generate_sequences


def eval_standard_sae(sae, gen_fn, k, device, n_samples=EVAL_N_SAMPLES):
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
    tfa.eval()
    total_mse = 0.0
    pred_energy_sum = 0.0
    novel_energy_sum = 0.0
    novel_l0_sum = 0.0
    n_tokens = 0
    batch_seqs = 128  # smaller batches for larger model
    with torch.no_grad():
        while n_tokens < n_samples:
            n_seq = min(batch_seqs, max(1, (n_samples - n_tokens) // SEQ_LEN))
            x = gen_fn(n_seq).to(device)
            recons, intermediates = tfa(x)
            B, T, D = x.shape
            n_tok = B * T
            x_flat = x.reshape(-1, D)
            r_flat = recons.reshape(-1, D)
            total_mse += (x_flat - r_flat).pow(2).sum().item()
            pred_r = intermediates["pred_recons"]
            novel_r = intermediates["novel_recons"]
            pred_energy_sum += pred_r.norm(dim=-1).pow(2).sum().item()
            novel_energy_sum += novel_r.norm(dim=-1).pow(2).sum().item()
            novel_codes = intermediates["novel_codes"]
            novel_l0_sum += (novel_codes > 0).float().sum(dim=-1).mean().item() * n_tok
            n_tokens += n_tok
    mse = total_mse / n_tokens
    total_energy = pred_energy_sum + novel_energy_sum + 1e-12
    rel_energy_pred = pred_energy_sum / total_energy
    novel_l0 = novel_l0_sum / n_tokens
    return {
        "model": "TFA", "k": k, "mse": mse,
        "rel_energy_pred": rel_energy_pred, "novel_l0": novel_l0,
    }


def per_feature_analysis(tfa, model, gen_fn, device, n_samples=50000):
    feat_dirs = model.feature_directions.to(device)
    feat_normed = feat_dirs / feat_dirs.norm(dim=1, keepdim=True)
    n_features = feat_dirs.shape[0]
    pred_energy_per_feat = torch.zeros(n_features, device=device)
    novel_energy_per_feat = torch.zeros(n_features, device=device)
    n_tokens = 0
    tfa.eval()
    with torch.no_grad():
        while n_tokens < n_samples:
            n_seq = min(128, max(1, (n_samples - n_tokens) // SEQ_LEN))
            x = gen_fn(n_seq).to(device)
            _, intermediates = tfa(x)
            B, T, D = x.shape
            n_tok = B * T
            pred_r = intermediates["pred_recons"].reshape(-1, D)
            novel_r = intermediates["novel_recons"].reshape(-1, D)
            pred_proj = pred_r @ feat_normed.T
            novel_proj = novel_r @ feat_normed.T
            pred_energy_per_feat += pred_proj.pow(2).sum(dim=0)
            novel_energy_per_feat += novel_proj.pow(2).sum(dim=0)
            n_tokens += n_tok
    total_recons = pred_energy_per_feat + novel_energy_per_feat + 1e-12
    pred_fraction = (pred_energy_per_feat / total_recons).cpu().tolist()
    return {
        "pred_fraction_per_feature": pred_fraction,
        "rho_per_feature": RHO,
    }


# ── Plotting ─────────────────────────────────────────────────────────


def plot_mse_comparison(sae_results, tfa_results, results_dir):
    """MSE vs k for SAE and TFA."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks_sae = [r["k"] for r in sae_results]
    mse_sae = [r["mse"] for r in sae_results]
    ks_tfa = [r["k"] for r in tfa_results]
    mse_tfa = [r["mse"] for r in tfa_results]

    ax.plot(ks_sae, mse_sae, "o-", label="SAE", color="tab:blue", linewidth=2)
    ax.plot(ks_tfa, mse_tfa, "s-", label="TFA", color="tab:orange", linewidth=2)
    ax.axvline(x=6.0, color="gray", linestyle="--", alpha=0.5, label="True L0=6")
    ax.set_xlabel("k (TopK sparsity)")
    ax.set_ylabel("MSE")
    ax.set_title(f"Larger Scale: {NUM_FEATURES} features, hidden_dim={HIDDEN_DIM}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "mse_vs_k_larger_scale.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_delta_mse(sae_results, tfa_results, results_dir):
    """Delta MSE vs k. Positive = TFA wins."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [r["k"] for r in sae_results]
    delta = [s["mse"] - t["mse"] for s, t in zip(sae_results, tfa_results)]

    ax.plot(ks, delta, "o-", color="tab:orange", linewidth=2)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.axvline(x=6.0, color="gray", linestyle="--", alpha=0.5, label="True L0=6")
    ax.set_xlabel("k (TopK sparsity)")
    ax.set_ylabel("MSE(SAE) - MSE(TFA)")
    ax.set_title(f"TFA Advantage ({NUM_FEATURES} features, hidden_dim={HIDDEN_DIM})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "tfa_advantage_larger_scale.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_pred_fraction_vs_rho(per_feature_results, results_dir):
    """Per-feature pred fraction vs rho, scatter."""
    pf_list = [p for p in per_feature_results if p["k"] == 6]
    if not pf_list:
        return
    pf = pf_list[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    rhos = pf["rho_per_feature"]
    fracs = pf["pred_fraction_per_feature"]

    # Color by rho group
    rho_vals = sorted(set(rhos))
    color_map = plt.cm.viridis(np.linspace(0, 1, len(rho_vals)))
    for ri, rv in enumerate(rho_vals):
        mask = [r == rv for r in rhos]
        xs = [r for r, m in zip(rhos, mask) if m]
        ys = [f for f, m in zip(fracs, mask) if m]
        jitter = np.random.default_rng(42).uniform(-0.02, 0.02, len(xs))
        ax.scatter([x + j for x, j in zip(xs, jitter)], ys, s=40,
                   color=color_map[ri], edgecolors="black", linewidth=0.5,
                   label=f"rho={rv}", zorder=5)

    ax.set_xlabel("Feature rho")
    ax.set_ylabel("Pred fraction")
    ax.set_title(f"Per-Feature Pred Fraction (k=6, true L0)")
    ax.legend(fontsize=7, loc="center left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = os.path.join(results_dir, "pred_fraction_vs_rho_larger.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_scale_comparison(sae_results, tfa_results, small_results_path, results_dir):
    """Compare TFA advantage between toy (10 feat) and larger (30 feat) scales."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Larger scale results: delta_MSE = SAE_MSE - TFA_MSE (positive = TFA wins)
    ks = [r["k"] for r in sae_results]
    delta = [s["mse"] - t["mse"] for s, t in zip(sae_results, tfa_results)]
    ax.plot(ks, delta, "o-", label="30 features", color="tab:orange", linewidth=2)

    # Load small-scale results if available
    if os.path.exists(small_results_path):
        with open(small_results_path) as f:
            small = json.load(f)
        small_sae = small.get("sae_results", [])
        small_tfa = small.get("tfa_results", [])
        if small_sae and small_tfa:
            small_ks = [r["k"] for r in small_sae]
            small_delta = [s["mse"] - t["mse"] for s, t in zip(small_sae, small_tfa)]
            ax.plot(small_ks, small_delta, "x--", label="10 features",
                    color="tab:blue", linewidth=1.5, alpha=0.6)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("k (TopK sparsity)")
    ax.set_ylabel("MSE(SAE) - MSE(TFA)")
    ax.set_title("TFA Advantage: Small vs Larger Scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "scale_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)
    true_l0 = sum(PI)

    print(f"\nLarger-scale benchmark")
    print(f"  {NUM_FEATURES} features, hidden_dim={HIDDEN_DIM}")
    print(f"  True L0: {true_l0}")
    print(f"  rho groups: {sorted(set(RHO))}")
    print(f"  k values: {K_VALUES}\n")

    t_start = time.time()

    set_seed(SEED)
    model = ToyModel(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    model.eval()

    scaling_factor, mean_norm = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"mean_norm={mean_norm:.4f}, scaling_factor={scaling_factor:.4f}")
    print(f"(target norm = sqrt({HIDDEN_DIM}) = {math.sqrt(HIDDEN_DIM):.4f})")

    gen_flat, gen_seq = make_data_generators(
        model, pi_t, rho_t, device, scaling_factor
    )

    sae_results = []
    tfa_results = []
    per_feature_results = []

    # SAE sweep
    print("\n--- Standard SAE sweep ---")
    for k in K_VALUES:
        set_seed(SEED)
        t0 = time.time()
        sae = create_sae(HIDDEN_DIM, SAE_D_SAE, k=float(k), device=device)
        cfg = TrainingConfig(
            k=float(k), d_sae=SAE_D_SAE,
            total_training_samples=SAE_TOTAL_SAMPLES,
            batch_size=SAE_BATCH_SIZE, lr=SAE_LR, seed=SEED,
        )
        sae = train_sae(sae, gen_flat, cfg, device)
        result = eval_standard_sae(sae, gen_flat, k, device)
        sae_results.append(result)
        dt = time.time() - t0
        print(f"  k={k:2d}: MSE={result['mse']:.6f} ({dt:.1f}s)")

    # TFA sweep
    print(f"\n--- TFA sweep ---")
    for k in K_VALUES:
        set_seed(SEED)
        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=TFA_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        result = eval_tfa_model(tfa, gen_seq, k, device)
        tfa_results.append(result)
        dt = time.time() - t0
        print(
            f"  k={k:2d}: MSE={result['mse']:.6f}, "
            f"pred_E={result['rel_energy_pred']:.3f}, "
            f"L0_novel={result['novel_l0']:.1f} ({dt:.1f}s)"
        )

        # Per-feature at k = true L0 (6)
        if k == 6:
            pf = per_feature_analysis(tfa, model, gen_seq, device)
            pf["k"] = k
            per_feature_results.append(pf)
            fracs = pf["pred_fraction_per_feature"]
            for rv in sorted(set(RHO)):
                group_fracs = [f for f, r in zip(fracs, RHO) if r == rv]
                mean_frac = np.mean(group_fracs)
                print(f"    rho={rv:.2f}: mean pred_frac={mean_frac:.3f} "
                      f"(n={len(group_fracs)})")

    # Summary
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"SUMMARY (runtime: {total_time / 60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"{'k':>3} | {'SAE MSE':>10} {'TFA MSE':>10} {'delta_MSE':>10} {'pred_E':>7}")
    print("-" * 50)
    for sr, tr in zip(sae_results, tfa_results):
        delta = sr["mse"] - tr["mse"]  # positive = TFA wins
        sign = "+" if delta >= 0 else ""
        print(
            f"{sr['k']:>3} | {sr['mse']:>10.6f} {tr['mse']:>10.6f} "
            f"{sign}{delta:>9.6f} {tr['rel_energy_pred']:>7.3f}"
        )

    # Plots
    plot_mse_comparison(sae_results, tfa_results, RESULTS_DIR)
    plot_delta_mse(sae_results, tfa_results, RESULTS_DIR)
    plot_pred_fraction_vs_rho(per_feature_results, RESULTS_DIR)

    small_results_path = os.path.join(
        os.path.dirname(__file__), "results", "extended_benchmark", "results.json"
    )
    plot_scale_comparison(sae_results, tfa_results, small_results_path, RESULTS_DIR)

    # Save
    results_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "pi": PI,
            "rho": RHO,
            "k_values": K_VALUES,
            "sae_d_sae": SAE_D_SAE,
            "tfa_width": TFA_WIDTH,
            "tfa_total_steps": TFA_TOTAL_STEPS,
            "sae_total_samples": SAE_TOTAL_SAMPLES,
            "scaling_factor": scaling_factor,
            "mean_norm": mean_norm,
            "seed": SEED,
        },
        "sae_results": sae_results,
        "tfa_results": tfa_results,
        "per_feature": per_feature_results,
    }
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
