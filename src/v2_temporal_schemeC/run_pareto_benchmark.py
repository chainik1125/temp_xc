"""Phase 2-3: Full Pareto sweep + per-feature diagnostics.

Trains standard SAE and TFA at each k value, produces:
1. MSE-vs-k Pareto frontier comparison
2. TFA energy decomposition
3. Per-feature analysis: which ground-truth features are captured by
   the predictable vs novel component, as a function of rho_i
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
K_VALUES = [1, 2, 3, 4, 6, 8, 10]

# Standard SAE training
SAE_TOTAL_SAMPLES = 10_000_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4
SAE_D_SAE = 40

# TFA training — 30K steps based on convergence study
TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3
TFA_WIDTH = 40
TFA_LOG_EVERY = 5000

EVAL_N_SAMPLES = 100_000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "pareto_benchmark")


# ── Scaling ─────────────────────────────────────────────────────────


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


# ── Data Generation ──────────────────────────────────────────────────


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


def generate_eval_data_with_support(model, pi_t, rho_t, device, scaling_factor,
                                     n_seq=2000):
    """Generate evaluation data that also returns ground-truth support."""
    acts, support = generate_markov_activations(
        n_seq, SEQ_LEN, pi_t, rho_t, device=device
    )
    hidden = model(acts) * scaling_factor
    return hidden, support  # (n_seq, T, d), (n_seq, T, n_features)


# ── Evaluation ───────────────────────────────────────────────────────


def eval_standard_sae(sae, gen_fn, device, n_samples=EVAL_N_SAMPLES):
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
    return total_mse / n_processed


def eval_tfa_basic(tfa, gen_fn, device, n_samples=EVAL_N_SAMPLES):
    """Basic TFA evaluation: MSE and energy decomposition."""
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
    return {
        "mse": mse,
        "rel_energy_pred": pred_energy_sum / total_energy,
        "novel_l0": novel_l0_sum / n_tokens,
    }


def eval_tfa_per_feature(tfa, model, hidden_data, support, device):
    """Per-feature analysis: measure how much of each ground-truth feature
    is captured by the predictable vs novel component.

    For each token where feature i is active, we project the predictable
    and novel reconstructions onto feature direction f_i and measure
    the fraction of the true projection explained by each component.
    """
    tfa.eval()
    feature_dirs = model.feature_directions.to(device)  # (n_features, d)
    n_features = feature_dirs.shape[0]

    # Process in batches
    pred_proj_sums = torch.zeros(n_features, device=device)
    novel_proj_sums = torch.zeros(n_features, device=device)
    total_proj_sums = torch.zeros(n_features, device=device)
    feature_counts = torch.zeros(n_features, device=device)

    batch_size = 256
    n_seq = hidden_data.shape[0]

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            x_batch = hidden_data[start:end].to(device)  # (B, T, d)
            s_batch = support[start:end].to(device)        # (B, T, n_features)

            recons, intermediates = tfa(x_batch)
            pred_r = intermediates["pred_recons"]   # (B, T, d)
            novel_r = intermediates["novel_recons"]  # (B, T, d)

            # Project onto each feature direction
            # feature_dirs: (n_features, d) -> (1, 1, n_features, d)
            fd = feature_dirs.unsqueeze(0).unsqueeze(0)

            # x_batch: (B, T, d) -> (B, T, 1, d)
            x_exp = x_batch.unsqueeze(2)
            pred_exp = pred_r.unsqueeze(2)
            novel_exp = novel_r.unsqueeze(2)

            # Projections: (B, T, n_features)
            x_proj = (x_exp * fd).sum(dim=-1)
            pred_proj = (pred_exp * fd).sum(dim=-1)
            novel_proj = (novel_exp * fd).sum(dim=-1)

            # Only count where feature is active
            active = s_batch > 0  # (B, T, n_features)

            for i in range(n_features):
                mask = active[:, :, i]  # (B, T)
                if mask.any():
                    pred_proj_sums[i] += pred_proj[:, :, i][mask].abs().sum()
                    novel_proj_sums[i] += novel_proj[:, :, i][mask].abs().sum()
                    total_proj_sums[i] += x_proj[:, :, i][mask].abs().sum()
                    feature_counts[i] += mask.float().sum()

    # Compute fractions
    results = []
    for i in range(n_features):
        total = total_proj_sums[i].item()
        if total > 0:
            pred_frac = pred_proj_sums[i].item() / total
            novel_frac = novel_proj_sums[i].item() / total
        else:
            pred_frac = novel_frac = 0.0
        results.append({
            "feature": i,
            "rho": RHO[i],
            "pred_frac": pred_frac,
            "novel_frac": novel_frac,
            "count": int(feature_counts[i].item()),
        })
    return results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_pareto(sae_results, tfa_results, results_dir):
    """MSE vs k Pareto frontier."""
    ks = [r["k"] for r in sae_results]
    mse_sae = [r["mse"] for r in sae_results]
    mse_tfa = [r["mse"] for r in tfa_results]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, mse_sae, "o-", label="Standard SAE", color="tab:blue",
            linewidth=2, markersize=8)
    ax.plot(ks, mse_tfa, "s-", label="TFA", color="tab:orange",
            linewidth=2, markersize=8)

    true_l0 = sum(PI)
    ax.axvline(x=true_l0, color="gray", linestyle="--", alpha=0.6,
               label=f"$E[L_0]$ = {true_l0}")

    ax.set_xlabel("k (TopK sparsity)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Reconstruction MSE vs Sparsity: TFA vs Standard SAE", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"pareto_mse_vs_k.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tfa_decomposition(tfa_results, results_dir):
    """TFA energy decomposition and novel L0."""
    ks = [r["k"] for r in tfa_results]
    rel_pred = [r["rel_energy_pred"] for r in tfa_results]
    novel_l0 = [r["novel_l0"] for r in tfa_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar([str(k) for k in ks], rel_pred,
            color="tab:green", alpha=0.8, label="Predictable")
    ax1.bar([str(k) for k in ks], [1 - r for r in rel_pred],
            bottom=rel_pred, color="tab:red", alpha=0.8, label="Novel")
    ax1.set_xlabel("k (TopK)")
    ax1.set_ylabel("Relative energy")
    ax1.set_title("TFA Energy Decomposition")
    ax1.legend()
    ax1.set_ylim(0, 1)

    ax2.plot(ks, novel_l0, "o-", color="tab:red", linewidth=2)
    ax2.set_xlabel("k (TopK)")
    ax2.set_ylabel("Novel L0")
    ax2.set_title("TFA Novel Code Sparsity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"tfa_decomposition.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_feature(per_feature_results, k, results_dir):
    """Per-feature predictable vs novel fraction, ordered by rho."""
    # Sort by rho
    sorted_results = sorted(per_feature_results, key=lambda r: r["rho"])
    labels = [f"f{r['feature']}\n($\\rho$={r['rho']})" for r in sorted_results]
    pred_fracs = [r["pred_frac"] for r in sorted_results]
    novel_fracs = [r["novel_frac"] for r in sorted_results]
    rhos = [r["rho"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.6

    ax.bar(x, pred_fracs, width, label="Predictable", color="tab:green", alpha=0.8)
    ax.bar(x, novel_fracs, width, bottom=pred_fracs,
           label="Novel", color="tab:red", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Fraction of feature energy")
    ax.set_title(f"Per-Feature Decomposition (k={k}): Predictable vs Novel")
    ax.legend()
    ax.set_ylim(0, max(1.2, max(p + n for p, n in zip(pred_fracs, novel_fracs)) * 1.1))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"per_feature_k{k}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pred_frac_vs_rho(all_per_feature, results_dir):
    """Scatter: predictable fraction vs rho across all k values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_per_feature)))

    for idx, (k, pf_results) in enumerate(sorted(all_per_feature.items())):
        rhos = [r["rho"] for r in pf_results]
        pred_fracs = [r["pred_frac"] for r in pf_results]
        ax.scatter(rhos, pred_fracs, color=colors[idx], s=60, alpha=0.8,
                   label=f"k={k}", zorder=3)
        # Connect points with same rho
        unique_rhos = sorted(set(rhos))
        for rho in unique_rhos:
            vals = [r["pred_frac"] for r in pf_results if r["rho"] == rho]
            ax.plot([rho] * len(vals), vals, ".", color=colors[idx],
                    markersize=3, zorder=2)

    ax.set_xlabel("Feature autocorrelation ($\\rho$)", fontsize=12)
    ax.set_ylabel("Predictable fraction of feature energy", fontsize=12)
    ax.set_title("TFA Routes High-$\\rho$ Features to Prediction", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0, 1.5)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"pred_frac_vs_rho.{ext}"),
                    dpi=150, bbox_inches="tight")
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
    print(f"rho: {RHO}")
    print(f"True L0: {true_l0}, k values: {K_VALUES}")
    print(f"TFA steps: {TFA_TOTAL_STEPS}\n")

    # ── Build toy model ──
    model = ToyModel(
        num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM,
    ).to(device)
    model.eval()

    scaling_factor, mean_norm = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling: mean_norm={mean_norm:.4f}, factor={scaling_factor:.4f}\n")

    gen_flat, gen_seq = make_data_generators(
        model, pi_t, rho_t, device, scaling_factor
    )

    # Generate per-feature eval data once
    set_seed(SEED + 100)
    eval_hidden, eval_support = generate_eval_data_with_support(
        model, pi_t, rho_t, device, scaling_factor, n_seq=2000
    )

    # ── Standard SAE sweep ──
    print("=" * 60)
    print("STANDARD SAE SWEEP")
    print("=" * 60)

    sae_results = []
    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n--- SAE k={k} ---")
        t0 = time.time()
        sae = create_sae(HIDDEN_DIM, SAE_D_SAE, k=float(k), device=device)
        cfg = TrainingConfig(
            k=float(k), d_sae=SAE_D_SAE,
            total_training_samples=SAE_TOTAL_SAMPLES,
            batch_size=SAE_BATCH_SIZE, lr=SAE_LR, seed=SEED,
        )
        sae = train_sae(sae, gen_flat, cfg, device)
        mse = eval_standard_sae(sae, gen_flat, device)
        dt = time.time() - t0
        print(f"  MSE={mse:.6f} ({dt:.1f}s)")
        sae_results.append({"model": "SAE", "k": k, "mse": mse})

    # ── TFA sweep ──
    print("\n" + "=" * 60)
    print("TFA SWEEP")
    print("=" * 60)

    tfa_results = []
    all_per_feature = {}
    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n--- TFA k={k} ---")
        t0 = time.time()

        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=TFA_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS,
            batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR,
            log_every=TFA_LOG_EVERY,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)

        # Basic eval
        basic = eval_tfa_basic(tfa, gen_seq, device)
        dt = time.time() - t0
        print(f"  MSE={basic['mse']:.6f}, pred_E={basic['rel_energy_pred']:.3f}, "
              f"novel_L0={basic['novel_l0']:.2f} ({dt:.1f}s)")

        tfa_results.append({
            "model": "TFA", "k": k,
            "mse": basic["mse"],
            "rel_energy_pred": basic["rel_energy_pred"],
            "novel_l0": basic["novel_l0"],
        })

        # Per-feature diagnostics
        print(f"  Running per-feature analysis...")
        pf = eval_tfa_per_feature(tfa, model, eval_hidden, eval_support, device)
        all_per_feature[k] = pf
        for r in sorted(pf, key=lambda x: x["rho"]):
            print(f"    f{r['feature']} (rho={r['rho']:.2f}): "
                  f"pred={r['pred_frac']:.3f}, novel={r['novel_frac']:.3f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PARETO SUMMARY")
    print("=" * 60)
    print(f"{'k':>3} | {'SAE MSE':>10} | {'TFA MSE':>10} {'pred_E':>7} | "
          f"{'delta':>10} {'winner':>6}")
    print("-" * 65)
    for sr, tr in zip(sae_results, tfa_results):
        delta = sr["mse"] - tr["mse"]
        sign = "+" if delta >= 0 else ""
        winner = "TFA" if delta > 0 else "SAE"
        print(f"{sr['k']:>3} | {sr['mse']:>10.6f} | {tr['mse']:>10.6f} "
              f"{tr['rel_energy_pred']:>7.3f} | {sign}{delta:>9.6f} {winner:>6}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_pareto(sae_results, tfa_results, RESULTS_DIR)
    plot_tfa_decomposition(tfa_results, RESULTS_DIR)
    for k, pf in all_per_feature.items():
        plot_per_feature(pf, k, RESULTS_DIR)
    plot_pred_frac_vs_rho(all_per_feature, RESULTS_DIR)

    # ── Save ──
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
            "scaling_factor": scaling_factor,
            "seed": SEED,
        },
        "sae_results": sae_results,
        "tfa_results": tfa_results,
        "per_feature": {str(k): v for k, v in all_per_feature.items()},
    }
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
