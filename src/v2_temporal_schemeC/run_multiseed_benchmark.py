"""Multi-seed benchmark: SAE vs TFA with confidence intervals.

Runs the benchmark across multiple seeds to get error bars on
MSE and delta_MSE. Uses key k values to keep runtime manageable.
"""

import json
import math
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.shared.configs import TrainingConfig
from src.shared.train_sae import create_sae, train_sae
from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.data.toy.toy_model import ToyModel
from src.data.toy.markov import generate_markov_activations
from src.training.train_tfa import (
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

K_VALUES = [1, 2, 4, 10]  # Key k values only for efficiency

SEEDS = [42, 123, 456]

# Standard SAE training
SAE_TOTAL_SAMPLES = 10_000_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4
SAE_D_SAE = 40

# TFA training
TFA_TOTAL_STEPS = 15000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3
TFA_WIDTH = 40

EVAL_N_SAMPLES = 100_000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "multiseed_benchmark")


# ── Helpers (reused from run_extended_benchmark.py) ──────────────────


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
            n_seq = min(256, max(1, (n_samples - n_tokens) // SEQ_LEN))
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


def plot_mse_with_errorbars(agg, results_dir):
    """MSE vs k with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ks = sorted(agg["sae_mse_mean"].keys())
    sae_mean = [agg["sae_mse_mean"][k] for k in ks]
    sae_std = [agg["sae_mse_std"][k] for k in ks]
    tfa_mean = [agg["tfa_mse_mean"][k] for k in ks]
    tfa_std = [agg["tfa_mse_std"][k] for k in ks]

    ax.errorbar(ks, sae_mean, yerr=sae_std, fmt="o-", label="SAE",
                 color="tab:blue", linewidth=2, capsize=4)
    ax.errorbar(ks, tfa_mean, yerr=tfa_std, fmt="s-", label="TFA",
                 color="tab:orange", linewidth=2, capsize=4)
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="True L0")
    ax.set_xlabel("k (TopK sparsity)")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"MSE vs Sparsity (mean +/- std, n={len(SEEDS)} seeds)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(results_dir, "mse_vs_k_errorbars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_delta_mse_with_errorbars(agg, results_dir):
    """Delta MSE (SAE - TFA) with error bars. Positive = TFA wins."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ks = sorted(agg["delta_mse_mean"].keys())
    delta_mean = [agg["delta_mse_mean"][k] for k in ks]
    delta_std = [agg["delta_mse_std"][k] for k in ks]
    ax.errorbar(ks, delta_mean, yerr=delta_std, fmt="o-",
                 color="tab:orange", linewidth=2, capsize=4)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="True L0")
    ax.set_xlabel("k (TopK sparsity)")
    ax.set_ylabel("MSE(SAE) - MSE(TFA)")
    ax.set_title(f"TFA Advantage (mean +/- std, n={len(SEEDS)} seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, "delta_mse_errorbars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_per_feature_multiseed(agg, results_dir):
    """Per-feature pred_fraction vs rho with error bars across seeds."""
    if "pf_mean" not in agg or 2 not in agg["pf_mean"]:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    pf_mean = agg["pf_mean"][2]  # k=2
    pf_std = agg["pf_std"][2]
    rhos = RHO

    ax.errorbar(rhos, pf_mean, yerr=pf_std, fmt="o", color="tab:orange",
                 capsize=4, markersize=8, markeredgecolor="black", zorder=5)
    for i, (r, m, s) in enumerate(zip(rhos, pf_mean, pf_std)):
        ax.annotate(f"f{i}", (r, m), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Feature rho (autocorrelation)")
    ax.set_ylabel("Pred fraction (mean +/- std)")
    ax.set_title(f"Per-Feature Pred Fraction vs Rho (k=2, n={len(SEEDS)} seeds)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.0)
    plt.tight_layout()
    path = os.path.join(results_dir, "pred_fraction_multiseed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Aggregation ──────────────────────────────────────────────────────


def aggregate_results(all_seed_results):
    """Compute mean +/- std across seeds for all metrics."""
    sae_mse_by_k = defaultdict(list)
    tfa_mse_by_k = defaultdict(list)
    delta_mse_by_k = defaultdict(list)
    pred_energy_by_k = defaultdict(list)
    pf_by_k = defaultdict(list)

    for seed_data in all_seed_results:
        for sr in seed_data["sae_results"]:
            sae_mse_by_k[sr["k"]].append(sr["mse"])
        for tr in seed_data["tfa_results"]:
            tfa_mse_by_k[tr["k"]].append(tr["mse"])
            pred_energy_by_k[tr["k"]].append(tr["rel_energy_pred"])

        # Compute delta per seed (matched k): positive = TFA wins
        sae_map = {r["k"]: r["mse"] for r in seed_data["sae_results"]}
        tfa_map = {r["k"]: r["mse"] for r in seed_data["tfa_results"]}
        for k in sae_map:
            if k in tfa_map:
                delta_mse_by_k[k].append(sae_map[k] - tfa_map[k])

        # Per-feature
        for pf in seed_data.get("per_feature", []):
            pf_by_k[pf["k"]].append(pf["pred_fraction_per_feature"])

    agg = {
        "sae_mse_mean": {k: np.mean(v) for k, v in sae_mse_by_k.items()},
        "sae_mse_std": {k: np.std(v) for k, v in sae_mse_by_k.items()},
        "tfa_mse_mean": {k: np.mean(v) for k, v in tfa_mse_by_k.items()},
        "tfa_mse_std": {k: np.std(v) for k, v in tfa_mse_by_k.items()},
        "delta_mse_mean": {k: np.mean(v) for k, v in delta_mse_by_k.items()},
        "delta_mse_std": {k: np.std(v) for k, v in delta_mse_by_k.items()},
        "pred_energy_mean": {k: np.mean(v) for k, v in pred_energy_by_k.items()},
        "pred_energy_std": {k: np.std(v) for k, v in pred_energy_by_k.items()},
    }

    if pf_by_k:
        pf_mean = {}
        pf_std = {}
        for k, pf_lists in pf_by_k.items():
            arr = np.array(pf_lists)
            pf_mean[k] = arr.mean(axis=0).tolist()
            pf_std[k] = arr.std(axis=0).tolist()
        agg["pf_mean"] = pf_mean
        agg["pf_std"] = pf_std

    return agg


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    print(f"\nMulti-seed benchmark: {len(SEEDS)} seeds")
    print(f"Seeds: {SEEDS}")
    print(f"k values: {K_VALUES}")
    print(f"TFA training steps: {TFA_TOTAL_STEPS}\n")

    # Build toy model — use fixed seed 42 for model construction
    # so feature directions are identical across training seeds
    set_seed(42)
    model = ToyModel(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    model.eval()

    # Scaling factor (same across seeds since model is identical)
    scaling_factor, mean_norm = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"scaling_factor={scaling_factor:.4f}")

    gen_flat, gen_seq = make_data_generators(
        model, pi_t, rho_t, device, scaling_factor
    )

    all_seed_results = []
    t_total_start = time.time()

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'#' * 70}")
        print(f"  SEED {seed} ({seed_idx + 1}/{len(SEEDS)})")
        print(f"{'#' * 70}")

        sae_results = []
        tfa_results = []
        per_feature_results = []

        # SAE sweep
        for k in K_VALUES:
            set_seed(seed)
            t0 = time.time()
            sae = create_sae(HIDDEN_DIM, SAE_D_SAE, k=float(k), device=device)
            cfg = TrainingConfig(
                k=float(k), d_sae=SAE_D_SAE,
                total_training_samples=SAE_TOTAL_SAMPLES,
                batch_size=SAE_BATCH_SIZE, lr=SAE_LR, seed=seed,
            )
            sae = train_sae(sae, gen_flat, cfg, device)
            result = eval_standard_sae(sae, gen_flat, k, device)
            result["seed"] = seed
            sae_results.append(result)
            dt = time.time() - t0
            print(f"  SAE k={k:2d}: MSE={result['mse']:.6f} ({dt:.1f}s)")

        # TFA sweep
        for k in K_VALUES:
            set_seed(seed)
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
            result["seed"] = seed
            tfa_results.append(result)
            dt = time.time() - t0
            print(
                f"  TFA k={k:2d}: MSE={result['mse']:.6f}, "
                f"pred_E={result['rel_energy_pred']:.3f} ({dt:.1f}s)"
            )

            # Per-feature at k=2
            if k == 2:
                pf = per_feature_analysis(tfa, model, gen_seq, device)
                pf["k"] = k
                pf["seed"] = seed
                per_feature_results.append(pf)
                print(
                    f"    pred_frac: "
                    + " ".join(f"{v:.2f}" for v in pf["pred_fraction_per_feature"])
                )

        # Summary for this seed
        print(f"\n  Summary (seed={seed}):")
        print(f"  {'k':>3} | {'SAE MSE':>10} {'TFA MSE':>10} {'delta':>10}")
        print(f"  {'-' * 40}")
        for sr, tr in zip(sae_results, tfa_results):
            delta = sr["mse"] - tr["mse"]  # positive = TFA wins
            sign = "+" if delta >= 0 else ""
            print(f"  {sr['k']:>3} | {sr['mse']:>10.6f} {tr['mse']:>10.6f} {sign}{delta:>9.6f}")

        all_seed_results.append({
            "sae_results": sae_results,
            "tfa_results": tfa_results,
            "per_feature": per_feature_results,
            "scaling_factor": scaling_factor,
            "mean_norm": mean_norm,
        })

    total_time = time.time() - t_total_start
    print(f"\n{'=' * 70}")
    print(f"Total runtime: {total_time / 60:.1f} min")
    print(f"{'=' * 70}")

    # Aggregate
    agg = aggregate_results(all_seed_results)

    # Print aggregated summary
    print(f"\nAGGREGATED RESULTS (n={len(SEEDS)} seeds)")
    print(f"{'k':>3} | {'SAE MSE':>16} {'TFA MSE':>16} {'delta_MSE':>18}")
    print("-" * 60)
    for k in sorted(agg["delta_mse_mean"].keys()):
        sae_m = agg["sae_mse_mean"][k]
        sae_s = agg["sae_mse_std"][k]
        tfa_m = agg["tfa_mse_mean"][k]
        tfa_s = agg["tfa_mse_std"][k]
        d_m = agg["delta_mse_mean"][k]
        d_s = agg["delta_mse_std"][k]
        sign = "+" if d_m >= 0 else ""
        print(
            f"{k:>3} | "
            f"{sae_m:.6f}+/-{sae_s:.6f} "
            f"{tfa_m:.6f}+/-{tfa_s:.6f} "
            f"{sign}{d_m:.6f}+/-{d_s:.6f}"
        )

    # Per-feature summary
    if "pf_mean" in agg and 2 in agg["pf_mean"]:
        print(f"\nPER-FEATURE PRED FRACTION (k=2, mean +/- std across seeds)")
        means = agg["pf_mean"][2]
        stds = agg["pf_std"][2]
        for i, (m, s, r) in enumerate(zip(means, stds, RHO)):
            print(f"  f{i} (rho={r:.2f}): {m:.3f} +/- {s:.3f}")

    # Plots
    plot_mse_with_errorbars(agg, RESULTS_DIR)
    plot_delta_mse_with_errorbars(agg, RESULTS_DIR)
    plot_per_feature_multiseed(agg, RESULTS_DIR)

    # Save
    results_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "pi": PI,
            "rho": RHO,
            "k_values": K_VALUES,
            "seeds": SEEDS,
            "sae_d_sae": SAE_D_SAE,
            "tfa_width": TFA_WIDTH,
            "tfa_total_steps": TFA_TOTAL_STEPS,
            "sae_total_samples": SAE_TOTAL_SAMPLES,
            "scaling_factor": scaling_factor,
        },
        "per_seed": all_seed_results,
        "aggregated": agg,
    }
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
