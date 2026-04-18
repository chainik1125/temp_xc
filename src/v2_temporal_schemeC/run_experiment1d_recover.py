"""Recovery script for Experiment 1d: re-run ρ=1 sweep and generate all plots.

The main run crashed during TXCDRv2-T5 at ρ=1. Main model training is cached.
This script re-runs ρ=1 (loading cached main models) and generates all plots.

Usage:
  TQDM_DISABLE=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_experiment1d_recover.py
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils.plot import save_figure
from src.pipeline.toy_models import (DataConfig, run_topk_sweep, TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec, ModelEntry)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]
K_VALUES_TXCDRV2_T5 = [k for k in K_VALUES if k * 5 <= 40]

NUM_FEATURES = 20
HIDDEN_DIM = 40
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "experiment1d_limiting_rho")

MODELS_MAIN = [
    ModelEntry("TFA-pos", TFAModelSpec(use_pos_encoding=True), "seq",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("Stacked-T2", StackedSAEModelSpec(T=2), "window_2",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("Stacked-T5", StackedSAEModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("TXCDRv2-T2", TXCDRv2ModelSpec(T=2), "window_2",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
]

MODELS_T5 = [
    ModelEntry("TXCDRv2-T5", TXCDRv2ModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
]


def make_data_config(rho: float) -> DataConfig:
    return DataConfig(
        num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, seq_len=64,
        pi=[0.5] * NUM_FEATURES, rho=[rho] * NUM_FEATURES,
        dict_width=HIDDEN_DIM, seed=42, eval_n_seq=2000,
    )


def results_to_serializable(results, k_values_map):
    out = {}
    for name, eval_results in results.items():
        ks = k_values_map[name]
        out[name] = [r.to_dict() | {"k": ks[i]} for i, r in enumerate(eval_results)]
    return out


def plot_sweep(results_ser, rho, results_dir):
    model_style = {
        "TFA-pos":      ("tab:brown",  "X",  "-"),
        "Stacked-T2":   ("tab:green",  "D",  "-"),
        "Stacked-T5":   ("tab:green",  "D",  "--"),
        "TXCDRv2-T2":   ("tab:purple", "P",  "-"),
        "TXCDRv2-T5":   ("tab:purple", "P",  "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, entries in results_ser.items():
        c, m, ls = model_style.get(name, ("gray", "o", "-"))
        ks = [e["k"] for e in entries]
        axes[0].plot(ks, [e["nmse"] for e in entries], marker=m, ls=ls, color=c, lw=2, ms=8, label=name)
        axes[1].plot(ks, [e["auc"] for e in entries], marker=m, ls=ls, color=c, lw=2, ms=8, label=name)
    rho_label = "0" if rho == 0.0 else "1"
    axes[0].set(xlabel="k (TopK)", ylabel="NMSE", title=f"NMSE vs k (ρ={rho})")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="k (TopK)", ylabel="AUC", title=f"AUC vs k (ρ={rho})")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Experiment 1d: TopK sweep at ρ={rho}", fontsize=13)
    plt.tight_layout()
    save_figure(fig, os.path.join(results_dir, f"exp1d_rho{rho_label}_nmse_auc.png"))
    plt.close(fig)


def load_experiment1_results(repro_dir):
    results = {}
    for fname in os.listdir(repro_dir):
        if not fname.endswith(".json"):
            continue
        name = fname.replace(".json", "")
        with open(os.path.join(repro_dir, fname)) as f:
            data = json.load(f)
        if "topk" in data:
            results[name] = data["topk"]
    return results


def plot_comparison(rho0_ser, rho1_ser, results_dir):
    repro_dir = os.path.join(os.path.dirname(__file__), "results", "reproduction")
    exp1 = load_experiment1_results(repro_dir)
    compare_models = {
        "TFA-pos":    {"color": "tab:brown",  "marker": "X"},
        "Stacked-T2": {"color": "tab:green",  "marker": "D"},
        "TXCDRv2-T2": {"color": "tab:purple", "marker": "P"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, style in compare_models.items():
        c, m = style["color"], style["marker"]
        if model_name in rho0_ser:
            ks = [e["k"] for e in rho0_ser[model_name]]
            lbl = f"{model_name} ρ=0"
            axes[0].plot(ks, [e["nmse"] for e in rho0_ser[model_name]],
                         marker=m, ls="-", color=c, lw=2, ms=7, alpha=0.5, label=lbl)
            axes[1].plot(ks, [e["auc"] for e in rho0_ser[model_name]],
                         marker=m, ls="-", color=c, lw=2, ms=7, alpha=0.5, label=lbl)
        if model_name in exp1:
            ks = [e["k"] for e in exp1[model_name]]
            lbl = f"{model_name} mixed-ρ"
            axes[0].plot(ks, [e["nmse"] for e in exp1[model_name]],
                         marker=m, ls="-", color=c, lw=2.5, ms=7, label=lbl)
            axes[1].plot(ks, [e["auc"] for e in exp1[model_name]],
                         marker=m, ls="-", color=c, lw=2.5, ms=7, label=lbl)
        if model_name in rho1_ser:
            ks = [e["k"] for e in rho1_ser[model_name]]
            lbl = f"{model_name} ρ=1"
            axes[0].plot(ks, [e["nmse"] for e in rho1_ser[model_name]],
                         marker=m, ls="--", color=c, lw=2, ms=7, alpha=0.5, label=lbl)
            axes[1].plot(ks, [e["auc"] for e in rho1_ser[model_name]],
                         marker=m, ls="--", color=c, lw=2, ms=7, alpha=0.5, label=lbl)
    axes[0].set(xlabel="k (TopK)", ylabel="NMSE", title="NMSE: ρ=0 vs mixed-ρ vs ρ=1")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="k (TopK)", ylabel="AUC", title="AUC: ρ=0 vs mixed-ρ vs ρ=1")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Experiment 1d: Comparison across ρ regimes", fontsize=13)
    plt.tight_layout()
    save_figure(fig, os.path.join(results_dir, "exp1d_comparison.png"))
    plt.close(fig)


def main():
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    print(f"Device: {DEVICE}", flush=True)
    t_start = time.time()

    k_map_main = {name: K_VALUES for name in
                  ["TFA-pos", "Stacked-T2", "Stacked-T5", "TXCDRv2-T2"]}
    k_map_t5 = {"TXCDRv2-T5": K_VALUES_TXCDRV2_T5}
    k_map_all = {**k_map_main, **k_map_t5}

    # Load existing ρ=0 results
    rho0_path = os.path.join(RESULTS_DIR, "rho0_results.json")
    with open(rho0_path) as f:
        rho0_ser = json.load(f)["results"]
    print(f"Loaded ρ=0 results from {rho0_path}", flush=True)

    # ── Re-run ρ=1 (cached main models + retrain TXCDRv2-T5) ─────────
    cfg = make_data_config(1.0)
    print(f"\n{'='*70}\nρ = 1.0 (recovery)\n{'='*70}", flush=True)

    print(f"\n  Main models (cached):", flush=True)
    results_main = run_topk_sweep(
        models=MODELS_MAIN, k_values=K_VALUES, data_config=cfg,
        device=DEVICE, compute_auc=True, cache_dir=cache_dir,
    )

    print(f"\n  TXCDRv2-T5 (k={K_VALUES_TXCDRV2_T5}):", flush=True)
    results_t5 = run_topk_sweep(
        models=MODELS_T5, k_values=K_VALUES_TXCDRV2_T5, data_config=cfg,
        device=DEVICE, compute_auc=True, cache_dir=cache_dir,
    )

    all_results = {**results_main, **results_t5}
    rho1_ser = results_to_serializable(all_results, k_map_all)

    rho1_path = os.path.join(RESULTS_DIR, "rho1_results.json")
    with open(rho1_path, "w") as f:
        json.dump({"rho": 1.0, "results": rho1_ser}, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nSaved ρ=1 results to {rho1_path}", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────
    plot_sweep(rho0_ser, 0.0, RESULTS_DIR)
    plot_sweep(rho1_ser, 1.0, RESULTS_DIR)
    plot_comparison(rho0_ser, rho1_ser, RESULTS_DIR)

    elapsed = time.time() - t_start
    print(f"\nRecovery complete in {elapsed/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
