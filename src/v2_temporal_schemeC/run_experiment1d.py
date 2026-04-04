"""Experiment 1d: TopK sweep at limiting correlations (ρ=0 and ρ=1).

Runs the full 12-point k sweep for 5 models at the two extreme temporal
correlation values:
  - ρ=0.0: i.i.d. features, no temporal structure
  - ρ=1.0: frozen features, constant support across entire sequence

Usage:
  TQDM_DISABLE=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_experiment1d.py
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
from src.v2_temporal_schemeC.experiment import (
    DataConfig, run_topk_sweep, save_results,
    TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec, ModelEntry,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]
# TXCDRv2 T=5: k*T > d_sae=40 at k>=9, so cap at k=8
K_VALUES_TXCDRV2_T5 = [k for k in K_VALUES if k * 5 <= 40]  # [1..8]

NUM_FEATURES = 20
HIDDEN_DIM = 40
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "experiment1d_limiting_rho")

# Models that work with all k values
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

# TXCDRv2 T=5 runs separately with capped k values
MODELS_T5 = [
    ModelEntry("TXCDRv2-T5", TXCDRv2ModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
]


def make_data_config(rho: float) -> DataConfig:
    return DataConfig(
        num_features=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        seq_len=64,
        pi=[0.5] * NUM_FEATURES,
        rho=[rho] * NUM_FEATURES,
        dict_width=HIDDEN_DIM,
        seed=42,
        eval_n_seq=2000,
    )


def run_sweep_for_rho(rho: float, cache_dir: str) -> dict[str, list]:
    """Run full TopK sweep for one rho value. Returns {model_name: [EvalResult]}."""
    print(f"\n{'='*70}", flush=True)
    print(f"ρ = {rho}", flush=True)
    print(f"{'='*70}", flush=True)

    cfg = make_data_config(rho)

    # Run main models (all k values)
    print(f"\n  Main models (k={K_VALUES}):", flush=True)
    results_main = run_topk_sweep(
        models=MODELS_MAIN,
        k_values=K_VALUES,
        data_config=cfg,
        device=DEVICE,
        compute_auc=True,
        cache_dir=cache_dir,
    )

    # Run TXCDRv2-T5 with capped k values
    print(f"\n  TXCDRv2-T5 (k={K_VALUES_TXCDRV2_T5}):", flush=True)
    results_t5 = run_topk_sweep(
        models=MODELS_T5,
        k_values=K_VALUES_TXCDRV2_T5,
        data_config=cfg,
        device=DEVICE,
        compute_auc=True,
        cache_dir=cache_dir,
    )

    # Merge
    all_results = {**results_main, **results_t5}
    return all_results


def results_to_serializable(results: dict, k_values_map: dict[str, list[int]]) -> dict:
    """Convert EvalResult objects to dicts with k values attached."""
    out = {}
    for name, eval_results in results.items():
        ks = k_values_map[name]
        out[name] = [r.to_dict() | {"k": ks[i]} for i, r in enumerate(eval_results)]
    return out


def plot_sweep(results_ser: dict, rho: float, results_dir: str):
    """Plot NMSE and AUC vs k for one rho value."""
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
        nmse = [e["nmse"] for e in entries]
        auc = [e["auc"] for e in entries]

        axes[0].plot(ks, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=8, label=name)
        axes[1].plot(ks, auc, marker=m, linestyle=ls, color=c, lw=2, ms=8, label=name)

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


def load_experiment1_results(repro_dir: str) -> dict:
    """Load Experiment 1 reproduction results for comparison overlay."""
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


def plot_comparison(rho0_ser: dict, rho1_ser: dict, results_dir: str):
    """Overlay ρ=0, mixed-ρ (Experiment 1), and ρ=1 for selected models."""
    repro_dir = os.path.join(os.path.dirname(__file__), "results", "reproduction")
    exp1 = load_experiment1_results(repro_dir)

    # Models to compare (use names matching both experiments)
    compare_models = {
        "TFA-pos":    {"color": "tab:brown",  "marker": "X"},
        "Stacked-T2": {"color": "tab:green",  "marker": "D"},
        "TXCDRv2-T2": {"color": "tab:purple", "marker": "P"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, style in compare_models.items():
        c, m = style["color"], style["marker"]

        # ρ=0
        if model_name in rho0_ser:
            ks = [e["k"] for e in rho0_ser[model_name]]
            axes[0].plot(ks, [e["nmse"] for e in rho0_ser[model_name]],
                         marker=m, ls="-", color=c, lw=2, ms=7, alpha=0.5,
                         label=f"{model_name} ρ=0")
            axes[1].plot(ks, [e["auc"] for e in rho0_ser[model_name]],
                         marker=m, ls="-", color=c, lw=2, ms=7, alpha=0.5)

        # Mixed ρ (Experiment 1)
        if model_name in exp1:
            ks = [e["k"] for e in exp1[model_name]]
            axes[0].plot(ks, [e["nmse"] for e in exp1[model_name]],
                         marker=m, ls="-", color=c, lw=2.5, ms=7,
                         label=f"{model_name} mixed-ρ")
            axes[1].plot(ks, [e["auc"] for e in exp1[model_name]],
                         marker=m, ls="-", color=c, lw=2.5, ms=7)

        # ρ=1
        if model_name in rho1_ser:
            ks = [e["k"] for e in rho1_ser[model_name]]
            axes[0].plot(ks, [e["nmse"] for e in rho1_ser[model_name]],
                         marker=m, ls="--", color=c, lw=2, ms=7, alpha=0.5,
                         label=f"{model_name} ρ=1")
            axes[1].plot(ks, [e["auc"] for e in rho1_ser[model_name]],
                         marker=m, ls="--", color=c, lw=2, ms=7, alpha=0.5)

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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    print(f"Device: {DEVICE}", flush=True)
    print(f"Results dir: {RESULTS_DIR}", flush=True)
    t_start = time.time()

    # k values per model for serialization
    k_map_main = {name: K_VALUES for name in
                  ["TFA-pos", "Stacked-T2", "Stacked-T5", "TXCDRv2-T2"]}
    k_map_t5 = {"TXCDRv2-T5": K_VALUES_TXCDRV2_T5}
    k_map_all = {**k_map_main, **k_map_t5}

    # ── ρ = 0.0 ──────────────────────────────────────────────────────
    results_rho0 = run_sweep_for_rho(0.0, cache_dir)
    rho0_ser = results_to_serializable(results_rho0, k_map_all)

    rho0_path = os.path.join(RESULTS_DIR, "rho0_results.json")
    with open(rho0_path, "w") as f:
        json.dump({"rho": 0.0, "results": rho0_ser}, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nSaved ρ=0 results to {rho0_path}", flush=True)

    # ── ρ = 1.0 ──────────────────────────────────────────────────────
    results_rho1 = run_sweep_for_rho(1.0, cache_dir)
    rho1_ser = results_to_serializable(results_rho1, k_map_all)

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
    print(f"\nExperiment 1d complete in {elapsed/60:.1f}m", flush=True)
    print(f"Results in {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
