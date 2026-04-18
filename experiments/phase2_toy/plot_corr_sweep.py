"""Regenerate correlation sweep plots from all JSON data files.

Combines per-token models (from results.json) with windowed models
(from *_corr.json files) into unified plots.

Usage:
  PYTHONPATH=/home/elysium/temp_xc python src/v2_temporal_schemeC/plot_corr_sweep.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting.save_figure import save_figure

BASE = os.path.join(os.path.dirname(__file__), "results", "correlation_sweep")
RHO_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]
K_VALUES = [3, 10]

# Style: same method = same color, T=2 solid, T=5 dashed
MODELS = [
    # Per-token models (from results.json)
    ("SAE",          "#1f77b4", "o", "-"),    # blue
    ("TFA",          "#ff7f0e", "s", "-"),    # orange
    ("TFA-shuf",     "#ff7f0e", "s", "--"),   # orange dashed
    ("TFA-pos",      "#2ca02c", "X", "-"),    # green
    ("TFA-pos-shuf", "#2ca02c", "X", "--"),   # green dashed
    # Windowed models (from *_corr.json)
    ("Stacked-T2",   "#9467bd", "D", "-"),    # purple solid
    ("Stacked-T5",   "#9467bd", "D", "--"),   # purple dashed
    ("TXCDR-T2",     "#d62728", "o", "-"),    # red solid
    ("TXCDR-T5",     "#d62728", "^", "--"),   # red dashed
    ("TXCDRv2-T2",   "#e377c2", "o", "-"),   # pink solid
    ("TXCDRv2-T5",   "#e377c2", "^", "--"),  # pink dashed
]


def load_all_data():
    """Load all correlation sweep data into unified structure.

    Returns: {rho: {model_name: [{k, nmse, auc, ...}, ...]}}
    """
    all_data = {rho: {} for rho in RHO_VALUES}

    # Per-token models from results.json
    path = os.path.join(BASE, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        for rho_str, models in d["results"].items():
            rho = float(rho_str)
            if rho in all_data:
                for model_name, entries in models.items():
                    all_data[rho][model_name] = entries

    # Also check per-rho files
    for rho in RHO_VALUES:
        path = os.path.join(BASE, f"rho_{rho}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            for model_name, entries in d["results"].items():
                if model_name not in all_data[rho]:
                    all_data[rho][model_name] = entries

    # Windowed models from *_corr.json
    for fname in os.listdir(BASE):
        if not fname.endswith("_corr.json"):
            continue
        model_name = fname.replace("_corr.json", "")
        with open(os.path.join(BASE, fname)) as f:
            d = json.load(f)
        for rho_str, entries in d["results"].items():
            rho = float(rho_str)
            if rho in all_data:
                all_data[rho][model_name] = entries

    return all_data


def main():
    all_data = load_all_data()

    for k in K_VALUES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for model_name, c, m, ls in MODELS:
            nmse_vals = []
            auc_vals = []
            rho_vals = []

            for rho in RHO_VALUES:
                if model_name not in all_data[rho]:
                    continue
                entries = all_data[rho][model_name]
                for e in entries:
                    if e["k"] == k:
                        nmse_vals.append(e["nmse"])
                        auc_vals.append(e["auc"])
                        rho_vals.append(rho)
                        break

            if nmse_vals:
                axes[0].plot(rho_vals, nmse_vals, marker=m, linestyle=ls,
                             color=c, lw=2, ms=7, label=model_name)
                axes[1].plot(rho_vals, auc_vals, marker=m, linestyle=ls,
                             color=c, lw=2, ms=7, label=model_name)

        axes[0].set(xlabel="ρ", ylabel="NMSE", title=f"NMSE vs ρ (k={k})")
        axes[0].set_yscale("log")
        axes[1].set(xlabel="ρ", ylabel="AUC (decoder-averaged)",
                    title=f"Feature Recovery AUC vs ρ (k={k})")
        for ax in axes:
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Correlation sweep (k={k}, all features at same ρ)", fontsize=13)
        plt.tight_layout()
        save_figure(fig, os.path.join(BASE, f"correlation_sweep_k{k}.png"))
        plt.close(fig)
        print(f"Saved correlation_sweep_k{k}.png")


if __name__ == "__main__":
    main()
