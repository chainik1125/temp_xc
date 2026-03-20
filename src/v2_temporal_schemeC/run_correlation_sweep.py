"""Sweep temporal correlation ρ with all features at the same value.

Tests how model performance changes as temporal persistence increases,
with all 20 features sharing one ρ value.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_correlation_sweep.py
"""

import json
import os
import sys
import time
from dataclasses import asdict

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils.plot import save_figure
from src.v2_temporal_schemeC.experiment import (
    DataConfig, SAEModelSpec, TFAModelSpec, ModelEntry,
    run_topk_sweep, save_results,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RHO_VALUES = [0.0, 0.3, 0.5, 0.7, 0.9]
K_VALUES = [3, 10]
NUM_FEATURES = 20
HIDDEN_DIM = 40
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "correlation_sweep")

MODELS = [
    ModelEntry("SAE", SAEModelSpec(), "flat",
               training_overrides={"total_steps": 30_000, "batch_size": 4096, "lr": 3e-4}),
    ModelEntry("TFA", TFAModelSpec(), "seq",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("TFA-shuf", TFAModelSpec(), "seq_shuffled",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("TFA-pos", TFAModelSpec(use_pos_encoding=True), "seq",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("TFA-pos-shuf", TFAModelSpec(use_pos_encoding=True), "seq_shuffled",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"ρ values: {RHO_VALUES}", flush=True)
    print(f"k values: {K_VALUES}", flush=True)
    t_start = time.time()

    all_results = {}

    for rho in RHO_VALUES:
        print(f"\n{'='*70}", flush=True)
        print(f"ρ = {rho}", flush=True)
        print(f"{'='*70}", flush=True)

        cfg = make_data_config(rho)
        results = run_topk_sweep(
            models=MODELS,
            k_values=K_VALUES,
            data_config=cfg,
            device=DEVICE,
        )
        all_results[rho] = {
            name: [r.to_dict() | {"k": K_VALUES[i]} for i, r in enumerate(rs)]
            for name, rs in results.items()
        }

    # Save
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"rho_values": RHO_VALUES, "k_values": K_VALUES,
                    "results": {str(k): v for k, v in all_results.items()}},
                  f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    # ── Plot ──────────────────────────────────────────────────────────

    model_style = {
        "SAE":          ("tab:blue",   "o", "-"),
        "TFA":          ("tab:orange", "s", "-"),
        "TFA-shuf":     ("tab:red",    "^", "--"),
        "TFA-pos":      ("tab:brown",  "X", "-"),
        "TFA-pos-shuf": ("tab:pink",   "v", "--"),
    }

    for ki, k in enumerate(K_VALUES):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for name, (c, m, ls) in model_style.items():
            nmse = [all_results[rho][name][ki]["nmse"] for rho in RHO_VALUES]
            auc = [all_results[rho][name][ki]["auc"] for rho in RHO_VALUES]
            axes[0].plot(RHO_VALUES, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=8, label=name)
            axes[1].plot(RHO_VALUES, auc, marker=m, linestyle=ls, color=c, lw=2, ms=8, label=name)

        # Temporal fraction
        for base, shuf, label, c, m in [
            ("TFA", "TFA-shuf", "TFA", "tab:orange", "s"),
            ("TFA-pos", "TFA-pos-shuf", "TFA-pos", "tab:brown", "X"),
        ]:
            fracs = []
            for rho in RHO_VALUES:
                sae_nmse = all_results[rho]["SAE"][ki]["nmse"]
                model_nmse = all_results[rho][base][ki]["nmse"]
                shuf_nmse = all_results[rho][shuf][ki]["nmse"]
                gap = sae_nmse - model_nmse
                if gap > 1e-6:
                    fracs.append((sae_nmse - shuf_nmse) / gap)
                else:
                    fracs.append(None)
            valid = [(r, f) for r, f in zip(RHO_VALUES, fracs) if f is not None]
            if valid:
                axes[2].plot([v[0] for v in valid], [1 - v[1] for v in valid],
                             marker=m, linestyle="-", color=c, lw=2, ms=8, label=f"{label} temporal %")

        axes[0].set(xlabel="ρ", ylabel="NMSE", title="NMSE vs ρ")
        axes[0].set_yscale("log")
        axes[1].set(xlabel="ρ", ylabel="AUC", title="AUC vs ρ")
        axes[2].set(xlabel="ρ", ylabel="Temporal fraction", title="Temporal fraction vs ρ")
        axes[2].axhline(0, color="gray", ls=":", lw=1)
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Correlation sweep (k={k}, all features at same ρ)", fontsize=13)
        plt.tight_layout()
        save_figure(fig, os.path.join(RESULTS_DIR, f"correlation_sweep_k{k}.png"))
        plt.close(fig)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.0f}m. Results in {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
