"""Fig 10: global vs local feature recovery AUC vs window size T.

Reads results/three_arch_sweep/sweep_results.json. The headline panel
shows global AUC vs T for each (rho, k) cell, with one line per model.
Supplementary panels show local AUC and NMSE on the same grid.

Two definitions of global vs local are reported (both populated by
sweep.py via metrics.evaluate):
  - decoder: local = mean across positions of per-position decoder AUC;
             global = AUC of the position-averaged decoder.
  - activation: local = best-latent per-token classification AUC of
             latent activation vs per-feature support s_jt;
             global = best-latent window-pooled (max-over-t) AUC vs
             feature-active-anywhere-in-window.

For non-temporal models (regular_sae, regular_sae_kT) decoder local ==
decoder global by construction.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


MODEL_COLORS = {
    "regular_sae": "C0",
    "regular_sae_kT": "C2",
    "stacked_sae": "C1",
    "txcdr": "C3",
}
MODEL_LABELS = {
    "regular_sae": "Regular SAE",
    "regular_sae_kT": "Regular SAE (k·T/token)",
    "stacked_sae": "Stacked SAE",
    "txcdr": "TXCDR",
}


def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def _plot_grid(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    rhos = sorted(df["rho"].unique())
    ks = sorted(df["k"].unique())
    fig, axes = plt.subplots(
        len(rhos), len(ks),
        figsize=(3.2 * len(ks), 2.8 * len(rhos)),
        sharex=True, sharey=True,
        squeeze=False,
    )
    for i, rho in enumerate(rhos):
        for j, k in enumerate(ks):
            ax = axes[i][j]
            cell = df[(df["rho"] == rho) & (df["k"] == k)]
            for model, g in cell.groupby("model"):
                g = g.sort_values("T")
                ax.plot(
                    g["T"], g[metric],
                    marker="o", color=MODEL_COLORS.get(model, "grey"),
                    label=MODEL_LABELS.get(model, model),
                )
            if i == 0:
                ax.set_title(f"k={k}")
            if j == 0:
                ax.set_ylabel(f"rho={rho}\n{ylabel}")
            if i == len(rhos) - 1:
                ax.set_xlabel("Window size T")
            ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="results/three_arch_sweep/sweep_results.json"
    )
    parser.add_argument(
        "--output-dir", default="docs/bill/results/three_arch"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load(args.input)

    panels = [
        ("auc_decoder_global", "Global decoder AUC vs T", "Global decoder AUC",
         "fig10_global_decoder_auc_vs_T.png"),
        ("auc_decoder_local", "Local decoder AUC vs T", "Local decoder AUC",
         "fig10_local_decoder_auc_vs_T.png"),
        ("auc_activation_global", "Global activation AUC vs T",
         "Global activation AUC", "fig10_global_activation_auc_vs_T.png"),
        ("auc_activation_local", "Local activation AUC vs T",
         "Local activation AUC", "fig10_local_activation_auc_vs_T.png"),
        ("nmse", "NMSE vs T", "NMSE", "fig10_nmse_vs_T.png"),
    ]

    for metric, title, ylabel, fname in panels:
        if metric not in df.columns:
            print(f"skipping {metric}: not in results")
            continue
        _plot_grid(
            df, metric, title, ylabel,
            os.path.join(args.output_dir, fname),
        )


if __name__ == "__main__":
    main()
