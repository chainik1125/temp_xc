"""Fig 5 analog: TXCDR advantage (DeltaAUC) heatmaps across (k, T) for each rho.

Reads results/three_arch_sweep/sweep_results.json and produces two 1x3 panels:
  - TXCDR minus regular SAE
  - TXCDR minus Stacked SAE
Each column is one rho value; cells are indexed by (k, T).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def pivot_auc(df: pd.DataFrame, model: str, rho: float) -> pd.DataFrame:
    sub = df[(df["model"] == model) & (df["rho"] == rho)]
    return (
        sub.groupby(["k", "T"], as_index=False)["auc"]
        .mean()
        .pivot(index="k", columns="T", values="auc")
        .sort_index(ascending=False)
    )


def plot_delta_panel(df: pd.DataFrame, baseline: str, out_path: str) -> None:
    rhos = sorted(df["rho"].unique())
    fig, axes = plt.subplots(
        1, len(rhos), figsize=(5 * len(rhos), 4), squeeze=False,
        constrained_layout=True,
    )
    vmin, vmax = -0.5, 0.5
    for ax, rho in zip(axes[0], rhos):
        auc_tx = pivot_auc(df, "txcdr", rho)
        auc_bl = pivot_auc(df, baseline, rho)
        delta = auc_tx - auc_bl
        im = ax.imshow(
            delta.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto"
        )
        ax.set_xticks(range(len(delta.columns)))
        ax.set_xticklabels([f"T={t}" for t in delta.columns])
        ax.set_yticks(range(len(delta.index)))
        ax.set_yticklabels([f"k={k}" for k in delta.index])
        for i, k in enumerate(delta.index):
            for j, t in enumerate(delta.columns):
                val = delta.iloc[i, j]
                if np.isfinite(val):
                    ax.text(
                        j, i, f"{val:+.02f}", ha="center", va="center",
                        color="black" if abs(val) < 0.35 else "white",
                        fontsize=9,
                    )
        label = "IID" if rho == 0.0 else f"rho={rho}"
        ax.set_title(f"Delta AUC — {label}")
    fig.suptitle(f"TXCDR minus {baseline}  (positive = TXCDR wins)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
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

    plot_delta_panel(
        df, "regular_sae",
        os.path.join(args.output_dir, "fig5_delta_auc_vs_regular_sae.png"),
    )
    plot_delta_panel(
        df, "stacked_sae",
        os.path.join(args.output_dir, "fig5_delta_auc_vs_stacked_sae.png"),
    )


if __name__ == "__main__":
    main()
