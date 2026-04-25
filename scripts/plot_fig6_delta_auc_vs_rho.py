"""Fig 6 analog: TXCDR advantage (DeltaAUC) vs rho, grouped by (k, T).

Reads results/three_arch_sweep/sweep_results.json. Produces a 1x2 panel:
left panel = TXCDR minus regular SAE; right panel = TXCDR minus Stacked SAE.
Each panel has one line per (k, T) pair, x-axis is rho.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def compute_delta(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    keys = ["rho", "k", "T"]
    tx = (
        df[df["model"] == "txcdr"]
        .groupby(keys, as_index=False)["auc"]
        .mean()
        .rename(columns={"auc": "auc_txcdr"})
    )
    bl = (
        df[df["model"] == baseline]
        .groupby(keys, as_index=False)["auc"]
        .mean()
        .rename(columns={"auc": "auc_bl"})
    )
    merged = tx.merge(bl, on=keys)
    merged["delta"] = merged["auc_txcdr"] - merged["auc_bl"]
    return merged


def plot_panel(ax, delta: pd.DataFrame, title: str) -> None:
    for (k, T), g in delta.groupby(["k", "T"]):
        g = g.sort_values("rho")
        ls = "-" if T == 5 else "--"
        ax.plot(
            g["rho"],
            g["delta"],
            marker="o",
            linestyle=ls,
            label=f"k={k}, T={T}",
        )
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("rho (lag-1 autocorrelation)")
    ax.set_ylabel("Delta AUC (TXCDR minus baseline)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")


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

    # One panel per non-TXCDR baseline that's present in the data.
    baselines = sorted(set(df["model"].unique()) - {"txcdr"})
    fig, axes = plt.subplots(1, len(baselines), figsize=(6 * len(baselines), 4.5), sharey=True)
    if len(baselines) == 1:
        axes = [axes]
    for ax, baseline in zip(axes, baselines):
        delta = compute_delta(df, baseline)
        plot_panel(ax, delta, f"TXCDR minus {baseline}")
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "fig6_delta_auc_vs_rho.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
