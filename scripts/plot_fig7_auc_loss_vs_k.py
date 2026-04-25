"""Fig 7 analog: AUC and reconstruction loss vs k, for each rho and T.

Reads results/three_arch_sweep/sweep_results.json. Produces a 2x3 panel:
row 1 = AUC vs k, row 2 = NMSE vs k; columns = rho values.
Three lines per subplot (regular SAE / Stacked SAE / TXCDR); T=2 dashed,
T=5 solid.
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
    rhos = sorted(df["rho"].unique())
    Ts = sorted(df["T"].unique())

    fig, axes = plt.subplots(
        2, len(rhos), figsize=(5 * len(rhos), 8), sharex=True
    )
    if len(rhos) == 1:
        axes = axes.reshape(2, 1)

    for col, rho in enumerate(rhos):
        ax_auc, ax_loss = axes[0, col], axes[1, col]
        sub = df[df["rho"] == rho]
        for model in [m for m in MODEL_LABELS if m in df["model"].unique()]:
            for T in Ts:
                g = (
                    sub[(sub["model"] == model) & (sub["T"] == T)]
                    .groupby("k", as_index=False)[["auc", "nmse"]]
                    .mean()
                    .sort_values("k")
                )
                if g.empty:
                    continue
                ls = "-" if T == max(Ts) else "--"
                label = f"{MODEL_LABELS[model]} T={T}"
                ax_auc.plot(
                    g["k"], g["auc"], marker="o", linestyle=ls,
                    color=MODEL_COLORS[model], label=label,
                )
                ax_loss.plot(
                    g["k"], g["nmse"], marker="o", linestyle=ls,
                    color=MODEL_COLORS[model], label=label,
                )
        label = "IID" if rho == 0.0 else f"rho={rho}"
        ax_auc.set_title(f"AUC — {label}")
        ax_auc.set_ylabel("Feature recovery AUC")
        ax_auc.set_ylim(0, 1.02)
        ax_auc.legend(fontsize=7, loc="best")
        ax_loss.set_title(f"NMSE — {label}")
        ax_loss.set_xlabel("k (per-position TopK)")
        ax_loss.set_ylabel("NMSE")

    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "fig7_auc_loss_vs_k.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
