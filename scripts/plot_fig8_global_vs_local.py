"""Fig 8 analog: global (latent vs hidden h) vs local (latent vs observed s) correlation.

Reads results/hmm_denoising/sweep_results.json and produces a scatter where
each point is one (model, T, k) cell. Stacked SAE and regular SAE should
cluster along the per-token floor (global / local = 0.77). TXCDR points
should rise above the floor, with larger T farther up.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_MARKERS = {"regular_sae": "s", "regular_sae_kT": "P", "stacked_sae": "D", "txcdr": "o"}
MODEL_LABELS = {
    "regular_sae": "Regular SAE",
    "regular_sae_kT": "Regular SAE (k·T/token)",
    "stacked_sae": "Stacked SAE",
    "txcdr": "TXCDR",
}
PER_TOKEN_FLOOR = 0.77


def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="results/hmm_denoising/sweep_results.json"
    )
    parser.add_argument(
        "--output-dir", default="docs/bill/results/hmm_denoising"
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load(args.input)
    fig, ax = plt.subplots(figsize=(7.5, 7))

    x_max = max(df["corr_local"].max(), 0.01) * 1.1
    y_max = max(df["corr_global"].max(), 0.01) * 1.1

    # Per-token floor line: global = 0.77 * local.
    xs = np.linspace(0, max(x_max, y_max), 2)
    ax.plot(
        xs, PER_TOKEN_FLOOR * xs, "--", color="grey",
        label=f"per-token floor (ratio={PER_TOKEN_FLOOR})",
    )
    ax.plot(xs, xs, ":", color="lightgrey", label="ratio = 1")

    for model, sub in df.groupby("model"):
        T_vals = sorted(sub["T"].unique())
        cmap = plt.get_cmap("viridis")
        for i, T in enumerate(T_vals):
            tsub = sub[sub["T"] == T]
            color = cmap(i / max(1, len(T_vals) - 1))
            ax.scatter(
                tsub["corr_local"],
                tsub["corr_global"],
                marker=MODEL_MARKERS.get(model, "x"),
                s=60,
                edgecolor="black",
                linewidth=0.5,
                color=color if model == "txcdr" else "lightgrey",
                label=f"{MODEL_LABELS.get(model, model)} T={T}" if model == "txcdr" or i == 0 else None,
            )

    ax.set_xlim(0, max(x_max, y_max))
    ax.set_ylim(0, max(x_max, y_max))
    ax.set_xlabel("Local correlation (latent vs observed s)")
    ax.set_ylabel("Global correlation (latent vs hidden h)")
    ax.set_title("Global vs local feature recovery under noisy HMM emissions")
    ax.legend(fontsize=7, loc="best")

    out_png = os.path.join(args.output_dir, "fig8_global_vs_local.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
