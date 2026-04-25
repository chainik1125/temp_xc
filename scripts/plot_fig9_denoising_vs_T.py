"""Fig 9 analog: denoising ratio (global / local correlation) vs window size T.

Reads results/hmm_denoising/sweep_results.json. For each architecture, plots
denoising_ratio_corr (or R^2-ratio with --metric r2) vs T, one line per k.
The per-token floor (ratio = 0.77) is the maximum achievable by any model
that processes tokens independently; crossing ratio = 1 indicates the model
tracks the hidden state better than the noisy observation.
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
    parser.add_argument(
        "--metric", choices=["corr", "r2"], default="corr",
        help="Which denoising ratio to plot (single-latent correlation or probe R^2).",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load(args.input)
    ratio_col = "denoising_ratio_corr" if args.metric == "corr" else "denoising_ratio_r2"

    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        for k, g in sub.groupby("k"):
            g = g.sort_values("T")
            ax.plot(g["T"], g[ratio_col], marker="o", label=f"k={k}")
        ax.axhline(1.0, color="lightgrey", linestyle=":", label="full denoising (ratio=1)")
        ax.axhline(PER_TOKEN_FLOOR, color="grey", linestyle="--", label=f"per-token floor ({PER_TOKEN_FLOOR})")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.set_xlabel("Window size T")
        ax.set_ylabel(f"Global / local ratio ({args.metric})")
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    out_png = os.path.join(
        args.output_dir, f"fig9_denoising_ratio_{args.metric}.png"
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
