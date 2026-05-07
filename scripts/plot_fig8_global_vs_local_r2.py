"""Fig 8 analog using Ridge-probe R^2 instead of single-latent correlation.

Same scatter as plot_fig8_global_vs_local.py but with x=r2_local, y=r2_global,
and styled to match the "Setup B: linear probe (γ=...)" figure circulating
on the team. Each (model, T) gets a distinct marker; the y=x line is the
"no denoising" reference (a model that processes tokens independently
cannot land above it).

Reads results/hmm_paperfig/sweep_results.json by default. Title shows
the data's γ amplitude prefactor:

    γ = μ(1-μ)(p_B - p_A)² / (μ_obs(1-μ_obs)),   μ_obs = (1-μ)p_A + μ p_B

For Bill's midterm bench (pi=0.15, p_A=0, p_B=0.625), γ ≈ 0.59.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_MARKERS = {
    "regular_sae": "P",
    "regular_sae_kT": "X",
    "stacked_sae": "D",
    "txcdr": "o",
}
MODEL_COLORS = {
    "regular_sae": "black",
    "regular_sae_kT": "tab:purple",
    "stacked_sae": "tab:purple",
    "txcdr": "tab:pink",
}
MODEL_LABELS = {
    "regular_sae": "TopK SAE",
    "regular_sae_kT": "TopK SAE (k·T/token)",
    "stacked_sae": "Stacked SAE",
    "txcdr": "TXCDR",
}


def load(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def gamma_prefactor(pi: float, p_A: float, p_B: float) -> float:
    """Amplitude prefactor on observed autocorrelation (Aniket HMM doc)."""
    mu = pi
    mu_obs = (1 - mu) * p_A + mu * p_B
    if mu_obs in (0.0, 1.0):
        return float("nan")
    return mu * (1 - mu) * (p_B - p_A) ** 2 / (mu_obs * (1 - mu_obs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="results/hmm_paperfig/sweep_results.json"
    )
    parser.add_argument(
        "--output-dir", default="docs/bill/results/hmm_paperfig"
    )
    parser.add_argument("--pi", type=float, default=0.15)
    parser.add_argument("--p-A", type=float, default=0.0)
    parser.add_argument("--p-B", type=float, default=0.625)
    parser.add_argument(
        "--metric-suffix", default="",
        help="Set to e.g. '_hidden' to plot gAUC scatter against eAUC.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load(args.input)
    # Drop the framing-B baseline by default — it sits on y=x like regular_sae
    # so it adds clutter without adding signal in this scatter. The control
    # value of regular_sae_kT shows up better in the corr-vs-T line plot
    # (fig9), where its trajectory across T can be compared to TXCDR's.
    df = df[df["model"] != "regular_sae_kT"]
    gamma = gamma_prefactor(args.pi, args.p_A, args.p_B)

    fig, ax = plt.subplots(figsize=(8, 7.5))

    # Reference lines:
    #   y = x        : "no denoising" — points here imply latent tracks h as
    #                  well as it tracks s, but no better.
    #   y = γ * x    : per-token floor on R² for any token-local model. With
    #                  R² = correlation², the per-token correlation bound
    #                  sqrt(γ) becomes γ for R². Token-local arches should
    #                  cluster ON this line; anything ABOVE it is genuinely
    #                  using cross-position information to denoise.
    xs = np.linspace(0, 1, 2)
    ax.plot(xs, xs, "--", color="lightgrey", label="y = x  (full denoising)")
    if np.isfinite(gamma):
        ax.plot(
            xs, gamma * xs,
            ":", color="grey",
            label=f"y = γ · x  (per-token floor, γ ≈ {gamma:.2f})",
        )

    # One scatter per (model, T) for distinct markers/colors
    for model, sub in df.groupby("model"):
        T_vals = sorted(sub["T"].unique())
        cmap = plt.get_cmap("viridis") if model == "txcdr" else None
        base_color = MODEL_COLORS.get(model, "grey")
        marker = MODEL_MARKERS.get(model, "x")
        for i, T in enumerate(T_vals):
            tsub = sub[sub["T"] == T]
            if model == "txcdr":
                color = cmap(i / max(1, len(T_vals) - 1))
                label = f"TXCDR T={T}"
            else:
                color = base_color
                # Only label once per non-txcdr arch (T-curve is degenerate
                # for token-local SAEs anyway).
                label = MODEL_LABELS.get(model, model) if i == 0 else None
            ax.scatter(
                tsub["r2_local"], tsub["r2_global"],
                marker=marker, s=70, edgecolor="black", linewidth=0.5,
                color=color, label=label, alpha=0.85,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Local $R^2$ ($z \rightarrow s_i$, noisy obs)")
    ax.set_ylabel(r"Global $R^2$ ($z \rightarrow h_i$, hidden state)")
    ax.set_title(f"Global vs local feature recovery (linear probe, γ ≈ {gamma:.2f})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    out_png = os.path.join(args.output_dir, "fig8_global_vs_local_r2.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
