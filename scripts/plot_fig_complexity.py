"""Part 3 figure: gAUC vs HMM complexity (n_parents) across architectures.

Reads results/coupled_complexity/sweep_results.json. Produces three plots:

  1. fig_complexity_gauc_vs_nparents.{png,pdf}
     Headline. gAUC (decoder cosine vs hidden-state directions) on y,
     n_parents on x, one line per model, panels per (T, k).

  2. fig_complexity_delta_gauc.{png,pdf}
     ΔgAUC = gAUC(arch) - gAUC(regular_sae at same T?) vs n_parents.
     Tests "TXCDR's global-recovery advantage grows with complexity."

  3. fig_complexity_eauc_vs_nparents.{png,pdf}
     Companion: eAUC (vs emission directions). Should be ~flat across
     n_parents for all archs since the emission inversion is unaffected
     by coupling structure.
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


def _grid_plot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    Ts = sorted(df["T"].unique())
    ks = sorted(df["k"].unique())
    fig, axes = plt.subplots(
        len(Ts), len(ks),
        figsize=(3.5 * len(ks), 2.8 * len(Ts)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, T in enumerate(Ts):
        for j, k in enumerate(ks):
            ax = axes[i][j]
            cell = df[(df["T"] == T) & (df["k"] == k)]
            for model, g in cell.groupby("model"):
                agg = (
                    g.groupby("n_parents")[metric]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .sort_values("n_parents")
                )
                yerr = agg["std"].fillna(0.0)
                ax.errorbar(
                    agg["n_parents"], agg["mean"], yerr=yerr,
                    marker="o", capsize=3,
                    color=MODEL_COLORS.get(model, "grey"),
                    label=MODEL_LABELS.get(model, model),
                )
            if i == 0:
                ax.set_title(f"k={k}")
            if j == 0:
                ax.set_ylabel(f"T={T}\n{ylabel}")
            if i == len(Ts) - 1:
                ax.set_xlabel("n_parents (HMM complexity →)")
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


def _delta_gauc_plot(df: pd.DataFrame, out_path: str, baseline: str = "regular_sae") -> None:
    """ΔgAUC = gAUC(arch) - gAUC(baseline at same T, k, n_parents, seed)."""
    if baseline not in df["model"].unique():
        print(f"skip ΔgAUC: baseline {baseline} not in results")
        return
    base = df[df["model"] == baseline][["n_parents", "T", "k", "seed", "auc_hidden"]]
    base = base.rename(columns={"auc_hidden": "auc_hidden_base"})
    merged = df.merge(base, on=["n_parents", "T", "k", "seed"], how="left")
    merged = merged[merged["model"] != baseline].copy()
    merged["delta_gauc"] = merged["auc_hidden"] - merged["auc_hidden_base"]

    Ts = sorted(merged["T"].unique())
    ks = sorted(merged["k"].unique())
    fig, axes = plt.subplots(
        len(Ts), len(ks),
        figsize=(3.5 * len(ks), 2.8 * len(Ts)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, T in enumerate(Ts):
        for j, k in enumerate(ks):
            ax = axes[i][j]
            cell = merged[(merged["T"] == T) & (merged["k"] == k)]
            for model, g in cell.groupby("model"):
                agg = (
                    g.groupby("n_parents")["delta_gauc"]
                    .agg(["mean", "std"])
                    .reset_index()
                    .sort_values("n_parents")
                )
                ax.errorbar(
                    agg["n_parents"], agg["mean"], yerr=agg["std"].fillna(0.0),
                    marker="o", capsize=3,
                    color=MODEL_COLORS.get(model, "grey"),
                    label=MODEL_LABELS.get(model, model),
                )
            ax.axhline(0.0, color="grey", linestyle=":")
            if i == 0:
                ax.set_title(f"k={k}")
            if j == 0:
                ax.set_ylabel(f"T={T}\nΔgAUC vs {MODEL_LABELS.get(baseline, baseline)}")
            if i == len(Ts) - 1:
                ax.set_xlabel("n_parents (HMM complexity →)")
            ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Global-recovery advantage over {MODEL_LABELS.get(baseline, baseline)} "
        "vs HMM complexity",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="results/coupled_complexity/sweep_results.json"
    )
    parser.add_argument(
        "--output-dir", default="docs/bill/results/coupled_complexity"
    )
    parser.add_argument(
        "--baseline", default="regular_sae",
        help="Reference model for the ΔgAUC plot.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load(args.input)

    _grid_plot(
        df, metric="auc_hidden",
        title="Global feature recovery (gAUC) vs HMM complexity",
        ylabel="gAUC",
        out_path=os.path.join(args.output_dir, "fig_complexity_gauc_vs_nparents.png"),
    )
    _grid_plot(
        df, metric="auc",
        title="Local feature recovery (eAUC) vs HMM complexity",
        ylabel="eAUC",
        out_path=os.path.join(args.output_dir, "fig_complexity_eauc_vs_nparents.png"),
    )
    _delta_gauc_plot(
        df,
        out_path=os.path.join(args.output_dir, "fig_complexity_delta_gauc.png"),
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()
