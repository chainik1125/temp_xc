"""Render nonlinear SWR summaries as a figure and Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODEL_KEYS = (
    ("ordered_pr_auc", "ordered", "#5E81AC"),
    ("mean_pool_param_matched_pr_auc", "invariant, parameter matched", "#A3BE8C"),
    ("mean_pool_same_rank_pr_auc", "invariant, same rank", "#88C0D0"),
    ("best_token_pr_auc", "best offset", "#D08770"),
)


def markdown_table(payload: dict) -> str:
    lines = [
        "| T | normalization | ordered | invariant (parameter matched) | "
        "invariant (same rank) | best offset | conservative G_order | grouped-fold 95% CI | "
        "folds > .02 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in payload["summaries"]:
        lines.append(
            "| {window} | {normalization} | {ordered:.3f} | {matched:.3f} | "
            "{same:.3f} | {token:.3f} | {gap:+.3f} | [{lower:+.3f}, {upper:+.3f}] | "
            "{above}/{folds} |".format(
                window=summary["window"],
                normalization=summary["normalization"],
                ordered=summary["ordered_pr_auc"]["mean"],
                matched=summary["mean_pool_param_matched_pr_auc"]["mean"],
                same=summary["mean_pool_same_rank_pr_auc"]["mean"],
                token=summary["best_token_pr_auc"]["mean"],
                gap=summary["swr_pr_auc_conservative"]["mean"],
                lower=summary["swr_pr_auc_conservative"]["grouped_fold_t_95"][0],
                upper=summary["swr_pr_auc_conservative"]["grouped_fold_t_95"][1],
                above=summary["swr_pr_auc_conservative"]["n_above_0_02"],
                folds=summary["swr_pr_auc_conservative"]["n_folds"],
            )
        )
    return "\n".join(lines) + "\n"


def plot_swr(payload: dict, output: Path) -> None:
    summaries = payload["summaries"]
    labels = [f"T={row['window']}\n{row['normalization']}" for row in summaries]
    x = np.arange(len(summaries), dtype=np.float64)
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for model_index, (key, label, color) in enumerate(MODEL_KEYS):
        means = np.asarray([row[key]["mean"] for row in summaries])
        std = np.asarray([row[key]["std_sample"] for row in summaries])
        folds = np.asarray([row[key]["n_folds"] for row in summaries])
        axes[0].bar(
            x + (model_index - 1.5) * width,
            means,
            yerr=1.96 * std / np.sqrt(folds),
            width=width,
            capsize=2,
            label=label,
            color=color,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set(ylabel="grouped-fold PR-AUC", title="Nonlinear raw-window gate")
    axes[0].legend(fontsize=7.5, frameon=False)

    for index, summary in enumerate(summaries):
        values = np.asarray(summary["swr_pr_auc_conservative"]["fold_values"])
        jitter = np.linspace(-0.05, 0.05, len(values))
        axes[1].scatter(np.full(len(values), index) + jitter, values, color="#B48EAD")
        axes[1].plot(
            [index - 0.14, index + 0.14],
            [values.mean(), values.mean()],
            color="#2E3440",
            linewidth=2,
        )
    axes[1].axhline(0.0, color="#4C566A", linewidth=1)
    axes[1].axhline(0.02, color="#BF616A", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels)
    axes[1].set(
        ylabel="ordered minus strongest null PR-AUC",
        title=r"Conservative $G_{\mathrm{order}}$",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    plot_swr(payload, args.figure)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_table(payload))


if __name__ == "__main__":
    main()
