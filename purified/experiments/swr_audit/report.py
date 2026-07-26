"""Render the E1 matched-filter gate as a figure and Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "ordered": "#5E81AC",
    "invariant": "#A3BE8C",
    "token": "#D08770",
    "gap": "#B48EAD",
}


def _format_interval(summary: dict) -> str:
    bootstrap = summary["g_order_pr_auc"]["cluster_bootstrap"]
    return f"[{bootstrap['lower_95']:.3f}, {bootstrap['upper_95']:.3f}]"


def markdown_table(payload: dict) -> str:
    lines = [
        "| T | offsets | ordered PR-AUC | invariant PR-AUC | best-offset PR-AUC | "
        "G_order | prompt-bootstrap 95% CI | selected offsets | task DC fraction |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for summary in sorted(payload["summaries"], key=lambda row: row["window"]):
        selected = ", ".join(str(value) for value in summary["best_offsets"])
        dc_fraction = summary["task_spectrum"]["mean_j_y_fraction"][0]
        lines.append(
            "| {window} | {offsets} | {ordered:.3f} | {invariant:.3f} | "
            "{token:.3f} | {gap:+.3f} | {interval} | {selected} | {dc:.3f} |".format(
                window=summary["window"],
                offsets="…".join(
                    [
                        str(summary["window_offsets"][0]),
                        str(summary["window_offsets"][-1]),
                    ]
                ),
                ordered=summary["ordered_pr_auc"]["mean"],
                invariant=summary["invariant_mean_pr_auc"]["mean"],
                token=summary["best_token_pr_auc"]["mean"],
                gap=summary["g_order_pr_auc"]["mean"],
                interval=_format_interval(summary),
                selected=selected,
                dc=dc_fraction,
            )
        )
    return "\n".join(lines) + "\n"


def _mean_and_sem(summary: dict, key: str) -> tuple[float, float]:
    values = np.asarray(summary[key]["fold_values"], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def plot_matched(payload: dict, output: Path) -> None:
    summaries = sorted(payload["summaries"], key=lambda row: row["window"])
    windows = np.asarray([row["window"] for row in summaries])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    for key, label, color in (
        ("ordered_pr_auc", "ordered matched filter", COLORS["ordered"]),
        ("invariant_mean_pr_auc", "exact invariant mean", COLORS["invariant"]),
        ("best_token_pr_auc", "best validation-selected offset", COLORS["token"]),
    ):
        values = np.asarray([_mean_and_sem(row, key) for row in summaries])
        axes[0].errorbar(
            windows,
            values[:, 0],
            yerr=1.96 * values[:, 1],
            marker="o",
            capsize=3,
            label=label,
            color=color,
        )
    axes[0].axhline(payload["positive_rate"], color="#4C566A", linestyle="--", linewidth=1)
    axes[0].set(xlabel="window length T", ylabel="grouped-fold PR-AUC", title="Raw detection")
    axes[0].legend(fontsize=8, frameon=False)

    gaps = np.asarray([row["g_order_pr_auc"]["mean"] for row in summaries])
    lower = np.asarray(
        [row["g_order_pr_auc"]["cluster_bootstrap"]["lower_95"] for row in summaries]
    )
    upper = np.asarray(
        [row["g_order_pr_auc"]["cluster_bootstrap"]["upper_95"] for row in summaries]
    )
    axes[1].errorbar(
        windows,
        gaps,
        yerr=np.vstack([gaps - lower, upper - gaps]),
        marker="o",
        capsize=4,
        color=COLORS["gap"],
    )
    axes[1].axhline(0.0, color="#4C566A", linewidth=1)
    axes[1].axhline(0.02, color="#BF616A", linestyle="--", linewidth=1, label="preregistered gate")
    axes[1].set(
        xlabel="window length T",
        ylabel="PR-AUC difference",
        title=r"Ordered opportunity $G_{\mathrm{order}}$",
    )
    axes[1].legend(fontsize=8, frameon=False)

    widest = summaries[-1]
    frequency = np.asarray(
        widest["task_spectrum"]["frequency_cycles_per_window_step"]
    )
    fraction = np.asarray(widest["task_spectrum"]["mean_j_y_fraction"])
    spread = np.asarray(widest["task_spectrum"]["std_j_y_fraction"])
    axes[2].bar(frequency, fraction, width=0.7 / widest["window"], color=COLORS["ordered"])
    axes[2].errorbar(frequency, fraction, yerr=spread, fmt="none", color="#2E3440", capsize=2)
    axes[2].set(
        xlabel="cycles / token step",
        ylabel="fraction of task J_y",
        title=f"Train-only task spectrum (T={widest['window']})",
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    plot_matched(payload, args.figure)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_table(payload))


if __name__ == "__main__":
    main()
