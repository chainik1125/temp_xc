"""Render the Spectral-v1 comparison against the three published baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    POWER_ROOT / "results" / "paper_synthetic_v1_comparison.json"
)
DEFAULT_OUTPUT = (
    POWER_ROOT / "figures" / "paper_synthetic_v1_comparison.png"
)
ARCHITECTURES = ("topk_sae", "tsae_paper", "txc_base", "spectral_v1")
COLORS = {
    "topk_sae": "#5F6368",
    "tsae_paper": "#A7A9AC",
    "txc_base": "#6F8FAF",
    "spectral_v1": "#0072B2",
}


def render(payload: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    task_specs = (
        (
            "denoising",
            "Denoising",
            r"Hidden-state linear probe  $R^2_{\mathrm{global}}$",
        ),
        (
            "coupling",
            "Coupling (max overlap)",
            r"Static decoder alignment  $g\mathrm{AUC}$",
        ),
    )
    for axis, (task_name, title, metric_label) in zip(
        axes,
        task_specs,
        strict=True,
    ):
        by_arch = {
            row["architecture"]: row
            for row in payload["tasks"][task_name]["rows"]
        }
        rows = [by_arch[architecture] for architecture in ARCHITECTURES]
        means = np.array([float(row["mean"]) for row in rows])
        lower = means - np.array([float(row["min"]) for row in rows])
        upper = np.array([float(row["max"]) for row in rows]) - means
        x = np.arange(len(rows))
        bars = axis.bar(
            x,
            means,
            yerr=np.vstack([lower, upper]),
            color=[COLORS[row["architecture"]] for row in rows],
            edgecolor="white",
            linewidth=0.7,
            error_kw={
                "elinewidth": 0.9,
                "capsize": 3,
                "ecolor": "#333333",
            },
        )
        for bar, row in zip(bars, rows, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                float(row["max"]) + 0.025,
                f"{float(row['mean']):.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                0.025,
                f"{row['t_label']}\nk={row['k_pos']}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
        axis.set_title(title, pad=10)
        axis.set_xlabel(metric_label, labelpad=8)
        axis.set_xticks(
            x,
            [row["label"] for row in rows],
            rotation=18,
            ha="right",
        )
        axis.set_ylim(0, 1.13)
        axis.grid(axis="y", linestyle=":", alpha=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Best seed-mean score (range across runs)")
    fig.suptitle(
        "Spectral v1 on the paper’s synthetic tasks",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(json.loads(args.input.read_text()), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
