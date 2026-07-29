"""Render the matched synthetic benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = POWER_ROOT / "results" / "overnight_remote" / "benchmark_analysis.json"
DEFAULT_FIGURES = POWER_ROOT / "figures"

MODELS = (
    "txc_pre",
    "txc_post",
    "spectral_v1",
    "v2_remove_dc",
    "v2_dominance",
    "v2_freq_matryoshka",
    "v2_combined",
    "v2_global",
    "v2_full_global",
)
MODEL_LABELS = {
    "txc_pre": "TXC-pre",
    "txc_post": "TXC-post",
    "spectral_v1": "Spectral v1",
    "v2_remove_dc": "−DC",
    "v2_dominance": "Band flatten.",
    "v2_freq_matryoshka": "Freq-Mat.",
    "v2_combined": "Combined",
    "v2_global": "Global top-k",
    "v2_full_global": "Full-band control",
}
MODEL_COLORS = {
    "txc_pre": "#777777",
    "txc_post": "#BBBBBB",
    "spectral_v1": "#0072B2",
    "v2_remove_dc": "#56B4E9",
    "v2_dominance": "#009E73",
    "v2_freq_matryoshka": "#E69F00",
    "v2_combined": "#D55E00",
    "v2_global": "#CC79A7",
    "v2_full_global": "#000000",
}
MODEL_MARKERS = {
    "txc_pre": "o",
    "txc_post": "s",
    "spectral_v1": "^",
    "v2_remove_dc": "v",
    "v2_dominance": "P",
    "v2_freq_matryoshka": "X",
    "v2_combined": "<",
    "v2_global": ">",
    "v2_full_global": "D",
}
TASKS = ("frequency", "multilane", "phasepair", "permuted", "colored")
TASK_LABELS = {
    "frequency": "Periodic velocity",
    "multilane": "Multilane periodic",
    "phasepair": "Phase-only sign",
    "permuted": "Permuted schedule",
    "colored": "Colored sources",
}


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["model"] = pd.Categorical(out["model"], categories=MODELS, ordered=True)
    out["task"] = pd.Categorical(out["task"], categories=TASKS, ordered=True)
    return out.sort_values(["task", "model"])


def plot_primary(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 4.2), sharey=True)
    for axis, task in zip(axes, TASKS, strict=True):
        part = frame[frame["task"] == task].set_index("model").reindex(MODELS)
        means = part["mean"].to_numpy(dtype=float)
        errors = part["std"].to_numpy(dtype=float)
        x = np.arange(len(MODELS))
        axis.bar(
            x,
            means,
            yerr=errors,
            color=[MODEL_COLORS[model] for model in MODELS],
            edgecolor="white",
            linewidth=0.6,
            capsize=2,
        )
        axis.axhline(0.0, color="0.45", linewidth=0.8)
        axis.set_title(TASK_LABELS[task])
        axis.set_xticks(x, [MODEL_LABELS[model] for model in MODELS], rotation=68, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Primary recovery metric (mean ± seed SD)")
    fig.suptitle("Matched synthetic benchmark", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(
    frame: pd.DataFrame,
    output: Path,
    *,
    baseline: str = "txc_pre",
) -> None:
    if baseline not in MODELS:
        raise ValueError(f"unknown baseline {baseline!r}")
    variants = [model for model in MODELS if model != baseline]
    delta_column = f"delta_vs_{baseline}"
    table = (
        frame.pivot(index="model", columns="task", values=delta_column)
        .reindex(index=variants, columns=TASKS)
        .astype(float)
    )
    values = table.to_numpy()
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(finite.max(initial=0.0)), 0.05)
    fig, axis = plt.subplots(figsize=(8.8, 5.1))
    image = axis.imshow(
        values,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                axis.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    axis.set_xticks(np.arange(len(TASKS)), [TASK_LABELS[task] for task in TASKS])
    axis.set_yticks(
        np.arange(len(variants)), [MODEL_LABELS[model] for model in variants]
    )
    axis.tick_params(axis="x", rotation=25)
    axis.set_title(
        "Paired primary-metric delta versus "
        f"{MODEL_LABELS[baseline]}"
    )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Recovery delta")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_nmse(frame: pd.DataFrame, output: Path) -> None:
    """Facet the task-specific recovery/reconstruction tradeoff."""
    fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 4.1))
    for axis, task in zip(axes, TASKS, strict=True):
        part = frame[frame["task"] == task].set_index("model").reindex(MODELS)
        for model in MODELS:
            row = part.loc[model]
            axis.errorbar(
                float(row["mean_nmse"]),
                float(row["mean"]),
                xerr=float(row["std_nmse"]),
                yerr=float(row["std"]),
                marker=MODEL_MARKERS[model],
                markersize=5.5,
                color=MODEL_COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.4,
                linewidth=0.8,
                capsize=2,
                linestyle="none",
                label=MODEL_LABELS[model],
            )
        axis.set_title(TASK_LABELS[task])
        axis.set_xlabel("NMSE (lower is better)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Primary recovery (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Recovery–reconstruction frontier", y=0.995)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    integrity = payload.get("integrity")
    if not integrity or not integrity.get("complete"):
        raise RuntimeError("refusing to plot a benchmark without a complete integrity gate")
    frame = _ordered(pd.DataFrame(payload["aggregates"]))
    expected = {(task, model) for task in TASKS for model in MODELS}
    observed = {(str(row.task), str(row.model)) for row in frame.itertuples()}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"incomplete summary, missing {missing}")
    wrong_n = frame[frame["n"] != 3][["task", "model", "n"]]
    if not wrong_n.empty:
        raise RuntimeError(
            "expected three paired seeds per cell: "
            f"{wrong_n.to_dict(orient='records')}"
        )
    plot_primary(frame, args.figures_dir / "benchmark_primary_metrics.png")
    plot_deltas(frame, args.figures_dir / "benchmark_delta_vs_txc_pre.png")
    plot_deltas(
        frame,
        args.figures_dir / "benchmark_delta_vs_spectral_v1.png",
        baseline="spectral_v1",
    )
    plot_recovery_nmse(frame, args.figures_dir / "benchmark_recovery_nmse.png")
    print(
        frame.pivot(index="model", columns="task", values="mean")
        .reindex(index=MODELS, columns=TASKS)
        .to_string(float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
