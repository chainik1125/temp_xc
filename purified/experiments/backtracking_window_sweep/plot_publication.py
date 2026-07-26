"""Create the reviewer-stage backtracking window-sweep figure package."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NORD = {
    "text": "#2E3440",
    "muted": "#4C566A",
    "grid": "#D8DEE9",
    "fill": "#ECEFF4",
    "blue": "#5E81AC",
    "red": "#BF616A",
    "orange": "#D08770",
}
SEEDS = (1, 2, 42)
WINDOWS = tuple(range(1, 7))

# Post-hoc T=6, seed-42 sensitivity result. The raw result was summarized in
# purified/docs/aniket/neurips-rebuttal/july23-aniket-workplan.md.
POSITIONAL_BUDGET_ROWS = (
    (32, 0.1399, 0.2585, 0.1186, 0.0995, 0.1383),
    (64, 0.1757, 0.2585, 0.0828, 0.0683, 0.0966),
    (128, 0.2417, 0.2585, 0.0168, 0.0047, 0.0282),
    (192, 0.2644, 0.2585, -0.0058, -0.0206, 0.0076),
    (256, 0.2779, 0.2585, -0.0194, -0.0351, -0.0038),
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.25,
            "axes.labelsize": 7.25,
            "axes.edgecolor": NORD["muted"],
            "axes.labelcolor": NORD["text"],
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": NORD["grid"],
            "grid.linewidth": 0.55,
            "xtick.color": NORD["muted"],
            "ytick.color": NORD["muted"],
            "xtick.labelsize": 6.75,
            "ytick.labelsize": 6.75,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _selected_txc(payload: dict) -> dict:
    matches = [
        row
        for row in payload["probes"]["txc"]
        if int(row["n_features"]) == 32
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one S=32 TXC probe for T={payload['window']} "
            f"seed={payload['seed']}, found {len(matches)}"
        )
    return matches[0]


def _selected_probe(payload: dict, name: str) -> dict:
    matches = [
        row
        for row in payload["probes"][name]
        if int(row["n_features"]) == 32
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one S=32 {name} probe for T={payload['window']} "
            f"seed={payload['seed']}, found {len(matches)}"
        )
    return matches[0]


def _load_rows(
    root: Path,
    *,
    allow_partial: bool,
    windows: tuple[int, ...] = WINDOWS,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    rows = []
    for path in sorted((root / "cells").glob("T*_seed*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        txc = _selected_txc(payload)
        controls = {
            name: float(txc["control_pr_auc"][name]["mean"])
            for name in ("shuffle", "reverse", "circular")
        }
        control_values = list(controls.values())
        ordered = float(txc["ordered_pr_auc"]["mean"])
        rows.append(
            {
                "window": int(payload["window"]),
                "seed": int(payload["seed"]),
                "txc_ordered_ap": ordered,
                "txc_shuffle_ap": controls["shuffle"],
                "txc_reverse_ap": controls["reverse"],
                "txc_circular_ap": controls["circular"],
                "ordered_minus_shuffle_ap": ordered - controls["shuffle"],
                "strongest_order_control_ap": max(control_values),
                "conservative_order_gap_ap": ordered - max(control_values),
                "sae_positional_ap": float(
                    _selected_probe(
                        payload,
                        "sae_positional",
                    )["ordered_pr_auc"]["mean"]
                ),
                "sae_invariant_ap": float(
                    _selected_probe(
                        payload,
                        "sae_invariant",
                    )["ordered_pr_auc"]["mean"]
                ),
                "sae_last_token_ap": float(
                    _selected_probe(
                        payload,
                        "sae_last_token",
                    )["ordered_pr_auc"]["mean"]
                ),
            }
        )

    observed = {(row["window"], row["seed"]) for row in rows}
    expected = {(window, seed) for window in windows for seed in seeds}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(f"unexpected cells: {unexpected}")
    if missing and not allow_partial:
        raise ValueError(
            f"missing {len(missing)} cells: {missing}; pass --allow-partial "
            "only for a diagnostic render"
        )
    return sorted(rows, key=lambda row: (row["window"], row["seed"]))


def _group(rows: list[dict], window: int) -> list[dict]:
    return [row for row in rows if row["window"] == window]


def _mean_sd(rows: list[dict], field: str) -> tuple[float, float]:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if len(values) == 0:
        return math.nan, math.nan
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), sd


def _base_axis(
    height: float, windows: tuple[int, ...]
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5.50, height))
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.24, top=0.96)
    ax.tick_params(direction="out", length=3.0, width=0.7)
    ax.set_xticks(windows)
    span = max(windows) - min(windows)
    margin = max(0.28, 0.035 * max(span, 1))
    ax.set_xlim(min(windows) - margin, max(windows) + margin)
    return fig, ax


def _plot_seed_trajectories(
    ax: plt.Axes,
    rows: list[dict],
    *,
    field: str,
    color: str = NORD["blue"],
    summary_marker: str = "s",
    raw_marker: str = "o",
    line_style: str = "-",
    x_shift: float = 0.0,
    label: str | None = None,
    seeds: tuple[int, ...] = SEEDS,
) -> tuple[list[float], list[float]]:
    offsets = (
        np.zeros(1)
        if len(seeds) == 1
        else np.linspace(-0.04, 0.04, num=len(seeds))
    )
    jitter = {seed: offset for seed, offset in zip(seeds, offsets)}
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        if len(seed_rows) > 1:
            ax.plot(
                [row["window"] + x_shift for row in seed_rows],
                [row[field] for row in seed_rows],
                color=color,
                alpha=0.24,
                linewidth=0.8,
                linestyle=line_style,
                zorder=1,
            )
        scatter_style = {
            "s": 12,
            "marker": raw_marker,
            "color": color,
            "linewidths": 0.45,
            "zorder": 3,
        }
        if raw_marker != "x":
            scatter_style["edgecolors"] = "white"
        ax.scatter(
            [
                row["window"] + jitter[seed] + x_shift
                for row in seed_rows
            ],
            [row[field] for row in seed_rows],
            **scatter_style,
        )

    available_windows = sorted({row["window"] for row in rows})
    means, sds = [], []
    for window in available_windows:
        mean, sd = _mean_sd(_group(rows, window), field)
        means.append(mean)
        sds.append(sd)
    ax.errorbar(
        [window + x_shift for window in available_windows],
        means,
        yerr=sds,
        color=color,
        marker=summary_marker,
        markersize=4.8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.45,
        linestyle=line_style,
        capsize=2.4,
        capthick=0.75,
        elinewidth=0.75,
        label=label,
        zorder=4,
    )
    return means, sds


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = stem.with_name(f"{stem.name}.tmp.pdf")
    temporary_png = stem.with_name(f"{stem.name}.tmp.png")
    fig.savefig(temporary_pdf)
    fig.savefig(temporary_png, dpi=300)
    plt.close(fig)
    os.replace(temporary_pdf, stem.with_suffix(".pdf"))
    os.replace(temporary_png, stem.with_suffix(".png"))


def _plot_window_curve(
    rows: list[dict],
    output_dir: Path,
    *,
    windows: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict:
    fig, ax = _base_axis(2.65, windows)
    fig.subplots_adjust(bottom=0.30)
    _plot_seed_trajectories(
        ax,
        rows,
        field="txc_ordered_ap",
        color=NORD["blue"],
        summary_marker="s",
        raw_marker="o",
        line_style="-",
        x_shift=-0.035,
        label="Ordered TXC",
        seeds=seeds,
    )
    _plot_seed_trajectories(
        ax,
        rows,
        field="txc_shuffle_ap",
        color=NORD["orange"],
        summary_marker="X",
        raw_marker="x",
        line_style=(0, (2, 1.5)),
        x_shift=0.035,
        label="Shuffled window",
        seeds=seeds,
    )
    ax.set_xlabel("Window length T (tokens)")
    ax.set_ylabel("Backtracking detection AP")
    absolute_values = np.asarray(
        [
            value
            for row in rows
            for value in (row["txc_ordered_ap"], row["txc_shuffle_ap"])
        ],
        dtype=np.float64,
    )
    ax.set_ylim(
        min(0.19, float(absolute_values.min()) - 0.01),
        max(0.29, float(absolute_values.max()) + 0.01),
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )

    endpoint_rows = [
        row for row in rows if row["window"] in {windows[0], windows[-1]}
    ]
    endpoints = {(row["window"], row["seed"]): row for row in endpoint_rows}
    paired_gains = [
        endpoints[(windows[-1], seed)]["txc_ordered_ap"]
        - endpoints[(windows[0], seed)]["txc_ordered_ap"]
        for seed in seeds
        if (windows[0], seed) in endpoints and (windows[-1], seed) in endpoints
    ]
    mean_gain = float(np.mean(paired_gains)) if paired_gains else math.nan
    if len(paired_gains) == len(seeds):
        endpoint_mean = np.mean(
            [
                endpoints[(windows[-1], seed)]["txc_ordered_ap"]
                for seed in seeds
            ]
        )
        y_low, y_high = ax.get_ylim()
        ax.annotate(
            f"T={windows[-1]} - T={windows[0]}: {mean_gain:+.3f} AP",
            xy=(
                float(windows[-1]),
                endpoint_mean,
            ),
            xytext=(
                float(windows[-1]) - 0.03 * max(windows[-1] - windows[0], 1),
                y_high - 0.08 * (y_high - y_low),
            ),
            ha="right",
            va="center",
            color=NORD["muted"],
            fontsize=6.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
                "pad": 1.2,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": NORD["muted"],
                "linewidth": 0.7,
            },
        )
    _save_figure(fig, output_dir / "txc_window_length")
    summary = {
        "endpoint_windows": [windows[0], windows[-1]],
        "paired_endpoint_gain": paired_gains,
        "paired_endpoint_gain_mean": mean_gain,
    }
    if windows[0] == 1 and windows[-1] == 6:
        summary.update(
            {
                "paired_t6_minus_t1": paired_gains,
                "paired_t6_minus_t1_mean": mean_gain,
            }
        )
    return summary


def _plot_order_gap(
    rows: list[dict],
    output_dir: Path,
    *,
    windows: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    fig, ax = _base_axis(2.20, windows)
    _plot_seed_trajectories(
        ax,
        rows,
        field="conservative_order_gap_ap",
        seeds=seeds,
    )
    ax.axhline(
        0.0,
        color=NORD["muted"],
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    values = np.asarray(
        [row["conservative_order_gap_ap"] for row in rows],
        dtype=np.float64,
    )
    upper = max(0.03, float(values.max()) + 0.006)
    lower = min(-0.006, float(values.min()) - 0.004)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Window length T (tokens)")
    ax.set_ylabel("Ordered - perturbed AP")
    _save_figure(fig, output_dir / "txc_order_sensitivity")


def _plot_shuffle_gap(
    rows: list[dict],
    output_dir: Path,
    *,
    windows: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    """Plot the temporal residual after removing the shuffled/DC-like curve."""

    fig, ax = _base_axis(2.20, windows)
    _plot_seed_trajectories(
        ax,
        rows,
        field="ordered_minus_shuffle_ap",
        color=NORD["blue"],
        seeds=seeds,
    )
    ax.axhline(
        0.0,
        color=NORD["muted"],
        linewidth=0.8,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    values = np.asarray(
        [row["ordered_minus_shuffle_ap"] for row in rows],
        dtype=np.float64,
    )
    upper = max(0.03, float(values.max()) + 0.006)
    lower = min(-0.006, float(values.min()) - 0.004)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Window length T (tokens)")
    ax.set_ylabel("Ordered - shuffled AP")
    _save_figure(fig, output_dir / "txc_ordered_minus_shuffled")


def _write_seed_metrics(rows: list[dict], output_dir: Path) -> None:
    path = output_dir / "window_sweep_seed_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_budget_csv(output_dir: Path) -> None:
    path = output_dir / "positional_sae_budget_sensitivity.csv"
    fields = (
        "n_features",
        "positional_sae_ap",
        "txc_ap",
        "txc_minus_sae_ap",
        "ci95_low",
        "ci95_high",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(POSITIONAL_BUDGET_ROWS)


def _write_markdown(
    output_dir: Path,
    *,
    n_cells: int,
    mean_gain: float,
    windows: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    lines = [
        "# Backtracking window-sweep reviewer figures",
        "",
        "![TXC detection AP across window lengths](txc_window_length.png)",
        "",
        "**Ordered and shuffled performance are reported separately because a "
        "longer window can improve detection without using order.** Small "
        "points are dictionary seeds, thin lines connect the same seed, and "
        "squares with whiskers show mean ± sample SD across seeds. A rise shared "
        "by both curves is consistent with denoising or recovery of an "
        "order-invariant/DC-like component; only their separation is evidence "
        "that the fixed ordered-trained representation depends on token order.",
        "",
        "![Ordered minus shuffled TXC AP](txc_ordered_minus_shuffled.png)",
        "",
        "**Ordered minus shuffled AP isolates the fixed-probe temporal "
        "residual.** Positive values mean the intact local trajectory carries "
        "signal that is damaged by within-window permutation. The control is "
        "still a covariate-shift sensitivity test, so it should be interpreted "
        "alongside retrained order-invariant baselines.",
        "",
        "![TXC order-perturbation sensitivity](txc_order_sensitivity.png)",
        "",
        "**Order sensitivity is smaller and less certain than the context-length "
        "effect.** Delta AP compares ordered TXC with the best-performing "
        "shuffle, reversal, or nonzero circular-shift control under the same "
        "fixed ordered-trained 32-feature probe. Points are dictionary seeds; "
        "squares and whiskers are mean ± sample SD. These perturbations induce "
        "covariate shift, so the plot measures representation sensitivity rather "
        "than a causal estimate of unique temporal information.",
        "",
        "## Reviewer-response table",
        "",
        "| T | Ordered TXC AP | Shuffled TXC AP | Last-token SAE AP | Invariant SAE AP |",
        "|---:|---:|---:|---:|---:|",
    ]
    seed_rows = list(csv.DictReader(
        (output_dir / "window_sweep_seed_metrics.csv").open(
            encoding="utf-8"
        )
    ))
    for window in windows:
        local = [
            row for row in seed_rows if int(row["window"]) == window
        ]
        formatted = []
        for field in (
            "txc_ordered_ap",
            "txc_shuffle_ap",
            "sae_last_token_ap",
            "sae_invariant_ap",
        ):
            mean, sd = _mean_sd(
                [
                    {field: float(row[field])}
                    for row in local
                ],
                field,
            )
            formatted.append(f"{mean:.3f} ± {sd:.3f}")
        lines.append(f"| {window} | " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            f"Entries are mean ± sample SD across {len(seeds)} dictionary "
            "seeds. "
            "Every method uses a 32-feature question-grouped sparse probe. "
            "The shuffled value applies the ordered-trained TXC probe after a "
            "deterministic within-window permutation; it is a fixed-probe "
            "sensitivity control rather than a retrained shuffled model.",
            "",
        "## Positional-SAE feature-budget sensitivity",
        "",
        "| S | Positional SAE AP | TXC AP | TXC - SAE AP [95% question-bootstrap CI] |",
        "|---:|---:|---:|---:|",
        ]
    )
    for budget, sae, txc, gap, low, high in POSITIONAL_BUDGET_ROWS:
        lines.append(
            f"| {budget} | {sae:.4f} | {txc:.4f} | "
            f"{gap:+.4f} [{low:+.4f}, {high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "**The positional-SAE comparison reverses as its probe budget grows.** "
            "This post-hoc T=6, seed-42 diagnostic ranks features and tunes L1 "
            "regularization using outer-training data with grouped inner CV. "
            "Intervals are paired 2,000-replicate question-group bootstraps "
            "within fixed outer test folds. It is a sensitivity analysis, not a "
            "preregistered three-seed result.",
            "",
            "## Machine-readable summary",
            "",
            f"- Complete sweep cells rendered: "
            f"{n_cells}/{len(windows) * len(seeds)}.",
            f"- Mean paired T={windows[-1]} minus T={windows[0]} gain: "
            f"{mean_gain:+.4f} AP.",
            "- Numeric sources: `window_sweep_seed_metrics.csv` and "
            "`positional_sae_budget_sensitivity.csv`.",
            "",
        ]
    )
    temporary = output_dir / "reviewer_figures.tmp.md"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, output_dir / "reviewer_figures.md")


def render(
    root: Path,
    output_dir: Path,
    *,
    allow_partial: bool,
    windows: tuple[int, ...] = WINDOWS,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    _configure_matplotlib()
    rows = _load_rows(
        root,
        allow_partial=allow_partial,
        windows=windows,
        seeds=seeds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_seed_metrics(rows, output_dir)
    _write_budget_csv(output_dir)
    summary = _plot_window_curve(
        rows, output_dir, windows=windows, seeds=seeds
    )
    _plot_shuffle_gap(rows, output_dir, windows=windows, seeds=seeds)
    _plot_order_gap(rows, output_dir, windows=windows, seeds=seeds)
    _write_markdown(
        output_dir,
        n_cells=len(rows),
        mean_gain=summary["paired_endpoint_gain_mean"],
        windows=windows,
        seeds=seeds,
    )
    payload = {
        "status": "complete",
        "n_cells": len(rows),
        **summary,
    }
    temporary = output_dir / "reviewer_figures.tmp.json"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_dir / "reviewer_figures.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--windows",
        default=",".join(str(window) for window in WINDOWS),
        help="comma-separated expected window grid",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="comma-separated expected seed grid",
    )
    args = parser.parse_args()
    windows = tuple(int(value) for value in args.windows.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not windows or not seeds:
        raise ValueError("--windows and --seeds must not be empty")
    print(
        json.dumps(
            render(
                args.root,
                args.output_dir,
                allow_partial=args.allow_partial,
                windows=windows,
                seeds=seeds,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
