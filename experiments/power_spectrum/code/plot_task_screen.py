"""Aggregate and plot the synthetic task-spectrum screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = POWER_ROOT / "results" / "task_screen.json"
DEFAULT_FIGURES = POWER_ROOT / "figures"
DEFAULT_RESULTS = POWER_ROOT / "results"

METHODS = {
    "power_ac": ("Power (AC)", "#0072B2", "o"),
    "cross_ac": ("Cross-spectrum", "#D55E00", "s"),
    "dc_vector": ("Signed DC", "#009E73", "^"),
}
TARGETS = (
    ("frequency_velocity", "Periodic velocity"),
    ("phasepair_pair", "Periodic magnitude"),
    ("phasepair_sign", "Phase-only sign"),
    ("signed_motion_sign", "Motion direction"),
    ("permuted_schedule", "Permuted schedule"),
    ("recipe_equality", "Higher-order equality"),
)
SUMMARY_LABELS = {
    "toy_backtracking_selfexcite_d64": "backtracking",
    "toy_changepoint_modes_d64": "changepoint",
    "toy_hedging_drift_d64": "hedging drift",
    "toy_cyclic_circle_M101_d128": "cyclic circle",
    "toy_phasepair_M101_d24": "phase pair",
    "toy_permuted_circle_M101_d128": "permuted circle",
    "toy_signed_motion_M19_d40": "signed motion",
    "toy_recipe_instruction_d64": "recipe",
    "toy_assumption_consequence_d64": "assumption",
    "toy_colored_sources_N32_D2_d32": "colored sources",
}
SUMMARY_OFFSETS = {
    "toy_backtracking_selfexcite_d64": (5, -15),
    "toy_changepoint_modes_d64": (6, 5),
    "toy_hedging_drift_d64": (5, 5),
    "toy_cyclic_circle_M101_d128": (5, 5),
    "toy_phasepair_M101_d24": (-62, 5),
    "toy_recipe_instruction_d64": (5, -13),
    "toy_assumption_consequence_d64": (5, 12),
}


def _probe_frame(payload: dict) -> pd.DataFrame:
    rows = []
    for probe in payload["probes"]:
        row = {
            "target": probe["target"],
            "datasource": probe["datasource"],
            "target_kind": probe["target_kind"],
            "seed": probe["seed"],
            "tile_size": probe["tile_size"],
        }
        for key in ("power_full", "power_ac", "cross_full", "cross_ac", "dc_vector"):
            row[f"{key}_score"] = probe[key]["score_mean"]
            row[f"{key}_fold_std"] = probe[key]["score_std"]
            row[f"{key}_null"] = probe[key]["shuffled_mean"]
            row[f"{key}_chance"] = probe[key]["chance"]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_probes(frame: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in frame
        if column.endswith(("_score", "_fold_std", "_null", "_chance"))
    ]
    grouped = frame.groupby(["target", "datasource", "target_kind", "tile_size"])
    mean = grouped[value_columns].mean().add_suffix("_mean")
    seed_std = grouped[[column for column in value_columns if column.endswith("_score")]].std(
        ddof=1
    )
    seed_std = seed_std.fillna(0.0).add_suffix("_seed_std")
    return mean.join(seed_std).reset_index()


def plot_separability(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.7), sharex=True)
    for axis, (target, title) in zip(axes.flat, TARGETS, strict=True):
        part = frame[frame["target"] == target]
        for key, (label, color, marker) in METHODS.items():
            summary = (
                part.groupby("tile_size")[f"{key}_score"]
                .agg(["mean", "std"])
                .reset_index()
            )
            axis.errorbar(
                summary["tile_size"],
                summary["mean"],
                yerr=summary["std"].fillna(0.0),
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.6,
                capsize=2,
                label=label,
            )
        chance = float(part["power_ac_chance"].mean())
        axis.axhline(chance, color="0.55", linestyle=":", linewidth=1.0)
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_xticks((4, 8, 16, 32), labels=("4", "8", "16", "32"))
        axis.set_ylim(-0.03, 1.04)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Cross-validated accuracy")
    axes[1, 0].set_ylabel("Cross-validated accuracy")
    for axis in axes[1]:
        axis.set_xlabel("Window length T (tokens)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Which second-order representation reveals the task?", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_global_summaries(payload: dict, output: Path) -> None:
    frame = pd.DataFrame(payload["summaries"])
    summary = (
        frame.groupby("datasource")
        .agg(
            ac_low_fraction=("ac_low_fraction", "mean"),
            ac_low_std=("ac_low_fraction", "std"),
            ac_entropy=("ac_entropy", "mean"),
            directionality=("max_directionality", "mean"),
            dc_fraction=("dc_fraction", "mean"),
        )
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.6))
    size = 40 + 900 * summary["dc_fraction"].to_numpy()
    color = summary["directionality"].to_numpy()
    scatter = axis.scatter(
        summary["ac_low_fraction"],
        summary["ac_entropy"],
        s=size,
        c=color,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.7,
    )
    for row in summary.itertuples(index=False):
        label = SUMMARY_LABELS.get(row.datasource, row.datasource)
        offset = SUMMARY_OFFSETS.get(row.datasource, (4, 3))
        axis.annotate(
            label,
            (row.ac_low_fraction, row.ac_entropy),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Low-frequency share of AC power (f ≤ 1/8)")
    axis.set_ylabel("Normalized AC spectral entropy")
    axis.set_xlim(0.20, 0.61)
    axis.set_ylim(0.72, 1.015)
    axis.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Maximum lag directionality")
    axis.set_title("Global activation spectra overlap across temporal tasks")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    probes = _probe_frame(payload)
    aggregate = aggregate_probes(probes)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.results_dir / "task_screen_aggregate.csv", index=False)
    plot_separability(probes, args.figures_dir / "task_screen_separability.png")
    plot_global_summaries(payload, args.figures_dir / "task_spectrum_summary.png")

    selected = aggregate[
        aggregate["target"].isin(target for target, _ in TARGETS)
        & aggregate["tile_size"].isin((4, 32))
    ]
    columns = [
        "target",
        "tile_size",
        "power_ac_score_mean",
        "cross_ac_score_mean",
        "dc_vector_score_mean",
    ]
    print(selected[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
