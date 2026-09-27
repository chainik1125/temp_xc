#!/usr/bin/env python3
"""Redraw three rebuttal previews from small saved numbers, with no inference.

Run from any directory: python purified/scripts/camera_ready/replot.py
Requires NumPy and Matplotlib only. Writes PNG/SVG previews, tidy source CSV,
and a JSON provenance/validation sidecar. Does not modify manuscript figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
PALETTE_PATH = Path(__file__).with_name("palette.json")
PALETTE = json.loads(PALETTE_PATH.read_text())
COLORS = PALETTE["architecture_colors"]
SEEDS = (1, 2, 42)
DETECTION = REPO / "purified/results/neurips_rebuttal/backtracking_300k_seeded/publication/raw_detection_metrics.csv"
DETECTION_SUMMARY = DETECTION.with_name("summary.json")
WINDOW = REPO / "purified/results/neurips_rebuttal/backtracking_window_sweep_t16/reviewer-five-point-v1/publication/window_sweep_seed_metrics.csv"
MEDICAL = REPO / "purified/artifacts/camera_ready_2026/hf_reviewer_results/reviewer_seed_audit_2026-07-27/medical_em/steering_three_seed_summary.json"
DEFAULT_OUTPUT = REPO / "purified/artifacts/camera_ready_2026/previews"


def relative(path: Path) -> str:
    return str(path.relative_to(REPO))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def source_row(figure, series, seed, x, value, source, protocol):
    return dict(figure=figure, series=series, seed=seed, x=x, value=value,
                source_path=relative(source), protocol=protocol)


def figure_axes(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.25)
    fig.text(0.12, 0.95, title, fontsize=14, weight="bold", va="top")
    fig.text(0.12, 0.895, subtitle, fontsize=9.4, color="#525D69", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, output: Path, name: str):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def detection_plot(output: Path, export: list[dict]) -> dict:
    rows = read_csv(DETECTION)
    metadata = json.loads(DETECTION_SUMMARY.read_text())
    grid = metadata["S_grid"]
    replicated = [r for r in rows if r["source"] == "corrected_replication"]
    values = np.array([[float(next(r["pr_auc"] for r in replicated
                                  if int(r["seed"]) == seed and int(r["S"]) == k))
                        for k in grid] for seed in SEEDS])
    means, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    headline = grid.index(8)
    assert np.isclose(means[headline], metadata["txc_headline_pr_auc_mean"])
    assert np.isclose(sd[headline], metadata["txc_headline_pr_auc_sd"])
    fig, ax = figure_axes("Backtracking detection · 300K training steps",
                         "Corrected TXC-base replication; T-SAE comparisons each use seed 42.")
    for i, seed in enumerate(SEEDS):
        ax.plot(grid, values[i], color=COLORS["txc_base"], alpha=0.22, lw=0.8,
                marker=PALETTE["seeds"][str(seed)], ms=3)
        for k, value in zip(grid, values[i]):
            export.append(source_row("detection_300k", "TXC-base 32k corrected", seed,
                                     k, value, DETECTION, "c7-detection-seeded-v1"))
    ax.fill_between(grid, means-sd, means+sd, color=COLORS["txc_base"], alpha=0.11)
    ax.plot(grid, means, color=COLORS["txc_base"], lw=2.2, marker="o", ms=5,
            label="TXC-base 32k · mean ± SD, 3 seeds")
    comparisons = [
        ("T-SAE 16k", "new_width_sensitivity", "-", "s", "T-SAE 16k · new width control, seed 42"),
        ("T-SAE 32k", "submitted_rounded_table_reference", ":", "D", "T-SAE 32k · historical rounded reference")
    ]
    for arch, source, style, marker, label in comparisons:
        selected = [r for r in rows if r["source"] == source and r["architecture"] == arch]
        y = [float(next(r["pr_auc"] for r in selected if int(r["S"]) == k)) for k in grid]
        ax.plot(grid, y, color=COLORS["tsae"], linestyle=style, marker=marker,
                fillstyle="none" if style == ":" else "full", ms=5, lw=1.7, label=label)
        for k, value in zip(grid, y):
            export.append(source_row("detection_300k", arch+" / "+source, 42, k,
                                     value, DETECTION, source))
    ax.axhline(metadata["positive_fraction"], color=PALETTE["neutral"], linestyle="--",
               lw=1, label=f"Class prior · {metadata['positive_fraction']:.4f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid, [str(k) for k in grid])
    ax.set_xlabel("Sparse-probe feature budget S")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0.115, 0.282)
    fig.subplots_adjust(bottom=0.34)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.105, 0.205), fontsize=8.7, frameon=False)
    save(fig, output, "backtracking_detection_300k")
    return dict(txc_S8_mean=float(means[headline]), txc_S8_sample_sd=float(sd[headline]),
                seeds=list(SEEDS), historical_reference_has_errorbars=False)


def window_plot(output: Path, export: list[dict]) -> dict:
    rows = read_csv(WINDOW)
    windows = sorted({int(r["window"]) for r in rows})
    assert len(rows) == len(windows)*len(SEEDS)
    fig, ax = figure_axes("Window length and order controls · 20K steps",
                         "Matched 32-feature probe; dictionary seeds 1, 2, 42. Bands show sample SD.")
    series = [("txc_ordered_ap", "TXC-base · ordered", "txc_base", "-", "o"),
              ("txc_shuffle_ap", "TXC-base · shuffled at evaluation", "txc_base", "--", "s"),
              ("sae_invariant_ap", "SAE · order-invariant readout", "topk_sae", "-.", "D")]
    check = {}
    for key, label, arch, style, marker in series:
        values = np.array([[float(next(r[key] for r in rows if int(r["seed"]) == seed
                                       and int(r["window"]) == window))
                            for window in windows] for seed in SEEDS])
        means, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.fill_between(windows, means-sd, means+sd, color=COLORS[arch], alpha=0.09)
        ax.plot(windows, means, color=COLORS[arch], linestyle=style,
                marker=marker, ms=5, lw=1.9, label=label)
        for i, seed in enumerate(SEEDS):
            for window, value in zip(windows, values[i]):
                export.append(source_row("window_sweep_20k", label, seed, window,
                                         value, WINDOW, "reviewer-five-point-v1"))
        check[key] = dict(windows=windows, mean=means.tolist(), sample_sd=sd.tolist())
    ax.set_xticks(windows)
    ax.set_xlabel("Window length T (tokens)")
    ax.set_ylabel("Detection average precision")
    ax.set_ylim(0.19, 0.278)
    fig.subplots_adjust(bottom=0.34)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.105, 0.205), fontsize=9, frameon=False)
    fig.text(0.12, 0.025, "Shuffling is a fixed-probe sensitivity test; it introduces covariate shift.",
             fontsize=8.4, color="#525D69")
    save(fig, output, "backtracking_window_20k")
    return check


def medical_plot(output: Path, export: list[dict]) -> dict:
    data = json.loads(MEDICAL.read_text())
    by_seed = {int(r["seed"]): r for r in data["seeds"]}
    assert set(by_seed) == set(SEEDS)
    values = np.array([by_seed[s]["alignment_delta"] for s in SEEDS], dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    fig, ax = figure_axes("Medical misalignment steering · TXC-base",
                         "Alignment dynamic range over canonical + dense alpha grid; coherence ≥ 70.")
    for index, (seed, value) in enumerate(zip(SEEDS, values)):
        ax.scatter(value, index, s=80, color=COLORS["txc_base"],
                   marker=PALETTE["seeds"][str(seed)], zorder=4)
        ax.text(value+0.45, index, f"{value:.3f}", va="center", fontsize=10)
        export.append(source_row("medical_em_steering", "TXC-base", seed, seed,
                                 value, MEDICAL, data["protocol"]))
    ax.axvline(mean, color=COLORS["txc_base"], alpha=0.65, linestyle="--", lw=1.2)
    ax.set_yticks(range(3), [f"Seed {s}" for s in SEEDS])
    ax.invert_yaxis()
    ax.set_ylim(2.5, -0.5)
    ax.set_xlim(0, 26.5)
    ax.set_xlabel("Alignment dynamic range (score points)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#E7E9EC", linewidth=0.7)
    fig.text(0.12, 0.13, f"Mean {mean:.3f}   ·   sample SD {sd:.3f}   ·   3 dictionary seeds",
             fontsize=10, color=COLORS["txc_base"])
    fig.text(0.12, 0.075, "Seed values are verified summaries; this preview adds no baseline or error bars.",
             fontsize=8.6, color="#525D69")
    save(fig, output, "medical_em_steering_seeds")
    return dict(mean=mean, sample_sd=sd, values=dict(zip(map(str, SEEDS), values.tolist())),
                baseline_included=False, errorbars_included=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.labelcolor": PALETTE["text"],
                         "text.color": PALETTE["text"], "axes.spines.top": False,
                         "axes.spines.right": False})
    export = []
    checks = dict(detection=detection_plot(args.output_dir, export),
                  window=window_plot(args.output_dir, export),
                  medical=medical_plot(args.output_dir, export))
    with (args.output_dir / "plot_source_values.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    sources = [DETECTION, DETECTION_SUMMARY, WINDOW, MEDICAL, PALETTE_PATH, Path(__file__)]
    result = dict(runtime_seconds=round(time.perf_counter()-started, 3),
                  source_rows=len(export), checks=checks,
                  sources={relative(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  outputs=[p.name for p in sorted(args.output_dir.glob("*")) if p.is_file()],
                  limits=["Preview design, not manuscript replacement.",
                          "300K replication, 20K window sweep, and medical steering remain separate experiments.",
                          "T-SAE historical values are rounded single-seed references without fabricated uncertainty."])
    (args.output_dir / "replot_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ["runtime_seconds", "source_rows", "outputs"]}, indent=2))


if __name__ == "__main__":
    main()
