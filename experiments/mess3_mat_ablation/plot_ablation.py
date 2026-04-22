"""Produce PDF plots for the Mess3 Mat-TopK-SAE ablation.

Reads per-cell JSONs under `results/cell_delta_*/results.json`, plus
the Bayes ceiling from `r2_ceiling.json` if present (else computes
R²/R²_max using the per-cell `linear_probe` number as a proxy ceiling).

Outputs two PDFs under `plots/`:
  fig1_gap_recovery_2x2.pdf   — R²/R²_max vs δ for the 4 arches
  fig2_feature_diversity.pdf  — |{argmax feature per component}| per cell

Both figures are laid out with publication-grade defaults (1-col, 300dpi,
white background, serif labels, single legend).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Four archs that form the 2x2 ablation. Order matters for plot legend.
# (name_in_config, legend_label, color, linestyle, marker)
ARCH_STYLE = [
    ("TopK SAE",                 "TopK SAE (no-matryoshka, no-window)",  "#4C72B0", "--", "o"),
    ("TXC",                      "TXC (no-matryoshka, window)",          "#4C72B0", "-",  "o"),
    ("MatryoshkaSAE (no-window)", "MatryoshkaSAE (matryoshka, no-window) ← ablation", "#DD8452", "--", "s"),
    ("MatryoshkaTXC",            "MatTXC (matryoshka, window)",          "#DD8452", "-",  "s"),
]


def _load_cell(cell_dir: Path) -> dict | None:
    p = cell_dir / "results.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _load_r2_ceiling(root: Path) -> dict[float, float]:
    """Map δ → R²_max from r2_ceiling.json. Empty dict if absent."""
    p = root / "r2_ceiling.json"
    if not p.exists():
        return {}
    with p.open() as f:
        data = json.load(f)
    # Expected schema: {"cells": [{"delta": 0.15, "r2_max": 0.36}, ...]}
    return {float(c["delta"]): float(c.get("r2_max", float("nan"))) for c in data.get("cells", [])}


def _arch_r2_over_delta(results: dict[float, dict], arch_name: str) -> list[float | None]:
    """Return the per-δ best-single-feature-R² summed across components for one arch.

    Matches the "all" column in Dmitry's separation_scaling.md table.
    Dmitry's `architectures` field is a DICT keyed by arch name (not a list).
    """
    out: list[float | None] = []
    for delta in sorted(results):
        cell = results[delta]
        arches_dict = cell.get("architectures", {})
        match = arches_dict.get(arch_name)
        if match is None:
            out.append(None)
            continue
        per_comp = match.get("per_component_best_r2")
        if not per_comp:
            out.append(None)
            continue
        out.append(float(np.sum(per_comp)))
    return out


def _feature_diversity(cell: dict, arch_name: str) -> int | None:
    """For one (cell, arch), count distinct argmax features across the
    C=3 Mess3 components. None if the arch didn't run on this cell, or
    if the per-component feature IDs weren't persisted.

    The IDs are populated by `run_ablation.py`'s wrappers around
    `run_architecture` + `run_one_cell` (Dmitry's driver computes them
    in `summarize_single_feature_probe` but drops them on the floor).
    """
    arches_dict = cell.get("architectures", {})
    match = arches_dict.get(arch_name)
    if match is None:
        return None
    idxs = match.get("per_component_best_feature")
    if idxs is None:
        return None
    return len(set(int(i) for i in idxs))


def _load_all(results_root: Path) -> dict[float, dict]:
    out: dict[float, dict] = {}
    for d in sorted(results_root.glob("cell_delta_*")):
        try:
            delta = float(d.name.replace("cell_delta_", ""))
        except ValueError:
            continue
        payload = _load_cell(d)
        if payload is not None:
            out[delta] = payload
    return out


def plot_gap_recovery(results: dict[float, dict], ceiling: dict[float, float], out_path: Path) -> None:
    """fig1: gap recovery (summed per-component best R² / R²_max) vs δ."""
    deltas = sorted(results)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    for arch_name, label, color, ls, marker in ARCH_STYLE:
        raw = _arch_r2_over_delta(results, arch_name)
        if ceiling:
            y = [
                (r / ceiling.get(d)) if (r is not None and ceiling.get(d, 0) > 0) else None
                for r, d in zip(raw, deltas)
            ]
        else:
            y = raw
        xs_ys = [(d, v) for d, v in zip(deltas, y) if v is not None]
        if not xs_ys:
            continue
        xs, ys = zip(*xs_ys)
        ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                markersize=6, linewidth=2.0, label=label)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("δ  (temporal-separation strength)")
    ylabel = "R² / R²_max  (gap recovery)" if ceiling else "R²  (summed best single-feature across components)"
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.05, 1.1 if ceiling else None)
    ax.set_title("Mess3: Mat-TopK-SAE ablation — 2×2 on matryoshka × temporal window")
    # Legend outside the axes so it never overlaps the curves (which
    # rise into the upper-right and used to collide with an upper-left
    # legend at low δ values).
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def plot_feature_diversity(results: dict[float, dict], out_path: Path) -> None:
    """fig2: distinct argmax features per component, grouped by arch × δ.

    Skips entirely if no arch has diversity data — which is the current
    default until we extend Dmitry's driver to persist argmax feature IDs.
    Rather than produce a blank bar chart, emits a placeholder PDF with
    a note.
    """
    deltas = sorted(results)
    arches = [name for name, *_ in ARCH_STYLE]
    colors = {name: color for name, _, color, _, _ in ARCH_STYLE}

    have_any = any(
        _feature_diversity(results[d], arch_name) is not None
        for d in deltas for arch_name in arches
    )
    if not have_any:
        fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=300)
        ax.text(
            0.5, 0.5,
            "Feature-diversity plot requires per_component_best_feature IDs,\n"
            "which run_ablation.py's wrappers should inject. None present in\n"
            "the loaded cell JSONs — re-run with the updated run_ablation.py\n"
            "(delete results/cell_delta_*/results.json first to defeat\n"
            "--skip-existing).",
            ha="center", va="center", fontsize=10,
        )
        ax.set_axis_off()
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[warn] wrote placeholder to {out_path} (no diversity data available)")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    width = 0.8 / len(arches)
    x = np.arange(len(deltas))

    for i, arch_name in enumerate(arches):
        heights = [_feature_diversity(results[d], arch_name) or 0 for d in deltas]
        offset = (i - (len(arches) - 1) / 2) * width
        ax.bar(x + offset, heights, width=width * 0.9, color=colors[arch_name], label=arch_name, alpha=0.85)

    ax.axhline(3.0, color="black", linestyle=":", linewidth=1, alpha=0.7,
               label="ideal (one feature per component)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.2f}" for d in deltas])
    ax.set_xlabel("δ")
    ax.set_ylabel("|{argmax feature per Mess3 component}|")
    ax.set_title("Feature diversity: does each arch recover the indicator basis?")
    ax.set_ylim(0, 3.6)
    # Legend outside the axes — bars at δ=0 reach the top (diversity=3)
    # and any in-axes legend collides with them.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--out-dir", type=Path, default=Path("plots"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = _load_all(args.results_root)
    if not results:
        raise SystemExit(f"[error] no cell_delta_* results found under {args.results_root}")

    ceiling = _load_r2_ceiling(args.results_root.parent)
    plot_gap_recovery(results, ceiling, args.out_dir / "fig1_gap_recovery_2x2.pdf")
    plot_feature_diversity(results, args.out_dir / "fig2_feature_diversity.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
