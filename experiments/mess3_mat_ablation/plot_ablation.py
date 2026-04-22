"""Produce PDF plots for the Mess3 Mat-TopK-SAE ablation.

Reads per-cell JSONs under `results/cell_delta_*/results.json`, plus
the Bayes ceiling from `r2_ceiling.json` if present (else computes
R²/R²_max using the per-cell `linear_probe` number as a proxy ceiling).

Outputs two PDFs under `plots/`:
  fig1_gap_recovery_2x2.pdf   — R² vs δ for the 4 arches
  fig2_feature_diversity.pdf  — |{argmax feature per component}| per cell

Styling aims for single-column publication use: serif typeface, ticks
inward, minimal grid, legend placed inside an empty region of the axes
with a faint boxed background so labels don't overlap data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Publication-style matplotlib defaults.
# ──────────────────────────────────────────────────────────────────────────
_PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.fancybox": False,
    "legend.edgecolor": "0.85",
    "pdf.fonttype": 42,  # embed TrueType so editors can edit text
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}


# Four archs that form the 2×2 ablation. Short legend labels — the
# "no-matryoshka / window" semantics is communicated via the linestyle
# (solid = window, dashed = no window) and color (blue = no-matryoshka,
# orange = matryoshka). The caption of the figure carries the decoder.
ARCH_STYLE = [
    # (name_in_config,             short_label,  color,     linestyle, marker)
    ("TopK SAE",                   "TopK SAE",   "#4C72B0", "--",      "o"),
    ("TXC",                        "TXC",        "#4C72B0", "-",       "o"),
    ("MatryoshkaSAE (no-window)",  "MatSAE",     "#DD8452", "--",      "s"),
    ("MatryoshkaTXC",              "MatTXC",     "#DD8452", "-",       "s"),
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
    return {float(c["delta"]): float(c.get("r2_max", float("nan"))) for c in data.get("cells", [])}


def _arch_r2_over_delta(results: dict[float, dict], arch_name: str) -> list[float | None]:
    """Per-δ summed best-single-feature R² across the 3 components."""
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
    """Count distinct argmax features across C=3 Mess3 components."""
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
    """fig1: summed-best-feature R² (or R²/R²_max if ceiling is known) vs δ."""
    deltas = sorted(results)
    with mpl.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))

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
                    markersize=4.5, linewidth=1.4, label=label)

        if ceiling:
            ax.axhline(1.0, color="0.5", linestyle=":", linewidth=0.8)

        ax.set_xlabel(r"$\delta$ (temporal-separation strength)")
        ylabel = r"$R^2 / R^2_{\max}$" if ceiling else r"summed best single-feature $R^2$"
        ax.set_ylabel(ylabel)
        if ceiling:
            ax.set_ylim(-0.05, 1.1)
        else:
            ax.set_ylim(-0.08, 0.78)
        ax.set_xlim(-0.01, 0.21)

        # Legend below the axes in a single row — matches the fig2
        # layout and keeps the plot area uncluttered.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=4, handlelength=2.0, handletextpad=0.5,
                  columnspacing=1.5, borderpad=0.4,
                  frameon=False)

        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.4)
        fig.tight_layout(pad=0.4)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"[done] wrote {out_path}")


def plot_feature_diversity(results: dict[float, dict], out_path: Path) -> None:
    """fig2: per-arch × δ bar chart of distinct argmax features per component."""
    deltas = sorted(results)
    arches = [name for name, *_ in ARCH_STYLE]
    labels = {name: label for name, label, *_ in ARCH_STYLE}
    colors = {name: color for name, _, color, *_ in ARCH_STYLE}
    # Distinguish no-window (faded) from window (saturated) within each color
    # family — avoids hatching clutter.
    alphas = {"TopK SAE": 0.55, "TXC": 1.0,
              "MatryoshkaSAE (no-window)": 0.55, "MatryoshkaTXC": 1.0}

    have_any = any(
        _feature_diversity(results[d], arch_name) is not None
        for d in deltas for arch_name in arches
    )
    if not have_any:
        with mpl.rc_context(_PUB_RC):
            fig, ax = plt.subplots(figsize=(4.2, 2.2))
            ax.text(
                0.5, 0.5,
                "Feature-diversity plot requires per_component_best_feature\n"
                "IDs. Re-run run_ablation.py after deleting per-cell\n"
                "results.json to defeat --skip-existing.",
                ha="center", va="center", fontsize=8,
            )
            ax.set_axis_off()
            fig.savefig(out_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
        print(f"[warn] wrote placeholder to {out_path}")
        return

    with mpl.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        width = 0.8 / len(arches)
        x = np.arange(len(deltas))

        for i, arch_name in enumerate(arches):
            heights = [_feature_diversity(results[d], arch_name) or 0 for d in deltas]
            offset = (i - (len(arches) - 1) / 2) * width
            ax.bar(x + offset, heights, width=width * 0.92,
                   color=colors[arch_name], edgecolor="white",
                   linewidth=0.4, alpha=alphas.get(arch_name, 1.0),
                   label=labels[arch_name])

        # Ideal reference line, labeled on the far left inside the axes
        # above the highest data there (3.0), so it never competes with
        # the legend region.
        ax.axhline(3.0, color="0.2", linestyle=":", linewidth=0.8,
                   zorder=0)
        ax.text(-0.4, 3.08, "ideal (one feature per component)",
                fontsize=7, color="0.3", va="bottom", ha="left",
                style="italic")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{d:.2f}" for d in deltas])
        ax.set_xlabel(r"$\delta$")
        ax.set_ylabel(r"$|\{\mathrm{argmax\ feature\ per\ component}\}|$")
        ax.set_ylim(0, 3.45)
        ax.set_yticks([0, 1, 2, 3])

        # Legend below the axes in a single row. No data can ever be
        # "below the x-axis" so this is collision-free and matches
        # common single-column paper layouts.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                  ncol=4, handlelength=1.3, handletextpad=0.5,
                  columnspacing=1.2, borderpad=0.4,
                  frameon=False)

        ax.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.4)
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
