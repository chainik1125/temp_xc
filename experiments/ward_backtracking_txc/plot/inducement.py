"""Inducement plot — Dmitry's pivot from rescue-correctness to inducement.

Y-axis = backtracking induced per generation, baseline-corrected vs mag=0.
Two metrics:
- **Δ keyword_rate** (cheap proxy: count("wait"|"hmm")/n_words)
- **Δ genuine_count** (Sonnet 4.6 judge for genuine backtracking events,
  filters out filler / pseudo-bt / loops)

Both are baseline-corrected (subtract the per-(arch, qid) value at mag=0).

Stability annotation per arch: number of magnitudes (out of 24 nonzero)
where Δ is positive.

Usage:
  python -m experiments.ward_backtracking_txc.plot.inducement \
      --inducement <out_root>/inducement.parquet \
      --out <out_root>
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.ward_backtracking_txc.plot.headline_steering import (
    ARCH_PALETTE, HEADLINE_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.plot.inducement")


def panel(ax, summary: pd.DataFrame, arches: list[str], col: str, ylabel: str):
    for arch in arches:
        sub = summary[summary["arch"] == arch].sort_values("magnitude")
        if col not in sub.columns or sub[col].isna().all():
            continue
        ax.plot(sub["magnitude"], sub[col], "-o",
                label=arch, color=ARCH_PALETTE.get(arch, "#888"),
                markersize=4, linewidth=1.6)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#888", linewidth=0.8, linestyle=":")
    ax.set_xlabel("steering magnitude (raw)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def render(summary: pd.DataFrame, out_path: Path,
           label_filter: set | None = None,
           include_genuine: bool = True):
    arches = sorted(summary["arch"].unique())
    if label_filter is not None:
        arches = [a for a in arches if a in label_filter]
    panels = [
        ("delta_keyword_rate",
         "Δ keyword_rate vs mag=0\n(cheap proxy: 'wait'+'hmm' / n_words)"),
        ("keyword_rate",
         "raw keyword_rate"),
    ]
    has_genuine = include_genuine and "delta_genuine_count" in summary.columns and not summary["delta_genuine_count"].isna().all()
    if has_genuine:
        panels.insert(0, ("delta_genuine_count",
                          "Δ Sonnet genuine_count vs mag=0\n(gold standard; filters filler/loops)"))
        panels.insert(2, ("genuine_count", "raw Sonnet genuine_count"))

    n = len(panels)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    for i, (col, ylabel) in enumerate(panels):
        ax = axes[i // cols][i % cols]
        panel(ax, summary, arches, col, ylabel)
    title = "Backtracking inducement (Dmitry pivot: did steering induce more backtracking?)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--inducement", type=Path,
                   default=Path("results/ward_backtracking_txc/b3_math500_cut25/inducement_summary.csv"),
                   help="inducement_summary.csv from build_inducement.py")
    p.add_argument("--out", type=Path,
                   default=Path("results/ward_backtracking_txc/b3_math500_cut25"))
    args = p.parse_args(argv)
    summary = pd.read_csv(args.inducement)
    args.out.mkdir(parents=True, exist_ok=True)
    render(summary, args.out / "inducement_headline.png", label_filter=HEADLINE_LABELS)
    render(summary, args.out / "inducement_appendix.png", label_filter=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
