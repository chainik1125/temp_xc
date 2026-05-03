"""Headline plot — BASELINE-CORRECTED.

Replaces the original headline plot. Y-axis is Δnet vs mag=0 baseline,
NOT raw net. The mag=0 row is treated as the cut-and-continue resampling
noise floor; the steering effect is the *additional* effect on top of it.

Main panel: Δnet rescues (extra_rescue − broke) vs raw magnitude.
Side panels: extra rescues vs raw magnitude, regressions caused by
steering (broke) vs raw magnitude.

Usage:
  python -m experiments.ward_backtracking_txc.plot.headline_baseline_corrected \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
      --steering-effect <out_root>/steering_effect.parquet \
      --out <out_root>
"""
from __future__ import annotations
import argparse
import json
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
log = logging.getLogger("ward_txc.plot.headline_corrected")

# arch lowercase (parquet) → display label
ARCH_TO_LABEL = {
    "txc": "TXC", "txc_h8": "TXC-H8", "topk_sae": "SAE",
    "tsae": "TSAE-paper", "tfa": "TFA", "mlc": "MLC",
}


def panel_delta(ax, df: pd.DataFrame, arches: list[str], col: str, ylabel: str):
    for arch in arches:
        sub = df[df["arch"] == arch].sort_values("magnitude")
        label = ARCH_TO_LABEL.get(arch, arch)
        ax.plot(sub["magnitude"], sub[col],
                "-o", label=label,
                color=ARCH_PALETTE.get(label, "#888"),
                markersize=4, linewidth=1.6)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#888", linewidth=0.8, linestyle=":")
    ax.set_xlabel("steering magnitude (raw)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def render(eff_summary: pd.DataFrame, out_path: Path,
           label_filter: set | None = None):
    arches = sorted(eff_summary["arch"].unique())
    if label_filter is not None:
        arches = [a for a in arches if ARCH_TO_LABEL.get(a) in label_filter]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    panel_delta(axes[0], eff_summary, arches, "delta_net_vs_baseline",
                "Δ net rescues vs mag=0 baseline\n(extra_rescue − broke)")
    panel_delta(axes[1], eff_summary, arches, "n_extra_rescue",
                "# extra rescues caused by steering\n(baseline-incorrect → steered-correct)")
    panel_delta(axes[2], eff_summary, arches, "n_broke_by_steering",
                "# regressions caused by steering\n(baseline-correct → steered-incorrect)")
    fig.suptitle(
        "Backtracking steering effect (baseline-corrected; mag=0 is cut-and-continue noise)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steering-effect-summary", type=Path, required=True,
                   help="path to steering_effect_summary.csv from build_steering_effect.py")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    args = p.parse_args(argv)

    eff = pd.read_csv(args.steering_effect_summary)
    args.out.mkdir(parents=True, exist_ok=True)
    render(eff, args.out / "headline_baseline_corrected_5arch.png",
           label_filter=HEADLINE_LABELS)
    render(eff, args.out / "headline_baseline_corrected_6arch_appendix.png",
           label_filter=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
