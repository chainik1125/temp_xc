"""Baseline-corrected per-arch flip-matrix grid.

Renders one 2×2 confusion per arch: rows = mag=0 baseline correctness,
cols = steered correctness. The lower-left cell is "extra rescue caused
by steering" (`baseline_incorrect → steered_correct`); upper-right is
"steering broke a baseline-correct" (`baseline_correct → steered_incorrect`).

Annotates raw counts and within-row proportions. Title shows per-arch
peak magnitude (= argmax of `delta_net_vs_baseline`) and Δnet vs the
mag=0 baseline.

Usage:
  python -m experiments.ward_backtracking_txc.plot.flip_matrix_grid_corrected \
      --steering-effect <out_root>/steering_effect.parquet \
      --out <out_root>
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ward_backtracking_txc.plot.headline_steering import (
    ARCH_PALETTE, HEADLINE_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.plot.flip_grid_corrected")

ARCH_TO_LABEL = {
    "txc": "TXC", "txc_h8": "TXC-H8", "topk_sae": "SAE",
    "tsae": "TSAE-paper", "tfa": "TFA", "mlc": "MLC",
}


def best_mag_per_arch(df: pd.DataFrame) -> dict[str, float]:
    rows = []
    for (arch, mag), g in df.groupby(["arch", "magnitude"]):
        ic = (g["change_direction"] == "extra_rescue").sum()
        ci = (g["change_direction"] == "broke").sum()
        rows.append({"arch": arch, "magnitude": mag, "delta_net": ic - ci})
    s = pd.DataFrame(rows)
    return {arch: g.loc[g["delta_net"].idxmax(), "magnitude"]
            for arch, g in s.groupby("arch")}


def confusion_at(df: pd.DataFrame, arch: str, mag: float) -> dict:
    sub = df[(df["arch"] == arch) & (df["magnitude"] == mag)]
    bc_sc = ((sub["after_baseline"]) & (sub["after_steered"])).sum()
    bc_si = ((sub["after_baseline"]) & (~sub["after_steered"])).sum()
    bi_sc = ((~sub["after_baseline"]) & (sub["after_steered"])).sum()
    bi_si = ((~sub["after_baseline"]) & (~sub["after_steered"])).sum()
    return {
        "n_bc_sc": int(bc_sc),
        "n_bc_si": int(bc_si),  # broke
        "n_bi_sc": int(bi_sc),  # extra rescue
        "n_bi_si": int(bi_si),
        "n_baseline_correct": int(sub["after_baseline"].sum()),
        "n_baseline_incorrect": int((~sub["after_baseline"]).sum()),
    }


def render_grid(df: pd.DataFrame, out_path: Path,
                label_filter: set | None = None,
                fixed_magnitude: float | None = None):
    bests = best_mag_per_arch(df)
    arches = sorted(bests.keys())
    if label_filter is not None:
        arches = [a for a in arches if ARCH_TO_LABEL.get(a) in label_filter]
    n = len(arches)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows),
                             squeeze=False)

    for i, arch in enumerate(arches):
        ax = axes[i // cols][i % cols]
        mag = fixed_magnitude if fixed_magnitude is not None else bests[arch]
        c = confusion_at(df, arch, mag)
        mat = np.array([[c["n_bc_sc"], c["n_bc_si"]],
                        [c["n_bi_sc"], c["n_bi_si"]]])
        row_totals = mat.sum(axis=1, keepdims=True).clip(min=1)
        prop = mat / row_totals
        delta_net = c["n_bi_sc"] - c["n_bc_si"]

        label = ARCH_TO_LABEL.get(arch, arch)
        color = ARCH_PALETTE.get(label, "#888")
        im = ax.imshow(prop, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for col in range(2):
                txt = f"{int(mat[r, col])}\n({prop[r, col]:.2f})"
                txt_color = "white" if prop[r, col] > 0.5 else "black"
                ax.text(col, r, txt, ha="center", va="center",
                        color=txt_color, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["after-steered\ncorrect", "after-steered\nincorrect"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([
            f"baseline correct\n(n={c['n_baseline_correct']})",
            f"baseline incorrect\n(n={c['n_baseline_incorrect']})",
        ])
        ax.set_title(
            f"{label}  @ mag={mag:+.1f}\n"
            f"Δnet vs baseline = {delta_net:+d}\n"
            f"(extra_rescue={c['n_bi_sc']}, broke={c['n_bc_si']})",
            color=color, fontsize=10,
        )
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    title = ("Baseline-corrected flip matrices: rows = mag=0 outcome, "
             "cols = steered outcome")
    if fixed_magnitude is not None:
        title += f"   (all at mag={fixed_magnitude:+.1f})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steering-effect", type=Path, required=True,
                   help="steering_effect.parquet from build_steering_effect.py")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.steering_effect)
    args.out.mkdir(parents=True, exist_ok=True)
    render_grid(df, args.out / "flip_matrix_corrected_headline.png",
                label_filter=HEADLINE_LABELS)
    render_grid(df, args.out / "flip_matrix_corrected_appendix.png",
                label_filter=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
