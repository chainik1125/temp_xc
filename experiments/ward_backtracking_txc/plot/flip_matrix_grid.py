"""Per-arch flip-matrix grid plot.

Renders a 2×N grid of 2×2 confusion matrices: one per architecture at its
per-arch best magnitude (= argmax of `n_ic - n_ci`). Each cell shows the
4-cell flip matrix (cc / ci / ic / ii) with cell counts + proportions
annotated. Useful as a compact visual summary of "what does steering do
to which questions, per arch."

Usage:
  python -m experiments.ward_backtracking_txc.plot.flip_matrix_grid \
      --flip-matrix <out_root>/flip_matrix.parquet \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
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
import numpy as np
import pandas as pd

from experiments.ward_backtracking_txc.plot.headline_steering import (
    ARCH_PALETTE, HEADLINE_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.plot.flip_grid")


# Map arch lowercase (as it appears in flip_matrix parquet) → display label.
ARCH_TO_LABEL = {
    "txc":        "TXC",
    "txc_h8":     "TXC-H8",
    "topk_sae":   "SAE",
    "tsae":       "TSAE-paper",
    "tfa":        "TFA",
    "mlc":        "MLC",
}


def best_magnitude_per_arch(df: pd.DataFrame) -> dict[str, float]:
    rows = []
    for (arch, mag), g in df.groupby(["arch", "magnitude"]):
        n_ic = (g["transition"] == "ic").sum()
        n_ci = (g["transition"] == "ci").sum()
        rows.append({"arch": arch, "magnitude": mag, "net": n_ic - n_ci})
    agg = pd.DataFrame(rows)
    return {arch: g.loc[g["net"].idxmax(), "magnitude"] for arch, g in agg.groupby("arch")}


def confusion_at(df: pd.DataFrame, arch: str, mag: float) -> dict:
    sub = df[(df["arch"] == arch) & (df["magnitude"] == mag)]
    counts = sub["transition"].value_counts().to_dict()
    return {
        "n_cc": int(counts.get("cc", 0)),
        "n_ci": int(counts.get("ci", 0)),
        "n_ic": int(counts.get("ic", 0)),
        "n_ii": int(counts.get("ii", 0)),
        "n_total": len(sub),
        "n_before_correct": int(sub["before_correct"].sum()),
        "n_before_incorrect": int((~sub["before_correct"]).sum()),
    }


def render_grid(df: pd.DataFrame, out_path: Path, label_filter: set | None = None,
                fixed_magnitude: float | None = None):
    bests = best_magnitude_per_arch(df)
    arches = sorted(bests.keys())
    if label_filter is not None:
        arches = [a for a in arches if ARCH_TO_LABEL.get(a) in label_filter]
    n = len(arches)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.2 * rows),
                             squeeze=False)

    for i, arch in enumerate(arches):
        ax = axes[i // cols][i % cols]
        mag = fixed_magnitude if fixed_magnitude is not None else bests[arch]
        c = confusion_at(df, arch, mag)
        # 2×2 layout (rows = before, cols = after); rows from top: correct, incorrect
        mat = np.array([[c["n_cc"], c["n_ci"]],
                        [c["n_ic"], c["n_ii"]]])
        # Per-row totals for proportion annotation
        row_totals = mat.sum(axis=1, keepdims=True).clip(min=1)
        prop = mat / row_totals

        label = ARCH_TO_LABEL.get(arch, arch)
        color = ARCH_PALETTE.get(label, "#888")
        # Imshow with proportions for color, counts overlaid as text.
        im = ax.imshow(prop, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for col in range(2):
                txt = f"{int(mat[r, col])}\n({prop[r, col]:.2f})"
                txt_color = "white" if prop[r, col] > 0.5 else "black"
                ax.text(col, r, txt, ha="center", va="center",
                        color=txt_color, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["after correct", "after incorrect"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["before correct\n(n=" + str(c["n_before_correct"]) + ")",
                                                    "before incorrect\n(n=" + str(c["n_before_incorrect"]) + ")"])
        net = c["n_ic"] - c["n_ci"]
        ax.set_title(f"{label}  @ mag={mag:+.1f}\nnet rescues = {net:+d}  (ic−ci)",
                     color=color, fontsize=11)
        ax.set_xlabel("after-steering correctness")
        ax.set_ylabel("before-steering correctness")
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Hide unused cells
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    title = "Flip matrices: 2×2 confusion of correctness (per-arch best magnitude)"
    if fixed_magnitude is not None:
        title = f"Flip matrices @ mag={fixed_magnitude:+.1f}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--flip-matrix", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.flip_matrix)
    log.info("[flip] %d rows, archs=%s", len(df), sorted(df["arch"].unique()))

    args.out.mkdir(parents=True, exist_ok=True)
    # Headline: 5 archs at per-arch best mag
    render_grid(df, args.out / "flip_matrix_grid_headline.png",
                label_filter=HEADLINE_LABELS)
    # Appendix: all 6 archs
    render_grid(df, args.out / "flip_matrix_grid_appendix.png",
                label_filter=None)
    # Also a "fixed-magnitude" pair for reference: at mag=0 (control) and mag=+8.
    render_grid(df, args.out / "flip_matrix_grid_at_mag_0.png",
                label_filter=HEADLINE_LABELS, fixed_magnitude=0.0)
    render_grid(df, args.out / "flip_matrix_grid_at_mag_p8.png",
                label_filter=HEADLINE_LABELS, fixed_magnitude=8.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
