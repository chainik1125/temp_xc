"""Paper-ready figure: per-arch SEMANTIC label-count at N=32 across
concat_A, concat_B, concat_random. Random-concat is the generalisation
control; emphasised visually.

Reads `results/autointerp/*__seed*__concat*__labels.json` produced by
`run_autointerp.py` and emits:

    results/phase61_rigorous_headline.png
    results/phase61_rigorous_headline.thumb.png

(PNG for docs + thumbnail for agent inspection per CLAUDE.md convention.)

Layout: grouped bar chart, one row per arch (sorted by random-concat
semantic count), three bars (A / B / random) per arch. Horizontal so
arch names are readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "experiments/phase6_qualitative_latents/results/autointerp"
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/results"

# Display names for the 10-arch cohort (sorted as logical grouping
# of TXC-family / MLC / T-SAE / TFA).
DISPLAY_NAMES = {
    "agentic_txc_02": "TXC (baseline)",
    "agentic_txc_02_batchtopk": "TXC+BatchTopK (Cycle F)",
    "agentic_txc_09_auxk": "TXC+AuxK (Cycle A)",
    "agentic_txc_10_bare": "TXC+anti-dead (Track 2)",
    "agentic_txc_11_stack": "TXC+BatchTopK+AuxK (Cycle H)",
    "agentic_txc_12_bare_batchtopk": "TXC+BatchTopK+anti-dead (2x2)",
    "phase62_c1_track2_matryoshka": "TXC+anti-dead+matr (C1)",
    "phase62_c2_track2_contrastive": "TXC+anti-dead+contr (C2)",
    "phase62_c3_track2_matryoshka_contrastive": "TXC+anti-dead+matr+contr (C3)",
    "agentic_mlc_08": "MLC (Phase 5.7)",
    "tsae_ours": "T-SAE (naive port)",
    "tsae_paper": "T-SAE (paper-faithful)",
    "tfa_big": "TFA",
}


def load_cells(seed: int = 42):
    rows = []
    for p in sorted(IN_DIR.glob(f"*__seed{seed}__concat*__labels.json")):
        d = json.loads(p.read_text())
        rows.append({
            "arch": d["arch"],
            "concat": d["concat"],
            "sem": d["metrics"]["semantic_count"],
            "N": d["metrics"]["N"],
            "cov": d["metrics"]["passage_coverage_count"],
            "P": d["metrics"]["n_passages"],
            "disagree": d["metrics"]["judge_disagreement_rate"],
        })
    return rows


def make_plot(rows, out_png: Path, seed: int = 42, figtitle: str | None = None):
    # Group by arch -> {concat: sem}
    by_arch: dict[str, dict[str, int]] = {}
    for r in rows:
        by_arch.setdefault(r["arch"], {})[r["concat"]] = r["sem"]

    # Sort by random-concat sem (descending), missing → 0
    archs = sorted(by_arch.keys(),
                   key=lambda a: by_arch[a].get("random", -1), reverse=True)

    n = len(archs)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * n + 1.5)))
    y = np.arange(n)[::-1]
    bar_h = 0.25
    concats = ("A", "B", "random")
    colors = {"A": "#6baed6", "B": "#4292c6", "random": "#d7301f"}

    for i, concat in enumerate(concats):
        vals = [by_arch[a].get(concat, 0) for a in archs]
        off = (i - 1) * bar_h
        bars = ax.barh(y + off, vals, bar_h,
                       color=colors[concat],
                       edgecolor="black" if concat == "random" else "none",
                       linewidth=1.2 if concat == "random" else 0,
                       label=f"concat_{concat}")
        # Label bars with the value
        for bar, v in zip(bars, vals):
            if v >= 0:
                ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                        str(v), ha="left", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES.get(a, a) for a in archs], fontsize=9)
    ax.set_xlabel("SEMANTIC label count (top-32 features, multi-Haiku judge, temp=0)")
    ax.set_xlim(0, 34)
    ax.axvline(32, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(figtitle or f"Rigorous qualitative metric (seed={seed}, N=32)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    # Thumbnail
    thumb = out_png.with_suffix(".thumb.png")
    fig.savefig(thumb, dpi=48, bbox_inches="tight")

    plt.close(fig)
    print(f"wrote {out_png} (and thumbnail)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str,
                   default=str(OUT_DIR / "phase61_rigorous_headline.png"))
    args = p.parse_args()

    rows = load_cells(seed=args.seed)
    if not rows:
        raise SystemExit(f"no cells found for seed={args.seed} in {IN_DIR}")
    make_plot(rows, Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
