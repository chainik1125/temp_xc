"""Render the global rose-diagram summary across all six case studies.

Each axis = the headline metric for one paper component, min-max normalised
to [0, 1] across the five target architectures. The 5-arch ordering and
palette match the rest of the paper (TopK SAE blue, T-SAE green, TFA grey,
TXC-base purple, TXC-pro gold).

Inputs (defaults resolve to in-repo canonical paths)::

    cd purified
    .venv/bin/python -m scripts.global_rose_renderer

Reads the precomputed ``global_rose_summary_data.json`` sidecar — that
file carries both the raw and the normalised per-axis values across the
five target architectures, exactly as harvested from the per-component
results files. Reviewers can regenerate the sidecar from scratch by
running each component's ``experiments/cN_*/run.py`` and rerunning the
component renderer (which writes the per-axis numbers consumed here).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARCHS = ("topk_sae", "tsae_paper", "tfa", "txc_base", "txc_pro")
ARCH_LABEL = {
    "topk_sae":   "TopK SAE",
    "tsae_paper": "T-SAE",
    "tfa":        "TFA",
    "txc_base":   "TXC-base",
    "txc_pro":    "TXC-pro",
}
# Paper-wide c7 palette (matches c7_paper_renderer / c2_paper_renderer).
ARCH_COLOR = {
    "topk_sae":   "#4C72B0",  # blue
    "tsae_paper": "#55A868",  # green
    "tfa":        "#777777",  # grey
    "txc_base":   "#8172B2",  # purple
    "txc_pro":    "#CCB974",  # gold
}


def render_rose(sidecar: Path, out_stem: Path) -> None:
    payload = json.loads(sidecar.read_text())
    case_studies = payload["case_studies"]
    norm = payload["normalised"]

    n_axes = len(case_studies)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=200,
                           subplot_kw=dict(projection="polar"))
    for arch in ARCHS:
        per_arch = norm.get(arch, {})
        vals: list[float] = []
        for cs in case_studies:
            v = per_arch.get(cs["axis"])
            vals.append(0.0 if v is None else float(v))
        vals_closed = vals + [vals[0]]
        col = ARCH_COLOR[arch]
        ax.plot(angles_closed, vals_closed, "-o", color=col, lw=1.6, ms=4.5,
                label=ARCH_LABEL[arch])
        ax.fill(angles_closed, vals_closed, color=col, alpha=0.10)

    tick_labels = [cs["axis"] for cs in case_studies]
    ax.set_xticks(angles)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"],
                       fontsize=7, color="#666")
    ax.set_ylim(0, 1.05)
    ax.set_rlabel_position(90)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["polar"].set_color("#aaaaaa")

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08),
              fontsize=9, frameon=False)
    fig.tight_layout()

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _purified_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cli() -> None:
    root = _purified_root()
    ap = argparse.ArgumentParser(description=(
        "Global rose-diagram renderer (one normalised score per arch per "
        "case study). Defaults to in-repo canonical paths."
    ))
    ap.add_argument(
        "--sidecar", type=Path,
        default=root / "notes" / "global_rose_summary_data.json",
        help=("Sidecar JSON with raw + normalised per-axis values "
              "(default: purified/notes/global_rose_summary_data.json)."),
    )
    ap.add_argument(
        "--output-stem", type=Path,
        default=root / "figs" / "global_rose_summary",
        help=("Output stem; writes {stem}.png and {stem}.pdf "
              "(default: purified/figs/global_rose_summary)."),
    )
    args = ap.parse_args()
    render_rose(args.sidecar, args.output_stem)
    print(f"wrote {args.output_stem.with_suffix('.png')} + .pdf")


if __name__ == "__main__":
    cli()
