#!/usr/bin/env python3
"""Per-architecture Pareto subplots with α-colored points.

One subplot per (variant, target) pair, showing the trajectory through
(kw fraction, Sonnet coherence) space as α varies. Marker colour encodes
α via a perceptually-uniform colormap; markers are connected in α order
so the Pareto trajectory is visible.

This makes it easy to compare the *shape* of each architecture's response
curve — TXC and TopKSAE have noticeably different transition profiles
(TXC sharp, TopKSAE gradual) which the cross-variant Pareto frontier
plot collapses into single dominant points.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.plotting.save_figure import save_figure  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    RESULTS_DIR,
)


# Curated set of variants to plot. Each is (variant_dir, mode_target_filter, label).
# Six-panel default focuses on the architectural comparisons; pass
# --include-stratified to also show the stratified-held-out reruns of
# the Llama-Scope baselines.
DEFAULT_VARIANTS = [
    ("intervene_32x",              ("sae_additive", "feat_71839"),   "Llama-Scope 32x feat_71839\n(billions of tokens)"),
    ("intervene",                  ("sae_additive", "feat_7792"),    "Llama-Scope 8x feat_7792\n(billions of tokens)"),
    ("intervene_llama_topk_30k",   ("sae_additive", "feat_28417"),   "Llama-trained TopKSAE@resid\n(30k steps, 3.8M tokens)"),
    ("intervene_llama_txc_resid_30k", ("sae_additive", "feat_5228"), "Llama-trained TXC@resid T=5\n(30k steps, 3.8M tokens)"),
    ("intervene_llama_txc_attn_30k_d8k_v2", ("sae_additive", "feat_8013"), "Llama-trained TXC@attn T=5\n(cross-hook steering)"),
    ("intervene",                  ("raw_dom", "raw_dom"),           "raw DoM\n(paper baseline)"),
]
EXTRA_STRATIFIED = [
    ("intervene_main_strat",       ("sae_additive", "feat_7792"),    "Llama-Scope 8x feat_7792\n(stratified held-out)"),
    ("intervene_32x_strat",        ("sae_additive", "feat_71839"),   "Llama-Scope 32x feat_71839\n(stratified held-out)"),
]


def load_curve(intervene_dir: Path, mode: str, target: str) -> list[tuple[float, float, float, float]]:
    """Return list of (alpha, kw, coh, sem_kw) sorted by α."""
    p = intervene_dir / "keyword_rates.csv"
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["mode"] != mode or r["target"] != target:
                continue
            try:
                alpha = float(r["magnitude"])
                kw = float(r["mean_keyword_fraction"])
                coh_raw = r.get("mean_coherence", "")
                if coh_raw in ("", None):
                    continue
                coh = float(coh_raw)
                sem = float(r.get("sem_keyword_fraction") or 0.0)
            except (ValueError, KeyError):
                continue
            out.append((alpha, kw, coh, sem))
    out.sort(key=lambda x: x[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="output png path (default plots_summary/pareto_per_arch.png)")
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--include-stratified", action="store_true",
                        help="also include stratified-held-out reruns of Llama-Scope baselines")
    args = parser.parse_args()

    variant_list = list(DEFAULT_VARIANTS)
    if args.include_stratified:
        variant_list.extend(EXTRA_STRATIFIED)
    curves = []
    for sub, (mode, target), label in variant_list:
        d = RESULTS_DIR / sub
        pts = load_curve(d, mode, target)
        if not pts:
            print(f"[skip] no data for {label} ({sub} / {mode} / {target})")
            continue
        curves.append((label, pts))

    if not curves:
        raise SystemExit("no curves to plot")

    n = len(curves)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.0 * rows), squeeze=False)
    # Generous vertical spacing so the bottom row's titles don't collide with the
    # top row's x-axis labels.
    fig.subplots_adjust(hspace=0.55, wspace=0.30)

    # Determine global α range for shared colorbar
    all_alphas = [a for _, pts in curves for a, *_ in pts]
    a_min, a_max = min(all_alphas), max(all_alphas)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=a_min, vmax=a_max)

    for i, (label, pts) in enumerate(curves):
        ax = axes[i // cols][i % cols]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        alphas = [p[0] for p in pts]
        # Connecting line (faded)
        ax.plot(xs, ys, color="grey", linewidth=1.0, alpha=0.4, zorder=1)
        # Coloured markers
        sc = ax.scatter(xs, ys, c=alphas, cmap=cmap, norm=norm, s=80, zorder=2,
                        edgecolors="black", linewidths=0.5)
        # Annotate with α value next to each marker
        for x, y, a in zip(xs, ys, alphas):
            ax.annotate(f"{a:g}", xy=(x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=7, alpha=0.7)
        ax.set_xlabel("backtracking rate (mean kw fraction in B = {wait, hmm})")
        ax.set_ylabel("Sonnet coherence (0 incoherent ↔ 3 coherent)")
        ax.set_ylim(-0.1, 3.1)
        ax.set_xlim(left=-0.005)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    # Shared colorbar (right of the figure). Re-apply right margin AFTER our
    # earlier subplots_adjust so the colorbar gets its own column without
    # eating into the per-axes padding.
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.91, hspace=0.55, wspace=0.30, top=0.92, bottom=0.08)
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("steering magnitude α (or clamp strength)", fontsize=9)

    fig.suptitle(
        "Backtracking case study — per-architecture Pareto trajectories  (colour = α)",
        fontsize=12, y=0.97,
    )

    out_path = Path(args.out) if args.out else RESULTS_DIR / "plots_summary" / "pareto_per_arch.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
