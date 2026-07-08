"""Shared matplotlib style + save helpers + frontier plot primitive.

Figures are NOT part of the record acceptance contract (only the ``AUTO`` blocks
and ``*_stats.json`` are), so this module holds the *common* styling and the one
repeated plotting loop; each bench composes its own figures (which panels,
ceilings, titles) in its driver — we do not force four figure sets into one
function.
"""

from __future__ import annotations

from pathlib import Path

# Paper-quality rcParams, shared verbatim by every bench renderer.
PAPER_STYLE = {
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.7,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
    "figure.facecolor": "white", "mathtext.default": "regular",
}

# Marker per window size T (T=1 = per-token). Superset over the benches.
MARK = {1: "o", 2: "s", 4: "^", 8: "D", 16: "P"}


def use_agg_style():
    """Select the Agg backend, apply :data:`PAPER_STYLE`, return ``pyplot``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)
    return plt


def save_fig(fig, fig_dir: Path, name: str, plt,
             variants=(("pdf", None), ("png", 200), ("thumb.png", 70))) -> None:
    """Save ``fig`` as ``<name>.{pdf,png,thumb.png}`` into ``fig_dir`` and close it."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in variants:
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"[fig] {fig_dir.name}/{name}.{{pdf,png}}")


def frontier_series(ax, arch_t, d_saes, value_fn, colors, per_token, label_fn, *,
                    ms=5.5, lw=1.9, capsize=2, elinewidth=1.0, marks=MARK):
    """Draw one errorbar series per ``(arch, T)`` over the ``d_sae`` axis.

    ``value_fn(arch, T, d) -> (mean, std, n)``; a point is drawn only where
    ``n>0``. Per-token ``(arch, T)`` get a dashed line; window archs solid. This
    is the arch×capacity loop shared by the benches' frontier panels; ceilings,
    shading, axes and titles stay in the caller.
    """
    for arch, T in arch_t:
        xs, ys, es = [], [], []
        for d in d_saes:
            m, s, n = value_fn(arch, T, d)
            if n:
                xs.append(d); ys.append(m); es.append(s)
        if xs:
            ls = "--" if (arch, T) in per_token else "-"
            ax.errorbar(xs, ys, yerr=es, marker=marks[T], ms=ms, lw=lw, ls=ls,
                        color=colors[(arch, T)], capsize=capsize,
                        elinewidth=elinewidth, label=label_fn(arch, T))
