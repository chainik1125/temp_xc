"""Program-level report figures for the interpretability researcher.

Three auto-generated views embedded in ``REPORT.md``, all from the *same*
per-token matched cells as the tables (so figure and table never disagree):

1. :func:`recovery_heatmap` — the B×A recovery matrix as colour: at a glance,
   which architecture linearly exposes which latent, and where it goes cold
   (additive / per-token columns on the AC-interaction rows).
2. :func:`capacity_frontiers` — recovery vs ``d_sae`` per (bench, latent), one
   line per arch, ``F`` marked: does the win survive into the scarce regime.
3. :func:`capability_gate` — latent recovery vs reconstruction NMSE: the
   README validity gate — a recovery number in the top-right (high recovery,
   poor reconstruction) is representing the latent while representing little else.

Everything reads ``groups`` (from :func:`report.group_cells`) or the matrix stats,
so the figures inherit the realized-L0 matching; no numbers are hand-placed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import report
from .figs import save_fig, use_agg_style

# One (colour, marker) per architecture column — token greys, then a distinct hue
# per decode structure (stacked purple, pre blue, post orange, spectral red).
ARCH_STYLE = {
    "batchtopk_sae":     ("#6e6e6e", "o"),
    "tsae":              ("#b0ac36", "o"),
    "stacked_batchtopk": ("#9467bd", "s"),
    "txc_batchtopk_pre": ("#1f77b4", "^"),
    "txc_batchtopk_post":("#ff7f0e", "D"),
    "spectral_txc":      ("#d62728", "P"),
}


def _rows(benches):
    """The matrix rows: one (bench, latent-axis) per row, in display order."""
    return [(b, ax) for b in benches for ax in b.axes]


def _row_label(b, ax):
    return f"{b.name}·{ax.key} ({ax.kind})"


def _primary_axis(b):
    """The bench's headline latent-axis (first ``primary`` one, else the first)."""
    return next((ax for ax in b.axes if ax.primary), b.axes[0])


def recovery_heatmap(mtx_stats, benches, archs, capacities_fn, op, out_dir) -> str:
    """B×A recovery heatmap, one panel per capacity ``{F, F//2}``. Returns filename."""
    plt = use_agg_style()
    rows = _rows(benches)
    ncap = len(op.capacity_fracs)
    cap_labels = report._cap_labels(op)
    fig, axes = plt.subplots(
        1, ncap, figsize=(1.15 * len(archs) * ncap + 1.5, 0.52 * len(rows) + 1.7),
        squeeze=False)
    im = None
    for ci in range(ncap):
        ax = axes[0][ci]
        M = np.full((len(rows), len(archs)), np.nan)
        for ri, (b, axl) in enumerate(rows):
            d = capacities_fn(b)[ci]
            for aj, a in enumerate(archs):
                cell = mtx_stats.get(f"{b.name}/{axl.key}/{a.name}")
                v = cell.get(d) if cell else None
                if v is not None:
                    M[ri, aj] = v["value"]
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels([a.label.split(" (")[0] for a in archs],
                           rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([_row_label(b, axl) for (b, axl) in rows] if ci == 0 else [],
                           fontsize=8)
        ax.set_title(f"d_sae = {cap_labels[ci]}", fontsize=10)
        ax.set_xticks(np.arange(-.5, len(archs), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
        for ri in range(len(rows)):
            for aj in range(len(archs)):
                val = M[ri, aj]
                txt = "—" if np.isnan(val) else f"{val:.2f}"
                ax.text(aj, ri, txt, ha="center", va="center", fontsize=7,
                        color="black" if (np.isnan(val) or 0.15 < val < 0.85) else "white")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02,
                 label="normalized recovery  [chance 0 → oracle 1]")
    fig.suptitle("Per-token matched recovery  (T_can=%d, B*=%g)"
                 % (op.T_can, op.B_star), fontsize=11, y=1.0)
    save_fig(fig, out_dir, "fig_recovery_heatmap", plt)
    return "fig_recovery_heatmap.png"


def capacity_frontiers(groups, benches, archs, op, out_dir) -> str:
    """Recovery vs d_sae, one small panel per (bench, latent-axis). Returns filename."""
    plt = use_agg_style()
    rows = _rows(benches)
    ncol = 3
    nrow = -(-len(rows) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.0 * nrow),
                             squeeze=False)
    for idx, (b, axl) in enumerate(rows):
        ax = axes[idx // ncol][idx % ncol]
        # the design sweep {F//2, F, 2F} — exclude any special control d_sae
        # (e.g. the frequency memorization-demo at d_sae≫2F).
        d_saes = sorted({d for (bn, an, T, d, kp) in groups
                         if bn == b.name and d <= 2 * b.F})
        for a in archs:
            T = 1 if not a.windowed else op.T_can
            xs, ys, es = [], [], []
            for d in d_saes:
                mg = report.matched_group(groups, b.name, a, d_sae=d, T_can=op.T_can,
                                          B_star=op.B_star)
                if mg is None:
                    continue
                m, s, n = mg[1]["metrics"].get(axl.metric, (np.nan, np.nan, 0))
                if n:
                    xs.append(d); ys.append(m); es.append(s)
            if xs:
                c, mk = ARCH_STYLE.get(a.name, ("#333", "o"))
                ls = "--" if not a.windowed else "-"
                ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, ls=ls, ms=5,
                            lw=1.7, capsize=2, elinewidth=0.9, label=a.label.split(" (")[0])
        ax.axvline(b.F, color="gray", ls=":", lw=1, alpha=0.7)
        ax.text(b.F, ax.get_ylim()[1], " F", fontsize=8, va="top", color="gray")
        ax.axhline(0.0, color="k", ls=":", lw=0.7, alpha=0.35)
        ax.set_title(_row_label(b, axl), fontsize=9.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(d_saes); ax.set_xticklabels(d_saes, fontsize=8)
        ax.set_xlabel("d_sae"); ax.set_ylabel("recovery")
    for j in range(len(rows), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(archs),
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Recovery vs capacity  (per-token matched, T_can=%d)" % op.T_can,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_fig(fig, out_dir, "fig_capacity_frontiers", plt)
    return "fig_capacity_frontiers.png"


def capability_gate(mtx_stats, nmse_stats, benches, archs, op, out_dir) -> str:
    """Scatter: primary-latent recovery (y) vs reconstruction NMSE (x). Returns filename."""
    plt = use_agg_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _markers = ["o", "s", "^", "D", "P", "v", "X", "*", "h", "<", ">", "p"]
    bench_marker = {b.name: _markers[i % len(_markers)]
                    for i, b in enumerate(benches)}
    for b in benches:
        axl = _primary_axis(b)
        d = b.F  # boundary capacity
        for a in archs:
            rc = mtx_stats.get(f"{b.name}/{axl.key}/{a.name}")
            nc = nmse_stats.get(f"{b.name}/{a.name}")
            rv = rc.get(d) if rc else None
            nv = nc.get(d) if nc else None
            if rv is None or nv is None:
                continue
            c, _mk = ARCH_STYLE.get(a.name, ("#333", "o"))
            ax.scatter(nv["value"], rv["value"], c=c, marker=bench_marker[b.name],
                       s=70, edgecolors="black", linewidths=0.5, zorder=3)
    # The genuine gate FAILURE is recovery with *near-trivial* reconstruction
    # (NMSE→1: representing ~nothing). A moderate NMSE is a cost, not a failure —
    # and a bench's irreducible noise floor (e.g. frequency σ) sits mid-axis for
    # every arch. Shade only the degenerate band; empty ⇒ all recovery is backed.
    ax.axvspan(0.85, 1.02, color="crimson", alpha=0.06, zorder=0)
    ax.text(0.935, 0.52, "NMSE→1: reconstructs ~nothing\n(empty ⇒ all recovery is backed)",
            rotation=90, ha="center", va="center", fontsize=7.5, color="crimson")
    ax.set_xlabel("reconstruction NMSE  (→ worse;  bench noise floor sits mid-axis)")
    ax.set_ylabel("primary-latent recovery  (↑ better)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.1, 1.08)
    ax.set_title("Capability gate — is the recovery reconstruction-backed?  (d_sae=F)")
    arch_handles = [plt.Line2D([], [], color=ARCH_STYLE[a.name][0], marker="o", ls="",
                               ms=8, label=a.label.split(" (")[0]) for a in archs]
    bench_handles = [plt.Line2D([], [], color="#444", marker=bench_marker[b.name], ls="",
                                ms=8, label=b.name) for b in benches]
    leg1 = ax.legend(handles=arch_handles, title="arch (colour)", loc="lower right",
                     fontsize=8, title_fontsize=8.5)
    ax.add_artist(leg1)
    ax.legend(handles=bench_handles, title="bench (marker)", loc="upper left",
              fontsize=8, title_fontsize=8.5)
    save_fig(fig, out_dir, "fig_capability_gate", plt)
    return "fig_capability_gate.png"


def render_all(groups, mtx_stats, nmse_stats, benches, archs, capacities_fn, op,
               out_dir: Path) -> list[str]:
    """Render all three program figures into ``out_dir``; return their filenames."""
    out_dir = Path(out_dir)
    return [
        recovery_heatmap(mtx_stats, benches, archs, capacities_fn, op, out_dir),
        capacity_frontiers(groups, benches, archs, op, out_dir),
        capability_gate(mtx_stats, nmse_stats, benches, archs, op, out_dir),
    ]
