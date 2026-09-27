"""Render the three synth-section panels at matched 4.8x4.0 canvas.

Panels (left → right in main.tex):
  (a) Bar chart: best global recovery per arch (Setup B & D side-by-side).
  (b) Setup B scatter: single-latent local vs global correlation.
  (c) Setup D scatter: dictionary AUC eAUC vs gAUC at np=10.

Style is minimal (no boxed axes, dotted y-grid, no heavy frame), matching
the bar-chart panel. Scatter panels keep the y=x diagonal.

Inputs:
  notes/c2_synthetic/data/denoising_probe_results.json    # Setup B per-cell
  notes/c2_synthetic/data/setup_d_leaderboard.jsonl       # Setup D per-cell
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "notes" / "c2_synthetic" / "data"
OUT  = REPO / "figs" / "c2"
OUT.mkdir(parents=True, exist_ok=True)

CANVAS = (4.8, 4.0)

# ---------------------------------------------------------------------------
# Architecture display catalogue (consistent across panels)
# ---------------------------------------------------------------------------
ARCH_STYLE = {
    # arch_name        marker, color, label
    "topk_sae":       ("o",  "#444444", "TopK SAE"),
    "tsae_paper":     ("h",  "#D81B60", "T-SAE"),
    "tfa_pos":        ("X",  "#43A047", "TFA-pos"),
    "txc_base":       ("^",  "#1E88E5", "TXC-base"),
    "txc_pro":        ("*",  "#E64A19", "TXC-pro"),
}
HEADLINE_ARCHS = ["topk_sae", "tsae_paper", "tfa_pos", "txc_base", "txc_pro"]

def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(direction="out", length=3)


# ---------------------------------------------------------------------------
# (a) bar chart — best global recovery per arch
# ---------------------------------------------------------------------------
def setup_b_best_with_errors() -> dict[str, tuple[float, float, float, str]]:
    """{arch: (mean, min_over_seeds, max_over_seeds, t_label)} for the
    seed-mean-best (T, k_pos) cell per arch.

    Headline metric: linear-probe R^2 against the clean hidden state
    (`lp_mean_global_r2` in the probe JSON). This reads dictionary-
    aggregated information rather than a single-latent's trace, which
    keeps the y-axis above the single-token noise floor.
    """
    rows = json.loads((DATA / "denoising_probe_results.json").read_text())
    cell = defaultdict(list)
    for r in rows:
        cell[(r["arch_name"], r.get("t_label", ""), int(r["k_pos"]))].append(float(r["lp_mean_global_r2"]))
    best: dict[str, tuple[float, float, float, str]] = {}
    for (arch, t, _k), vs in cell.items():
        if len(vs) < 2:
            continue  # need at least 2 seeds for an error bar
        m = statistics.mean(vs)
        if m > best.get(arch, (-1.0, 0.0, 0.0, ""))[0]:
            best[arch] = (m, min(vs), max(vs), t)
    return best


def setup_d_best_with_errors() -> dict[str, tuple[float, float, float, str]]:
    """Same shape, computed from the np=10 leaderboard rows."""
    lines = (DATA / "setup_d_leaderboard.jsonl").read_text().splitlines()
    cell = defaultdict(list)
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        if r["datasource"] != "toy_coupled_noisy_K10_M20_d256_pB05_np10":
            continue
        cell[(r["arch"], r["eval_cfg"].get("t_label", "default"), int(r["eval_cfg"]["k_pos"]))].append(
            float(r["metrics"]["gauc"])
        )
    best: dict[str, tuple[float, float, float, str]] = {}
    for (arch, t, _k), vs in cell.items():
        if len(vs) < 2:
            continue
        m = statistics.mean(vs)
        if m > best.get(arch, (-1.0, 0.0, 0.0, ""))[0]:
            best[arch] = (m, min(vs), max(vs), t)
    return best


def render_bars():
    setup_b = setup_b_best_with_errors()
    setup_d = setup_d_best_with_errors()
    archs = HEADLINE_ARCHS
    labels = [ARCH_STYLE[a][2] for a in archs]
    b_vals  = [setup_b.get(a, (np.nan, np.nan, np.nan, ""))[0] for a in archs]
    b_lo    = [setup_b.get(a, (np.nan, np.nan, np.nan, ""))[1] for a in archs]
    b_hi    = [setup_b.get(a, (np.nan, np.nan, np.nan, ""))[2] for a in archs]
    d_vals  = [setup_d.get(a, (np.nan, np.nan, np.nan, ""))[0] for a in archs]
    d_lo    = [setup_d.get(a, (np.nan, np.nan, np.nan, ""))[1] for a in archs]
    d_hi    = [setup_d.get(a, (np.nan, np.nan, np.nan, ""))[2] for a in archs]

    # Asymmetric err bars: yerr=[mean-min, max-mean]; replace NaN with 0 to
    # avoid matplotlib errors on missing cells.
    def _err(vals, lo, hi):
        lower = [(v - l) if not np.isnan(v) and not np.isnan(l) else 0 for v, l in zip(vals, lo)]
        upper = [(h - v) if not np.isnan(v) and not np.isnan(h) else 0 for v, h in zip(vals, hi)]
        return [lower, upper]

    fig, ax = plt.subplots(figsize=CANVAS)
    x = np.arange(len(archs))
    w = 0.38

    bb = ax.bar(x - w/2, b_vals, w, yerr=_err(b_vals, b_lo, b_hi),
                label=r"Denoising  (linear-probe $R^2_{\mathrm{global}}$)",
                color="#7E57C2", edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#222"})
    bd = ax.bar(x + w/2, d_vals, w, yerr=_err(d_vals, d_lo, d_hi),
                label=r"Coupling  ($g\mathrm{AUC}$, $n_{\mathrm{parents}}{=}10$)",
                color="#E64A19", edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#222"})

    for rect, val, hi in list(zip(bb, b_vals, b_hi)) + list(zip(bd, d_vals, d_hi)):
        if not np.isnan(val):
            top = hi if not np.isnan(hi) else val
            ax.text(rect.get_x() + rect.get_width()/2, top + 0.012,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(d_vals):
        if np.isnan(v):
            ax.text(x[i] + w/2, 0.04, "in flight", ha="center", va="bottom",
                    rotation=90, fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Best global recovery")
    ax.set_title("Best global recovery per architecture", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=7,
              labelspacing=0.3, handletextpad=0.4)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "c2_synth_global_headline.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) Setup B scatter: single-latent local vs global, seed-averaged cells
# ---------------------------------------------------------------------------
def render_setup_b_scatter():
    rows = json.loads((DATA / "denoising_probe_results.json").read_text())
    cell_loc = defaultdict(list)
    cell_glb = defaultdict(list)
    for r in rows:
        key = (r["arch_name"], r.get("t_label", "default"), int(r["k_pos"]))
        cell_loc[key].append(float(r["sl_mean_local"]))
        cell_glb[key].append(float(r["sl_mean_global"]))

    fig, ax = plt.subplots(figsize=CANVAS)
    legend_handles = {}
    for (arch, t, k), vs_loc in cell_loc.items():
        if arch not in ARCH_STYLE:
            continue
        vs_glb = cell_glb[(arch, t, k)]
        x = statistics.mean(vs_loc)
        y = statistics.mean(vs_glb)
        marker, color, lbl = ARCH_STYLE[arch]
        # Slightly different alpha for non-default t to show window variants.
        h = ax.scatter(x, y, marker=marker, c=color, s=42, alpha=0.85,
                       edgecolors="white", linewidth=0.5)
        if arch not in legend_handles:
            legend_handles[arch] = h

    # y = x diagonal
    lim = 0.18
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=0.8, label=r"$y = x$")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel(r"local  $\bar r(z_{j^*}, a_{i,t})$")
    ax.set_ylabel(r"global $\bar r(z_{j^*}, s_{i,t})$")
    ax.set_title("Denoising: single-latent correlation", pad=8)
    handles = [legend_handles[a] for a in HEADLINE_ARCHS if a in legend_handles]
    labels  = [ARCH_STYLE[a][2] for a in HEADLINE_ARCHS if a in legend_handles]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=7,
              ncol=1, labelspacing=0.3, handletextpad=0.4,
              bbox_to_anchor=(0.0, 1.0))
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "c2_setup_b_singlelatent.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (c) Setup D scatter: eAUC vs gAUC at np=10, seed-averaged cells
# ---------------------------------------------------------------------------
def render_setup_d_scatter():
    lines = (DATA / "setup_d_leaderboard.jsonl").read_text().splitlines()
    rows = []
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        if r["datasource"] != "toy_coupled_noisy_K10_M20_d256_pB05_np10":
            continue
        rows.append(r)

    cell_e = defaultdict(list)
    cell_g = defaultdict(list)
    for r in rows:
        key = (r["arch"], r["eval_cfg"].get("t_label", "default"), int(r["eval_cfg"]["k_pos"]))
        cell_e[key].append(float(r["metrics"]["eauc"]))
        cell_g[key].append(float(r["metrics"]["gauc"]))

    fig, ax = plt.subplots(figsize=CANVAS)
    legend_handles = {}
    for (arch, t, k), vs_e in cell_e.items():
        if arch not in ARCH_STYLE:
            continue
        vs_g = cell_g[(arch, t, k)]
        x = statistics.mean(vs_e)
        y = statistics.mean(vs_g)
        marker, color, lbl = ARCH_STYLE[arch]
        h = ax.scatter(x, y, marker=marker, c=color, s=42, alpha=0.85,
                       edgecolors="white", linewidth=0.5)
        if arch not in legend_handles:
            legend_handles[arch] = h

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$e\mathrm{AUC}$ (local)")
    ax.set_ylabel(r"$g\mathrm{AUC}$ (global)")
    ax.set_title(r"Coupling: dictionary alignment, $n_{\mathrm{parents}}{=}10$", pad=8)
    handles = [legend_handles[a] for a in HEADLINE_ARCHS if a in legend_handles]
    labels  = [ARCH_STYLE[a][2] for a in HEADLINE_ARCHS if a in legend_handles]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=7,
              ncol=1, labelspacing=0.3, handletextpad=0.4)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "c2_setup_d_scatter_clean.png", dpi=180)
    plt.close(fig)


def main():
    render_bars()
    render_setup_b_scatter()
    render_setup_d_scatter()
    print("wrote 3 panels to", OUT)


if __name__ == "__main__":
    main()
