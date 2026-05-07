"""Render the three synth-section panels of Figure 2 (synthetic_overview).

Panels (left → right in the paper):
  (a) Bar chart: best global recovery per arch (Setup B & D side-by-side).
  (b) Setup B scatter: single-latent local vs global correlation (Denoising).
  (c) Setup D scatter: dictionary eAUC vs gAUC at n_parents=10 (Coupling).

The bar chart uses the paper-wide arch palette (TopK SAE blue, T-SAE green,
TFA-pos grey, TXC-base purple, TXC-pro gold). The bench dimension
(Denoising vs Coupling) is encoded by alpha + hatch, with neutral-grey
swatches in the legend.

Inputs (defaults resolve to in-repo canonical paths)::

    cd purified
    .venv/bin/python -m scripts.c2_synth_paper_renderer

  experiments/c1_noisy_filler/denoising_probe_results.json     # Setup B
  experiments/c2_hierarchical/setup_d_leaderboard.jsonl        # Setup D
  figs/c2/                                                     # output

The Setup B file ships from the c1_noisy_filler experiment, which generates
both the c1 NMSE/AUC sweep and the Denoising probe results. The Setup D
slice ships as a snapshot of the leaderboard rows whose datasource is
``toy_coupled_noisy_K10_M20_d256_pB05_np10`` (n_parents=10 — the maximal
overlap regime that defines the Coupling axis).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


CANVAS = (4.8, 4.0)

# ---------------------------------------------------------------------------
# Architecture display catalogue (paper-wide c7-style palette)
# ---------------------------------------------------------------------------
ARCH_STYLE = {
    # arch_name        marker, color, label
    "topk_sae":       ("D",  "#4C72B0", "TopK SAE"),    # blue
    "tsae_paper":     ("s",  "#55A868", "T-SAE"),       # green
    "tfa_pos":        ("X",  "#777777", "TFA-pos"),     # grey
    "txc_base":       ("^",  "#8172B2", "TXC-base"),    # purple
    "txc_pro":        ("*",  "#CCB974", "TXC-pro"),     # gold
}
HEADLINE_ARCHS = ("topk_sae", "tsae_paper", "tfa_pos", "txc_base", "txc_pro")

SETUP_D_DATASOURCE = "toy_coupled_noisy_K10_M20_d256_pB05_np10"


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(direction="out", length=3)


def _save_png_pdf(fig, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=180)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")


# ---------------------------------------------------------------------------
# (a) bar chart — best global recovery per arch
# ---------------------------------------------------------------------------
def setup_b_best_with_errors(
    probe_json: Path,
) -> dict[str, tuple[float, float, float, str]]:
    """{arch: (mean, min_over_seeds, max_over_seeds, t_label)} for the
    seed-mean-best (T, k_pos) cell per arch.

    Headline metric is the linear-probe R² against the clean hidden state
    (``lp_mean_global_r2``): a dictionary-aggregated read that stays above
    the single-token noise floor.
    """
    rows = json.loads(probe_json.read_text())
    cell = defaultdict(list)
    for r in rows:
        cell[
            (r["arch_name"], r.get("t_label", ""), int(r["k_pos"]))
        ].append(float(r["lp_mean_global_r2"]))
    best: dict[str, tuple[float, float, float, str]] = {}
    for (arch, t, _k), vs in cell.items():
        if len(vs) < 2:
            continue  # need at least 2 seeds for an error bar
        m = statistics.mean(vs)
        if m > best.get(arch, (-1.0, 0.0, 0.0, ""))[0]:
            best[arch] = (m, min(vs), max(vs), t)
    return best


def setup_d_best_with_errors(
    leaderboard_jsonl: Path,
) -> dict[str, tuple[float, float, float, str]]:
    """Same shape, computed from the np=10 leaderboard rows."""
    cell = defaultdict(list)
    for line in leaderboard_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        if r.get("datasource") != SETUP_D_DATASOURCE:
            continue
        cell[
            (r["arch"], r["eval_cfg"].get("t_label", "default"),
             int(r["eval_cfg"]["k_pos"]))
        ].append(float(r["metrics"]["gauc"]))
    best: dict[str, tuple[float, float, float, str]] = {}
    for (arch, t, _k), vs in cell.items():
        if len(vs) < 2:
            continue
        m = statistics.mean(vs)
        if m > best.get(arch, (-1.0, 0.0, 0.0, ""))[0]:
            best[arch] = (m, min(vs), max(vs), t)
    return best


def render_bars(probe_json: Path, leaderboard: Path, out_dir: Path) -> None:
    setup_b = setup_b_best_with_errors(probe_json)
    setup_d = setup_d_best_with_errors(leaderboard)
    archs = HEADLINE_ARCHS
    labels = [ARCH_STYLE[a][2] for a in archs]
    nan_tuple = (np.nan, np.nan, np.nan, "")
    b_vals = [setup_b.get(a, nan_tuple)[0] for a in archs]
    b_lo   = [setup_b.get(a, nan_tuple)[1] for a in archs]
    b_hi   = [setup_b.get(a, nan_tuple)[2] for a in archs]
    d_vals = [setup_d.get(a, nan_tuple)[0] for a in archs]
    d_lo   = [setup_d.get(a, nan_tuple)[1] for a in archs]
    d_hi   = [setup_d.get(a, nan_tuple)[2] for a in archs]

    def _err(vals, lo, hi):
        lower = [(v - l) if not np.isnan(v) and not np.isnan(l) else 0
                 for v, l in zip(vals, lo)]
        upper = [(h - v) if not np.isnan(v) and not np.isnan(h) else 0
                 for v, h in zip(vals, hi)]
        return [lower, upper]

    fig, ax = plt.subplots(figsize=CANVAS)
    x = np.arange(len(archs))
    w = 0.38
    arch_colors = [ARCH_STYLE[a][1] for a in archs]

    bb = ax.bar(
        x - w / 2, b_vals, w, yerr=_err(b_vals, b_lo, b_hi),
        color=arch_colors, edgecolor="black", linewidth=0.6, alpha=0.95,
        error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#222"},
    )
    bd = ax.bar(
        x + w / 2, d_vals, w, yerr=_err(d_vals, d_lo, d_hi),
        color=arch_colors, edgecolor="black", linewidth=0.6, alpha=0.55,
        hatch="//",
        error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#222"},
    )

    bench_handles = [
        Patch(facecolor="#888888", edgecolor="black", alpha=0.95,
              label=r"Denoising  (linear-probe $R^2_{\mathrm{global}}$)"),
        Patch(facecolor="#888888", edgecolor="black", alpha=0.55,
              hatch="//",
              label=r"Coupling  ($g\mathrm{AUC}$, $n_{\mathrm{parents}}{=}10$)"),
    ]

    for rect, val, hi in (
        list(zip(bb, b_vals, b_hi)) + list(zip(bd, d_vals, d_hi))
    ):
        if not np.isnan(val):
            top = hi if not np.isnan(hi) else val
            ax.text(rect.get_x() + rect.get_width() / 2, top + 0.012,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(d_vals):
        if np.isnan(v):
            ax.text(x[i] + w / 2, 0.04, "in flight",
                    ha="center", va="bottom",
                    rotation=90, fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Best global recovery")
    ax.set_title("Best global recovery per architecture", pad=8)
    ax.legend(handles=bench_handles, loc="upper left", frameon=False,
              fontsize=7, labelspacing=0.3, handletextpad=0.4)
    _clean_axes(ax)
    fig.tight_layout()
    _save_png_pdf(fig, out_dir / "c2_synth_global_headline")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) Setup B scatter: single-latent local vs global, seed-averaged cells
# ---------------------------------------------------------------------------
def render_setup_b_scatter(probe_json: Path, out_dir: Path) -> None:
    rows = json.loads(probe_json.read_text())
    cell_loc = defaultdict(list)
    cell_glb = defaultdict(list)
    for r in rows:
        key = (r["arch_name"], r.get("t_label", "default"), int(r["k_pos"]))
        cell_loc[key].append(float(r["sl_mean_local"]))
        cell_glb[key].append(float(r["sl_mean_global"]))

    fig, ax = plt.subplots(figsize=CANVAS)
    legend_handles: dict[str, object] = {}
    for (arch, _t, _k), vs_loc in cell_loc.items():
        if arch not in ARCH_STYLE:
            continue
        vs_glb = cell_glb[(arch, _t, _k)]
        x = statistics.mean(vs_loc)
        y = statistics.mean(vs_glb)
        marker, color, _lbl = ARCH_STYLE[arch]
        h = ax.scatter(x, y, marker=marker, c=color, s=42, alpha=0.85,
                       edgecolors="white", linewidth=0.5)
        if arch not in legend_handles:
            legend_handles[arch] = h

    lim = 0.18
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--",
            linewidth=0.8, label=r"$y = x$")
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    ax.set_xlabel(r"local  $\bar r(z_{j^*}, a_{i,t})$")
    ax.set_ylabel(r"global $\bar r(z_{j^*}, s_{i,t})$")
    ax.set_title("Denoising: single-latent correlation", pad=8)
    handles = [legend_handles[a] for a in HEADLINE_ARCHS if a in legend_handles]
    labels = [ARCH_STYLE[a][2] for a in HEADLINE_ARCHS if a in legend_handles]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=7,
              ncol=1, labelspacing=0.3, handletextpad=0.4,
              bbox_to_anchor=(0.0, 1.0))
    _clean_axes(ax)
    fig.tight_layout()
    _save_png_pdf(fig, out_dir / "c2_setup_b_singlelatent")
    plt.close(fig)


# ---------------------------------------------------------------------------
# (c) Setup D scatter: eAUC vs gAUC at np=10, seed-averaged cells
# ---------------------------------------------------------------------------
def render_setup_d_scatter(leaderboard: Path, out_dir: Path) -> None:
    rows = []
    for line in leaderboard.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        if r.get("datasource") != SETUP_D_DATASOURCE:
            continue
        rows.append(r)

    cell_e = defaultdict(list)
    cell_g = defaultdict(list)
    for r in rows:
        key = (r["arch"], r["eval_cfg"].get("t_label", "default"),
               int(r["eval_cfg"]["k_pos"]))
        cell_e[key].append(float(r["metrics"]["eauc"]))
        cell_g[key].append(float(r["metrics"]["gauc"]))

    fig, ax = plt.subplots(figsize=CANVAS)
    legend_handles: dict[str, object] = {}
    for (arch, _t, _k), vs_e in cell_e.items():
        if arch not in ARCH_STYLE:
            continue
        vs_g = cell_g[(arch, _t, _k)]
        x = statistics.mean(vs_e)
        y = statistics.mean(vs_g)
        marker, color, _lbl = ARCH_STYLE[arch]
        h = ax.scatter(x, y, marker=marker, c=color, s=42, alpha=0.85,
                       edgecolors="white", linewidth=0.5)
        if arch not in legend_handles:
            legend_handles[arch] = h

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$e\mathrm{AUC}$ (local)")
    ax.set_ylabel(r"$g\mathrm{AUC}$ (global)")
    ax.set_title(
        r"Coupling: dictionary alignment, $n_{\mathrm{parents}}{=}10$", pad=8,
    )
    handles = [legend_handles[a] for a in HEADLINE_ARCHS if a in legend_handles]
    labels = [ARCH_STYLE[a][2] for a in HEADLINE_ARCHS if a in legend_handles]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=7,
              ncol=1, labelspacing=0.3, handletextpad=0.4)
    _clean_axes(ax)
    fig.tight_layout()
    _save_png_pdf(fig, out_dir / "c2_setup_d_scatter_clean")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _purified_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(probe_json: Path, leaderboard: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    render_bars(probe_json, leaderboard, out_dir)
    render_setup_b_scatter(probe_json, out_dir)
    render_setup_d_scatter(leaderboard, out_dir)
    print(f"wrote 3 panels to {out_dir}")


def cli() -> None:
    root = _purified_root()
    ap = argparse.ArgumentParser(description=(
        "C2 synthetic Fig 2 paper renderer (Denoising + Coupling overview). "
        "Defaults to in-repo canonical paths."
    ))
    ap.add_argument(
        "--probe-json", type=Path,
        default=root / "experiments" / "c1_noisy_filler"
                     / "denoising_probe_results.json",
        help=("Setup B probe JSON "
              "(default: purified/experiments/c1_noisy_filler/"
              "denoising_probe_results.json)."),
    )
    ap.add_argument(
        "--leaderboard", type=Path,
        default=root / "experiments" / "c2_hierarchical"
                     / "setup_d_leaderboard.jsonl",
        help=("Setup D leaderboard slice "
              "(default: purified/experiments/c2_hierarchical/"
              "setup_d_leaderboard.jsonl). "
              "Falls back to purified/results/leaderboard.jsonl if missing."),
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=root / "figs" / "c2",
        help="Output directory (default: purified/figs/c2/).",
    )
    args = ap.parse_args()

    leaderboard = args.leaderboard
    if not leaderboard.exists():
        fallback = root / "results" / "leaderboard.jsonl"
        if fallback.exists():
            leaderboard = fallback
        else:
            raise SystemExit(
                f"Setup D leaderboard not found at {args.leaderboard} "
                f"or fallback {fallback}"
            )
    main(
        probe_json=args.probe_json,
        leaderboard=leaderboard,
        out_dir=args.output_dir,
    )


if __name__ == "__main__":
    cli()
