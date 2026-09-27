"""Alternative headline: mean global/local ratio per architecture.

For each (arch, T, k_pos) cell we compute the ratio
    Denoising bench: sl_mean_global / sl_mean_local  (single-latent)
    Coupling bench:  gAUC / eAUC                     (dictionary AUC)
seed-averaged over the 3 seeds, then take the mean across all cells per
arch. Ratio > 1 means the architecture recovers the global feature better
than the local one on average.

Output: figs/c2/c2_synth_ratio_headline_preview.png  (preview only).
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
OUT  = REPO / "figs" / "c2" / "c2_synth_ratio_headline_preview.png"

ARCH_LABEL = {
    "topk_sae":   "TopK SAE",
    "tsae_paper": "T-SAE",
    "tfa_pos":    "TFA-pos",
    "txc_base":   "TXC-base",
    "txc_pro":    "TXC-pro",
}
HEADLINE_ARCHS = list(ARCH_LABEL.keys())

EPS = 1e-6


def denoising_ratios() -> dict[str, tuple[float, float, float]]:
    """{arch: (mean_ratio, min_ratio, max_ratio)} over (T, k) cells."""
    rows = json.loads((DATA / "denoising_probe_results.json").read_text())
    cell_loc = defaultdict(list)
    cell_glb = defaultdict(list)
    for r in rows:
        key = (r["arch_name"], r.get("t_label", "default"), int(r["k_pos"]))
        cell_loc[key].append(float(r["sl_mean_local"]))
        cell_glb[key].append(float(r["sl_mean_global"]))
    by_arch = defaultdict(list)
    for key, locs in cell_loc.items():
        glbs = cell_glb[key]
        if len(locs) < 2:
            continue
        loc_mean = statistics.mean(locs)
        glb_mean = statistics.mean(glbs)
        if abs(loc_mean) < EPS:
            continue
        by_arch[key[0]].append(glb_mean / loc_mean)
    out = {}
    for a, vs in by_arch.items():
        out[a] = (statistics.mean(vs), min(vs), max(vs))
    return out


def coupling_ratios() -> dict[str, tuple[float, float, float]]:
    lines = (DATA / "setup_d_leaderboard.jsonl").read_text().splitlines()
    cell_e = defaultdict(list)
    cell_g = defaultdict(list)
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        if r["datasource"] != "toy_coupled_noisy_K10_M20_d256_pB05_np10":
            continue
        key = (r["arch"], r["eval_cfg"].get("t_label", "default"), int(r["eval_cfg"]["k_pos"]))
        cell_e[key].append(float(r["metrics"]["eauc"]))
        cell_g[key].append(float(r["metrics"]["gauc"]))
    by_arch = defaultdict(list)
    for key, es in cell_e.items():
        gs = cell_g[key]
        if len(es) < 2:
            continue
        e_mean = statistics.mean(es)
        g_mean = statistics.mean(gs)
        if abs(e_mean) < EPS:
            continue
        by_arch[key[0]].append(g_mean / e_mean)
    out = {}
    for a, vs in by_arch.items():
        out[a] = (statistics.mean(vs), min(vs), max(vs))
    return out


def main():
    den = denoising_ratios()
    coup = coupling_ratios()

    archs = HEADLINE_ARCHS
    labels = [ARCH_LABEL[a] for a in archs]
    den_v = [den.get(a, (np.nan, np.nan, np.nan))[0] for a in archs]
    den_lo = [den.get(a, (np.nan, np.nan, np.nan))[1] for a in archs]
    den_hi = [den.get(a, (np.nan, np.nan, np.nan))[2] for a in archs]
    coup_v = [coup.get(a, (np.nan, np.nan, np.nan))[0] for a in archs]
    coup_lo = [coup.get(a, (np.nan, np.nan, np.nan))[1] for a in archs]
    coup_hi = [coup.get(a, (np.nan, np.nan, np.nan))[2] for a in archs]

    def _err(vals, lo, hi):
        lower = [(v - l) if not np.isnan(v) and not np.isnan(l) else 0 for v, l in zip(vals, lo)]
        upper = [(h - v) if not np.isnan(v) and not np.isnan(h) else 0 for v, h in zip(vals, hi)]
        return [lower, upper]

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    x = np.arange(len(archs))
    w = 0.38
    bb = ax.bar(x - w/2, den_v, w, yerr=_err(den_v, den_lo, den_hi),
                label=r"Denoising  ($\bar r_{\mathrm{global}} / \bar r_{\mathrm{local}}$)",
                color="#7E57C2", edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#222"})
    bd = ax.bar(x + w/2, coup_v, w, yerr=_err(coup_v, coup_lo, coup_hi),
                label=r"Coupling  ($g\mathrm{AUC} / e\mathrm{AUC}$)",
                color="#E64A19", edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "capsize": 2.5, "ecolor": "#222"})
    for rect, val, hi in list(zip(bb, den_v, den_hi)) + list(zip(bd, coup_v, coup_hi)):
        if not np.isnan(val):
            top = hi if not np.isnan(hi) else val
            ax.text(rect.get_x() + rect.get_width()/2, top + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label=r"$=1$ (no denoising)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean global/local ratio")
    ax.set_title("Mean global/local ratio per architecture", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=7,
              labelspacing=0.3, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")
    print("\nDenoising bench: mean (min, max) ratio per arch:")
    for a in archs:
        if a in den:
            m, lo, hi = den[a]
            print(f"  {a:14} {m:.3f}  ({lo:.3f}, {hi:.3f})")
    print("\nCoupling bench: mean (min, max) ratio per arch:")
    for a in archs:
        if a in coup:
            m, lo, hi = coup[a]
            print(f"  {a:14} {m:.3f}  ({lo:.3f}, {hi:.3f})")


if __name__ == "__main__":
    main()
