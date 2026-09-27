"""Generate the c3 (sparse probing) paper figures from Han's purified results.

Outputs six PDFs to ``temp_xc_tex/figs/`` (paired BASE/IT subfigs for three
parent figures: bar summary, zoomed curves without TFA, full curves with TFA):

- ``c3_sparse_probing_summary_gemma.pdf``         — BASE peak-AUC bar
- ``c3_sparse_probing_summary_gemma_it.pdf``      — IT  peak-AUC bar
- ``c3_sparse_probing_curves_gemma.pdf``          — BASE AUC vs k_feats, no TFA (zoom)
- ``c3_sparse_probing_curves_gemma_it.pdf``       — IT  AUC vs k_feats, no TFA (zoom)
- ``c3_sparse_probing_full_gemma.pdf``            — BASE AUC vs k_feats, with TFA (appendix)
- ``c3_sparse_probing_full_gemma_it.pdf``         — IT  AUC vs k_feats, with TFA (appendix)

Inputs (read via `git show origin/final:...`):
- ``purified/experiments/c3_probing/results.json`` (IT, top-level dict keyed by arch)
- ``purified/experiments/c3_probing_base/results.json`` (BASE, dict with `by_arch` nested)

Usage::

    python scripts/make_c3_sparse_probing_figures.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_DATA = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc"
)
REPO_TEX = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc_tex"
)
FIGS_OUT = REPO_TEX / "figs"

K_GRID = (5, 10, 20, 40, 80, 160, 320, 640)

ARCH_LABEL = {
    "topk_sae":     "TopK SAE",
    "tsae_paper":   "T-SAE",
    "tfa":          "TFA",
    "mlc":          "MLC",
    "txc_base":     "TXC-base",
    "txc_base_T5":  "TXC-base (T=5)",
    "txc_base_T10": "TXC-base (T=10)",
    "txc_base_T20": "TXC-base (T=20)",
    "txc_pro":      "TXC-pro",
}
ARCH_COLOR = {
    "topk_sae":     "#4477AA",   # blue (per-token baseline)
    "tsae_paper":   "#229922",   # green
    "tfa":          "#888888",   # gray (the laggard)
    "mlc":          "#AA3377",   # magenta (multi-layer)
    "txc_base":     "#EE6677",   # red
    "txc_base_T5":  "#EE6677",   # red — same family
    "txc_base_T10": "#CC4444",   # darker red
    "txc_base_T20": "#992222",   # darkest red
    "txc_pro":      "#CC8800",   # orange
}
# Display order: per-token baselines, then temporal family, then TFA last.
DISPLAY_ORDER = [
    "topk_sae", "mlc", "tsae_paper",
    "txc_base", "txc_base_T5", "txc_base_T10", "txc_base_T20",
    "txc_pro", "tfa",
]


def git_show(branch_path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_DATA), "show", branch_path], text=True
    )


def load_it() -> dict[str, dict[int, dict]]:
    """IT results: {arch: {k_feats: {mean_auc, std_seeds_auc, n_seeds, ...}}}."""
    txt = git_show("origin/final:purified/experiments/c3_probing/results.json")
    raw = json.loads(txt)
    out: dict[str, dict[int, dict]] = {}
    for arch, by_k in raw.items():
        out[arch] = {int(k): v for k, v in by_k.items()}
    return out


def load_base() -> dict[str, dict[int, dict]]:
    """BASE results: {by_arch: {arch: {kN: {...}}}, datasource, ...}.
    Returns same shape as load_it for downstream uniformity."""
    txt = git_show("origin/final:purified/experiments/c3_probing_base/results.json")
    raw = json.loads(txt)
    out: dict[str, dict[int, dict]] = {}
    for arch, by_k in raw.get("by_arch", {}).items():
        out[arch] = {int(k.lstrip("k")): v for k, v in by_k.items()}
    return out


def archs_in(d: dict, drop_tfa: bool = False) -> list[str]:
    """Return arch keys present in `d`, sorted by DISPLAY_ORDER, optionally dropping TFA."""
    archs = [a for a in DISPLAY_ORDER if a in d]
    if drop_tfa:
        archs = [a for a in archs if a != "tfa"]
    return archs


# ── Plot 1: "AUC of the AUC" sweep summary ──────────────────────────

def auc_of_auc(d_arch: dict[int, dict]) -> tuple[float, float]:
    """Per-arch summary: mean SAEBench AUC averaged across the log-spaced
    k_feats sweep, weighted by log2(k_feats) to give equal weight to each
    doubling. Equivalently, trapezoidal integral of mean_auc over log2 k_feats
    divided by the sweep span. Returns (mean, propagated_seed_std)."""
    items = sorted(d_arch.items())
    ks = np.array([k for k, _ in items], dtype=float)
    aucs = np.array([v["mean_auc"] for _, v in items])
    stds = np.array([v.get("std_seeds_auc", 0.0) or 0.0 for _, v in items])
    x = np.log2(ks)
    span = x[-1] - x[0]
    aoa_mean = float(np.trapz(aucs, x) / span)
    # Propagate seed-σ through the trapezoidal weights (assume per-k seed errors
    # are independent, which is approximately true since k changes the probe
    # selection per fold but the underlying activations are shared).
    w = np.zeros_like(x)
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2
    w[1:-1] = (x[2:] - x[:-2]) / 2
    w = w / span  # so sum(w) == 1
    aoa_std = float(np.sqrt(np.sum((w * stds) ** 2)))
    return aoa_mean, aoa_std


def plot_auc_of_auc_bar(d: dict, out_path: Path, title_suffix: str) -> None:
    archs = archs_in(d, drop_tfa=False)
    means: list[float] = []
    stds: list[float] = []
    for a in archs:
        m, s = auc_of_auc(d[a])
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(4.0, 3.4), dpi=200)
    x = np.arange(len(archs))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, width=0.65,
        color=[ARCH_COLOR[a] for a in archs], edgecolor="black", linewidth=0.7,
        error_kw=dict(ecolor="black", lw=0.9, capthick=0.9),
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width()/2, m + 0.005,
            f"{m:.3f}", ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs], fontsize=8.5,
                       rotation=30, ha="right")
    ax.set_ylabel(r"$\overline{\mathrm{AUC}}$ across $\log_2 k_{\mathrm{feats}}$ sweep",
                  fontsize=10)
    ax.set_title(f"Sparse-probing summary — {title_suffix}", fontsize=10)
    y_lo = max(0.55, min(means) - 0.05)
    ax.set_ylim(y_lo, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    print(f"  → {out_path.name} (+ .png)")


# ── Plot 2 / 3: AUC vs k_feats curves ────────────────────────────────

def plot_curves(d: dict, out_path: Path, *, drop_tfa: bool, title_suffix: str) -> None:
    archs = archs_in(d, drop_tfa=drop_tfa)
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=200)
    for a in archs:
        items = sorted(d[a].items())
        ks = np.array([k for k, _ in items], dtype=float)
        aucs = np.array([v["mean_auc"] for _, v in items])
        stds = np.array([v.get("std_seeds_auc", 0.0) or 0.0 for _, v in items])
        col = ARCH_COLOR[a]
        # Different linestyle for the T-sweep variants of TXC-base to keep them
        # distinguishable from the headline (canonical) cells.
        ls = "-"
        lw = 1.4 if a in ("topk_sae", "tsae_paper", "mlc", "txc_pro", "tfa", "txc_base") else 1.0
        ms = 4.0 if lw >= 1.4 else 2.8
        ax.fill_between(ks, aucs - stds, aucs + stds, color=col, alpha=0.18, lw=0)
        ax.plot(ks, aucs, ls + "o", color=col, lw=lw, ms=ms, label=ARCH_LABEL[a])
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(K_GRID))
    ax.set_xticklabels([str(k) for k in K_GRID])
    ax.minorticks_off()
    ax.set_xlabel(r"$k_{\mathrm{feats}}$ (top-$S$ probe features)", fontsize=10)
    ax.set_ylabel("Mean SAEBench AUC", fontsize=10)
    if drop_tfa:
        # Zoom: tight band around the cluster of temporal archs + TopK.
        all_aucs = [v["mean_auc"] for a in archs for _, v in d[a].items()]
        y_lo = max(0.78, min(all_aucs) - 0.01)
        y_hi = min(0.97, max(all_aucs) + 0.01)
        ax.set_ylim(y_lo, y_hi)
        title = f"AUC vs $k_{{\\mathrm{{feats}}}}$ — {title_suffix} (TFA dropped)"
    else:
        ax.set_ylim(0.55, 0.97)
        title = f"AUC vs $k_{{\\mathrm{{feats}}}}$ — {title_suffix}"
    ax.set_title(title, fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    print(f"  → {out_path.name} (+ .png)")


def main():
    print("== loading c3 results (Gemma-2-2B-IT only) ==")
    it = load_it()
    print(f"  IT archs: {sorted(it.keys())}")
    print()
    print("== AUC of AUC (mean over log-k_feats sweep) per arch ==")
    for a in archs_in(it, drop_tfa=False):
        m, s = auc_of_auc(it[a])
        print(f"  {ARCH_LABEL[a]:18}  AoA = {m:.4f} ± {s:.4f}")
    print()
    print("== writing figures ==")
    FIGS_OUT.mkdir(parents=True, exist_ok=True)

    # AUC-of-AUC summary bar (with TFA)
    plot_auc_of_auc_bar(it, FIGS_OUT / "c3_sparse_probing_auc_of_auc_gemma_it.pdf",
                        "Gemma-2-2B-IT")

    # Zoomed curves (drop TFA)
    plot_curves(it, FIGS_OUT / "c3_sparse_probing_curves_gemma_it.pdf",
                drop_tfa=True, title_suffix="Gemma-2-2B-IT")

    # Full curves (with TFA, appendix)
    plot_curves(it, FIGS_OUT / "c3_sparse_probing_full_gemma_it.pdf",
                drop_tfa=False, title_suffix="Gemma-2-2B-IT")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
