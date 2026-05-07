"""Paper-grade Setup A plot for c2.md ((decision 2026-05-07): missing).

Reads c2 leaderboard, plots gAUC vs k_pos per (arch, T_label) at the
canonical ρ=0.7. One line per arch family with error bars.

Mirrors c1_noisy_filler/analysis.py:_save_plot style:
- Okabe-Ito + RdPu palette (color-blind safe)
- 150 dpi paper output + 72 dpi thumb
- Log x-scale on k_pos
- Setup A explicitly named in title

Outputs:
- experiments/c2_synthetic_coupled/plots/c2_coupled_gauc_vs_k.png
- experiments/c2_synthetic_coupled/plots/c2_coupled_eauc_vs_k.png

Invoke:
    .venv/bin/python -m experiments.c2_synthetic_coupled.plot_setup_a
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
import numpy as np

from temp_bench.cache import _read_jsonl, leaderboard_path


COMPONENT = "c2"

CANONICAL_ARCH_TS: list[tuple[str, str]] = [
    ("topk_sae",    "default"),
    ("tsae_paper",  "default"),
    ("stacked_sae", "T=2"),
    ("stacked_sae", "default"),  # T=5
    ("txc_base",    "default"),  # T=5
    ("txc_pro",     "T=2"),
    ("txc_pro",     "T=5"),
    ("txc_pro",     "T=12"),
]

# Paper-friendly Okabe-Ito + RdPu cmap for txc_pro T-sweep.
_txc_cmap = _cm.get_cmap("RdPu", 6)
PLOT_STYLE: dict[tuple[str, str], dict] = {
    ("topk_sae",    "default"): {"label": "TopK-SAE",       "color": "#000000",   "ls": "-",  "marker": "P"},
    ("tsae_paper",  "default"): {"label": "T-SAE",          "color": "#CC79A7",   "ls": "-",  "marker": "h"},
    ("stacked_sae", "T=2"):     {"label": "Stacked T=2",    "color": "#9467bd",   "ls": "-",  "marker": "o"},
    ("stacked_sae", "default"): {"label": "Stacked T=5",    "color": "#7E57A0",   "ls": "--", "marker": "^"},
    ("txc_base",    "default"): {"label": "TXC-base T=5",   "color": "#882255",   "ls": "-",  "marker": "X"},
    ("txc_pro",     "T=2"):     {"label": "TXC-pro T=2",    "color": _txc_cmap(2), "ls": "-", "marker": "v"},
    ("txc_pro",     "T=5"):     {"label": "TXC-pro T=5",    "color": _txc_cmap(3), "ls": "-", "marker": "s"},
    ("txc_pro",     "T=12"):    {"label": "TXC-pro T=12",   "color": _txc_cmap(5), "ls": "-", "marker": "D"},
}


def collect():
    """Aggregate c2 leaderboard at ρ=0.7 (legacy + explicit)."""
    by_cell: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"eauc": [], "gauc": []}
    )
    for r in _read_jsonl(leaderboard_path()):
        if r.get("component") != COMPONENT:
            continue
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        cfg = r["eval_cfg"]
        # Setup A is canonical ρ=0.7; legacy cells have no rho field.
        rho = cfg.get("rho", 0.7)
        if rho != 0.7:
            continue
        k = cfg.get("k_pos")
        if k is None:
            continue
        key = (r["arch"], cfg.get("t_label", "default"), int(k))
        by_cell[key]["eauc"].append(float(r["metrics"].get("eauc", float("nan"))))
        by_cell[key]["gauc"].append(float(r["metrics"].get("gauc", float("nan"))))
    agg = {}
    for key, vals in by_cell.items():
        agg[key] = {
            "eauc_mean": float(np.nanmean(vals["eauc"])),
            "eauc_std":  float(np.nanstd(vals["eauc"], ddof=1)) if len(vals["eauc"]) > 1 else 0.0,
            "gauc_mean": float(np.nanmean(vals["gauc"])),
            "gauc_std":  float(np.nanstd(vals["gauc"], ddof=1)) if len(vals["gauc"]) > 1 else 0.0,
            "n":         len(vals["gauc"]),
        }
    return agg


def _plot(agg: dict, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ks_global = sorted({k for (_, _, k) in agg.keys()})

    for arch, t_label in CANONICAL_ARCH_TS:
        ks = []
        ys = []
        es = []
        for k in ks_global:
            stat = agg.get((arch, t_label, k))
            if stat is None:
                continue
            ks.append(k)
            ys.append(stat[f"{metric}_mean"])
            es.append(stat[f"{metric}_std"])
        if not ks:
            continue
        s = PLOT_STYLE.get((arch, t_label), {"label": f"{arch} {t_label}",
                                              "color": "gray", "marker": "o", "ls": "-"})
        ax.errorbar(
            ks, ys, yerr=es,
            label=s["label"], color=s["color"],
            linestyle=s.get("ls", "-"), marker=s["marker"],
            markersize=6, capsize=3, linewidth=1.6, alpha=0.9,
        )

    ax.set_xlabel(r"Per-token sparsity $k_{\rm pos}$", fontsize=11)
    if metric == "gauc":
        ax.set_ylabel(r"gAUC (global feature recovery)", fontsize=11)
        ax.set_title(r"Setup A: hidden-feature recovery vs $k_{\rm pos}$  "
                     r"(coupled, $\rho{=}0.7$)", fontsize=12)
    else:
        ax.set_ylabel(r"eAUC (emission feature recovery)", fontsize=11)
        ax.set_title(r"Setup A: emission-feature recovery vs $k_{\rm pos}$  "
                     r"(coupled, $\rho{=}0.7$)", fontsize=12)
    ax.set_ylim(0.25, 1.02)
    ax.set_xscale("log")
    ax.set_xticks(ks_global)
    ax.set_xticklabels([str(k) for k in ks_global])
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".thumb.png"), bbox_inches="tight", dpi=72)
    plt.close(fig)


def main():
    agg = collect()
    print(f"Setup A cells (ρ=0.7): {len(agg)}")
    plots_dir = Path("experiments/c2_synthetic_coupled/plots")
    _plot(agg, "gauc", plots_dir / "c2_coupled_gauc_vs_k.png")
    _plot(agg, "eauc", plots_dir / "c2_coupled_eauc_vs_k.png")
    print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
