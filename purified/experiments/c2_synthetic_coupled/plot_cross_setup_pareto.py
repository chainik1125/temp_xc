"""Cross-setup Pareto plot — TXC-vs-SAE on the eAUC × gAUC plane,
aggregating Setups A, D-np10, E into one figure.

(decision 2026-05-07) (NeurIPS deadline rescue): one paper-grade figure
showing **TXC dictionaries upper-left (high gAUC, low/mid eAUC),
per-token SAE dictionaries lower-right (low gAUC, high eAUC),
robustly across multiple synthetic generative processes.**

Reads c2 leaderboard rows (component='c2', smoke=False) at ρ=0.7
across 3 datasources:
- toy_coupled_K10_M20_d256                  (Setup A)
- toy_coupled_noisy_K10_M20_d256_pB05_np10  (Setup D pB05_np10)
- toy_hierarchical_Kg10_Kl30_d256           (Setup E)

Each (arch, T, k_pos, datasource) cell averaged over seeds → one
point. Color = arch family (TXC family magenta/pink, SAE family
black/grey, baseline encoders muted). Marker = setup.

Output: experiments/c2_synthetic_coupled/plots/c2_cross_setup_pareto.png

Invoke:
    .venv/bin/python -m experiments.c2_synthetic_coupled.plot_cross_setup_pareto
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
import numpy as np

from temp_bench.cache import _read_jsonl, leaderboard_path


COMPONENT = "c2"

DATASOURCES = {
    "toy_coupled_K10_M20_d256":                "Setup A (coupled)",
    "toy_coupled_noisy_K10_M20_d256_pB05_np10": "Setup D-np10 (noisy+max-overlap)",
    "toy_hierarchical_Kg10_Kl30_d256":          "Setup E (hierarchical)",
}

# Color = arch family. Lighter shades for non-TXC.
ARCH_COLOR = {
    "topk_sae":    "#000000",   # black (per-token TopK)
    "tsae_paper":  "#7E57C2",   # purple (T-SAE)
    "tfa_pos":     "#2ca02c",   # green
    "stacked_sae": "#9467bd",   # purple (Stacked)
    "txc_base":    "#CC79A7",   # magenta
    "txc_pro":     "#882255",   # dark magenta
}

# Marker = setup.
SETUP_MARKER = {
    "Setup A (coupled)":              "o",
    "Setup D-np10 (noisy+max-overlap)": "s",
    "Setup E (hierarchical)":         "^",
}


def _arch_family_label(arch, t_label):
    """Compact label for legend."""
    if arch == "txc_base":
        if t_label == "default":
            return "TXC-base T=5"
        return f"TXC-base {t_label}"
    if arch == "txc_pro":
        return f"TXC-pro {t_label}"
    if arch == "stacked_sae":
        return f"Stacked {t_label if t_label != 'default' else 'T=5'}"
    if arch == "tfa_pos":
        return "TFA-pos"
    if arch == "topk_sae":
        return "TopK-SAE"
    if arch == "tsae_paper":
        return "T-SAE"
    return f"{arch} {t_label}"


def collect():
    by_cell = defaultdict(lambda: {"eauc": [], "gauc": []})
    for r in _read_jsonl(leaderboard_path()):
        if r.get("component") != COMPONENT:
            continue
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        cfg = r.get("eval_cfg", {})
        ds = r.get("datasource", "")
        if ds not in DATASOURCES:
            continue
        # Setup A is canonical ρ=0.7; D and E datasources have ρ=0.9 (max temporal coherence).
        rho = cfg.get("rho")
        if ds == "toy_coupled_K10_M20_d256":
            if rho not in (None, 0.7):
                continue
        # D and E don't have a meaningful ρ filter (their ρ is baked in).
        k = cfg.get("k_pos")
        if k is None:
            continue
        key = (DATASOURCES[ds], r["arch"], cfg.get("t_label", "default"), int(k))
        e = r.get("metrics", {}).get("eauc")
        g = r.get("metrics", {}).get("gauc")
        if e is None or g is None:
            continue
        by_cell[key]["eauc"].append(float(e))
        by_cell[key]["gauc"].append(float(g))

    agg = {}
    for key, vals in by_cell.items():
        if not vals["eauc"]:
            continue
        agg[key] = {
            "eauc_mean": float(np.nanmean(vals["eauc"])),
            "gauc_mean": float(np.nanmean(vals["gauc"])),
            "n":         len(vals["eauc"]),
        }
    return agg


def plot(agg, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, lw=1)

    # Group + plot.
    legend_handles = {}
    for (setup, arch, t_label, k), stats in sorted(agg.items()):
        marker = SETUP_MARKER.get(setup, "x")
        color = ARCH_COLOR.get(arch, "gray")
        label = _arch_family_label(arch, t_label)
        # Edge: black on TXC family, gray otherwise.
        edge = "black" if arch in ("txc_base", "txc_pro") else "0.4"
        sc = ax.scatter(stats["eauc_mean"], stats["gauc_mean"],
                        color=color, marker=marker,
                        s=80, alpha=0.7, edgecolors=edge, linewidths=0.5)
        if label not in legend_handles:
            legend_handles[label] = sc

    # Axes + title.
    ax.set_xlabel(r"eAUC (local: emission-feature recovery)", fontsize=12)
    ax.set_ylabel(r"gAUC (global: hidden-feature recovery)", fontsize=12)
    ax.set_title(r"C2 cross-setup Pareto: TXC dictionaries skew GLOBAL, "
                 r"per-token SAE skew LOCAL"
                 "\n"
                 r"(setups A coupled · D-np10 noisy+overlap · E hierarchical, "
                 r"$d_{\rm sae}{=}40$, all k_pos × 3 seeds averaged)",
                 fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Two legends: arch family (color) + setup (marker).
    leg1 = ax.legend(
        legend_handles.values(), legend_handles.keys(),
        title="arch", loc="upper left", fontsize=8, framealpha=0.85, ncol=2,
    )
    ax.add_artist(leg1)

    setup_handles = []
    setup_labels = []
    for setup, mk in SETUP_MARKER.items():
        h = ax.scatter([], [], color="gray", marker=mk, s=70, edgecolors="0.4")
        setup_handles.append(h)
        setup_labels.append(setup)
    ax.legend(setup_handles, setup_labels, title="setup",
              loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".thumb.png"), bbox_inches="tight", dpi=72)
    plt.close(fig)


def main():
    agg = collect()
    print(f"Cross-setup Pareto cells: {len(agg)}")
    out = Path("experiments/c2_synthetic_coupled/plots/c2_cross_setup_pareto.png")
    plot(agg, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
