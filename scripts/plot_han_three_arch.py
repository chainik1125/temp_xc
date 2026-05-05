"""Plot AUC and NMSE vs rho for the four-arch Han-recipe three-arch sweep.

Reads results/han_three_arch/results.json (produced by
run_han_three_arch_sweep.py) and writes:
    results/han_three_arch/auc_vs_rho.png
    results/han_three_arch/nmse_vs_rho.png

One line per architecture, three rho points each. Y-axis is the metric.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARCH_ORDER = ["regular_sae", "stacked_sae", "txc_base", "txc_pro"]
ARCH_LABEL = {
    "regular_sae": "regular SAE (k=20)",
    "stacked_sae": "Stacked SAE (k=20, T=5)",
    "txc_base": "TXC-base (Han, T=5)",
    "txc_pro": "TXC-pro / H8 (Han, T_max=10)",
}
ARCH_COLOR = {
    "regular_sae": "#888888",
    "stacked_sae": "#d95f02",
    "txc_base": "#1b9e77",
    "txc_pro": "#7570b3",
}


def main() -> None:
    in_path = "results/han_three_arch/results.json"
    out_dir = "results/han_three_arch"
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path) as f:
        results = json.load(f)

    rhos = sorted({r["rho"] for r in results})
    by_arch = {a: {} for a in ARCH_ORDER}
    for r in results:
        by_arch[r["model"]][r["rho"]] = r

    for metric, label, fname in [
        ("auc", "Feature recovery AUC", "auc_vs_rho.png"),
        ("nmse", "Reconstruction NMSE", "nmse_vs_rho.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for arch in ARCH_ORDER:
            ys = [by_arch[arch][r][metric] for r in rhos]
            ax.plot(
                rhos, ys, marker="o", lw=2, color=ARCH_COLOR[arch],
                label=ARCH_LABEL[arch],
            )
        ax.set_xlabel("ρ (lag-1 autocorrelation)")
        ax.set_ylabel(label)
        ax.set_title(
            f"{label}  —  d_sae=2048, k_pos=20, n_features=128, d_in=256"
        )
        if metric == "auc":
            ax.set_ylim(0.45, 1.02)
        else:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
