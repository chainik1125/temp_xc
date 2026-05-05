"""Plots for results/rho_k_sweep/results.json (ρ × k sweep with Han recipe).

Two figures:
  - rho_k_auc_grid.png:  AUC vs ρ, one panel per k_pos (4 panels), one line per arch
  - rho_k_delta_vs_sae.png:  ΔAUC = arch − regular_sae, vs ρ, one panel per k_pos
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARCH_ORDER = ["regular_sae", "txcdr_t2", "txcdr_t5", "txc_pro"]
ARCH_LABEL = {
    "regular_sae": "regular SAE (per-token k)",
    "txcdr_t2": "plain TXCDR T=2",
    "txcdr_t5": "plain TXCDR T=5",
    "txc_pro": "TXC-pro / H8 (T_max=10)",
}
ARCH_COLOR = {
    "regular_sae": "#7f7f7f",
    "txcdr_t2": "#ff7f0e",
    "txcdr_t5": "#d62728",
    "txc_pro": "#1f77b4",
}
ARCH_MARKER = {
    "regular_sae": "P",
    "txcdr_t2": "o",
    "txcdr_t5": "s",
    "txc_pro": "D",
}


def main() -> None:
    in_path = "results/rho_k_sweep/results.json"
    out_dir = "results/rho_k_sweep"
    os.makedirs(out_dir, exist_ok=True)

    with open(in_path) as f:
        results = json.load(f)

    rhos = sorted({r["rho"] for r in results})
    ks = sorted({r["k_pos"] for r in results})
    by = {(r["model"], r["rho"], r["k_pos"]): r for r in results}

    # ── Panel grid: AUC vs ρ, one panel per k ──
    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.2),
                             sharey=True)
    for ki, k in enumerate(ks):
        ax = axes[ki]
        for arch in ARCH_ORDER:
            ys = [by[(arch, rho, k)]["auc"] for rho in rhos]
            ax.plot(
                rhos, ys, marker=ARCH_MARKER[arch], lw=2, ms=8,
                color=ARCH_COLOR[arch], label=ARCH_LABEL[arch],
            )
        ax.set_xlabel("ρ (lag-1 autocorr)")
        if ki == 0:
            ax.set_ylabel("Feature recovery AUC")
        ax.set_title(f"k_pos = {k}  (raw_k: SAE={k}, TXCDR-T2={k*2}, "
                     f"TXCDR-T5={k*5}, H8={k*10})", fontsize=9)
        ax.set_ylim(0.70, 1.005)
        ax.grid(True, alpha=0.3)
        if ki == len(ks) - 1:
            ax.legend(fontsize=7, loc="lower right")

    plt.suptitle(
        "ρ × k sweep — d_sae=2048 (=8×d_in), Bill three-arch DataConfig "
        "(n_features=128, d_model=256, π=0.05)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "rho_k_auc_grid.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ── Δ vs SAE: ΔAUC = arch − regular_sae, vs ρ, one panel per k ──
    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.2),
                             sharey=True)
    for ki, k in enumerate(ks):
        ax = axes[ki]
        sae_y = [by[("regular_sae", rho, k)]["auc"] for rho in rhos]
        ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
        for arch in ARCH_ORDER:
            if arch == "regular_sae":
                continue
            ys = [
                by[(arch, rho, k)]["auc"] - sae_y[i]
                for i, rho in enumerate(rhos)
            ]
            ax.plot(
                rhos, ys, marker=ARCH_MARKER[arch], lw=2, ms=8,
                color=ARCH_COLOR[arch], label=ARCH_LABEL[arch],
            )
        ax.set_xlabel("ρ (lag-1 autocorr)")
        if ki == 0:
            ax.set_ylabel("Δ AUC vs regular SAE")
        ax.set_title(f"k_pos = {k}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ki == len(ks) - 1:
            ax.legend(fontsize=7, loc="best")

    plt.suptitle("ΔAUC vs regular SAE — d_sae=2048", fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(out_dir, "rho_k_delta_vs_sae.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
