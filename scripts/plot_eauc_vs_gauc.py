"""eAUC (local) vs gAUC (global) scatter — analogue of Bill's fig8.

Bill's fig8 plots per-feature single-latent correlation against observed
support `s` (local) vs hidden state `h` (global), one point per feature.
We don't have per-feature corr data for the coupled bench, but we do
have decoder-cosine eAUC (vs emission features f_m) and gAUC (vs hidden
features h_feat_k) per (arch, ρ, k_pos) cell. That's the same conceptual
local-x vs global-y axis at the cell level.

Reads experiments/phase3_coupled/results/coupled_rho_sweep/results.json
and writes docs/bill/results/hmm_spec_eauc_vs_gauc.png.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


IN_PATH = (
    "experiments/phase3_coupled/results/coupled_rho_sweep/results.json"
)
OUT_PATH = "../temp_xc/docs/bill/results/hmm_spec_eauc_vs_gauc.png"

ARCH_COLOR = {
    "regular_sae": "#7f7f7f",
    "txcdr_t2": "#ff7f0e",
    "txcdr_t5": "#d62728",
    "txc_pro": "#1f77b4",
}
ARCH_LABEL = {
    "regular_sae": "regular SAE",
    "txcdr_t2": "plain TXCDR T=2",
    "txcdr_t5": "plain TXCDR T=5",
    "txc_pro": "TXC-pro / H8",
}
RHO_MARKER = {0.0: "o", 0.6: "s", 0.9: "D"}
RHO_SIZE = {0.0: 70, 0.6: 90, 0.9: 130}


def main() -> None:
    with open(IN_PATH) as f:
        results = json.load(f)
    print(f"loaded {len(results)} cells")

    fig, ax = plt.subplots(1, 1, figsize=(8, 7.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.25, lw=1, zorder=0)

    for r in results:
        arch = r["model"]
        rho = r["rho"]
        ax.scatter(
            r["emission_auc"], r["hidden_auc"],
            color=ARCH_COLOR[arch], marker=RHO_MARKER[rho],
            s=RHO_SIZE[rho], alpha=0.85,
            edgecolors="white", linewidths=1.0, zorder=4,
        )
        # Annotate with raw_k inside the marker for context
        ax.text(
            r["emission_auc"], r["hidden_auc"], str(r["raw_k"]),
            fontsize=6, ha="center", va="center", color="white",
            zorder=5,
        )

    ax.set_xlabel("eAUC (local) — decoder cosine vs emission features f_m")
    ax.set_ylabel("gAUC (global) — decoder cosine vs hidden features h_feat_k")
    ax.set_title(
        f"Local vs global feature recovery on coupled bench  "
        f"({len(results)} cells)\n"
        f"4 arches × ρ ∈ {{0.0, 0.6, 0.9}} × k_pos ∈ {{1, 2, 5, 10}};  "
        f"label inside marker = raw_k",
        fontsize=10,
    )
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, alpha=0.3)

    arch_handles = [
        mlines.Line2D([], [], marker="o", linestyle="", markersize=10,
                      color=c, label=ARCH_LABEL[a],
                      markeredgecolor="white", markeredgewidth=1.0)
        for a, c in ARCH_COLOR.items()
    ]
    rho_handles = [
        mlines.Line2D([], [], marker=m, linestyle="", markersize=10,
                      color="black", label=f"ρ = {rho}",
                      markeredgecolor="white", markeredgewidth=1.0)
        for rho, m in RHO_MARKER.items()
    ]
    leg1 = ax.legend(handles=arch_handles, loc="lower right", fontsize=9,
                     title="architecture")
    ax.add_artist(leg1)
    ax.legend(handles=rho_handles, loc="upper left", fontsize=9,
              title="ρ")

    fig.tight_layout()
    out = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
