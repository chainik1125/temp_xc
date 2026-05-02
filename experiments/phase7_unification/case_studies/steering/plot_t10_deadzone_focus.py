"""Focused Pareto plot — T=10 deadzone-escape chain only.

Filters the unified inventory to T-SAE k=20 (anchor) + the 8 T=10 archs
W trained for the deadzone-escape thread. Two panels: right-edge (V1) and
tiled-broadcast (V7). Per-position is omitted because the T=10 archs were
not evaluated under PP.

The headline cell — subseq H8 (T_max=10, t_samp=5) + Gaussian-mixture
position sampler — is highlighted with thicker linewidth + a star marker
at its cliff15 point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.phase7_unification.case_studies.steering.plot_unified_pareto import (
    INVENTORY, get_curve_avg, ANCHOR_15,
)
from src.plotting.save_figure import save_figure


T10_ARCH_IDS = {
    "tsae_paper_k20",
    "txc_h8_t10_kpos20_shifts10",
    "txc_h8_t10_kpos20_shifts2",
    "subseq_h8_tmax10_tsamp5_kpos20_shifts2_ctg",
    "subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2",
    "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_uniform_contr",
    "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_nested_uniform_contr",
    "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_gauss_s1.5_3.0_g2_contr",
    "spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_nested_gauss_s1.5_3.0_g2_contr",
}
HEADLINE_ARCH = "subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/workspace/temp_xc/experiments/phase7_unification/results/case_studies/plots/unified_pareto_t10_focus.png"
        ),
    )
    args = p.parse_args()

    # Filter inventory and gather curves
    curves = {}  # (arch_id, proto) -> (s, succ, coh, n, label, color)
    for arch_id, label, color, subdir_list in INVENTORY:
        if arch_id not in T10_ARCH_IDS:
            continue
        per_proto = get_curve_avg(arch_id, subdir_list)
        for proto, (s, succ, coh, n) in per_proto.items():
            curves[(arch_id, proto)] = (s, succ, coh, n, label, color)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    panels = [("right-edge", axes[0]), ("tiled-broadcast", axes[1])]

    for proto_filter, ax in panels:
        for (arch_id, proto), (s, succ, coh, n, label, color) in curves.items():
            # T-SAE has T=1 — show its right-edge curve on both panels
            if arch_id == "tsae_paper_k20":
                if proto != "right-edge":
                    continue
                display_label = (
                    f"{label} (n={n})" if proto_filter == "right-edge"
                    else f"{label} (T=1, RE=PP=V7) (n={n})"
                )
                ax.plot(coh, succ, marker="o", markersize=6, color=color,
                        linewidth=2.5, linestyle="--", label=display_label,
                        alpha=0.9, zorder=10)
                continue
            if proto != proto_filter:
                continue
            is_headline = arch_id == HEADLINE_ARCH
            lw = 2.5 if is_headline else 1.2
            alpha = 0.95 if is_headline else 0.65
            zorder = 9 if is_headline else 3
            marker = "o" if proto == "right-edge" else "s"
            ax.plot(coh, succ, marker=marker, markersize=5, color=color,
                    linewidth=lw, alpha=alpha, zorder=zorder,
                    label=f"{label} (n={n})")
            # Star the headline cell at its cliff15 point
            if is_headline:
                pts15 = [(co, su) for su, co in zip(succ, coh)
                         if su is not None and co is not None and co >= 1.5]
                if pts15:
                    co_, su_ = max(pts15, key=lambda r: r[1])
                    ax.plot(co_, su_, marker="*", markersize=22, color=color,
                            markeredgecolor="black", markeredgewidth=1.2,
                            zorder=15, label=None)

        # Reference lines
        ax.axhline(ANCHOR_15, color="blue", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(0.62, ANCHOR_15 + 0.02, f"T-SAE k=20 anchor = {ANCHOR_15}",
                fontsize=8, color="blue")
        ax.axhline(ANCHOR_15 + 0.27, color="green", linestyle=":",
                   linewidth=0.8, alpha=0.6)
        ax.text(0.62, ANCHOR_15 + 0.27 + 0.02,
                f"WIN threshold = {ANCHOR_15 + 0.27:.2f}",
                fontsize=8, color="green")
        ax.axvline(1.5, color="grey", linestyle=":", linewidth=0.8)
        ax.text(1.51, 0.02, "coh=1.5", fontsize=8, color="grey")

        proto_title = {
            "right-edge": "Right-edge (V1)",
            "tiled-broadcast": "Tiled-broadcast (V7)",
        }[proto_filter]
        ax.set_xlabel("mean coherence")
        ax.set_ylabel("mean success")
        ax.set_title(f"{proto_title}")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.55, 3.1)
        ax.set_ylim(-0.05, 2.0)

    fig.suptitle(
        "T=10 deadzone-escape chain (W) — single seed, k_pos=20  |  "
        "subseq H8 + Gaussian-mixture sampler (orange) clears deadzone @ Δ = +0.234 RE",
        fontsize=11, y=0.99,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, str(args.out))
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
