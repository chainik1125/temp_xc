"""Pareto frontier — T-SAE k=20 + the 3 most Pareto-dominant TXC architectures.

"Most dominant" = highest count of T-SAE strength points (succ_t, coh_t)
that the TXC arch dominates with at least one of its own (succ_x, coh_x)
strengths satisfying succ_x >= succ_t AND coh_x >= coh_t.

T-SAE k=20 has 7 strength points. None of the 63 TXC cells in the Y+W
inventory dominate all 7. The ceiling is 5/7 (T=3 H8 shifts=(T,) RE,
single seed). Tied at 4/7 are several n=3 multi-seed cells. We pick the
top 3 by (dom desc, peak15 desc):

  1. T=3 H8 shifts=(T,) RE                    — 5/7  n=1  peak15 0.93
  2. Galaxy 18 SoftMaxPool T=3 PP             — 4/7  n=3  peak15 1.36
  3. Galaxy 8 SoftMaxPool T=2 V7              — 4/7  n=3  peak15 1.33

T-SAE's signature point (succ=1.68, coh=1.36) is dominated by 0/63 cells
— it sits below the prereg coh-floor of 1.5, so it's excluded from the
cliff15 metric, but no TXC arch matches it on the raw Pareto frontier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.phase7_unification.case_studies.steering.plot_unified_pareto import (
    INVENTORY, get_curve_avg, ANCHOR_15,
)
from src.plotting.save_figure import save_figure


# (arch_id, protocol, label, color, marker, dom_score)
TOP3 = [
    ("txc_h8_t3_kpos20_shifts3", "right-edge",
     "T=3 H8 shifts=(T,) RE",                "#d62728", "o", "5/7"),
    ("txc_softmaxpool_t3_kpos20", "per-position",
     "T=3 Galaxy 18 SoftMaxPool PP",         "#2ca02c", "^", "4/7"),
    ("txc_softmaxpool_t2_kpos20", "tiled-broadcast",
     "T=2 Galaxy 8 SoftMaxPool V7",          "#9467bd", "s", "4/7"),
]

ANCHOR_KEY = ("tsae_paper_k20", "right-edge", "T-SAE k=20", "#1f77b4", "D", "—")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path,
        default=Path(
            "/workspace/temp_xc/experiments/phase7_unification/results/case_studies/plots/unified_pareto_top3_vs_tsae.png"
        ),
    )
    args = p.parse_args()

    # Load curves keyed by (arch_id, protocol)
    inventory_by_arch = {a: (label, color, subdirs) for a, label, color, subdirs in INVENTORY}
    series = []
    for arch_id, proto, label, color, marker, dom in [ANCHOR_KEY, *TOP3]:
        if arch_id not in inventory_by_arch:
            print(f"WARN: {arch_id} missing from INVENTORY")
            continue
        _, _, subdirs = inventory_by_arch[arch_id]
        per_proto = get_curve_avg(arch_id, subdirs)
        if proto not in per_proto:
            print(f"WARN: {arch_id} has no {proto} data")
            continue
        s, succ, coh, n = per_proto[proto]
        series.append((arch_id, proto, label, color, marker, dom, n, s, succ, coh))

    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot each curve
    for arch_id, proto, label, color, marker, dom, n, s, succ, coh in series:
        is_anchor = arch_id == "tsae_paper_k20"
        lw = 3.0 if is_anchor else 2.0
        alpha = 1.0 if is_anchor else 0.85
        ls = "--" if is_anchor else "-"
        zorder = 12 if is_anchor else 6
        ms = 10 if is_anchor else 7
        # Compute peak15 for the legend
        pts15 = [(co, su) for su, co in zip(succ, coh)
                 if su is not None and co is not None and co >= 1.5]
        peak15 = max(pts15, key=lambda r: r[1])[1] if pts15 else None
        peak_unc = max(s_ for s_, c_ in zip(succ, coh)
                       if s_ is not None and c_ is not None)
        p15s = f"{peak15:.2f}" if peak15 is not None else "—"
        legend = (
            f"{label} (n={n}, dom {dom}, peak15={p15s}, peak_unc={peak_unc:.2f})"
        )
        ax.plot(coh, succ, marker=marker, markersize=ms, color=color,
                linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder,
                label=legend)
        # Star the cliff15 point for non-anchor
        if not is_anchor and pts15:
            co_, su_ = max(pts15, key=lambda r: r[1])
            ax.plot(co_, su_, marker="*", markersize=22, color=color,
                    markeredgecolor="black", markeredgewidth=1.2,
                    zorder=20)

    # Highlight T-SAE's undominated peak (succ=1.68, coh=1.36)
    ax.annotate(
        "T-SAE's UNDOMINATED peak\n(succ=1.68, coh=1.36)\n0 of 63 TXC cells dominate this",
        xy=(1.36, 1.68),
        xytext=(1.85, 1.85),
        fontsize=9, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="#e6f0fa", ec="#1f77b4", alpha=0.9),
        zorder=30,
    )
    ax.plot(1.36, 1.68, marker="o", markersize=14, markerfacecolor="none",
            markeredgecolor="#1f77b4", markeredgewidth=2.5, zorder=25)

    # Reference lines
    ax.axhline(ANCHOR_15, color="blue", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(0.62, ANCHOR_15 + 0.02,
            f"T-SAE cliff15 = {ANCHOR_15} (peak succ at coh ≥ 1.5)",
            fontsize=8, color="blue")
    ax.axhline(ANCHOR_15 + 0.27, color="green", linestyle=":",
               linewidth=0.8, alpha=0.6)
    ax.text(0.62, ANCHOR_15 + 0.27 + 0.02,
            f"prereg WIN threshold = {ANCHOR_15 + 0.27:.2f}",
            fontsize=8, color="green")
    ax.axvline(1.5, color="grey", linestyle=":", linewidth=0.8)
    ax.text(1.51, 0.04, "coh=1.5 (prereg floor)", fontsize=8, color="grey")

    ax.set_xlabel("mean coherence", fontsize=11)
    ax.set_ylabel("mean success", fontsize=11)
    ax.set_title(
        "T-SAE k=20 vs the 3 most-Pareto-dominant TXC architectures\n"
        "(dom = # of T-SAE's 7 strength points the TXC arch (weakly) dominates)",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 3.1)
    ax.set_ylim(-0.05, 2.0)

    plt.tight_layout()
    save_figure(fig, str(args.out))
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
