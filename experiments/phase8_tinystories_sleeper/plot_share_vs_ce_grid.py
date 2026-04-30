"""3 (arch) × 5 (hookpoint) grid of sleeper-share × CE-ratio scatters.

Each subplot shows all individual sample points (per prompt × per
matched-RNG seed) for that (arch, hookpoint) cell, color-coded by
training seed (s0/s1/s2). The cell's overall mean is drawn with
black error bars.

Reads outputs/seeded_logs/recovery.json (which stores
sleeper_shares_steered and recoveries as per-sample lists).
Writes outputs/seeded_logs/share_vs_ce_grid.png + .thumb.png.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
JSON_PATH = ROOT / "outputs/seeded_logs/recovery.json"
OUT_PNG = ROOT / "outputs/seeded_logs/share_vs_ce_grid.png"


HOOK_ORDER = [
    ("l0_ln1",  "ln1.0"),
    ("l0_pre",  "resid_pre.0"),
    ("l0_mid",  "resid_mid.0"),
    ("l0_post", "resid_post.0"),
    ("l1_ln1",  "ln1.1"),
]
ARCH_ORDER = ["sae", "tsae", "txc"]
ARCH_LABELS = {"sae": "SAE", "tsae": "T-SAE", "txc": "TXC"}
SEED_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}

# Equalised axis limits — sample CE ratios go up to ~5 in the worst
# coherence-collapse cells.
X_LIM = (-0.05, 1.1)
Y_LIM = (-0.2, 5.5)


def parse_tag(tag: str) -> tuple[str, str, int]:
    m = re.match(r"^(?P<a>sae|tsae|txc)_(?P<h>.+)_s(?P<s>\d+)$", tag)
    return m.group("a"), m.group("h"), int(m.group("s"))


def main() -> None:
    data = json.loads(JSON_PATH.read_text())
    cells = data["cells"]

    # Group by (arch, hook) → {seed: cell-dict}.
    grouped: dict[tuple[str, str], dict[int, dict]] = {}
    for tag, info in cells.items():
        a, h, s = parse_tag(tag)
        grouped.setdefault((a, h), {})[s] = info

    n_arch, n_hook = len(ARCH_ORDER), len(HOOK_ORDER)
    fig, axes = plt.subplots(
        n_arch, n_hook,
        figsize=(2.7 * n_hook, 2.5 * n_arch),
        sharex=True, sharey=True,
    )

    for r, arch in enumerate(ARCH_ORDER):
        for c, (hookkey, hooklabel) in enumerate(HOOK_ORDER):
            ax = axes[r, c]
            seed_map = grouped.get((arch, hookkey), {})

            # Per-sample dots (64 per seed × up-to-3 seeds = up to 192).
            cell_xs, cell_ys = [], []
            for seed in (0, 1, 2):
                d = seed_map.get(seed)
                if d is None:
                    continue
                shares = d.get("sleeper_shares_steered", [])
                recoveries = d.get("recoveries", [])
                if not shares or not recoveries:
                    continue
                # CE ratio per sample = 1 - recovery
                ce_ratios = [1.0 - r for r in recoveries]
                color = SEED_COLORS[seed]
                ax.scatter(shares, ce_ratios, color=color, s=12, alpha=0.45,
                           edgecolor="none", label=f"s{seed}")
                cell_xs.extend(shares); cell_ys.extend(ce_ratios)
                # Per-seed mean as a star.
                ax.scatter(
                    [statistics.mean(shares)], [statistics.mean(ce_ratios)],
                    marker="*", s=160, color=color, edgecolor="black",
                    linewidths=0.6, zorder=10,
                )

            # Cell-level mean with error bars across all samples.
            if cell_xs and cell_ys:
                mx = statistics.mean(cell_xs); my = statistics.mean(cell_ys)
                sx = statistics.stdev(cell_xs) if len(cell_xs) > 1 else 0.0
                sy = statistics.stdev(cell_ys) if len(cell_ys) > 1 else 0.0
                ax.errorbar(mx, my, xerr=sx, yerr=sy,
                            fmt="o", color="black", markersize=6,
                            capsize=3, alpha=0.7, zorder=11)

            ax.axhline(1.0, linestyle="--", color="grey", alpha=0.4, linewidth=0.7)
            ax.axvline(1.0, linestyle=":", color="grey", alpha=0.3, linewidth=0.7)
            ax.axhspan(-0.2, 1.0, xmin=0, xmax=0.2 / 1.15, alpha=0.06, color="green")

            if r == 0:
                ax.set_title(hooklabel, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{ARCH_LABELS[arch]}\nCE ratio", fontsize=10)
            if r == n_arch - 1:
                ax.set_xlabel('"IHY" token share', fontsize=9)
            ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM)
            ax.grid(alpha=0.2)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=SEED_COLORS[s],
                   markersize=8, label=f"seed {s}")
        for s in (0, 1, 2)
    ]
    handles.append(plt.Line2D([0], [0], marker="*", linestyle="",
                              color="grey", markeredgecolor="black",
                              markersize=12, label="per-seed mean"))
    handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                              color="black", markersize=6,
                              label="cell mean ± std"))
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=5,
               bbox_to_anchor=(0.99, 1.03))

    fig.suptitle(
        "Phase 8 — graded suppression × coherence per (arch, hookpoint, seed)\n"
        "rows = arch, cols = hookpoint; lower-left (green) = ideal; "
        "y = CE_steered / CE_pois (>1 = collapse)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PNG.with_suffix(".thumb.png"), dpi=24, bbox_inches="tight")
    print(f"[plot] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
