"""Per-variant coh/align frontier panels.

One subplot per (arch × training-step × hookpoint) combination — every
distinct training setup we have a Wang bundle k=30 frontier for. Same x/y
axes across all panels for direct visual comparison of where each variant's
peak lands in the (coherence, alignment) plane.

    uv run python -m experiments.em_features.plot_hookpoint_frontier_panels \\
        --root docs/dmitry/results/em_features \\
        --out  docs/dmitry/results/em_features/hookpoint_compare/txc_vs_tsae_frontier_panels.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_curve(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("rows", []):
        a = r.get("mean_alignment", r.get("mean_align"))
        c = r.get("mean_coherence", r.get("mean_coh"))
        if a is None or c is None: continue
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(c, float) and math.isnan(c)): continue
        rows.append({"alpha": float(r["alpha"]), "align": float(a), "coh": float(c)})
    rows.sort(key=lambda r: r["alpha"])
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha_clip", type=float, default=15.0)
    p.add_argument("--ncols", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root

    # Each panel: (title, json path, color, family-tag-for-grouping-color)
    variants = [
        ("SAE arditi 100k @ resid_post",
         root / "wang/sae_bundle30_frontier.json", "navy"),
        ("Han 100k @ resid_post",
         root / "wang/han_bundle30_frontier.json", "darkred"),
        ("T-SAE 30k @ resid_post",
         root / "wang/tsae_30k_bundle30_frontier.json", "darkorange"),
        ("T-SAE 100k @ resid_post",
         root / "wang/tsae_100k_bundle30_frontier.json", "chocolate"),
        ("T-SAE 30k @ resid_mid",
         root / "hookpoint_compare/tsae_residmid_30k_bundle30_frontier.json", "orange"),
        ("T-SAE 30k @ ln1_normalized",
         root / "hookpoint_compare/tsae_ln1_30k_bundle30_frontier.json", "gold"),
        ("TXC brickenauxk 30k @ resid_mid",
         root / "hookpoint_compare/txc_residmid_30k/results/wang_txc_residmid_step30000_bundle30_frontier.json",
         "darkgreen"),
        ("TXC brickenauxk 30k @ ln1_normalized",
         root / "hookpoint_compare/txc_ln1_30k/results/wang_txc_ln1_step30000_bundle30_frontier.json",
         "limegreen"),
    ]

    # First pass: load + global axis bounds
    loaded = []
    all_coh, all_align = [], []
    for title, p, color in variants:
        rows = load_curve(p)
        if rows and args.alpha_clip is not None:
            rows = [r for r in rows if abs(r["alpha"]) <= args.alpha_clip]
        loaded.append({"title": title, "rows": rows, "color": color, "path": p})
        if rows:
            all_coh += [r["coh"] for r in rows]
            all_align += [r["align"] for r in rows]

    if not all_coh:
        raise SystemExit("no data found")
    xpad = (max(all_coh) - min(all_coh)) * 0.07
    ypad = (max(all_align) - min(all_align)) * 0.07
    xlim = (min(all_coh) - xpad, max(all_coh) + xpad)
    ylim = (min(all_align) - ypad, max(all_align) + ypad)

    n = len(loaded)
    ncols = args.ncols
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(vmin=-args.alpha_clip, vmax=args.alpha_clip)

    for ax, v in zip(axes, loaded):
        rows = v["rows"]
        if not rows:
            ax.set_title(f"{v['title']}\n(NO DATA)", fontsize=10, color="grey")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=0.3)
            continue
        alphas = np.array([r["alpha"] for r in rows])
        coh = np.array([r["coh"] for r in rows])
        align = np.array([r["align"] for r in rows])

        ax.plot(coh, align, "-", color=v["color"], alpha=0.4, zorder=1)
        ax.scatter(coh, align, c=alphas, cmap=cmap, norm=norm,
                   s=70, edgecolor=v["color"], linewidth=1.0, zorder=2)
        peak_i = int(np.argmax(align))
        ax.scatter([coh[peak_i]], [align[peak_i]], s=240,
                   facecolors="none", edgecolors=v["color"], linewidths=2.0, zorder=3)
        ax.annotate(f"α={alphas[peak_i]:+.1f}\nalign={align[peak_i]:.2f}\ncoh={coh[peak_i]:.2f}",
                    (coh[peak_i], align[peak_i]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8.5, color=v["color"], fontweight="bold")
        zi = int(np.argmin(np.abs(alphas)))
        ax.scatter([coh[zi]], [align[zi]], marker="*", s=160, c="black", zorder=4,
                   label=f"α=0  align={align[zi]:.1f}")

        ax.set_title(v["title"], fontsize=11, color=v["color"])
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.legend(loc="lower right", fontsize=8)

    # blank the unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # column / row labels
    for j in range(ncols):
        try: axes[(nrows - 1) * ncols + j].set_xlabel("mean coherence")
        except IndexError: pass
    for i in range(nrows):
        axes[i * ncols].set_ylabel("mean alignment")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[:n].tolist(), location="right", shrink=0.85, pad=0.02)
    cbar.set_label("steering coefficient α")

    fig.suptitle(
        f"Wang bundle k=30 — coherence/alignment frontier per (arch × step × hookpoint), |α|≤{args.alpha_clip:g}\n"
        f"Qwen-7B PEFT-LoRA EM organism, layer 15. Black ★ = α=0 baseline.",
        fontsize=12, y=0.997)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
