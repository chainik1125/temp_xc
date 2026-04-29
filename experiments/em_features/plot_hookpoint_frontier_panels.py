"""Per-arch coh/align frontier panels.

Each subplot is one SAE-side architecture; curves within a panel are the
hookpoints we have for that arch. Same x/y axes across panels for visual
comparison of where each arch's bundle peak lands in the (coherence, alignment)
plane.

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
    p.add_argument("--alpha_clip", type=float, default=15.0,
                   help="Drop points with |α| > this so extreme α don't compress the plot")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root

    panels = [
        ("SAE arditi (k=128, T=1)", [
            ("100k @ resid_post", root / "wang/sae_bundle30_frontier.json", "navy", "o"),
        ]),
        ("T-SAE (k=128, per-token + adjacent contrastive)", [
            ("30k @ resid_post", root / "wang/tsae_30k_bundle30_frontier.json", "darkorange", "s"),
            ("30k @ resid_mid",  root / "hookpoint_compare/tsae_residmid_30k_bundle30_frontier.json", "orange", "^"),
            ("30k @ ln1_normalized", root / "hookpoint_compare/tsae_ln1_30k_bundle30_frontier.json", "gold", "D"),
        ]),
        ("TXC brickenauxk (k_total=128, T=5 windowed)", [
            ("30k @ resid_mid",  root / "hookpoint_compare/txc_residmid_30k/results/wang_txc_residmid_step30000_bundle30_frontier.json", "darkgreen", "^"),
            ("30k @ ln1_normalized", root / "hookpoint_compare/txc_ln1_30k/results/wang_txc_ln1_step30000_bundle30_frontier.json", "limegreen", "D"),
        ]),
    ]

    # First pass: load + collect global axis bounds
    loaded = []
    all_coh, all_align = [], []
    for arch_label, curves in panels:
        loaded_curves = []
        for label, p, color, marker in curves:
            rows = load_curve(p)
            if not rows:
                print(f"  MISSING: {label} ({p})")
                loaded_curves.append((label, None, color, marker))
                continue
            if args.alpha_clip is not None:
                rows = [r for r in rows if abs(r["alpha"]) <= args.alpha_clip]
            loaded_curves.append((label, rows, color, marker))
            all_coh += [r["coh"] for r in rows]
            all_align += [r["align"] for r in rows]
        loaded.append((arch_label, loaded_curves))

    xpad = (max(all_coh) - min(all_coh)) * 0.05
    ypad = (max(all_align) - min(all_align)) * 0.05
    xlim = (min(all_coh) - xpad, max(all_coh) + xpad)
    ylim = (min(all_align) - ypad, max(all_align) + ypad)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(vmin=-args.alpha_clip, vmax=args.alpha_clip)

    for ax, (arch_label, curves) in zip(axes, loaded):
        for label, rows, color, marker in curves:
            if rows is None:
                ax.plot([], [], marker, color=color, label=f"{label} (missing)")
                continue
            alphas = np.array([r["alpha"] for r in rows])
            coh = np.array([r["coh"] for r in rows])
            align = np.array([r["align"] for r in rows])

            # connect curve in α order
            ax.plot(coh, align, "-", color=color, alpha=0.4, zorder=1)
            # α-colored markers (color encodes intervention strength)
            ax.scatter(coh, align, c=alphas, cmap=cmap, norm=norm,
                       marker=marker, s=70, edgecolor=color, linewidth=1.0, zorder=2)
            # peak annotation
            peak_i = int(np.argmax(align))
            ax.scatter([coh[peak_i]], [align[peak_i]], s=240,
                       facecolors="none", edgecolors=color, linewidths=2.0, zorder=3)
            ax.annotate(f"α={alphas[peak_i]:+.1f}\nalign={align[peak_i]:.2f}",
                        (coh[peak_i], align[peak_i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color=color, fontweight="bold")
            # α=0 black star
            zi = int(np.argmin(np.abs(alphas)))
            ax.scatter([coh[zi]], [align[zi]], marker="*", s=140,
                       c="black", zorder=4)
            # legend handle in panel color
            ax.plot([], [], marker, color=color, markersize=8,
                    markeredgecolor="k", linestyle="-", linewidth=1.5, label=label)

        ax.set_title(arch_label, fontsize=11)
        ax.set_xlabel("mean coherence")
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("mean alignment (Wang bundle k=30)")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("steering coefficient α")

    fig.suptitle(f"Coherence/alignment frontier — Wang bundle k=30, |α|≤{args.alpha_clip:g}\n"
                 f"Qwen-7B PEFT-LoRA EM organism, layer 15. Black ★ = α=0 baseline.",
                 fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    # thumbnail
    img_h, img_w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
    fig.savefig(args.out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
