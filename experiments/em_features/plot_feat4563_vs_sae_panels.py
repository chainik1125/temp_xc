"""TXC paper-faithful k=100 single-feat 4563 vs SAE arditi 100k bundle k=30
vs T-SAE paper-faithful 30k bundle k=30 — one subplot per (arch, hookpoint)
with shared axes.

    uv run python -m experiments.em_features.plot_feat4563_vs_sae_panels \\
        --root  docs/dmitry/results/em_features \\
        --out   docs/dmitry/results/em_features/hookpoint_compare/feat4563_vs_sae_panels.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out",  type=Path, required=True)
    p.add_argument("--alpha_clip", type=float, default=15.0)
    return p.parse_args()


def load_finalist(path: Path, feature_id: int):
    d = json.loads(path.read_text())
    for f in d["finalists"]:
        if f["feature_id"] == feature_id:
            return [{"alpha": float(r["alpha"]),
                     "align": float(r["mean_align"]),
                     "coh": float(r["mean_coh"])} for r in f["rows"]]
    return None


def load_bundle(path: Path):
    d = json.loads(path.read_text())
    rows = []
    for r in d["rows"]:
        a = r.get("mean_alignment", r.get("mean_align"))
        c = r.get("mean_coherence", r.get("mean_coh"))
        if a is None or c is None: continue
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(c, float) and math.isnan(c)): continue
        rows.append({"alpha": float(r["alpha"]), "align": float(a), "coh": float(c)})
    return rows


def main():
    args = parse_args()
    R = args.root
    H = R / "hookpoint_compare"

    panels = [
        ("TXC paper-faithful k=100 @ resid_post\nsingle feat 4563  (k_bundle=1!)",
         load_finalist(H / "txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000/stage4_final_frontier.json", 4563),
         "darkgreen", "limegreen", "o"),
        ("SAE arditi 100k @ resid_post\nbundle k=30 (prior champion)",
         load_bundle(R / "wang/sae_bundle30_frontier.json"),
         "navy", "skyblue", "s"),
        ("T-SAE paper-faithful 30k @ resid_post\nbundle k=30",
         load_bundle(H / "tsae_paper_30k/results/wang_tsae_paper_k20_d16k_step30000_bundle30_frontier.json"),
         "saddlebrown", "wheat", "D"),
    ]

    # alpha clip + global axes
    panels = [(t, [r for r in rows if abs(r["alpha"]) <= args.alpha_clip], e, f, m)
              for t, rows, e, f, m in panels if rows is not None]
    all_coh, all_align = [], []
    for _, rows, *_ in panels:
        all_coh += [r["coh"] for r in rows]
        all_align += [r["align"] for r in rows]
    xpad = (max(all_coh) - min(all_coh)) * 0.07
    ypad = (max(all_align) - min(all_align)) * 0.07
    xlim = (min(all_coh) - xpad, max(all_coh) + xpad)
    ylim = (min(all_align) - ypad, max(all_align) + ypad)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
    cmap = plt.cm.coolwarm_r
    norm = TwoSlopeNorm(vmin=-args.alpha_clip, vcenter=0, vmax=args.alpha_clip)

    for ax, (title, rows, edge, fill, marker) in zip(axes, panels):
        rows = sorted(rows, key=lambda r: r["alpha"])
        alphas = np.array([r["alpha"] for r in rows])
        coh    = np.array([r["coh"]   for r in rows])
        align  = np.array([r["align"] for r in rows])
        ax.scatter(coh, align, c=alphas, cmap=cmap, norm=norm,
                   marker=marker, s=130, edgecolor=edge, linewidth=1.4, zorder=2)
        peak_i = int(np.argmax(align))
        ax.scatter([coh[peak_i]], [align[peak_i]], facecolors="none",
                   edgecolors=edge, linewidths=2.5, s=320, zorder=3)
        ax.annotate(f"α={alphas[peak_i]:+.1f}\nalign={align[peak_i]:.2f}\ncoh={coh[peak_i]:.2f}",
                    (coh[peak_i], align[peak_i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, color=edge, fontweight="bold")
        zi = int(np.argmin(np.abs(alphas)))
        ax.scatter([coh[zi]], [align[zi]], marker="*", s=200, c="black", zorder=4,
                   label=f"α=0  align={align[zi]:.1f}")
        ax.set_title(title, fontsize=11, color=edge)
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlabel("mean coherence")

    axes[0].set_ylabel("mean alignment")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("steering coefficient α")

    fig.suptitle(
        f"Per-(arch, hookpoint) Wang frontier: TXC single feat 4563 vs SAE arditi 100k bundle vs T-SAE paper-faithful bundle\n"
        f"Qwen-7B PEFT-LoRA EM organism, layer 15 resid_post. Black ★ = α=0 baseline. |α|≤{args.alpha_clip:g}.",
        fontsize=12, y=0.995)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    import matplotlib.image as mpimg
    img = mpimg.imread(args.out)
    h, w = img.shape[:2]
    target_w = 288
    fig2, ax2 = plt.subplots(figsize=(target_w/48, h*target_w/w/48), dpi=48)
    ax2.imshow(img); ax2.axis("off")
    fig2.savefig(args.out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight", pad_inches=0)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
