"""Frontier plot for the Qwen-14B finance pivot.

One subpanel per (arch, hookpoint, organism). α-colored scatter, no connecting
lines (per user preference). Peak circled and annotated. Black ★ = α=0.
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


def load_finalist(path: Path, feature_id: int):
    if not path.exists(): return None
    d = json.loads(path.read_text())
    finalists = d.get("finalists", [d]) if "finalists" in d else [d]
    for f in finalists:
        if f.get("feature_id") == feature_id:
            return [{"alpha": float(r["alpha"]),
                     "align": r.get("mean_align"),
                     "coh": r.get("mean_coh"),
                     "n_align": r.get("n_align")}
                    for r in f["rows"]
                    if r.get("mean_align") is not None]
    return None


def load_partial_finalist(path: Path):
    if not path.exists(): return None, None
    d = json.loads(path.read_text())
    fid = d.get("feature_id")
    rows = [{"alpha": float(r["alpha"]),
             "align": r.get("mean_align"),
             "coh": r.get("mean_coh"),
             "n_align": r.get("n_align")}
            for r in d.get("rows", [])
            if r.get("mean_align") is not None]
    return fid, rows


def load_bundle(path: Path):
    if not path.exists(): return None
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("rows", []):
        a = r.get("mean_alignment", r.get("mean_align"))
        c = r.get("mean_coherence", r.get("mean_coh"))
        if a is None or c is None: continue
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(c, float) and math.isnan(c)): continue
        rows.append({"alpha": float(r["alpha"]), "align": float(a), "coh": float(c)})
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha_clip", type=float, default=15.0)
    return p.parse_args()


def main():
    args = parse_args()
    R = args.root
    H = R / "hookpoint_compare"
    EN = R / "em_nanda" / "results" / "em_nanda_sae_arditi_step10000_wang"

    panels = []

    # Two finalists from em_nanda Qwen-14B
    fid1, rows1 = load_partial_finalist(EN / "stage4_finalist_11086.partial.json")
    if rows1:
        panels.append(("Qwen-14B finance / SAE arditi 10k @ L24 / feat 11086\n(em-nanda CHAMPION)", rows1, "darkred", "navy"))
    fid2, rows2 = load_partial_finalist(EN / "stage4_finalist_17837.partial.json")
    if rows2:
        panels.append(("Qwen-14B finance / SAE arditi 10k @ L24 / feat 17837", rows2, "indianred", "navy"))

    # Prior champions for head-to-head (Qwen-7B medical)
    txc_champion = load_finalist(
        H / "txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000/stage4_final_frontier.json",
        4563,
    )
    if txc_champion:
        panels.append(("Qwen-7B medical / TXC paper k=100 30k @ L15 / feat 4563\n(prior single-feat champion)",
                       txc_champion, "darkgreen", "navy"))

    sae_arditi_bundle = load_bundle(R / "wang/sae_bundle30_frontier.json")
    if sae_arditi_bundle:
        panels.append(("Qwen-7B medical / SAE arditi 100k bundle k=30 @ L15\n(prior bundle champion)",
                       sae_arditi_bundle, "navy", "navy"))

    # Apply alpha clip
    panels_clipped = []
    all_coh, all_align = [], []
    for title, rows, color, _ in panels:
        rows = [r for r in rows if abs(r["alpha"]) <= args.alpha_clip]
        if not rows: continue
        all_coh += [r["coh"] for r in rows]
        all_align += [r["align"] for r in rows]
        panels_clipped.append((title, rows, color))

    xpad = (max(all_coh) - min(all_coh)) * 0.07
    ypad = (max(all_align) - min(all_align)) * 0.07
    xlim = (min(all_coh) - xpad, max(all_coh) + xpad)
    ylim = (min(all_align) - ypad, max(all_align) + ypad)

    n = len(panels_clipped)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    cmap = plt.cm.coolwarm_r
    norm = TwoSlopeNorm(vmin=-args.alpha_clip, vcenter=0, vmax=args.alpha_clip)

    for ax, (title, rows, color) in zip(axes, panels_clipped):
        alphas = np.array([r["alpha"] for r in rows])
        coh = np.array([r["coh"] for r in rows])
        align = np.array([r["align"] for r in rows])
        ax.scatter(coh, align, c=alphas, cmap=cmap, norm=norm,
                   s=160, edgecolor=color, linewidth=1.5, zorder=2)
        peak_i = int(np.argmax(align))
        ax.scatter([coh[peak_i]], [align[peak_i]], facecolors="none",
                   edgecolors=color, linewidths=3, s=400, zorder=3)
        ax.annotate(f"α={alphas[peak_i]:+.1f}\nalign={align[peak_i]:.2f}\ncoh={coh[peak_i]:.2f}",
                    (coh[peak_i], align[peak_i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=11, color=color, fontweight="bold")
        zi = int(np.argmin(np.abs(alphas)))
        ax.scatter([coh[zi]], [align[zi]], marker="*", s=240, c="black", zorder=4,
                   label=f"α=0  align={align[zi]:.1f}")
        ax.set_title(title, fontsize=11, color=color)
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.legend(loc="lower right", fontsize=9)

    for ax in axes[n:]:
        ax.set_visible(False)

    for j in range(ncols):
        axes[(nrows - 1) * ncols + j].set_xlabel("mean coherence", fontsize=11)
    for i in range(nrows):
        axes[i * ncols].set_ylabel("mean alignment", fontsize=11)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[:n].tolist(), location="right", shrink=0.85, pad=0.02)
    cbar.set_label("steering coefficient α")

    fig.suptitle(
        f"Wang frontier: Qwen-14B finance (em-nanda) vs Qwen-7B medical priors  |α|≤{args.alpha_clip:g}\n"
        f"em-nanda: SAE arditi 10k @ resid_post L24, single-feat steering on bad-finance organism",
        fontsize=13, y=0.998)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    import matplotlib.image as mpimg
    img = mpimg.imread(args.out)
    h_, w_ = img.shape[:2]
    target_w = 320
    fig2, ax2 = plt.subplots(figsize=(target_w/48, h_*target_w/w_/48), dpi=48)
    ax2.imshow(img); ax2.axis("off")
    fig2.savefig(args.out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight", pad_inches=0)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
