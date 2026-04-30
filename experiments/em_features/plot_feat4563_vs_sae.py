"""TXC paper-faithful k=100 single-feature 4563 vs SAE arditi 100k bundle k=30:
coh / align frontier Pareto comparison.

Shows that TXC's single feature 4563 is comparable to / better than the prior
champion SAE arditi 100k bundle on the alignment axis, traded slightly on coh.

    uv run python -m experiments.em_features.plot_feat4563_vs_sae \\
        --root  docs/dmitry/results/em_features \\
        --out   docs/dmitry/results/em_features/hookpoint_compare/feat4563_vs_sae.png
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
    p.add_argument("--alpha_clip", type=float, default=15.0,
                   help="Hide |α|>clip points (the -100/+100 outliers compress the plot)")
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

    feat4563 = load_finalist(
        args.root / "hookpoint_compare/txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000/stage4_final_frontier.json",
        4563,
    )
    sae_arditi = load_bundle(args.root / "wang/sae_bundle30_frontier.json")
    tsae_paper = load_bundle(args.root / "hookpoint_compare/tsae_paper_30k/results/wang_tsae_paper_k20_d16k_step30000_bundle30_frontier.json")

    if args.alpha_clip is not None:
        feat4563 = [r for r in feat4563 if abs(r["alpha"]) <= args.alpha_clip]
        sae_arditi = [r for r in sae_arditi if abs(r["alpha"]) <= args.alpha_clip]
        tsae_paper = [r for r in tsae_paper if abs(r["alpha"]) <= args.alpha_clip]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.coolwarm_r
    norm = TwoSlopeNorm(vmin=-args.alpha_clip, vcenter=0, vmax=args.alpha_clip)

    for rows, label, marker, color, edge in [
        (feat4563, "TXC paper k=100 single-feat 4563  (k_bundle=1!)", "o", "limegreen", "darkgreen"),
        (sae_arditi, "SAE arditi 100k bundle k=30 (prior champion)", "s", "skyblue", "navy"),
        (tsae_paper, "T-SAE paper-faithful 30k bundle k=30", "D", "wheat", "saddlebrown"),
    ]:
        rows = sorted(rows, key=lambda r: r["alpha"])
        alphas = np.array([r["alpha"] for r in rows])
        coh    = np.array([r["coh"]   for r in rows])
        align  = np.array([r["align"] for r in rows])
        ax.scatter(coh, align, c=alphas, cmap=cmap, norm=norm,
                   marker=marker, s=130, edgecolor=edge, linewidth=1.4, zorder=2,
                   label=label)
        # Peak annotation
        peak_i = int(np.argmax(align))
        ax.scatter([coh[peak_i]], [align[peak_i]], facecolors="none",
                   edgecolors=edge, linewidths=2.5, s=320, zorder=3)
        ax.annotate(f"α={alphas[peak_i]:+.1f}\nalign={align[peak_i]:.2f}\ncoh={coh[peak_i]:.2f}",
                    (coh[peak_i], align[peak_i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=10, color=edge, fontweight="bold")
        # α=0 marker
        zi = int(np.argmin(np.abs(alphas)))
        ax.scatter([coh[zi]], [align[zi]], marker="*", s=200, c="black", zorder=4)

    ax.set_xlabel("Mean coherence (Wang OpenAI judge)")
    ax.set_ylabel("Mean alignment (Wang OpenAI judge)")
    ax.set_title("TXC single-feat 4563 vs SAE arditi 100k bundle vs T-SAE paper-faithful bundle\n"
                 "Qwen-7B PEFT-LoRA EM organism, layer 15, 30k–100k steps. Black ★ = α=0 baseline.")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label("steering coefficient α")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")

    import matplotlib.image as mpimg
    img = mpimg.imread(args.out)
    h, w = img.shape[:2]
    target_w = 288
    fig2, ax2 = plt.subplots(figsize=(target_w/48, h*target_w/w/48), dpi=48)
    ax2.imshow(img); ax2.axis("off")
    fig2.savefig(args.out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight", pad_inches=0)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
