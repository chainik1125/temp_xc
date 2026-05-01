"""Overnight session: per-(arch, hookpoint, recipe) panel layout, shared axes.

Each subplot is one variant's full coh/align frontier (α-colored).

    uv run python -m experiments.em_features.plot_overnight_panels \\
        --root  docs/dmitry/results/em_features \\
        --out   docs/dmitry/results/em_features/hookpoint_compare/overnight_panels.png
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha_clip", type=float, default=15.0)
    p.add_argument("--ncols", type=int, default=4)
    return p.parse_args()


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
    rows.sort(key=lambda r: r["alpha"])
    return rows


def load_finalist(path: Path, feature_id: int):
    if not path.exists(): return None
    d = json.loads(path.read_text())
    for f in d["finalists"]:
        if f["feature_id"] == feature_id:
            return [{"alpha": float(r["alpha"]),
                     "align": float(r["mean_align"]),
                     "coh": float(r["mean_coh"])} for r in f["rows"]]
    return None


def main():
    args = parse_args()
    root = args.root
    H = root / "hookpoint_compare"

    # (subplot title, rows-or-None, color, kind)
    PANELS = [
        # ROW 1: prior champions
        ("SAE arditi 100k @ resid_post (champion)",
         load_bundle(root / "wang/sae_bundle30_frontier.json"), "navy"),
        ("Han 100k @ resid_post",
         load_bundle(root / "wang/han_bundle30_frontier.json"), "darkred"),
        ("T-SAE 100k @ resid_post (ours: k=128, α=1.0)",
         load_bundle(root / "wang/tsae_100k_bundle30_frontier.json"), "chocolate"),
        ("T-SAE paper-faithful 30k @ resid_post (k=20 BatchTopK, α=0.1)",
         load_bundle(H / "tsae_paper_30k/results/wang_tsae_paper_k20_d16k_step30000_bundle30_frontier.json"),
         "saddlebrown"),
        # ROW 2: TXC paper-faithful k-sweep
        ("TXC paper-faithful 30k k=20 @ resid_post",
         load_bundle(H / "txc_paper_k20_30k/results/wang_txc_paper_k20bt_d16k_step30000_bundle30_frontier.json"),
         "yellowgreen"),
        ("TXC paper-faithful 30k k=50 @ resid_post",
         load_bundle(H / "txc_paper_k50_30k/results/wang_txc_paper_k50bt_d16k_step30000_bundle30_frontier.json"),
         "lightgreen"),
        ("TXC paper-faithful 30k k=100 @ resid_post  (bundle k=30)",
         load_bundle(H / "txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000_bundle30_frontier.json"),
         "darkgreen"),
        ("TXC paper-faithful 30k k=200 @ resid_post",
         load_bundle(H / "txc_paper_k200_30k/results/wang_txc_paper_k200bt_d16k_step30000_bundle30_frontier.json"),
         "forestgreen"),
        ("TXC paper k=100 SINGLE FEAT 4563  (k_bundle=1!) @ resid_post",
         load_finalist(H / "txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000/stage4_final_frontier.json", 4563),
         "limegreen"),
        # ROW 3: TXC ours + windowed_tsae variants
        ("TXC ours 30k @ resid_mid (k=128, no batch_topk)",
         load_bundle(H / "txc_residmid_30k/results/wang_txc_residmid_step30000_bundle30_frontier.json"),
         "teal"),
        ("TXC ours 30k @ ln1_normalized (k=128, no batch_topk)",
         load_bundle(H / "txc_ln1_30k/results/wang_txc_ln1_step30000_bundle30_frontier.json"),
         "darkcyan"),
        ("WindowedTSAE T=2 30k (paper recipe, M=I)",
         load_bundle(H / "wtsae_T2_30k/results/wang_wtsae_T2_30000step_bundle30_frontier.json"),
         "purple"),
        ("WindowedTSAE T=2 + mix_positions 30k",
         load_bundle(H / "wtsae_T2_mix_30k/results/wang_wtsae_T2_mix_30000step_bundle30_frontier.json"),
         "mediumpurple"),
        ("WindowedTSAE T=2 + mix + matryoshka 20% 30k  (BUNDLE)",
         load_bundle(H / "wtsae_T2_mix_matr_30000step/wang_wtsae_T2_mix_matr_30000step_bundle30_frontier.json"),
         "magenta"),
        ("WindowedTSAE T=2 + mix + matr 20%  SINGLE FEAT 14496",
         load_finalist(H / "wtsae_T2_mix_matr_30000step/wang_wtsae_T2_mix_matr_30000step/stage4_final_frontier.json", 14496),
         "deeppink"),
        ("WindowedTSAE T=2 vanilla (no contrastive) d=32k k=128 + mix 30k  (BUNDLE)",
         load_bundle(H / "wtsae_T2_vanilla_d32k_k128_mix/wang_wtsae_T2_vanilla_d32k_k128_mix_30000step_bundle30_frontier.json"),
         "slategray"),
        ("TXC paper k=100 60k extension @ resid_post  (BUNDLE — judge-NaN-heavy)",
         load_bundle(H / "txc_paper_k100bt_d16k_60k_step60000/results/wang_txc_paper_k100bt_d16k_60k_step60000_bundle30_frontier.json"),
         "olivedrab"),
        ("T-SAE 30k @ resid_post (ours: k=128, α=1.0)",
         load_bundle(root / "wang/tsae_30k_bundle30_frontier.json"), "darkorange"),
    ]

    # Drop missing panels
    PANELS = [(t, r, c) for t, r, c in PANELS if r is not None and len(r) > 0]
    print(f"loaded {len(PANELS)} panels")

    # alpha clip + global axes
    all_coh, all_align = [], []
    panels = []
    for title, rows, color in PANELS:
        rows = [r for r in rows if abs(r["alpha"]) <= args.alpha_clip]
        all_coh += [r["coh"] for r in rows]
        all_align += [r["align"] for r in rows]
        panels.append((title, rows, color))
    xpad = (max(all_coh) - min(all_coh)) * 0.07
    ypad = (max(all_align) - min(all_align)) * 0.07
    xlim = (min(all_coh) - xpad, max(all_coh) + xpad)
    ylim = (min(all_align) - ypad, max(all_align) + ypad)

    n = len(panels)
    ncols = args.ncols
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(vmin=-args.alpha_clip, vmax=args.alpha_clip)

    for ax, (title, rows, color) in zip(axes, panels):
        alphas = np.array([r["alpha"] for r in rows])
        coh = np.array([r["coh"] for r in rows])
        align = np.array([r["align"] for r in rows])
        ax.scatter(coh, align, c=alphas, cmap=cmap, norm=norm, s=70,
                   edgecolor=color, linewidth=1.0, zorder=2)
        peak_i = int(np.argmax(align))
        ax.scatter([coh[peak_i]], [align[peak_i]], facecolors="none",
                   edgecolors=color, linewidths=2.5, s=240, zorder=3)
        ax.annotate(f"α={alphas[peak_i]:+.1f}\n{align[peak_i]:.2f} / {coh[peak_i]:.1f}",
                    (coh[peak_i], align[peak_i]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=8.5, color=color, fontweight="bold")
        zi = int(np.argmin(np.abs(alphas)))
        ax.scatter([coh[zi]], [align[zi]], marker="*", s=140, c="black", zorder=4,
                   label=f"α=0  align={align[zi]:.1f}")
        ax.set_title(title, fontsize=9.5, color=color)
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.legend(loc="lower right", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    for j in range(ncols):
        try: axes[(nrows - 1) * ncols + j].set_xlabel("mean coherence")
        except IndexError: pass
    for i in range(nrows):
        axes[i * ncols].set_ylabel("mean alignment")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[:n].tolist(), location="right", shrink=0.85, pad=0.02)
    cbar.set_label("steering coefficient α")

    fig.suptitle(
        f"Overnight session — Wang frontier per (arch, hookpoint, recipe), |α|≤{args.alpha_clip:g}\n"
        f"Qwen-7B PEFT-LoRA EM organism, layer 15. Black ★ = α=0 baseline.",
        fontsize=12, y=0.997)
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
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
