"""Bundle-size sweep plot: bundle k_bundle vs peak align, fixed arch ckpt.

Shows how the Wang bundle peak align changes as a function of bundle size
k_bundle ∈ {1, 2, 3, 5, 10, 30}. Used to test the "bundle dilution" hypothesis
that summing many ~orthogonal feature directions hurts the steering peak.

    uv run python -m experiments.em_features.plot_bundle_size_sweep \\
        --root  docs/dmitry/results/em_features \\
        --out   docs/dmitry/results/em_features/hookpoint_compare/bundle_size_sweep.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out",  type=Path, required=True)
    return p.parse_args()


def peak_for_ckpt(prefix: Path, k_bundle: int) -> tuple[float, float, float] | None:
    """Return (alpha, peak_align, peak_coh) for a bundle frontier file."""
    p = prefix.parent / f"{prefix.name}_bundle{k_bundle}_frontier.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rs = d.get("rows", [])
    if not rs: return None
    p_ = max(rs, key=lambda r: r.get("mean_alignment", -1))
    return float(p_["alpha"]), float(p_["mean_alignment"]), float(p_["mean_coherence"])


def main():
    args = parse_args()

    # Discover variants by their bundle JSON family. Each entry:
    #   (label, prefix, color, marker)
    variants = [
        ("TXC paper k=100 (resid_post 30k)",
         args.root / "hookpoint_compare/txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000",
         "darkgreen", "o"),
        ("T-SAE paper-faithful 30k (resid_post)",
         args.root / "hookpoint_compare/tsae_paper_30k/results/wang_tsae_paper_k20_d16k_step30000",
         "saddlebrown", "s"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    bundle_sizes = [1, 2, 3, 5, 10, 30]
    for label, prefix, color, marker in variants:
        xs, ys = [], []
        for kb in bundle_sizes:
            r = peak_for_ckpt(prefix, kb)
            if r is None: continue
            alpha, align, coh = r
            xs.append(kb); ys.append(align)
            ax.annotate(f"α={alpha:+.1f}", (kb, align),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color)
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=label,
                    markersize=10, linewidth=2)

    # Reference horizontal lines
    for ref_label, val, c, ls in [
        ("SAE arditi 100k bundle k=30 (champion)", 57.42, "navy", "--"),
        ("T-SAE paper-faithful bundle k=30",        56.23, "saddlebrown", ":"),
        ("TXC paper k=100 single-feat 4563",        58.47, "limegreen", "-."),
    ]:
        ax.axhline(val, linestyle=ls, color=c, alpha=0.6, linewidth=1.2)
        ax.text(31, val + 0.1, ref_label, fontsize=8, color=c, va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_xticks(bundle_sizes)
    ax.set_xticklabels([str(k) for k in bundle_sizes])
    ax.set_xlabel("Bundle size k_bundle (sum of top-k_bundle decoder rows)")
    ax.set_ylabel("Peak Wang bundle alignment (best α)")
    ax.set_title("Bundle-size sweep: Wang peak alignment vs k_bundle\n"
                 "Qwen-7B PEFT-LoRA EM organism, layer 15 resid_post, 30k steps")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")

    import matplotlib.image as mpimg
    img = mpimg.imread(args.out)
    h, w = img.shape[:2]
    target_w = 288
    fig2, ax2 = plt.subplots(figsize=(target_w/48, h*target_w/w/48), dpi=48)
    ax2.imshow(img); ax2.axis("off")
    fig2.savefig(args.out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight", pad_inches=0)
    print(f"wrote {args.out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
