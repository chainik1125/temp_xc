"""Master comparison plot for the paper-faithful overnight session.

Single panel: peak alignment per variant, with α=0 baseline shown as the start
of an arrow pointing to the peak. Shows all paper-faithful TXC k variants plus
T-SAE paper-faithful + windowed-T-SAE T=2 + champion references.

    uv run python -m experiments.em_features.plot_paper_faithful_comparison \\
        --root  docs/dmitry/results/em_features \\
        --out   docs/dmitry/results/em_features/hookpoint_compare/paper_faithful_overnight.png
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


def peak(d: dict) -> tuple[float, float, float, float]:
    """(peak_alpha, peak_align, peak_coh, alpha0_align) for one frontier file."""
    rs = d["rows"]
    p_ = max(rs, key=lambda r: r.get("mean_alignment", -1))
    a0 = next((r for r in rs if abs(r["alpha"]) < 1e-6), None)
    return (
        float(p_["alpha"]),
        float(p_["mean_alignment"]),
        float(p_["mean_coherence"]),
        float(a0["mean_alignment"]) if a0 else float("nan"),
    )


def main():
    args = parse_args()
    root = args.root
    H = root / "hookpoint_compare"

    # (label, frontier path, color)
    variants = [
        ("SAE arditi 100k bundle k=30 (prior champion)",
         root / "wang/sae_bundle30_frontier.json", "navy"),
        ("Han 100k bundle k=30",
         root / "wang/han_bundle30_frontier.json", "darkred"),
        ("T-SAE paper-faithful 30k bundle k=30",
         H / "tsae_paper_30k/results/wang_tsae_paper_k20_d16k_step30000_bundle30_frontier.json", "saddlebrown"),
        ("TXC paper-faithful k=50 bundle k=30",
         H / "txc_paper_k50_30k/results/wang_txc_paper_k50bt_d16k_step30000_bundle30_frontier.json", "lightgreen"),
        ("TXC paper-faithful k=100 bundle k=30",
         H / "txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000_bundle30_frontier.json", "darkgreen"),
        ("TXC paper-faithful k=200 bundle k=30",
         H / "txc_paper_k200_30k/results/wang_txc_paper_k200bt_d16k_step30000_bundle30_frontier.json", "forestgreen"),
        ("TXC paper-faithful k=100 bundle k=5 (dilution-corrected)",
         H / "txc_paper_k100_30k/results/wang_txc_paper_k100bt_d16k_step30000_bundle5_frontier.json", "limegreen"),
        ("TXC paper k=100 single feat 4563 (manual stage 4 peak)",
         None, "yellowgreen"),  # special — hard-coded peak
        ("WindowedTSAE T=2 30k bundle k=30",
         H / "wtsae_T2_30k/results/wang_wtsae_T2_30000step_bundle30_frontier.json", "purple"),
    ]

    rows = []
    for label, path, color in variants:
        if path is None:
            # special case: manual single-feature peak from stage 4
            rows.append({"label": label, "peak_alpha": -8.0, "peak_align": 58.47,
                         "peak_coh": 30.86, "a0": 47.56, "color": color})
            continue
        if not path.exists():
            print(f"missing: {path}"); continue
        d = json.loads(path.read_text())
        a, peak_align, peak_coh, a0_align = peak(d)
        rows.append({"label": label, "peak_alpha": a, "peak_align": peak_align,
                     "peak_coh": peak_coh, "a0": a0_align, "color": color})

    fig, ax = plt.subplots(figsize=(13, 7))

    y = np.arange(len(rows))[::-1]   # top-down order
    for i, r in enumerate(rows):
        yi = y[i]
        # arrow from α=0 to peak
        if not np.isnan(r["a0"]):
            ax.annotate(
                "",
                xy=(r["peak_align"], yi),
                xytext=(r["a0"], yi),
                arrowprops=dict(arrowstyle="->", color=r["color"], lw=2),
            )
            ax.scatter([r["a0"]], [yi], color="black", marker="*", s=80,
                       zorder=5)
        ax.scatter([r["peak_align"]], [yi], color=r["color"], s=160,
                   edgecolor="k", zorder=5)
        ax.text(r["peak_align"] + 0.4, yi,
                f"  align={r['peak_align']:.2f} (α={r['peak_alpha']:+.1f}, coh={r['peak_coh']:.1f})",
                fontsize=9, va="center", color=r["color"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_xlabel("Mean alignment (Wang procedure peak)")
    ax.set_xlim(35, 70)
    ax.axvline(57.42, linestyle="--", color="navy", alpha=0.4)
    ax.text(57.42, len(rows), "  SAE arditi 100k peak", fontsize=8,
            color="navy", va="bottom")
    ax.set_title("Paper-faithful overnight comparison — Wang peak alignment per variant\n"
                 "Qwen-7B PEFT-LoRA EM organism, layer 15, 30k steps. Black ★ = α=0 baseline.")
    ax.grid(alpha=0.3, axis="x")

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
