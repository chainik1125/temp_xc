"""TXC vs T-SAE hookpoint comparison: bundle k=30 frontier scatter.

Reads the 30k Wang-bundle frontier JSONs for T-SAE and TXC at resid_post /
resid_mid / ln1_normalized, plus the SAE arditi 100k @ resid_post reference,
and produces a single scatter (coh vs align, α-colored) plus a summary bar.

    uv run python -m experiments.em_features.plot_hookpoint_comparison \\
        --root docs/dmitry/results/em_features \\
        --out  docs/dmitry/results/em_features/hookpoint_compare/txc_vs_tsae_hookpoint_comparison.png
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


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("rows", []):
        a = r.get("mean_alignment", r.get("mean_align"))
        c = r.get("mean_coherence", r.get("mean_coh"))
        if a is None or c is None:
            continue
        if (isinstance(a, float) and math.isnan(a)) or (isinstance(c, float) and math.isnan(c)):
            continue
        rows.append({"alpha": float(r["alpha"]), "align": float(a), "coh": float(c)})
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root

    panels = [
        # (label, path, panel-color for legend dot)
        ("SAE arditi 100k @ resid_post (champion)",
         root / "wang" / "sae_bundle30_frontier.json", "navy"),
        ("T-SAE 30k @ resid_post",
         root / "wang" / "tsae_30k_bundle30_frontier.json", "darkorange"),
        ("T-SAE 30k @ resid_mid",
         root / "hookpoint_compare" / "tsae_residmid_30k_bundle30_frontier.json", "orange"),
        ("T-SAE 30k @ ln1_normalized",
         root / "hookpoint_compare" / "tsae_ln1_30k_bundle30_frontier.json", "gold"),
        ("TXC brickenauxk 30k @ resid_mid",
         root / "hookpoint_compare" / "txc_residmid_30k" / "results"
              / "wang_txc_residmid_step30000_bundle30_frontier.json", "darkgreen"),
        ("TXC brickenauxk 30k @ ln1_normalized",
         root / "hookpoint_compare" / "txc_ln1_30k" / "results"
              / "wang_txc_ln1_step30000_bundle30_frontier.json", "limegreen"),
    ]

    data = []
    for label, p, color in panels:
        rows = load_rows(p)
        peak_i = int(np.argmax([r["align"] for r in rows])) if rows else None
        zero_row = next((r for r in rows if abs(r["alpha"]) < 1e-6), None)
        data.append({"label": label, "rows": rows, "color": color,
                     "peak": rows[peak_i] if peak_i is not None else None,
                     "zero": zero_row,
                     "missing": len(rows) == 0,
                     "path": p})

    # Summary print
    print("=" * 76)
    print(f"{'variant':50s}  peak α   align    coh   Δalign(vs α=0)")
    print("=" * 76)
    for d in data:
        if d["missing"]:
            print(f"{d['label']:50s}  -- MISSING --  ({d['path']})")
            continue
        peak = d["peak"]; zero = d["zero"]
        delta = (peak["align"] - zero["align"]) if zero else float("nan")
        print(f"{d['label']:50s}  {peak['alpha']:+6.2f}  {peak['align']:6.2f}  "
              f"{peak['coh']:6.2f}  {delta:+6.2f}")
    print("=" * 76)

    # ---- Figure: 2x3 grid of coh-vs-align scatter, α-colored
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    cmap = plt.cm.RdBu_r

    for ax, d in zip(axes, data):
        if d["missing"]:
            ax.set_title(f"{d['label']}\n(NO DATA)", fontsize=9, color="grey")
            continue
        rows = d["rows"]
        alphas = np.array([r["alpha"] for r in rows])
        aligns = np.array([r["align"] for r in rows])
        cohs = np.array([r["coh"] for r in rows])
        sc = ax.scatter(cohs, aligns, c=alphas, cmap=cmap, norm=norm,
                        s=70, edgecolors="k", linewidths=0.5)
        peak_i = int(np.argmax(aligns))
        ax.scatter([cohs[peak_i]], [aligns[peak_i]], s=240,
                   facecolors="none", edgecolors="darkgreen", linewidths=2.0)
        ax.annotate(f"α={alphas[peak_i]:+.2f}\nalign={aligns[peak_i]:.2f}\ncoh={cohs[peak_i]:.2f}",
                    (cohs[peak_i], aligns[peak_i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=8, color="darkgreen", fontweight="bold")
        zero_i = int(np.argmin(np.abs(alphas)))
        ax.scatter([cohs[zero_i]], [aligns[zero_i]], marker="*", s=220,
                   c="black", zorder=5, label=f"α=0 (align={aligns[zero_i]:.1f})")
        ax.set_title(d["label"], fontsize=10, color=d["color"])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Wang bundle k=30: TXC brickenauxk vs T-SAE across hookpoints (Qwen-7B PEFT-LoRA EM organism, layer 15, 30k steps)",
        fontsize=12, y=0.995)
    for ax in axes[3:]:
        ax.set_xlabel("Mean coherence")
    for i in (0, 3):
        axes[i].set_ylabel("Mean alignment")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("α (intervention strength)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"wrote {args.out.with_suffix('.thumb.png')}")

    # ---- Bar summary
    fig2, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(13, 5))
    labels, peaks, deltas, colors = [], [], [], []
    for d in data:
        if d["missing"]:
            continue
        peak = d["peak"]; zero = d["zero"]
        labels.append(d["label"].replace(" 30k", "").replace(" 100k", ""))
        peaks.append(peak["align"])
        deltas.append((peak["align"] - zero["align"]) if zero else 0.0)
        colors.append(d["color"])
    x = np.arange(len(labels))
    ax_a.bar(x, peaks, color=colors)
    for i, v in enumerate(peaks):
        ax_a.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=9)
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_a.set_ylabel("Peak alignment (Wang bundle k=30)")
    ax_a.set_title("Peak alignment by arch × hookpoint")
    ax_a.grid(alpha=0.3, axis="y")

    ax_d.bar(x, deltas, color=colors)
    for i, v in enumerate(deltas):
        ax_d.text(i, v + 0.15, f"{v:+.2f}", ha="center", fontsize=9)
    ax_d.set_xticks(x); ax_d.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_d.set_ylabel("Δ alignment (peak − α=0)")
    ax_d.set_title("Causal lift from steering")
    ax_d.grid(alpha=0.3, axis="y")

    fig2.suptitle("Hookpoint sweep summary: TXC vs T-SAE (30k) + SAE arditi 100k reference",
                  fontsize=12)
    bar_out = args.out.with_name(args.out.stem + "_bars.png")
    fig2.tight_layout()
    fig2.savefig(bar_out, dpi=150, bbox_inches="tight")
    fig2.savefig(bar_out.with_suffix(".thumb.png"), dpi=48, bbox_inches="tight")
    print(f"wrote {bar_out}")


if __name__ == "__main__":
    main()
