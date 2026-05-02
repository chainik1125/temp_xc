"""Coherence/suppression frontier plot for the em_nanda Qwen-14B SAE arditi 10k
Wang stage 4 result.

One subpanel per finalist feature. Each panel: scatter of (mean coherence, mean
alignment) across α, colored by α (diverging). Peak circled. Black star marks
α=0. No connecting lines (α is a nominal index across the discrete sweep grid,
not a smooth path).

    python -m experiments.em_features.plot_em_nanda_sae_arditi_frontier \\
        --in docs/dmitry/results/em_features/data/em_nanda_sae_arditi_10k_stage4.json \\
        --out docs/dmitry/results/em_features/plots/em_nanda_sae_arditi_10k_frontier.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha_clip", type=float, default=15.0,
                   help="Drop |α| > this (suppresses α=±100 outliers from the visual; "
                        "they collapse to incoherence and squash the colorbar).")
    p.add_argument("--zoom", action="store_true",
                   help="Zoom axes to 60–105 instead of 0–105 (the action lives in the top-right).")
    return p.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.inp.read_text())
    finalists = data["finalists"]
    meta = data["meta"]

    n = len(finalists)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 5.6), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    # Outliers (|α| > clip) reported as a per-panel side note since they're
    # the diagnostic α=±100 collapse points (incoherent / refusal-y).
    side_notes = []

    for ax, f in zip(axes, finalists):
        all_rows = [r for r in f["rows"]
                    if r.get("mean_align") is not None and r.get("mean_coh") is not None]
        kept = [r for r in all_rows if abs(r["alpha"]) <= args.alpha_clip]
        dropped = [r for r in all_rows if abs(r["alpha"]) > args.alpha_clip]

        alphas = np.array([r["alpha"] for r in kept])
        aligns = np.array([r["mean_align"] for r in kept])
        cohs = np.array([r["mean_coh"] for r in kept])

        amax = max(abs(alphas.min()), abs(alphas.max())) or 1.0
        norm = TwoSlopeNorm(vmin=-amax, vcenter=0.0, vmax=amax)
        sc = ax.scatter(cohs, aligns, c=alphas, cmap="RdBu_r", norm=norm,
                        s=140, edgecolor="k", linewidth=0.7, zorder=3)

        # Peak: max alignment among coherent points (Turner uses coh > 50)
        coherent_mask = cohs >= 50
        score = np.where(coherent_mask, aligns, -1e6)
        peak_idx = int(np.argmax(score))
        ax.scatter(cohs[peak_idx], aligns[peak_idx], s=520,
                   facecolor="none", edgecolor="limegreen", linewidth=2.6, zorder=4,
                   label=f"peak α={alphas[peak_idx]:+g}\n  align={aligns[peak_idx]:.1f}\n  coh={cohs[peak_idx]:.1f}")

        # α=0 baseline as black star
        zero_idx = np.where(alphas == 0.0)[0]
        if len(zero_idx):
            zi = int(zero_idx[0])
            ax.scatter(cohs[zi], aligns[zi], marker="*", s=320,
                       facecolor="black", edgecolor="white", linewidth=0.8, zorder=5,
                       label=f"α=0  align={aligns[zi]:.1f} coh={cohs[zi]:.1f}")

        # Light EM-region shading (Turner def: align<30 AND coh>50)
        ax.axhspan(0, 30, xmin=0.5, xmax=1.0, color="red", alpha=0.06, zorder=0)
        ax.axhline(30, color="red", linestyle=":", alpha=0.3, linewidth=1, zorder=0)

        ax.set_xlabel("mean coherence")
        ax.set_title(f"feat {f['feature_id']}   Δz̄={f['delta_z']:.3f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        if args.zoom:
            ax.set_xlim(60, 105)
            ax.set_ylim(60, 105)
        else:
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 105)
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
        plt.colorbar(sc, ax=ax, label="steering α", pad=0.02)

        if dropped:
            d_str = ", ".join(f"α={r['alpha']:+g}: align={r['mean_align']:.0f}/coh={r['mean_coh']:.0f}"
                              for r in dropped)
            side_notes.append(f"feat {f['feature_id']} (off-plot): {d_str}")

    axes[0].set_ylabel("mean alignment")

    title = (f"em_nanda Qwen-14B finance — SAE arditi 10k @ resid_post L24\n"
             f"Wang stage 4 coherence/suppression frontier  "
             f"({meta.get('n_rollouts', '?')} rollouts × 8 prompts; "
             f"upper-right = aligned + coherent; |α| clipped to {args.alpha_clip:g})")
    fig.suptitle(title, y=1.02, fontsize=12)

    if side_notes:
        fig.text(0.5, -0.02, "Off-plot α extremes: " + " | ".join(side_notes),
                 ha="center", fontsize=8.5, style="italic", color="dimgray")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
