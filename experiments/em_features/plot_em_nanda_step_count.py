"""Step-count line plot: peak align as a function of training steps.

Two architectures (SAE arditi T=1, TXC paper k=100), three step counts each
({5k, 10k, 30k}). Two markers per (arch, steps): edge-α peak (α=−10) and
mid-α peak (best |α|≤8, conventionally α=−6 for SAE arditi and α∈[−2,−1.5]
for TXC). Per the 15:00 UTC convention, this small step-count plot is exempt
from the "no connecting lines" frontier-plot policy — connect 3 dots per
(arch, peak-type) line so the trajectory is legible.

Reads stage-4 frontier JSONs from
docs/dmitry/results/em_features/data/em_nanda_<arch>_<steps>_stage4.json.

    python -m experiments.em_features.plot_em_nanda_step_count \\
        --out docs/dmitry/results/em_features/plots/em_nanda_step_count_trajectory.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = [
    ("sae_arditi", "SAE arditi T=1",  "tab:blue"),
    ("txc_paper_k100", "TXC paper k=100", "tab:orange"),
]
STEPS = [5_000, 10_000, 30_000]
DATA_TAGS = {5_000: "5k", 10_000: "10k", 30_000: "30k"}


def best_per_finalist(rows, alpha_pred):
    """For each finalist, pick best row matching alpha_pred(row), then take max
    across finalists. Returns (peak_align, peak_coh, feat_id, alpha)."""
    best = None
    for f in rows:
        cands = [r for r in f["rows"]
                 if r.get("mean_align") is not None
                 and r.get("mean_coh") is not None
                 and alpha_pred(r["alpha"])]
        if not cands:
            continue
        # Filter out finalists whose α=0 baseline is anomalously high (>=88) —
        # those are sign-symmetric artifacts not clean directional re-alignment
        # (per synthesis convention).
        a0 = next((r for r in f["rows"] if r["alpha"] == 0.0
                   and r.get("mean_align") is not None), None)
        if a0 is not None and a0["mean_align"] >= 88.0:
            continue
        peak_row = max(cands, key=lambda r: r["mean_align"])
        candidate = (peak_row["mean_align"], peak_row["mean_coh"],
                     f["feature_id"], peak_row["alpha"])
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path,
                   default=Path("docs/dmitry/results/em_features/data"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.4))

    for arch_key, arch_label, color in CASES:
        edge_y, mid_y = [], []
        edge_meta, mid_meta = [], []
        for s in STEPS:
            tag = DATA_TAGS[s]
            path = args.data_dir / f"em_nanda_{arch_key}_{tag}_stage4.json"
            if not path.exists():
                edge_y.append(None); mid_y.append(None)
                edge_meta.append(None); mid_meta.append(None)
                continue
            d = json.loads(path.read_text())
            finalists = d["finalists"]
            edge = best_per_finalist(finalists, lambda a: a == -10.0)
            mid = best_per_finalist(finalists, lambda a: abs(a) <= 8.0)
            edge_y.append(edge[0] if edge else None)
            mid_y.append(mid[0] if mid else None)
            edge_meta.append(edge); mid_meta.append(mid)

        ax.plot(STEPS, edge_y, marker="o", linestyle="-", color=color,
                label=f"{arch_label}: edge α=−10",
                markersize=9, linewidth=1.6)
        ax.plot(STEPS, mid_y, marker="s", linestyle="--", color=color,
                label=f"{arch_label}: mid α (|α|≤8 best)",
                markersize=8, linewidth=1.4, alpha=0.85)

        # Annotate each point with its peak feat id
        for s, e, m in zip(STEPS, edge_meta, mid_meta):
            if e is not None:
                ax.annotate(f"f{e[2]}", (s, e[0]),
                            textcoords="offset points", xytext=(7, 6),
                            fontsize=7, color=color, alpha=0.9)
            if m is not None and (e is None or m[2] != e[2]):
                ax.annotate(f"f{m[2]} α={m[3]:+g}", (s, m[0]),
                            textcoords="offset points", xytext=(7, -11),
                            fontsize=7, color=color, alpha=0.7)

    # Reference: prior Qwen-7B medical champion
    ax.axhline(58.47, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(30000, 58.47 + 0.8, "Qwen-7B medical champion (58.47)",
            fontsize=8, color="gray", ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(STEPS)
    ax.set_xticklabels(["5k", "10k", "30k"])
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Single-feat peak align (Wang stage-4 8 rollouts/cell)")
    ax.set_title("EM Nanda — step-count trajectory on Qwen-14B + R1 finance\n"
                 "(stage-4 resolved peaks; 5k = cheapest winning recipe)")
    ax.set_ylim(55, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
