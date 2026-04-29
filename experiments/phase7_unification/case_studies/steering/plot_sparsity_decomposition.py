"""Sparsity decomposition plot: T-SAE k=20 vs k=500 vs window archs.

Visualises the key Q1.3 finding — at MATCHED effective sparsity (k≈500),
the cross-family steering gap shrinks to ~0.27. T-SAE k=20's lead is
dominantly about its sparsity choice, not about per-token vs window
architecture.

Two-panel bar chart:
  Panel A: peak success bars, sorted by k_eff (or families coloured)
  Panel B: same but with annotation showing the k=20 vs k=500 split

Reads from `results/case_studies/steering_paper_normalised/<arch>/grades.jsonl`.

Output: results/case_studies/plots/phase7_steering_v2_sparsity_decomp.png
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from experiments.phase7_unification._paths import OUT_DIR, banner
from experiments.phase7_unification.case_studies._paths import CASE_STUDIES_DIR


# (arch, k_eff_per_position, family, display_label)
ARCH_INFO = [
    ("tsae_paper_k20",                              20,  "per-token", "T-SAE\n(T=1, k=20)"),
    ("agentic_txc_02",                             100,  "window",    "TXC matryoshka\n(T=5, k_pos=100)"),
    ("phase57_partB_h8_bare_multidistance_t5",     100,  "window",    "H8 multi-dist\n(T=5, k_pos=100)"),
    ("phase5b_subseq_h8",                           100, "window",    "SubseqH8\n(T=10, k_pos=100)"),
    ("topk_sae",                                   500,  "per-token", "TopKSAE\n(T=1, k=500)"),
    ("tsae_paper_k500",                            500,  "per-token", "T-SAE\n(T=1, k=500)"),
]
FAMILY_COLOR = {"per-token": "#d62728", "window": "#2ca02c", "mlc": "#9467bd"}


def _peak(arch_id):
    p = CASE_STUDIES_DIR / "steering_paper_normalised" / arch_id / "grades.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    by_s = collections.defaultdict(list)
    for r in rows:
        if r.get("success_grade") is None:
            continue
        by_s[float(r["strength"])].append((r["success_grade"], r["coherence_grade"]))
    if not by_s:
        return None
    means = [(s, np.mean([p[0] for p in pairs]), np.mean([p[1] for p in pairs])) for s, pairs in by_s.items()]
    return max(means, key=lambda t: t[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_v2_sparsity_decomp.png")
    args = ap.parse_args()
    banner(__file__)

    # Build data
    rows = []
    for arch, k, family, label in ARCH_INFO:
        peak = _peak(arch)
        if peak is None:
            continue
        rows.append({"arch": arch, "k": k, "family": family, "label": label,
                     "peak_suc": peak[1], "peak_coh": peak[2], "peak_s": peak[0]})

    # Sort by k_eff (descending sparsity = ascending k)
    rows.sort(key=lambda r: (r["k"], r["arch"]))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.arange(len(rows))
    suc = [r["peak_suc"] for r in rows]
    colors = [FAMILY_COLOR[r["family"]] for r in rows]
    bars = ax.bar(xs, suc, color=colors, alpha=0.85)
    for r, b in zip(rows, bars):
        # value label on top
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{r['peak_suc']:.2f}", ha="center", fontsize=10)
        # k label below
        ax.text(b.get_x() + b.get_width()/2, -0.15, f"k_eff={r['k']}",
                ha="center", fontsize=9, color="#555")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_ylabel("Peak success (mean Sonnet 4.6 grade, 0-3)")
    ax.set_title(
        "Sparsity decomposition: T-SAE k=20 leads, but at matched sparsity (k≈500),\n"
        "cross-family spread is 0.27 — within concept noise on 30 concepts.",
        fontsize=12,
    )
    ax.set_ylim(0, 2.2)
    ax.grid(axis="y", alpha=0.25)

    # Family legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=FAMILY_COLOR["per-token"], label="Per-token (T=1)"),
        Patch(color=FAMILY_COLOR["window"], label="Window (T≥5)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    # Vertical separator between k=20 and k≈500 groups
    if any(r["k"] == 20 for r in rows) and any(r["k"] > 20 for r in rows):
        # find boundary
        split_x = next(i for i, r in enumerate(rows) if r["k"] > 20) - 0.5
        ax.axvline(split_x, color="#888", linestyle="--", alpha=0.5)
        ax.text(split_x - 0.25, 2.05, "k=20", ha="right", fontsize=10, color="#444")
        ax.text(split_x + 0.25, 2.05, "k_eff ≈ 500", ha="left", fontsize=10, color="#444")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(args.out))
    plt.close(fig)
    print(f"  saved {args.out}")

    # Markdown table
    print("\n## Peak success per arch under normalised paper-clamp, grouped by k_eff")
    print()
    print("| arch | family | T | k_eff | peak success |")
    print("|---|---|---|---|---|")
    for r in rows:
        T = r["arch"].replace("_", " ")  # display hint only
        print(f"| `{r['arch']}` | {r['family']} | {('1' if r['family']=='per-token' else 'multi')} | {r['k']} | {r['peak_suc']:.2f} |")


if __name__ == "__main__":
    main()
