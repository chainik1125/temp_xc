"""Multi-seed peak success plot — sd42 vs sd1.

Bar chart with seeds side-by-side per arch. Highlights variance across
seeds for each arch.

Output: results/case_studies/plots/phase7_steering_v2_multiseed.png
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


ARCHS = [
    ("tsae_paper_k20",       "T-SAE\n(T=1, k=20)",      "#d62728"),
    ("tsae_paper_k500",      "T-SAE\n(T=1, k=500)",     "#ff7f0e"),
    ("topk_sae",             "TopKSAE\n(T=1, k=500)",   "#1f77b4"),
    ("agentic_txc_02",       "TXC matryoshka\n(T=5)",   "#2ca02c"),
    ("phase5b_subseq_h8",    "SubseqH8\n(T=10)",        "#8c564b"),
]


def _peak(arch_id, subdir):
    p = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
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
    means = {s: (np.mean([p[0] for p in pairs]), np.mean([p[1] for p in pairs])) for s, pairs in by_s.items()}
    return max(means.values(), key=lambda v: v[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_v2_multiseed.png")
    args = ap.parse_args()
    banner(__file__)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # ─── Panel A: Q1.3 normalised paper-clamp
    x = np.arange(len(ARCHS))
    bar_w = 0.35

    sd42_vals = []
    sd1_vals = []
    means = []
    for arch_id, label, color in ARCHS:
        p42 = _peak(arch_id, "steering_paper_normalised")
        p1 = _peak(arch_id, "steering_paper_normalised_seed1")
        sd42_vals.append(p42[0] if p42 else 0)
        sd1_vals.append(p1[0] if p1 else 0)
        means.append(((p42[0] if p42 else 0) + (p1[0] if p1 else 0)) / 2)

    bars1 = ax.bar(x - bar_w/2, sd42_vals, bar_w, label="seed=42", alpha=0.85, color="#1f77b4")
    bars2 = ax.bar(x + bar_w/2, sd1_vals, bar_w, label="seed=1", alpha=0.85, color="#ff7f0e")

    # ─── Panel B: Q2.C per-position write (3 window archs)
    Q2C_ARCHS = [
        ("agentic_txc_02",       "TXC matryoshka\n(T=5)",   "#2ca02c"),
        ("phase5b_subseq_h8",    "SubseqH8\n(T=10)",        "#8c564b"),
    ]
    x2 = np.arange(len(Q2C_ARCHS))
    sd42_q2 = []
    sd1_q2 = []
    for arch_id, label, color in Q2C_ARCHS:
        p42 = _peak(arch_id, "steering_paper_window_perposition")
        p1 = _peak(arch_id, "steering_paper_window_perposition_seed1")
        sd42_q2.append(p42[0] if p42 else 0)
        sd1_q2.append(p1[0] if p1 else 0)
    ax2.bar(x2 - bar_w/2, sd42_q2, bar_w, label="seed=42", alpha=0.85, color="#1f77b4")
    ax2.bar(x2 + bar_w/2, sd1_q2, bar_w, label="seed=1", alpha=0.85, color="#ff7f0e")
    for xi, (s42, s1) in enumerate(zip(sd42_q2, sd1_q2)):
        ax2.text(xi - bar_w/2, s42 + 0.03, f"{s42:.2f}", ha="center", fontsize=9)
        ax2.text(xi + bar_w/2, s1 + 0.03, f"{s1:.2f}", ha="center", fontsize=9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([label for _, label, _ in Q2C_ARCHS])
    ax2.set_title("Q2.C per-position window-clamp (multi-seed)", fontsize=11)
    ax2.set_ylim(0, 2.2)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.axhline(1.38, color="#d62728", linestyle="--", alpha=0.6,
                label="T-SAE k=500 mean")

    # Error bars showing the seed variance
    for i, (s42, s1, m) in enumerate(zip(sd42_vals, sd1_vals, means)):
        ax.plot([i - bar_w/2, i + bar_w/2], [s42, s1], "k--", alpha=0.3)

    # Annotate with peak values
    for b, v in zip(bars1, sd42_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03, f"{v:.2f}",
                ha="center", fontsize=9)
    for b, v in zip(bars2, sd1_vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03, f"{v:.2f}",
                ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in ARCHS])
    ax.set_ylabel("Peak success (Sonnet 4.6 grade, 0-3)")
    ax.set_title(
        "Q1.3 normalised paper-clamp (multi-seed)",
        fontsize=11,
    )
    ax.set_ylim(0, 2.2)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=10)
    fig.suptitle(
        "Multi-seed validation: T-SAE k=20 robust at 1.80; "
        "TXC Q2.C mean peak (1.37) TIES T-SAE k=500 mean (1.38) at matched sparsity",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(args.out))
    plt.close(fig)
    print(f"  saved {args.out}")

    print("\n## Multi-seed peak success comparison")
    print()
    print("| arch | seed=42 | seed=1 | mean |")
    print("|---|---|---|---|")
    for (arch_id, label, _), s42, s1, m in zip(ARCHS, sd42_vals, sd1_vals, means):
        print(f"| `{arch_id}` | {s42:.2f} | {s1:.2f} | {m:.2f} |")


if __name__ == "__main__":
    main()
