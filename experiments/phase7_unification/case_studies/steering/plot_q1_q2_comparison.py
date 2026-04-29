"""Q1.3 vs Q2.C comparison plot — per-strength curves under right-edge
vs per-position window-clamp.

For each window arch, overlays the s_norm curves under:
  Q1.3: paper-clamp normalised + right-edge attribution
  Q2.C: paper-clamp normalised + per-position write

Output:
  results/case_studies/plots/phase7_steering_v2_q1_vs_q2.png
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


WINDOW_ARCHS = [
    ("agentic_txc_02",                          "TXC matryoshka (T=5)"),
    ("phase5b_subseq_h8",                       "SubseqH8 (T=10)"),
    ("phase57_partB_h8_bare_multidistance_t5",  "H8 multi-dist (T=5)"),
]
ARCH_COLOR = {
    "agentic_txc_02": "#2ca02c",
    "phase5b_subseq_h8": "#8c564b",
    "phase57_partB_h8_bare_multidistance_t5": "#e377c2",
}


def _per_strength(arch_id, subdir):
    p_grades = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    p_gens = CASE_STUDIES_DIR / subdir / arch_id / "generations.jsonl"
    if not p_grades.exists():
        return {}
    s_to_norm = {}
    for line in p_gens.open():
        g = json.loads(line)
        s_to_norm[float(g["strength"])] = g.get("s_norm")
    by_s = collections.defaultdict(list)
    for line in p_grades.open():
        r = json.loads(line)
        if r.get("success_grade") is None:
            continue
        by_s[float(r["strength"])].append((r["success_grade"], r["coherence_grade"]))
    out = {}
    for s, pairs in by_s.items():
        sa = np.array([p[0] for p in pairs])
        ca = np.array([p[1] for p in pairs])
        sn = s_to_norm.get(s)
        out[s] = (float(sa.mean()), float(ca.mean()), sn)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_v2_q1_vs_q2.png")
    args = ap.parse_args()
    banner(__file__)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (arch_id, label) in zip(axes, WINDOW_ARCHS):
        q1 = _per_strength(arch_id, "steering_paper_normalised")
        q2 = _per_strength(arch_id, "steering_paper_window_perposition")
        if not q1 or not q2:
            ax.set_title(f"{label}\n(missing data)")
            continue
        # Sort by s_norm
        s1 = sorted(q1.items(), key=lambda kv: (kv[1][2] or kv[0]))
        s2 = sorted(q2.items(), key=lambda kv: (kv[1][2] or kv[0]))
        x1 = [v[2] for _, v in s1]
        y1 = [v[0] for _, v in s1]
        x2 = [v[2] for _, v in s2]
        y2 = [v[0] for _, v in s2]
        color = ARCH_COLOR.get(arch_id, "black")
        ax.plot(x1, y1, "o--", color=color, alpha=0.7, label="Q1.3 right-edge", linewidth=2, markersize=8)
        ax.plot(x2, y2, "o-", color=color, alpha=1.0, label="Q2.C per-position", linewidth=2.5, markersize=9)
        ax.set_xscale("log")
        ax.set_xlabel("s_norm = s_abs / <|z|>_arch")
        ax.set_title(label)
        ax.set_ylim(-0.05, 2.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Mean steering success (Sonnet 4.6, 0-3)")

    fig.suptitle("Q2.C: per-position window-clamp vs Q1.3 right-edge\n"
                 "TXC matryoshka clearly benefits (+0.13); SubseqH8 + H8 multi-dist improvement is smaller", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(args.out))
    plt.close(fig)
    print(f"  saved {args.out}")

    print("\n## Q1.3 (right-edge) vs Q2.C (per-position) — peak success per arch")
    print()
    print("| arch | Q1.3 peak | Q2.C peak | Δ |")
    print("|---|---|---|---|")
    for arch_id, label in WINDOW_ARCHS:
        q1 = _per_strength(arch_id, "steering_paper_normalised")
        q2 = _per_strength(arch_id, "steering_paper_window_perposition")
        if not q1 or not q2:
            print(f"| {label} | — | — | — |")
            continue
        peak1 = max(q1.values(), key=lambda v: v[0])
        peak2 = max(q2.values(), key=lambda v: v[0])
        d = peak2[0] - peak1[0]
        print(f"| {label} | {peak1[0]:.2f} (s_norm={peak1[2]:.0f}) | {peak2[0]:.2f} (s_norm={peak2[2]:.0f}) | {d:+.2f} |")


if __name__ == "__main__":
    main()
