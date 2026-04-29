"""Headline plot for the steering paper-rescue: protocol-comparison bars.

Shows per-arch peak (success, coherence) under THREE protocols on the
same 6-arch shortlist:

  1. AxBench-additive (Agent C's protocol; data from `steering/<arch>/grades.jsonl`).
  2. Paper-clamp at PAPER_STRENGTHS (Dmitry's reported numbers — hardcoded
     since his outputs are gitignored on his pod).
  3. Paper-clamp normalised at s ∈ {0.5..50} × <|z|>_arch (Q1.3 — this
     run; data from `steering_paper_normalised/<arch>/grades.jsonl`).

Output:
  results/case_studies/plots/phase7_steering_v2_protocol_comparison.png

Reads:
  results/case_studies/steering/<arch>/grades.jsonl              (AxBench)
  results/case_studies/steering_paper_normalised/<arch>/grades.jsonl (Q1.3)
  Hardcoded Dmitry peaks (commented section in this file).
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


# ─────────────────────────────────────────── Dmitry's reported paper-clamp peaks
# Source: docs/dmitry/case_studies/rlhf/summary.md on origin/dmitry-rlhf,
#         commits 0550f2f..229d1f0 (cherry-picked onto unification).
# Format: arch_id -> (peak_strength, peak_success, peak_coherence)
DMITRY_PAPER_CLAMP_PEAKS = {
    "topk_sae":            (100,  1.07, 1.40),
    "tsae_paper_k500":     (100,  1.33, 1.50),
    "tsae_paper_k20":      (100,  1.93, 1.37),
    "agentic_txc_02":      (500,  0.97, 1.20),
    "phase5b_subseq_h8":   (500,  1.10, 1.53),
    "phase57_partB_h8_bare_multidistance_t5": (500, 1.13, 1.10),
    # mlc not in Dmitry's set
}


DISPLAY = {
    "topk_sae": "TopKSAE\n(T=1, k=500)",
    "tsae_paper_k500": "T-SAE\n(T=1, k=500)",
    "tsae_paper_k20": "T-SAE\n(T=1, k=20)",
    "agentic_txc_02": "TXC matryoshka\n(T=5)",
    "phase5b_subseq_h8": "SubseqH8\n(T=10)",
    "phase57_partB_h8_bare_multidistance_t5": "H8 multidist\n(T=5)",
    "mlc_contrastive_alpha100_batchtopk": "MLC\n(L=5)",
}
ARCH_FAMILY = {
    "topk_sae": "per-token",
    "tsae_paper_k500": "per-token",
    "tsae_paper_k20": "per-token",
    "agentic_txc_02": "window",
    "phase5b_subseq_h8": "window",
    "phase57_partB_h8_bare_multidistance_t5": "window",
    "mlc_contrastive_alpha100_batchtopk": "mlc",
}


def _peak_from_grades(arch_id: str, subdir: str) -> tuple[float, float, float] | None:
    """Returns (peak_strength, peak_success, peak_coherence) — peak by mean success."""
    p = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    if not p.exists():
        return None
    by_s = collections.defaultdict(list)
    for line in p.open():
        r = json.loads(line)
        if r.get("success_grade") is None or r.get("coherence_grade") is None:
            continue
        by_s[float(r["strength"])].append((r["success_grade"], r["coherence_grade"]))
    if not by_s:
        return None
    means = []
    for s, pairs in by_s.items():
        sa = np.array([p[0] for p in pairs])
        ca = np.array([p[1] for p in pairs])
        means.append((s, float(sa.mean()), float(ca.mean())))
    best = max(means, key=lambda t: t[1])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(DMITRY_PAPER_CLAMP_PEAKS.keys()))
    ap.add_argument("--out-path", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_v2_protocol_comparison.png")
    args = ap.parse_args()
    banner(__file__)

    table_rows = []
    for arch in args.archs:
        ax_peak = _peak_from_grades(arch, "steering")
        norm_peak = _peak_from_grades(arch, "steering_paper_normalised")
        paper_peak = DMITRY_PAPER_CLAMP_PEAKS.get(arch)
        table_rows.append({
            "arch": arch,
            "family": ARCH_FAMILY.get(arch, "?"),
            "axbench": ax_peak,
            "paper_clamp_baseline": paper_peak,
            "paper_clamp_normalised": norm_peak,
        })
        print(f"  {arch:42s}  axbench={ax_peak}  paper_baseline={paper_peak}  paper_normalised={norm_peak}")

    # Bar chart: 3 protocols × N archs, two panels (success + coherence)
    fig, (ax_s, ax_c) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    arch_labels = [DISPLAY.get(r["arch"], r["arch"]) for r in table_rows]
    x = np.arange(len(table_rows))
    bar_w = 0.27
    colors = {"axbench": "#9467bd", "paper_clamp_baseline": "#d62728",
              "paper_clamp_normalised": "#2ca02c"}
    legend = {"axbench": "AxBench-additive (Agent C)",
              "paper_clamp_baseline": "Paper-clamp baseline (Dmitry)",
              "paper_clamp_normalised": "Paper-clamp normalised (Q1.3)"}

    def _y(r, key, idx):
        v = r[key]
        if v is None:
            return 0.0
        return v[idx]

    for offset, key in enumerate(["axbench", "paper_clamp_baseline", "paper_clamp_normalised"]):
        ax_s.bar(x + (offset - 1) * bar_w,
                 [_y(r, key, 1) for r in table_rows],
                 bar_w, label=legend[key], color=colors[key], alpha=0.85)
        ax_c.bar(x + (offset - 1) * bar_w,
                 [_y(r, key, 2) for r in table_rows],
                 bar_w, color=colors[key], alpha=0.85)

    ax_s.set_ylabel("Peak success (mean Sonnet 4.6 grade, 0-3)")
    ax_s.set_title("Steering protocol comparison — peak per-arch success\n"
                   "Same 6 archs × 30 concepts. Sonnet 4.6 grader.", fontsize=12)
    ax_s.set_ylim(0, 2.5)
    ax_s.grid(axis="y", alpha=0.25)
    ax_s.legend(loc="upper left", fontsize=10, frameon=True)

    ax_c.set_ylabel("Peak coherence (at success-peak strength)")
    ax_c.set_xlabel("")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(arch_labels)
    ax_c.set_ylim(0, 3.05)
    ax_c.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(args.out_path))
    plt.close(fig)
    print(f"  saved {args.out_path}")

    # Markdown leaderboard
    print("\n## Peak success/coherence per arch under three protocols")
    print()
    print("| arch | family | AxBench peak (suc, coh) | Paper-clamp baseline (suc, coh) | Paper-clamp normalised (suc, coh) |")
    print("|---|---|---|---|---|")
    for r in table_rows:
        arch = r["arch"]
        fam = r["family"]
        def _fmt(v):
            if v is None:
                return "—"
            return f"{v[1]:.2f} / {v[2]:.2f} (s={v[0]:g})"
        print(f"| `{arch}` | {fam} | {_fmt(r['axbench'])} | {_fmt(r['paper_clamp_baseline'])} | {_fmt(r['paper_clamp_normalised'])} |")


if __name__ == "__main__":
    main()
