"""Pareto plot + per-strength curves for Agent C's case study C.ii.

Reads each arch's grades.jsonl and produces:

  results/case_studies/plots/phase7_steering_pareto.png
    Per-arch (mean coherence, mean success) Pareto curve. Points are
    strength levels; lines connect strengths in order. The arch whose
    curve lies in the upper-right Pareto-dominates the others.

  results/case_studies/plots/phase7_steering_strength_curves.png
    2-panel side-by-side: (top) mean success vs strength, (bottom)
    mean coherence vs strength, lines per arch. Reveals where each
    arch's steering kicks in and where coherence collapses.

Also prints a per-arch summary table: AUC under the Pareto curve (rough
"how dominated is this arch by the best") plus best-case (success,
coherence) point.
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
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, STAGE_1_ARCHS, STEERING_STRENGTHS,
)


DISPLAY = {
    "topk_sae": "TopKSAE (per-token, k=500)",
    "tsae_paper_k500": "T-SAE (per-token, k=500)",
    "tsae_paper_k20": "T-SAE (paper-faithful, k=20)",
    "agentic_txc_02": "TXC (matryoshka multi-scale, T=5)",
    "mlc_contrastive_alpha100_batchtopk": "MLC (contrastive, k=500)",
    "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
    "phase57_partB_h8_bare_multidistance_t5": "H8 (multi-distance, T=5)",
}

ARCH_COLOR = {
    "topk_sae": "#1f77b4",
    "tsae_paper_k500": "#ff7f0e",
    "tsae_paper_k20": "#d62728",
    "agentic_txc_02": "#2ca02c",
    "mlc_contrastive_alpha100_batchtopk": "#9467bd",
    "phase5b_subseq_h8": "#8c564b",
    "phase57_partB_h8_bare_multidistance_t5": "#e377c2",
}


def _load_per_strength(arch_id: str, base_subdir: str = "steering") -> dict[float, tuple[float, float, int]]:
    """Return {strength: (mean_success, mean_coherence, n_valid)} for one arch."""
    p = CASE_STUDIES_DIR / base_subdir / arch_id / "grades.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(line) for line in p.open()]
    by_s: dict[float, list[tuple[int, int]]] = collections.defaultdict(list)
    for r in rows:
        s = r.get("success_grade")
        c = r.get("coherence_grade")
        if s is None or c is None:
            continue
        by_s[float(r["strength"])].append((s, c))
    out = {}
    for strength, pairs in by_s.items():
        sa = np.array([p[0] for p in pairs], dtype=float)
        ca = np.array([p[1] for p in pairs], dtype=float)
        out[strength] = (float(sa.mean()), float(ca.mean()), len(pairs))
    return out


def make_pareto(arch_data: dict[str, dict[float, tuple]], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for arch_id, per_s in arch_data.items():
        if not per_s:
            continue
        strengths_sorted = sorted(per_s.keys())
        coh = [per_s[s][1] for s in strengths_sorted]
        suc = [per_s[s][0] for s in strengths_sorted]
        color = ARCH_COLOR.get(arch_id, "black")
        ax.plot(coh, suc, "o-", color=color, label=DISPLAY.get(arch_id, arch_id),
                linewidth=2, markersize=8)
        # Annotate strength next to each point.
        for s, x, y in zip(strengths_sorted, coh, suc):
            ax.annotate(f"{s:g}", (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color=color)
    ax.set_xlabel("Coherence  (mean Sonnet 4.6 grade, 0-3)")
    ax.set_ylabel("Steering success  (mean Sonnet 4.6 grade, 0-3)")
    ax.set_xlim(0, 3.05)
    ax.set_ylim(-0.05, 3.05)
    ax.grid(alpha=0.25)
    ax.set_title(
        "AxBench-style steering: success vs coherence per arch\n"
        "(30 concepts × 8 strengths, decoder-direction additive at L12 of Gemma-2-2b base)"
    )
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    # Pareto-frontier annotation: top-right is dominant.
    ax.text(2.95, 0.05, "↑ better\nsteering",
            ha="right", va="bottom", fontsize=9, color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa"))
    ax.text(0.05, 2.95, "more coherent →",
            ha="left", va="top", fontsize=9, color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa"))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(out_path))
    plt.close(fig)


def make_strength_curves(arch_data: dict[str, dict[float, tuple]], out_path: Path) -> None:
    fig, (ax_s, ax_c) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for arch_id, per_s in arch_data.items():
        if not per_s:
            continue
        strengths_sorted = sorted(per_s.keys())
        suc = [per_s[s][0] for s in strengths_sorted]
        coh = [per_s[s][1] for s in strengths_sorted]
        color = ARCH_COLOR.get(arch_id, "black")
        ax_s.plot(strengths_sorted, suc, "o-", color=color,
                  label=DISPLAY.get(arch_id, arch_id), linewidth=2, markersize=8)
        ax_c.plot(strengths_sorted, coh, "o-", color=color, linewidth=2, markersize=8)
    ax_s.set_ylabel("Steering success (0-3)")
    ax_s.set_ylim(0, 3.05)
    ax_s.grid(alpha=0.25)
    ax_s.set_title("Per-strength steering curves (Stage 1 archs)")
    ax_s.legend(loc="lower right", fontsize=9, frameon=True)
    ax_c.set_xlabel("Strength (decoder-direction multiplier)")
    ax_c.set_ylabel("Coherence (0-3)")
    ax_c.set_ylim(0, 3.05)
    ax_c.set_xscale("log")
    ax_c.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(out_path))
    plt.close(fig)


def emit_table(arch_data: dict[str, dict[float, tuple]]) -> str:
    lines = []
    lines.append("## Per-arch C.ii steering summary (Stage 1)")
    lines.append("")
    lines.append("| arch | mean success | mean coherence | best (suc, coh) | best strength |")
    lines.append("|---|---|---|---|---|")
    for arch_id, per_s in arch_data.items():
        if not per_s:
            continue
        all_pts = [(s, ms, mc) for s, (ms, mc, _) in per_s.items()]
        ms_all = np.mean([ms for _, ms, _ in all_pts])
        mc_all = np.mean([mc for _, _, mc in all_pts])
        # "Best" = strength that maximises (suc + coh) — dominant point.
        best = max(all_pts, key=lambda t: t[1] + t[2])
        lines.append(
            f"| `{arch_id}` | {ms_all:.2f} | {mc_all:.2f} | "
            f"({best[1]:.2f}, {best[2]:.2f}) | {best[0]:g} |"
        )
    lines.append("")
    lines.append("Per-strength values:")
    lines.append("")
    lines.append("| arch | s=0.5 | s=1 | s=2 | s=4 | s=8 | s=12 | s=16 | s=24 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arch_id, per_s in arch_data.items():
        cells = []
        for s in STEERING_STRENGTHS:
            if s in per_s:
                ms, mc, _ = per_s[s]
                cells.append(f"({ms:.2f}, {mc:.2f})")
            else:
                cells.append("—")
        lines.append(f"| `{arch_id}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS))
    ap.add_argument("--pareto-path", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_pareto.png")
    ap.add_argument("--curves-path", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_strength_curves.png")
    ap.add_argument("--base-subdir", default="steering",
                    help="subdir under results/case_studies/ to read grades.jsonl from "
                         "(steering | steering_paper)")
    args = ap.parse_args()
    banner(__file__)

    arch_data: dict[str, dict[float, tuple]] = {}
    for arch_id in args.archs:
        per_s = _load_per_strength(arch_id, base_subdir=args.base_subdir)
        if per_s:
            arch_data[arch_id] = per_s

    if not arch_data:
        print("no graded archs found; run grade_with_sonnet first")
        return

    print(f"  plotting {len(arch_data)} archs to {args.pareto_path} + {args.curves_path}")
    make_pareto(arch_data, args.pareto_path)
    make_strength_curves(arch_data, args.curves_path)
    print()
    print(emit_table(arch_data))


if __name__ == "__main__":
    main()
