"""Analysis + plots for Q1.3 (normalised paper-clamp).

Reads `results/case_studies/steering_paper_normalised/<arch>/grades.jsonl`
and produces:

  results/case_studies/plots/phase7_steering_v2_pareto.png
    Per-arch (mean coherence, mean success) Pareto curves under
    family-normalised paper-clamp.

  results/case_studies/plots/phase7_steering_v2_curves.png
    Per-arch curves of (success, coherence) vs s_norm = s_abs / <|z|>_arch.
    If the magnitude-scale story holds, all archs should peak at roughly
    the same s_norm.

  results/case_studies/plots/phase7_steering_v2_peak_comparison.png
    Bar chart: peak (success, coherence) per arch under
      (a) Dmitry's PAPER_STRENGTHS (absolute, baseline)
      (b) family-normalised strengths (this run)
    Shows whether normalisation closes the per-token / window gap.

Also prints a leaderboard table.
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


DISPLAY = {
    "topk_sae": "TopKSAE (T=1, k=500)",
    "tsae_paper_k500": "T-SAE (T=1, k=500)",
    "tsae_paper_k20": "T-SAE (T=1, k=20)",
    "agentic_txc_02": "TXC matryoshka (T=5)",
    "phase5b_subseq_h8": "SubseqH8 (T=10)",
    "phase57_partB_h8_bare_multidistance_t5": "H8 multidist (T=5)",
    "mlc_contrastive_alpha100_batchtopk": "MLC contrastive (L=5)",
}
ARCH_COLOR = {
    "topk_sae": "#1f77b4",
    "tsae_paper_k500": "#ff7f0e",
    "tsae_paper_k20": "#d62728",
    "agentic_txc_02": "#2ca02c",
    "phase5b_subseq_h8": "#8c564b",
    "phase57_partB_h8_bare_multidistance_t5": "#e377c2",
    "mlc_contrastive_alpha100_batchtopk": "#9467bd",
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


def _load_grades(arch_id: str, subdir: str) -> list[dict]:
    p = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.open()]
    # Some Q1.3 generations carry s_norm; for the abs-strength baseline
    # subdir, fall back to using strength directly. We attach abs_z_mean
    # if present; otherwise None. Plot code can decide what to plot.
    return rows


def _load_gen_meta(arch_id: str, subdir: str) -> list[dict]:
    """Load generations to recover s_norm/abs_z_mean (grader doesn't carry them)."""
    p = CASE_STUDIES_DIR / subdir / arch_id / "generations.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open()]


def _per_strength_summary(grades: list[dict], gens: list[dict]) -> dict[float, tuple]:
    # Build a (concept_id, strength) -> {s_norm, abs_z} map from gens
    meta = {}
    for g in gens:
        key = (g["concept_id"], float(g["strength"]))
        meta[key] = {"s_norm": g.get("s_norm"), "abs_z_mean": g.get("abs_z_mean")}

    by_s: dict[float, list[tuple[int, int, float | None]]] = collections.defaultdict(list)
    for r in grades:
        s = r.get("success_grade")
        c = r.get("coherence_grade")
        if s is None or c is None:
            continue
        strength = float(r["strength"])
        m = meta.get((r["concept_id"], strength), {})
        by_s[strength].append((s, c, m.get("s_norm")))

    out: dict[float, tuple[float, float, float | None, int]] = {}
    for s, triples in by_s.items():
        sa = np.array([t[0] for t in triples], dtype=float)
        ca = np.array([t[1] for t in triples], dtype=float)
        sn = triples[0][2]  # all rows for same strength share s_norm
        out[s] = (float(sa.mean()), float(ca.mean()), sn, len(triples))
    return out


def make_pareto(
    arch_data: dict[str, dict[float, tuple]],
    out_path: Path,
    title: str,
    annotate_what: str = "s_abs",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for arch_id, per_s in arch_data.items():
        if not per_s:
            continue
        strengths_sorted = sorted(per_s.keys())
        coh = [per_s[s][1] for s in strengths_sorted]
        suc = [per_s[s][0] for s in strengths_sorted]
        color = ARCH_COLOR.get(arch_id, "black")
        family = ARCH_FAMILY.get(arch_id, "")
        linestyle = "-" if family == "window" else ("--" if family == "per-token" else ":")
        ax.plot(coh, suc, "o" + linestyle, color=color,
                label=DISPLAY.get(arch_id, arch_id), linewidth=2, markersize=8)
        for s, x, y in zip(strengths_sorted, coh, suc):
            label = f"{s:g}"
            if annotate_what == "s_norm":
                sn = per_s[s][2]
                if sn is not None:
                    label = f"{sn:g}"
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=color)
    ax.set_xlabel("Coherence (mean Sonnet 4.6 grade, 0-3)")
    ax.set_ylabel("Steering success (mean Sonnet 4.6 grade, 0-3)")
    ax.set_xlim(0, 3.05)
    ax.set_ylim(-0.05, 3.05)
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.plot import save_figure
    save_figure(fig, str(out_path))
    plt.close(fig)


def make_curves_vs_s_norm(
    arch_data: dict[str, dict[float, tuple]],
    out_path: Path,
    title: str,
) -> None:
    fig, (ax_s, ax_c) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for arch_id, per_s in arch_data.items():
        if not per_s:
            continue
        # Order by s_norm
        items = [(per_s[s][2], per_s[s][0], per_s[s][1])
                 for s in per_s.keys() if per_s[s][2] is not None]
        if not items:
            continue
        items.sort()
        s_norms = [it[0] for it in items]
        suc = [it[1] for it in items]
        coh = [it[2] for it in items]
        color = ARCH_COLOR.get(arch_id, "black")
        family = ARCH_FAMILY.get(arch_id, "")
        ls = "-" if family == "window" else "--"
        ax_s.plot(s_norms, suc, "o" + ls, color=color,
                  label=DISPLAY.get(arch_id, arch_id), linewidth=2, markersize=8)
        ax_c.plot(s_norms, coh, "o" + ls, color=color, linewidth=2, markersize=8)
    ax_s.set_ylabel("Steering success (0-3)")
    ax_s.set_ylim(-0.05, 3.05)
    ax_s.grid(alpha=0.25)
    ax_s.set_title(title, fontsize=12)
    ax_s.legend(loc="lower right", fontsize=9, frameon=True)
    ax_c.set_xlabel("s_norm = absolute strength / <|z[picked]|>_arch  (log scale)")
    ax_c.set_ylabel("Coherence (0-3)")
    ax_c.set_ylim(-0.05, 3.05)
    ax_c.set_xscale("log")
    ax_c.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from src.utils.plot import save_figure
    save_figure(fig, str(out_path))
    plt.close(fig)


def emit_leaderboard(arch_data: dict[str, dict[float, tuple]]) -> str:
    lines = []
    lines.append("| arch | family | T | best s | best s_norm | peak suc | peak coh |")
    lines.append("|---|---|---|---|---|---|---|")
    for arch_id, per_s in arch_data.items():
        if not per_s:
            continue
        family = ARCH_FAMILY.get(arch_id, "?")
        all_pts = [(s, ms, mc, sn) for s, (ms, mc, sn, _) in per_s.items()]
        # peak by success
        best = max(all_pts, key=lambda t: t[1])
        s_abs, ms, mc, sn = best
        T = "?"  # populated below from arch tag
        if "agentic_txc" in arch_id or "matryoshka" in arch_id:
            T = "5"
        elif "subseq_h8" in arch_id:
            T = "10"
        elif "h8_bare" in arch_id:
            T = "5"
        elif "tsae" in arch_id or "topk_sae" in arch_id:
            T = "1"
        elif "mlc" in arch_id:
            T = "L=5"
        sn_str = f"{sn:.2f}" if sn is not None else "—"
        lines.append(f"| `{arch_id}` | {family} | {T} | {s_abs:g} | {sn_str} | {ms:.2f} | {mc:.2f} |")
    return "\n".join(lines)


DEFAULT_ARCHS = (
    "topk_sae",
    "tsae_paper_k20",
    "tsae_paper_k500",
    "agentic_txc_02",
    "phase5b_subseq_h8",
    "phase57_partB_h8_bare_multidistance_t5",
    "mlc_contrastive_alpha100_batchtopk",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS))
    ap.add_argument("--subdir", default="steering_paper_normalised")
    ap.add_argument("--out-pareto", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_v2_pareto.png")
    ap.add_argument("--out-curves", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_v2_curves.png")
    ap.add_argument("--title-suffix", default="(family-normalised paper-clamp, s × <|z|>_arch)")
    args = ap.parse_args()
    banner(__file__)

    arch_data: dict[str, dict[float, tuple]] = {}
    for arch_id in args.archs:
        grades = _load_grades(arch_id, args.subdir)
        gens = _load_gen_meta(arch_id, args.subdir)
        per_s = _per_strength_summary(grades, gens)
        if per_s:
            arch_data[arch_id] = per_s
            n_strengths = len(per_s)
            best_s = max(per_s.keys(), key=lambda s: per_s[s][0])
            print(f"  {arch_id}: {n_strengths} strengths  peak suc={per_s[best_s][0]:.2f}@s={best_s:g}")
        else:
            print(f"  {arch_id}: no grades found at {CASE_STUDIES_DIR / args.subdir / arch_id}/grades.jsonl")

    if not arch_data:
        print("no data; bail")
        return

    title_pareto = f"Steering: success vs coherence per arch — {args.title_suffix}"
    title_curves = f"Steering vs normalised strength — {args.title_suffix}"
    make_pareto(arch_data, args.out_pareto, title=title_pareto, annotate_what="s_norm")
    make_curves_vs_s_norm(arch_data, args.out_curves, title=title_curves)

    print()
    print(emit_leaderboard(arch_data))


if __name__ == "__main__":
    main()
