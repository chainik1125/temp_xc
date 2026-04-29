"""Build the final v2 Pareto plot for Y's recommended protocol (AxBench-additive).

Pulls from two grader sources:

  1. Agent C's v1 grades at strengths {0.5, 1, 2, 4, 8, 12, 16, 24} —
     `results/case_studies/steering/<arch>/grades.jsonl`. Graded with the
     pre-rubric Sonnet pipeline.
  2. Y's extended grades at strengths {-100, -50, -25, -10, 0, 10, 25, 50,
     100} — `results/case_studies/steering_axbench_extended/<arch>/grades.jsonl`.
     Graded with the prompt-cached system rubric (slightly stricter, ~10-15%
     lower absolute scores for the same generation).

The v2 Pareto plot stitches both: low-strength regime from (1), extended
regime from (2). Per-arch points are connected only within a single source
to avoid mixing graders. Strength=0 (no-op control) provides a calibration
anchor between the two.

Output:
  results/case_studies/plots/phase7_steering_v2.png
  results/case_studies/plots/phase7_steering_v2.json   (per-arch peak summary)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.phase7_unification.case_studies._paths import CASE_STUDIES_DIR


SHORTLIST = [
    "topk_sae",
    "tsae_paper_k500",
    "tsae_paper_k20",
    "mlc_contrastive_alpha100_batchtopk",
    "agentic_txc_02",
    "phase5b_subseq_h8",
]
PALETTE = {
    "topk_sae": "#1f77b4",
    "tsae_paper_k500": "#ff7f0e",
    "tsae_paper_k20": "#d62728",
    "mlc_contrastive_alpha100_batchtopk": "#9467bd",
    "agentic_txc_02": "#2ca02c",
    "phase5b_subseq_h8": "#17becf",
}
LABEL = {
    "topk_sae": "TopKSAE (per-token, k=500)",
    "tsae_paper_k500": "T-SAE (per-token, k=500)",
    "tsae_paper_k20": "T-SAE (per-token, k=20)",
    "mlc_contrastive_alpha100_batchtopk": "MLC contrastive (5-layer)",
    "agentic_txc_02": "TXC matryoshka (T=5)",
    "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
}
PLOTS_DIR = Path("experiments/phase7_unification/results/case_studies/plots")


def _aggregate(arch_id: str, subdir: str) -> dict[float, tuple[float, float, int]]:
    path = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    if not path.exists():
        return {}
    by_s: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        if r.get("success_grade") is None or r.get("coherence_grade") is None:
            continue
        s = float(r["strength"])
        by_s[s].append((float(r["success_grade"]), float(r["coherence_grade"])))
    return {s: (float(np.mean([p[0] for p in pairs])),
                float(np.mean([p[1] for p in pairs])),
                len(pairs))
            for s, pairs in sorted(by_s.items())}


def _build_payload() -> dict:
    archs = {}
    for arch_id in SHORTLIST:
        v1 = _aggregate(arch_id, "steering")
        ext = _aggregate(arch_id, "steering_axbench_extended")
        all_pos = {s: pair for s, pair in {**v1, **ext}.items() if s >= 0}
        # Pick "best" (success, coherence) pareto-front operating point per arch.
        best_suc = max(all_pos.values(), key=lambda x: x[0]) if all_pos else None
        archs[arch_id] = {
            "label": LABEL[arch_id],
            "agent_c_v1": {f"{s:g}": list(pair) for s, pair in v1.items()},
            "extended_y_rubric": {f"{s:g}": list(pair) for s, pair in ext.items()},
            "peak_overall_success": best_suc,
        }
    return {
        "shortlist": SHORTLIST,
        "v1_source": "results/case_studies/steering/<arch>/grades.jsonl (Agent C, pre-rubric)",
        "extended_source": ("results/case_studies/steering_axbench_extended/<arch>/grades.jsonl "
                            "(Y, prompt-cached system rubric)"),
        "archs": archs,
    }


def _plot(payload: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    fig, (ax_pareto, ax_strength) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Pareto (success vs coherence). One marker per (arch, strength)
    # cell; v1 = circle (low-strength), extended = triangle (high-strength).
    for arch_id in SHORTLIST:
        info = payload["archs"][arch_id]
        c = PALETTE[arch_id]
        v1 = sorted(((float(s), tuple(pair)) for s, pair in info["agent_c_v1"].items()))
        ex = sorted(((float(s), tuple(pair)) for s, pair in info["extended_y_rubric"].items()))
        # Points
        for source, marker, alpha, dataset in [
            (v1, "o", 0.55, "agent_c_v1"),
            (ex, "^", 0.55, "extended_y"),
        ]:
            for s_val, pair in source:
                if s_val < 0:
                    continue
                suc, coh, _n = pair[0], pair[1], pair[2]
                ax_pareto.scatter(coh, suc, color=c, marker=marker, alpha=alpha,
                                  s=70, edgecolor="black", linewidths=0.5)
        # Best peak marker (star)
        if info["peak_overall_success"]:
            best_suc, best_coh, _n = info["peak_overall_success"]
            ax_pareto.scatter([best_coh], [best_suc], color=c, marker="*",
                              s=320, edgecolor="black", linewidths=1.0,
                              label=f"{info['label']}  (peak suc={best_suc:.2f})",
                              zorder=5)

    ax_pareto.set_xlabel("mean coherence (0-3)")
    ax_pareto.set_ylabel("mean steering success (0-3)")
    ax_pareto.set_title("Pareto: success vs coherence (AxBench-additive)\n"
                        "circle=Agent C v1 (s in 0.5-24); triangle=extended (s in 0-100); star=peak per arch")
    ax_pareto.grid(True, ls=":", alpha=0.4)
    ax_pareto.legend(loc="lower left", fontsize=8)
    ax_pareto.set_xlim(0, 3.05)
    ax_pareto.set_ylim(0, 3.05)

    # Right panel: success vs strength on log-x.
    for arch_id in SHORTLIST:
        info = payload["archs"][arch_id]
        c = PALETTE[arch_id]
        v1 = sorted(((float(s), tuple(pair)) for s, pair in info["agent_c_v1"].items()))
        ex = sorted(((float(s), tuple(pair)) for s, pair in info["extended_y_rubric"].items()))
        v1_pos = [(s, p) for s, p in v1 if s > 0]
        ex_pos = [(s, p) for s, p in ex if s > 0]
        if v1_pos:
            ss = [s for s, _ in v1_pos]; sucs = [p[0] for _, p in v1_pos]
            ax_strength.plot(ss, sucs, "-", color=c, lw=1.5, alpha=0.9,
                             marker="o", label=f"{LABEL[arch_id]} v1")
        if ex_pos:
            ss = [s for s, _ in ex_pos]; sucs = [p[0] for _, p in ex_pos]
            ax_strength.plot(ss, sucs, "--", color=c, lw=1.5, alpha=0.6,
                             marker="^", label=f"{LABEL[arch_id]} ext")
    ax_strength.set_xscale("log")
    ax_strength.set_xlabel("AxBench strength (positive multiplier of unit decoder, log)")
    ax_strength.set_ylabel("mean steering success (0-3)")
    ax_strength.set_title("Success vs strength, two grader sources overlaid")
    ax_strength.grid(True, ls=":", alpha=0.4)
    ax_strength.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        "Phase 7 — recommended-protocol Pareto (AxBench-additive)\n"
        "v1 (Agent C, pre-rubric) at low strengths; extended (Y, system-rubric) at high strengths",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, str(out_path))
    plt.close(fig)


def _print_summary(payload: dict) -> None:
    print(f"\n  v2 Pareto peaks (best success across both grader sources):")
    print(f"  {'arch':<48}  {'best_suc':>9}  {'at_coh':>7}")
    for arch_id in SHORTLIST:
        info = payload["archs"][arch_id]
        if info["peak_overall_success"]:
            suc, coh, n = info["peak_overall_success"]
            print(f"  {info['label']:<48}  {suc:>9.2f}  {coh:>7.2f}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_payload()
    json_path = PLOTS_DIR / "phase7_steering_v2.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_path}")
    png_path = PLOTS_DIR / "phase7_steering_v2.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")
    _print_summary(payload)


if __name__ == "__main__":
    main()
