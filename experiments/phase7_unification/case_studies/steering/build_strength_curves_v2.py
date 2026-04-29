"""Build phase7_steering_strength_curves_v2.png — per-arch success-vs-strength
under AxBench-additive (the recommended protocol from Y's Q2 synthesis).

Combines Agent C v1 grades at strengths {0.5..24} with Y's extended grades at
{-100..+100}. v1 grader: pre-rubric Sonnet 4.6. Extended grader: Sonnet 4.6
with prompt-cached system rubric (~10-15% stricter on absolute scores).
The two source labels are visually distinguished so the rubric shift is
explicit; the rank ordering is unchanged across the boundary.
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


def _plot(out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_suc, ax_coh = axes

    for arch_id in SHORTLIST:
        c = PALETTE[arch_id]
        v1 = _aggregate(arch_id, "steering")
        ext = _aggregate(arch_id, "steering_axbench_extended")

        v1_pos = sorted([(s, suc, coh, n) for s, (suc, coh, n) in v1.items() if s > 0])
        ext_pos = sorted([(s, suc, coh, n) for s, (suc, coh, n) in ext.items() if s > 0])

        if v1_pos:
            ss = [v[0] for v in v1_pos]
            sucs = [v[1] for v in v1_pos]
            cohs = [v[2] for v in v1_pos]
            ax_suc.plot(ss, sucs, "-", color=c, lw=2.0, marker="o",
                        label=f"{LABEL[arch_id]} v1")
            ax_coh.plot(ss, cohs, "-", color=c, lw=2.0, marker="o",
                        label=f"{LABEL[arch_id]} v1")
        if ext_pos:
            ss = [v[0] for v in ext_pos]
            sucs = [v[1] for v in ext_pos]
            cohs = [v[2] for v in ext_pos]
            ax_suc.plot(ss, sucs, "--", color=c, lw=1.5, marker="^", alpha=0.7,
                        label=f"{LABEL[arch_id]} ext")
            ax_coh.plot(ss, cohs, "--", color=c, lw=1.5, marker="^", alpha=0.7,
                        label=f"{LABEL[arch_id]} ext")

    for ax, title in [(ax_suc, "Success"), (ax_coh, "Coherence")]:
        ax.set_xscale("log")
        ax.set_xlabel("AxBench-additive strength s (positive multiplier of unit decoder, log)")
        ax.set_title(f"{title} vs strength (AxBench-additive)")
        ax.grid(True, ls=":", alpha=0.4)
    ax_suc.set_ylabel("mean steering success (0-3)")
    ax_coh.set_ylabel("mean coherence (0-3)")
    ax_suc.legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(
        "Phase 7 — strength curves under recommended protocol (AxBench-additive)\n"
        "solid = Agent C v1 (s in 0.5-24, pre-rubric); dashed = Y extended (s in 10-100, system-rubric)",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, str(out_path))
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "phase7_steering_strength_curves_v2.png"
    _plot(out_path)
    print(f"wrote {out_path}  (+ thumb)")


if __name__ == "__main__":
    main()
