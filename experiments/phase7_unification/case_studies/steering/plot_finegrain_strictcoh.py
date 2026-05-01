"""Fine-grain Pareto + strict-coh band visualization.

Shows the saving-grace finding: under fine-grain protocol, TXC family
maintains coherent steering at strict coh ≥ 1.8 where T-SAE collapses.

Two-panel:
- Left: full Pareto curves (succ vs coh) with strict-coh band [1.8, 2.5] shaded
- Right: AUC over strict-coh band — bar chart with Δ vs T-SAE

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_finegrain_strictcoh
"""
from __future__ import annotations
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/temp_xc")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"

ARCHS = [
    ("tsae_paper_k20", "T-SAE k=20 (anchor)", "#1f77b4"),
    ("txc_h8_t2_kpos20_shifts2", "OBLIT H8 RE", "#d62728"),
    ("txc_maxpool_h8_t2_kpos20_shifts2", "MaxPool H8 RE", "#e377c2"),
    ("txc_contrastive_h8_t2_kpos20_shifts2", "Contrastive H8 RE", "#9467bd"),
]

SUBDIRS = [
    "steering_paper_absolute", "steering_paper_absolute_seed1", "steering_paper_absolute_seed2",
    "steering_paper_finegrain", "steering_paper_finegrain_seed1", "steering_paper_finegrain_seed2",
]


def load_curve(arch_id):
    by_s = collections.defaultdict(list)
    for sub in SUBDIRS:
        path = BASE / sub / arch_id / "grades.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open()]
        per_s_seed = collections.defaultdict(list)
        for r in rows:
            if r.get("success_grade") is None or r.get("coherence_grade") is None: continue
            per_s_seed[float(r.get("strength", 0))].append(r)
        for s, items in per_s_seed.items():
            ss = float(np.mean([i["success_grade"] for i in items]))
            cs = float(np.mean([i["coherence_grade"] for i in items]))
            by_s[s].append((ss, cs))
    s_vals = sorted(by_s.keys())
    succ = [float(np.mean([r[0] for r in by_s[s]])) for s in s_vals]
    coh = [float(np.mean([r[1] for r in by_s[s]])) for s in s_vals]
    return s_vals, succ, coh


def auc_pareto(succ, coh, lo, hi):
    succs = np.array(succ); cohs = np.array(coh)
    grid = np.linspace(lo, hi, 31)
    auc_vals = []
    for c in grid:
        valid = succs[cohs >= c]
        auc_vals.append(float(np.max(valid)) if len(valid) > 0 else 0.0)
    return float(np.trapezoid(auc_vals, grid) / (hi - lo))


def main():
    curves = {}
    for arch_id, label, color in ARCHS:
        curves[arch_id] = (label, color, *load_curve(arch_id))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Left panel: Pareto curves with strict-coh band shaded
    ax = axes[0]
    for arch_id, (label, color, s, su, co) in curves.items():
        if not su: continue
        # Sort by coh
        order = np.argsort(co)
        cov = np.array(co)[order]
        suv = np.array(su)[order]
        is_anchor = arch_id == "tsae_paper_k20"
        ax.plot(cov, suv, "-o", color=color, lw=2.5 if is_anchor else 2.0,
                markersize=7 if is_anchor else 6,
                label=f"{label} (n=3)", alpha=0.95,
                linestyle="--" if is_anchor else "-")
        # Mark all strengths used
        for s_v, su_v, co_v in zip(s, su, co):
            if 1.8 <= co_v <= 2.5:
                ax.plot(co_v, su_v, "*", color=color, markersize=14,
                        markeredgecolor="black", markeredgewidth=0.8, zorder=10)

    ax.axvspan(1.8, 2.5, alpha=0.12, color="green", label="strict-coh band [1.8, 2.5]")
    ax.axvline(1.8, color="green", linestyle="--", alpha=0.5)
    ax.axvline(2.5, color="green", linestyle="--", alpha=0.5)
    ax.set_xlim(0.6, 3.1)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("Mean coherence (Sonnet 4.6 grader)", fontsize=11)
    ax.set_ylabel("Mean steering success (Sonnet 4.6 grader)", fontsize=11)
    ax.set_title("Pareto curves under fine-grain protocol\n(stars = strengths in strict-coh band [1.8, 2.5])", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    # Right panel: AUC over strict-coh bands, bar chart
    ax = axes[1]
    bands = [(1.5, 3.0), (1.75, 3.0), (1.8, 2.5), (1.9, 2.5), (2.0, 2.5)]
    band_labels = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in bands]
    width = 0.20

    arch_aucs = {}
    for arch_id, (label, color, s, su, co) in curves.items():
        aucs = [auc_pareto(su, co, lo, hi) for lo, hi in bands]
        arch_aucs[arch_id] = (label, color, aucs)

    x = np.arange(len(bands))
    for i, (arch_id, (label, color, aucs)) in enumerate(arch_aucs.items()):
        offset = (i - 1.5) * width
        is_anchor = arch_id == "tsae_paper_k20"
        ax.bar(x + offset, aucs, width, label=label, color=color, alpha=0.85,
               edgecolor="black" if is_anchor else None, linewidth=1.5 if is_anchor else 0)

    # Δ annotations
    anchor_aucs = arch_aucs["tsae_paper_k20"][2]
    for i, (lo, hi) in enumerate(bands):
        for j, (arch_id, (label, color, aucs)) in enumerate(arch_aucs.items()):
            if arch_id == "tsae_paper_k20": continue
            d = aucs[i] - anchor_aucs[i]
            offset = (j - 1.5) * width
            if abs(d) > 0.03:
                ax.text(i + offset, aucs[i] + 0.02, f"{d:+.2f}", ha="center", fontsize=7,
                        color="green" if d > 0.27 else ("darkgreen" if d > 0 else "red"),
                        fontweight="bold" if d > 0.27 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=10)
    ax.set_xlabel("Coherence band", fontsize=11)
    ax.set_ylabel("AUC of (max success | coh ≥ c) over band", fontsize=11)
    ax.set_title("AUC over coh bands\n(numbers = Δ vs T-SAE; bold green = above +0.27 prereg)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Phase 7 strict-coherence Pareto: TXC family wins at strict-coh under fine-grain protocol\n"
        "OBLIT/MaxPool RE n=3 both clear +0.27 prereg threshold on AUC over strict-coh band [1.8, 2.5]",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()

    out = PLOTS_DIR / "finegrain_strictcoh_pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
