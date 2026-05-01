"""Paper-strength absolute Pareto — direct apples-to-apples vs T-SAE paper.

Two-panel view:
- Left: NORMALISED grid (s_norm × abs_mean per arch — fair cross-arch comparison)
- Right: PAPER-FAITHFUL absolute grid (paper's exact strengths, all archs same)

Shows the dramatic shift in TXC's Δ vs T-SAE between the two protocols.

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_paper_protocol_pareto
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"
ANCHOR_15_NORM = 1.133
ANCHOR_15_ABS = 0.244

# (arch_id, label, color)
ARCHS = [
    ("tsae_paper_k20", "T-SAE k=20", "#1f77b4"),
    ("txc_h8_t2_kpos20_shifts2", "OBLIT H8 RE", "#d62728"),
    ("txc_maxpool_h8_t2_kpos20_shifts2", "MaxPool H8 RE", "#e377c2"),
    ("txc_contrastive_h8_t2_kpos20_shifts2", "Contrastive H8 RE", "#9467bd"),
]

NORM_DIRS = {
    42: "steering_paper_normalised",
    1: "steering_paper_normalised_seed1",
    2: "steering_paper_normalised_seed2",
}
ABS_DIRS = {
    42: "steering_paper_absolute",
    1: "steering_paper_absolute_seed1",
    2: "steering_paper_absolute_seed2",
}


def get_curve(arch_id, dirs_dict):
    by_strength_succ = collections.defaultdict(list)
    by_strength_coh = collections.defaultdict(list)
    for sd, sub in dirs_dict.items():
        path = BASE / sub / arch_id / "grades.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open()]
        per_s = collections.defaultdict(list)
        for r in rows:
            if r.get("success_grade") is None or r.get("coherence_grade") is None:
                continue
            per_s[float(r.get("strength", 0))].append(r)
        s_vals = sorted(per_s.keys())
        for s in s_vals:
            grades = per_s[s]
            ms = float(np.mean([g["success_grade"] for g in grades]))
            mc = float(np.mean([g["coherence_grade"] for g in grades]))
            by_strength_succ[s].append(ms)
            by_strength_coh[s].append(mc)
    s_vals = sorted(by_strength_succ.keys())
    succ = [float(np.mean(by_strength_succ[s])) for s in s_vals]
    coh = [float(np.mean(by_strength_coh[s])) for s in s_vals]
    return s_vals, succ, coh


def cliff_at(succ, coh, thr):
    valid = [s for s, c in zip(succ, coh) if c >= thr]
    return float(max(valid)) if valid else 0.0


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    for ax, dirs_dict, anchor, title_proto in [
        (axes[0], NORM_DIRS, ANCHOR_15_NORM, "Normalised per-arch grid (fair cross-arch)"),
        (axes[1], ABS_DIRS, ANCHOR_15_ABS, "Paper-faithful absolute grid (paper-comparable)"),
    ]:
        for arch_id, label, color in ARCHS:
            curve = get_curve(arch_id, dirs_dict)
            if not curve or not curve[0]:
                continue
            s, su, co = curve
            order = np.argsort(co)  # plot ordered by coh
            cov = np.array(co)
            suv = np.array(su)
            sv = np.array(s)
            is_anchor = arch_id == "tsae_paper_k20"
            ax.plot(cov, suv, "-o", color=color, lw=2.5 if is_anchor else 2.0,
                    markersize=7 if is_anchor else 6,
                    label=f"{label} (n=3)", alpha=0.95,
                    linestyle="--" if is_anchor else "-")
            valid_15 = [(s_, su_, co_) for s_, su_, co_ in zip(sv, suv, cov) if co_ >= 1.5]
            if valid_15:
                peak15 = max(valid_15, key=lambda v: v[1])
                ax.plot(peak15[2], peak15[1], "*", color=color,
                        markersize=18, markeredgecolor="black",
                        markeredgewidth=1.0, zorder=10)

        ax.axvline(1.5, color="grey", linestyle=":", alpha=0.6)
        ax.text(1.51, 0.05, "coh=1.5 (prereg)", fontsize=9, color="grey")
        ax.axhline(anchor, color="#1f77b4", linestyle=":", alpha=0.5)
        ax.text(0.7, anchor + 0.03, f"T-SAE peak15={anchor}",
                fontsize=8, color="#1f77b4")
        win_thresh = anchor + 0.27
        ax.axhline(win_thresh, color="green", linestyle="--", alpha=0.4)
        ax.text(2.5, win_thresh + 0.03, f"+0.27 prereg WIN line ({win_thresh:.2f})",
                fontsize=8, color="green")
        ax.set_xlim(0.6, 3.1)
        ax.set_ylim(0, 2.0)
        ax.set_xlabel("Mean coherence (Sonnet 4.6 grader)", fontsize=11)
        ax.set_ylabel("Mean steering success (Sonnet 4.6 grader)", fontsize=11)
        ax.set_title(title_proto, fontsize=12)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Phase 7 protocol comparison: T-SAE k=20 baseline vs 3 best TXC architectures (n=3 multi-seed)\n"
        "Left: normalised per-arch s_norm grid. Right: paper-faithful absolute strength grid {10,100,150,500,...,15000}.\n"
        "stars = cliff @ coh >= 1.5; dashed green = +0.27 WIN line.\n"
        "Under paper-faithful protocol, ALL 3 TXCs win by Delta = +1.0+ (T-SAE coh-stable peak isn't in paper grid).",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()

    out = PLOTS_DIR / "paper_protocol_pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.thumb.png')}")

    # Also print cliff comparison
    print("\n--- Cliff comparison @ coh ≥ 1.5 ---")
    print(f"{'cell':25s}  {'normalised':>10s}  {'paper-abs':>10s}")
    for arch_id, label, color in ARCHS:
        norm_curve = get_curve(arch_id, NORM_DIRS)
        abs_curve = get_curve(arch_id, ABS_DIRS)
        norm_c = cliff_at(norm_curve[1], norm_curve[2], 1.5) if norm_curve and norm_curve[1] else 0
        abs_c = cliff_at(abs_curve[1], abs_curve[2], 1.5) if abs_curve and abs_curve[1] else 0
        print(f"{label:25s}  {norm_c:>10.3f}  {abs_c:>10.3f}")


if __name__ == "__main__":
    main()
