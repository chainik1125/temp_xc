"""Reviewer-robust Pareto trade-off figure: 3-seed mean ± stderr on
both axes for the 5 paper-load-bearing archs (baseline TXC, Track 2,
2×2 cell, Cycle F, tsae_paper). Phase 6.2 ablation points (C1/C2/C3/
C5/C6) shown as fainter secondary markers at seed=42 to demonstrate
the TXC plateau.

Axes:
    x = mean_pool sparse-probing AUC (Phase 5 benchmark, k=5)
    y = concat_random SEMANTIC label count (rigorous metric, N=32)

Emits `results/phase61_pareto_robust.png` + thumbnail. This figure is
the paper's main Pareto claim.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
AUTOIN = REPO / "experiments/phase6_qualitative_latents/results/autointerp"

# Primary (3-seed-available) archs with rich seed variance
PRIMARY = {
    "agentic_txc_02":              ("TXC baseline",   "#3182bd", "o"),
    "agentic_txc_10_bare":         ("Track 2 (TXC+anti-dead)", "#08519c", "D"),
    "agentic_txc_12_bare_batchtopk": ("2×2 cell",     "#4292c6", "s"),
    "agentic_txc_02_batchtopk":    ("Cycle F",         "#9ecae1", "v"),
    "tsae_paper":                  ("T-SAE (paper)",   "#d62728", "*"),
}

# Phase 6.2 secondary points (seed=42 only, show ablation plateau)
SECONDARY = {
    "phase62_c1_track2_matryoshka":               ("C1 (Track 2+matr)",         "#9e9ac8"),
    "phase62_c2_track2_contrastive":              ("C2 (Track 2+contr)",        "#756bb1"),
    "phase62_c3_track2_matryoshka_contrastive":   ("C3 (Track 2+matr+contr)",   "#54278f"),
    "phase62_c5_track2_longer":                   ("C5 (Track 2 longer)",       "#bcbddc"),
    "phase62_c6_bare_batchtopk_longer":           ("C6 (2×2 longer)",           "#dadaeb"),
}


def load_probing(agg="mean_pool", k=5):
    """Return {arch: [per-seed AUC]}."""
    per_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    with PROBE.open() as f:
        for line in f:
            d = json.loads(line)
            if d["aggregation"] != agg or d["k_feat"] != k:
                continue
            rid = d["run_id"]
            if "__seed" not in rid:
                continue
            arch, sd = rid.rsplit("__seed", 1)
            try:
                sd = int(sd)
            except ValueError:
                continue
            key = (rid, d["task_name"])
            if key in seen:
                continue
            seen.add(key)
            per_seed[arch][sd].append(d["test_auc"])
    out = {}
    for arch, seed_dict in per_seed.items():
        seed_means = [sum(v) / len(v) for v in seed_dict.values() if len(v) == 36]
        if seed_means:
            out[arch] = seed_means
    return out


def load_random_sem():
    """Return {arch: [per-seed semantic_count]}."""
    out: dict[str, list[int]] = defaultdict(list)
    for p in AUTOIN.glob("*__seed*__concatrandom__labels.json"):
        d = json.loads(p.read_text())
        out[d["arch"]].append(d["metrics"]["semantic_count"])
    return dict(out)


def _mean_se(vals):
    if not vals:
        return float("nan"), 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="mean_pool",
                    choices=["last_position", "mean_pool"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=str,
                    default=str(REPO / "experiments/phase6_qualitative_latents/results/phase61_pareto_robust.png"))
    args = ap.parse_args()

    aucs = load_probing(agg=args.agg, k=args.k)
    sems = load_random_sem()

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Secondary (Phase 6.2) points first (behind primary)
    for arch, (label, color) in SECONDARY.items():
        if arch not in aucs or arch not in sems:
            continue
        a_mean, _ = _mean_se(aucs[arch])
        s_mean, _ = _mean_se(sems[arch])
        ax.scatter([a_mean], [s_mean], s=60, color=color, alpha=0.7,
                   marker="o", edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(label, (a_mean, s_mean), xytext=(6, 3),
                    textcoords="offset points", fontsize=7, color="#555555")

    # --- Primary (3-seed) points with error bars
    for arch, (label, color, marker) in PRIMARY.items():
        if arch not in aucs or arch not in sems:
            continue
        a_mean, a_se = _mean_se(aucs[arch])
        s_mean, s_se = _mean_se(sems[arch])
        n_probe = len(aucs[arch])
        n_autoin = len(sems[arch])
        ax.errorbar([a_mean], [s_mean],
                    xerr=[a_se] if a_se > 0 else None,
                    yerr=[s_se] if s_se > 0 else None,
                    fmt=marker, markersize=14, color=color,
                    markeredgecolor="black", markeredgewidth=1.5,
                    ecolor="black", elinewidth=1.2, capsize=4,
                    zorder=5, label=f"{label}  (n_p={n_probe}, n_q={n_autoin})")
        ax.annotate(label, (a_mean, s_mean), xytext=(12, 8),
                    textcoords="offset points", fontsize=10,
                    fontweight="bold", zorder=6)

    # --- Pareto frontier line: tsae_paper → Track 2
    if "tsae_paper" in aucs and "agentic_txc_10_bare" in aucs:
        xs = [_mean_se(aucs["tsae_paper"])[0], _mean_se(aucs["agentic_txc_10_bare"])[0]]
        ys = [_mean_se(sems["tsae_paper"])[0], _mean_se(sems["agentic_txc_10_bare"])[0]]
        # Sort by x for a clean line
        order = np.argsort(xs)
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                "--", color="#555555", linewidth=1.5, alpha=0.7, zorder=2,
                label="Pareto frontier")

    # --- Gap annotations
    if "tsae_paper" in aucs and "agentic_txc_10_bare" in aucs:
        tsp_x = _mean_se(aucs["tsae_paper"])[0]
        tsp_y = _mean_se(sems["tsae_paper"])[0]
        tr2_x = _mean_se(aucs["agentic_txc_10_bare"])[0]
        tr2_y = _mean_se(sems["agentic_txc_10_bare"])[0]

        # Horizontal gap annotation (probing)
        ax.annotate(
            "", xy=(tsp_x, 1.0), xytext=(tr2_x, 1.0),
            arrowprops=dict(arrowstyle="<->", color="grey", alpha=0.6,
                            lw=1.2),
        )
        ax.text((tsp_x + tr2_x) / 2, 1.8,
                f"Δ AUC = {tr2_x - tsp_x:+.3f}",
                ha="center", fontsize=9, color="grey",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

        # Vertical gap annotation (qualitative)
        ax.annotate(
            "", xy=(tsp_x + 0.005, tsp_y), xytext=(tsp_x + 0.005, tr2_y),
            arrowprops=dict(arrowstyle="<->", color="grey", alpha=0.6,
                            lw=1.2),
        )
        ax.text(tsp_x + 0.012, (tsp_y + tr2_y) / 2,
                f"Δ = {tsp_y - tr2_y:+.1f}\nlabels",
                va="center", fontsize=9, color="grey",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_xlabel(f"Probing mean AUC  ({args.agg}, k={args.k})  "
                  f"— Phase 5 SAEBench protocol, 36 binary tasks",
                  fontsize=11)
    ax.set_ylabel("SEMANTIC label count on concat_random  (/32, Haiku temp=0, multi-judge)",
                  fontsize=11)
    ax.set_title(
        "TXC vs T-SAE Pareto trade-off\n"
        "Error bars = stderr across 3 training seeds  ·  Phase 6.2 ablations in light purple",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Shaded regions
    ax.axhspan(0, 6, color="#08519c", alpha=0.04, zorder=0)
    ax.axhspan(6, 20, color="#d62728", alpha=0.04, zorder=0)
    ax.text(ax.get_xlim()[0] + 0.002, 0.5,
            "TXC family\n(probing-favored region)",
            fontsize=9, color="#08519c", alpha=0.7)
    ax.text(ax.get_xlim()[0] + 0.002, 10.5,
            "T-SAE paper\n(qualitative-favored region)",
            fontsize=9, color="#d62728", alpha=0.7)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    thumb = Path(args.out).with_suffix(".thumb.png")
    fig.savefig(thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} + {thumb}")


if __name__ == "__main__":
    main()
