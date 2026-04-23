"""Paper-ready Pareto-front figure: probing AUC (x) vs qualitative
generalisation (concat_random x/32 semantic count, y).

Each arch is a single point; the upper-right region is the Pareto
frontier (wins both). This figure is the paper's main argument for
"TXC family retains probing utility while lifting qualitative".

Reads:
  - experiments/phase5_downstream_utility/results/probing_results.jsonl
    → probing AUC (mean_pool, k=5) per arch, seed=42 only.
  - experiments/phase6_qualitative_latents/results/autointerp/
    *__seed42__concatrandom__labels.json → random /32 count.

Emits:
  - results/phase61_pareto_tradeoff.png (+ .thumb.png)
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
OUT = REPO / "experiments/phase6_qualitative_latents/results/phase61_pareto_tradeoff.png"

# Archs to show + display names
ARCHS = {
    "agentic_txc_02": ("TXC (baseline)", "#3182bd"),
    "agentic_txc_02_batchtopk": ("TXC+BatchTopK (Cycle F)", "#9ecae1"),
    "agentic_txc_09_auxk": ("TXC+AuxK (Cycle A)", "#6baed6"),
    "agentic_txc_11_stack": ("TXC+BatchTopK+AuxK (Cycle H)", "#c6dbef"),
    "agentic_txc_10_bare": ("TXC+anti-dead (Track 2)", "#08519c"),
    "agentic_mlc_08": ("MLC", "#31a354"),
    "tsae_paper": ("T-SAE (paper)", "#d62728"),
    # tsae_ours + tfa_big have no Phase 5 probing; skipped unless we add it
}


def load_probing_aucs(agg: str = "mean_pool", k: int = 5):
    aucs: dict[str, list[float]] = defaultdict(list)
    with PROBE.open() as f:
        for line in f:
            d = json.loads(line)
            if d["aggregation"] != agg or d["k_feat"] != k:
                continue
            rid = d["run_id"]
            # Parse arch from run_id (strip trailing __seedN)
            if "__seed" in rid:
                arch = rid.rsplit("__seed", 1)[0]
            else:
                arch = rid
            # Only seed=42 for this plot (apples-to-apples)
            if rid.endswith("__seed42"):
                aucs[arch].append(d["test_auc"])
    return {a: (sum(v) / len(v), len(v)) for a, v in aucs.items() if v}


def load_random_sem_counts():
    sems = {}
    for p in AUTOIN.glob("*__seed42__concatrandom__labels.json"):
        d = json.loads(p.read_text())
        sems[d["arch"]] = d["metrics"]["semantic_count"]
    return sems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agg", default="mean_pool",
                   choices=["last_position", "mean_pool"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    aucs = load_probing_aucs(agg=args.agg, k=args.k)
    sems = load_random_sem_counts()

    # Intersect archs
    keys = [a for a in ARCHS if a in aucs and a in sems]
    missing = [a for a in ARCHS if a not in keys]
    print(f"plotting {len(keys)} archs; missing: {missing}")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for arch in keys:
        auc, n = aucs[arch]
        sem = sems[arch]
        label, color = ARCHS[arch]
        ax.scatter([auc], [sem], s=180, color=color, edgecolor="black",
                   linewidth=1.2, zorder=5, label=label)
        ax.annotate(
            label, (auc, sem),
            xytext=(8, 5), textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(f"Probing mean AUC  ({args.agg}, k={args.k})  — Phase 5 benchmark")
    ax.set_ylabel("SEMANTIC labels on concat_random  (x/32, N=32, multi-judge)")
    ax.set_ylim(-1, 14)
    ax.grid(alpha=0.3)
    ax.axhline(12, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[0] + 0.005, 12.2, "T-SAE paper target (12/32)",
            fontsize=8, color="#d62728", alpha=0.8)
    ax.set_title(
        "Pareto: probing utility (x) vs qualitative generalisation (y)\n"
        "Upper-right = wins both. seed=42, N=32, temp=0, multi-judge.",
        fontsize=11,
    )

    # Arrow pointing to Pareto ideal
    ax.annotate(
        "Pareto\nideal",
        xy=(ax.get_xlim()[1] - 0.01, 12), xytext=(-40, -10),
        textcoords="offset points",
        fontsize=9, color="grey",
        arrowprops=dict(arrowstyle="->", color="grey"),
    )

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    thumb = Path(args.out).with_suffix(".thumb.png")
    fig.savefig(thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} + {thumb}")


if __name__ == "__main__":
    main()
