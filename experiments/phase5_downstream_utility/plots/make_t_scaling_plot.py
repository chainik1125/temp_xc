"""T-scaling line plot — mp/lp AUC vs T for various TXC families.

The paper-critical T-scaling result:
  - vanilla TXCDR × {TopK, BatchTopK}: known to FAIL (mono ≤ 0.57)
  - agentic_txc_02 × TopK: known to FAIL (mono ≤ 0.7)
  - H1 ConvTXCDR: FAILED (mono 0.4)
  - H8 multi-distance + anti-dead: TESTED HERE

If H8's curve climbs monotonically through T=20+, the paper headline is saved.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/temp_xc")
JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
OUT_DIR = REPO / "experiments/phase5_downstream_utility/results/plots"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}
K_FEAT = 5

FAMILIES = {
    "txcdr_t<T>": {
        T: f"txcdr_t{T}" for T in (2, 3, 5, 6, 7, 8, 10, 15, 20)
    },
    "txcdr_t<T>_batchtopk": {
        T: f"txcdr_t{T}_batchtopk" for T in (2, 3, 5, 6, 7, 8, 10, 15, 20, 24, 28, 30, 32, 36)
    },
    "agentic_txc_02_t<T>": {
        T: f"agentic_txc_02_t{T}" for T in (2, 3, 6, 7, 8)
    } | {5: "agentic_txc_02"},
    "conv_txcdr_t<T>": {
        T: f"conv_txcdr_t{T}" for T in (5, 10, 15, 20, 30)
    },
    "H8 phase57_partB_h8_bare_multidistance_t<T>": {
        5: "phase57_partB_h8_bare_multidistance",
        **{T: f"phase57_partB_h8_bare_multidistance_t{T}"
           for T in (6, 7, 8, 10, 15, 20, 30)}
    },
    "H3 log_matryoshka_t<T>": {
        T: f"log_matryoshka_t{T}" for T in (5, 10, 15, 20, 30)
    },
}


def _mean_auc(arch: str, seed: int = 42, agg: str = "last_position") -> float | None:
    aucs = []
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != agg or r.get("k_feat") != K_FEAT:
                continue
            if r.get("run_id") != f"{arch}__seed{seed}":
                continue
            v = r.get("test_auc")
            if v is None:
                continue
            v = float(v)
            if r.get("task_name") in FLIP_TASKS:
                v = max(v, 1.0 - v)
            aucs.append(v)
    if len(aucs) < 30:
        return None
    return st.mean(aucs)


def make_plot(agg: str = "mean_pool") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (label, t_to_arch) in enumerate(FAMILIES.items()):
        Ts, ys = [], []
        for T, arch in sorted(t_to_arch.items()):
            auc = _mean_auc(arch, agg=agg)
            if auc is not None:
                Ts.append(T)
                ys.append(auc)
        if not Ts:
            continue
        is_h8 = "H8" in label
        ax.plot(Ts, ys, marker='o',
                color=colors[i % len(colors)],
                label=label,
                linewidth=3 if is_h8 else 1.5,
                markersize=8 if is_h8 else 5,
                zorder=10 if is_h8 else 5)
        for T, y in zip(Ts, ys):
            ax.annotate(f"{y:.3f}", (T, y), textcoords="offset points",
                        xytext=(0, 8 if is_h8 else 5),
                        ha='center', fontsize=7 if not is_h8 else 9,
                        fontweight='bold' if is_h8 else 'normal')
    ax.set_xlabel("T (window length)")
    ax.set_ylabel(f"mean test AUC ({agg}, k_feat=5, 36 tasks)")
    ax.set_title(f"T-scaling — {agg}")
    ax.axhline(0.80, color='black', linestyle='--', alpha=0.3,
               label='target ≥ 0.80')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out = OUT_DIR / f"t_scaling_{agg}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    thumb = OUT_DIR / f"t_scaling_{agg}.thumb.png"
    fig.savefig(thumb, dpi=48, bbox_inches='tight')
    print(f"wrote {out}")


def main():
    for agg in ("mean_pool", "last_position"):
        make_plot(agg)


if __name__ == "__main__":
    main()
