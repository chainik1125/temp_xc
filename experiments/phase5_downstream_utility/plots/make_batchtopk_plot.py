"""BatchTopK bar charts — Figure 4 for the paper.

Two plots per aggregation:

  1. **Headline BatchTopK bar** — same format as Figure 1/2 but filtered
     to archs ending in `_batchtopk`. Sorted by AUC desc, coloured by
     family.
  2. **Paired TopK vs BatchTopK** — for each arch that has BOTH a TopK
     and a BatchTopK row, two bars side-by-side so the Δ is visually
     obvious.

Reads per-task results from `probing_results.jsonl` directly (does NOT
require `headline_summary_*.json`). Writes to `results/plots/`.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from experiments.phase5_downstream_utility.plots.make_headline_plot import (  # noqa: E402
    FLAVOR_COLORS,
    flavor_of,
)
from src.plotting.save_figure import save_figure  # noqa: E402

JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
PLOTS = REPO / "experiments/phase5_downstream_utility/results/plots"
HEADLINE_K = 5


def _load_last_write(aggregation: str, seed: int = 42, k_feat: int = HEADLINE_K):
    """{arch: {task: test_auc}} using last-write-wins per (arch, task)."""
    by_key: dict[tuple[str, str], float] = {}
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != aggregation or r.get("k_feat") != k_feat:
                continue
            rid = r["run_id"]
            if rid.startswith("baseline"):
                continue
            if not rid.endswith(f"__seed{seed}"):
                continue
            auc = r.get("test_auc")
            if auc is None:
                continue
            arch = rid.rsplit(f"__seed{seed}", 1)[0]
            by_key[(arch, r["task_name"])] = float(auc)
    per_arch: dict[str, dict[str, float]] = defaultdict(dict)
    for (arch, t), auc in by_key.items():
        per_arch[arch][t] = auc
    return per_arch


def _load_baselines(aggregation: str, k_feat: int = HEADLINE_K) -> dict[str, float]:
    """{baseline_arch: mean_auc across tasks} for baseline rows."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != aggregation or r.get("k_feat") != k_feat:
                continue
            rid = r["run_id"]
            if not rid.startswith("baseline"):
                continue
            auc = r.get("test_auc")
            if auc is None:
                continue
            out[rid][r["task_name"]] = float(auc)
    return {b: st.mean(d.values()) for b, d in out.items() if len(d) >= 30}


def _mean_std(d: dict[str, float]) -> tuple[float, float, int]:
    vals = list(d.values())
    if not vals:
        return 0.0, 0.0, 0
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def _baseof(bt_arch: str) -> str:
    return bt_arch.removesuffix("_batchtopk")


def headline_batchtopk_bar(per_arch: dict, baselines: dict, aggregation: str,
                           out_path: Path) -> None:
    """Bar chart of only BatchTopK archs, sorted by AUC desc, coloured by family.

    Matches Figure 1/2 styling.
    """
    bt_archs = [a for a in per_arch if a.endswith("_batchtopk")
                and len(per_arch[a]) >= 30]
    if not bt_archs:
        print(f"[batchtopk plot] {aggregation}: no BatchTopK archs found")
        return
    stats = [(a, *_mean_std(per_arch[a])) for a in bt_archs]
    stats.sort(key=lambda r: -r[1])  # by mean desc

    labels = [r[0] for r in stats]
    means = np.array([r[1] for r in stats])
    stds = np.array([r[2] for r in stats])

    # Colour by base family (strip _batchtopk suffix for flavor_of).
    flavors = [flavor_of(_baseof(a)) for a in labels]
    colors = [FLAVOR_COLORS[f] for f in flavors]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors,
           edgecolor="#333", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"mean AUC (k={HEADLINE_K})")
    ax.set_title(
        f"Phase 5: BatchTopK sparse-probing AUC by arch "
        f"[{aggregation}] — sorted by score, coloured by family"
    )
    ax.set_ylim(0.5, 1.0)

    used = []
    for f in flavors:
        if f not in used:
            used.append(f)
    flav_handles = [Patch(facecolor=FLAVOR_COLORS[f], label=f) for f in used]
    leg_flav = ax.legend(handles=flav_handles, title="family",
                         loc="upper right", fontsize=8, title_fontsize=8,
                         framealpha=0.9)
    ax.add_artist(leg_flav)

    bh = []
    baseline_colors = {"baseline_last_token_lr": "black",
                       "baseline_attn_pool": "gray"}
    for ba, m in baselines.items():
        bc = baseline_colors.get(ba, "black")
        line = ax.axhline(m, color=bc, ls="--", lw=1.2,
                          label=f"{ba.replace('baseline_', '')} ({m:.3f})")
        bh.append(line)
    if bh:
        ax.legend(handles=bh, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def paired_topk_batchtopk(per_arch: dict, aggregation: str, out_path: Path) -> None:
    """Grouped bar: TopK vs BatchTopK per base arch. One pair per arch that
    has both counterparts. Sorted by TopK mean AUC desc (so the visual order
    mirrors Figure 1/2)."""
    pairs = []
    for a in per_arch:
        if a.endswith("_batchtopk"):
            continue
        bt = f"{a}_batchtopk"
        if bt in per_arch and len(per_arch[bt]) >= 30 and len(per_arch[a]) >= 30:
            pairs.append((a, bt))
    if not pairs:
        print(f"[paired plot] {aggregation}: no paired TopK/BatchTopK archs")
        return
    pairs.sort(key=lambda p: -st.mean(per_arch[p[0]].values()))

    labels = [p[0] for p in pairs]
    topk_means = [st.mean(per_arch[p[0]].values()) for p in pairs]
    topk_stds = [st.stdev(per_arch[p[0]].values()) if len(per_arch[p[0]]) > 1
                 else 0.0 for p in pairs]
    bt_means = [st.mean(per_arch[p[1]].values()) for p in pairs]
    bt_stds = [st.stdev(per_arch[p[1]].values()) if len(per_arch[p[1]]) > 1
               else 0.0 for p in pairs]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(labels) + 2), 5.5))
    ax.bar(x - w/2, topk_means, width=w, yerr=topk_stds, capsize=3,
           label="TopK", color="#4c72b0", edgecolor="#222", linewidth=0.4)
    ax.bar(x + w/2, bt_means, width=w, yerr=bt_stds, capsize=3,
           label="BatchTopK", color="#dd8452", edgecolor="#222", linewidth=0.4)

    # Annotate Δ above each pair
    for i, (tm, bm) in enumerate(zip(topk_means, bt_means)):
        d = bm - tm
        colour = "green" if d > 0 else "red" if d < -0.005 else "black"
        ax.annotate(f"{d:+.3f}", xy=(x[i], max(tm, bm) + 0.01),
                    ha="center", fontsize=7, color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"mean AUC (k={HEADLINE_K})")
    ax.set_title(
        f"Phase 5: TopK vs BatchTopK pairwise — [{aggregation}] "
        "(label annotates Δ = BatchTopK − TopK)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.55, max(max(topk_means), max(bt_means)) + 0.05)
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for agg in ["last_position", "mean_pool"]:
        per_arch = _load_last_write(agg)
        baselines = _load_baselines(agg)
        headline_batchtopk_bar(
            per_arch, baselines, agg,
            PLOTS / f"batchtopk_bar_k5_{agg}_auc.png",
        )
        paired_topk_batchtopk(
            per_arch, agg,
            PLOTS / f"batchtopk_paired_k5_{agg}_auc.png",
        )
    print("done")


if __name__ == "__main__":
    main()
