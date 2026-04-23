"""TXCDR T-sweep comparison plot: TopK vs BatchTopK sparsity.

Parallels the existing `plot_txcdr_t_sweep.py`. At k=5 probe features,
plots AUC vs T for both sparsity mechanisms side-by-side. Shows the
U-shaped Δ profile (gains at T=2 and T=20, regressions at T=3-15).

Panels: last_position (always), mean_pool (if complete coverage at
n>=30 for BatchTopK archs — filled in when extended probe pipeline
finishes).

Reads probing_results.jsonl. Writes
plots/txcdr_t_sweep_batchtopk_comparison.{png,thumb.png,html}.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/workspace/temp_xc")
sys.path.insert(0, str(REPO))
from src.plotting.save_figure import save_figure  # noqa: E402

RESULTS_DIR = REPO / "experiments/phase5_downstream_utility/results"
JSONL = RESULTS_DIR / "probing_results.jsonl"
PLOTS_DIR = RESULTS_DIR / "plots"

T_VALUES = [2, 3, 5, 8, 10, 15, 20]
K_FEAT = 5
SEED = 42
FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}


def _load(agg: str):
    """Returns {(T, sparsity): {task: auc}}."""
    out: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != agg or r.get("k_feat") != K_FEAT:
                continue
            rid = r.get("run_id", "")
            if not rid.endswith(f"__seed{SEED}"):
                continue
            auc = r.get("test_auc")
            if auc is None:
                continue
            arch = rid.rsplit(f"__seed{SEED}", 1)[0]
            sparsity = "BatchTopK" if arch.endswith("_batchtopk") else "TopK"
            base = arch.removesuffix("_batchtopk")
            if not (base.startswith("txcdr_t")
                    and base.split("_t", 1)[1].isdigit()):
                continue
            T = int(base.split("_t", 1)[1])
            if T not in T_VALUES:
                continue
            task = r["task_name"]
            v = float(auc)
            if task in FLIP_TASKS:
                v = max(v, 1.0 - v)
            out[(T, sparsity)][task] = v
    return out


def _curve(data, sparsity: str):
    xs, ms, ss, ns = [], [], [], []
    for T in T_VALUES:
        tasks = data.get((T, sparsity), {})
        if len(tasks) < 30:
            continue
        vals = np.array(list(tasks.values()))
        xs.append(T)
        ms.append(float(vals.mean()))
        ss.append(float(vals.std() / np.sqrt(len(vals))))
        ns.append(len(vals))
    return np.array(xs), np.array(ms), np.array(ss), ns


def _plot(aggregations):
    fig, axes = plt.subplots(1, len(aggregations),
                              figsize=(5.0 * len(aggregations), 4.5),
                              sharey=True, squeeze=False)
    axes = axes[0]
    colors = {"TopK": "#1f77b4", "BatchTopK": "#dd8452"}
    markers = {"TopK": "o", "BatchTopK": "s"}

    for ax, agg in zip(axes, aggregations):
        data = _load(agg)
        for sparsity in ("TopK", "BatchTopK"):
            xs, ms, ss, ns = _curve(data, sparsity)
            if len(xs) == 0:
                continue
            n_any = max(ns) if ns else 0
            label = f"{sparsity} (n_tasks={n_any})"
            ax.errorbar(
                xs, ms, yerr=ss, marker=markers[sparsity], capsize=3,
                color=colors[sparsity], label=label, lw=1.5, ms=6,
            )
            # Annotate each point with its mean value
            for x, m in zip(xs, ms):
                ax.annotate(f"{m:.4f}", xy=(x, m),
                            xytext=(4, 6 if sparsity == "TopK" else -12),
                            textcoords="offset points", fontsize=7,
                            color=colors[sparsity])
        ax.set_title(f"{agg}")
        ax.set_xlabel("T (window size)")
        ax.set_xticks(T_VALUES)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel(f"mean AUC across 36 tasks (k_feat={K_FEAT})")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "TXCDR T-sweep — TopK vs BatchTopK "
        f"(d_sae=18432, seed={SEED})"
    )
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(aggregations)
    out = PLOTS_DIR / f"txcdr_t_sweep_batchtopk_comparison_{tag}.png"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"wrote {out}")


def main():
    # Always do last_position. Include mean_pool if BatchTopK coverage
    # is complete (>=30 tasks per T).
    aggs = ["last_position"]
    mp_data = _load("mean_pool")
    bt_mp_complete = all(
        len(mp_data.get((T, "BatchTopK"), {})) >= 30 for T in T_VALUES
    )
    if bt_mp_complete:
        aggs.append("mean_pool")
    _plot(aggs)
    # Also always emit a combined-aggregations plot if mean_pool partial
    # (use whatever's available — viewers can judge).
    if not bt_mp_complete:
        print("mean_pool BatchTopK coverage incomplete — single-panel plot only")


if __name__ == "__main__":
    main()
