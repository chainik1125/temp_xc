"""Consolidated T-sweep 4-panel plot: {TopK, BatchTopK} × {last_position, mean_pool}.

Supersedes `plot_txcdr_t_sweep.py` and `plot_txcdr_t_sweep_batchtopk.py`
for the summary.md §T-sweep matrix section (A6 of 2026-04-23 handover).

Data source: probing_results.jsonl, seed=42, k_feat=5.
Families plotted per panel (same color scheme across panels):
- vanilla TXCDR at T ∈ all available values
- agentic_txc_02 T-sweep (overlay on TopK panels only)
- BatchTopK at matching T values (on the right panels)

T > 20 is plotted at last_position only (mean_pool infeasible;
LAST_N=20 → K = LN − T + 1 ≤ 0).

Writes plots/txcdr_t_sweep_4panel.{png,thumb.png}.
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

T_VALUES = [2, 3, 5, 8, 10, 15, 20, 24, 28, 32, 36]
K_FEAT = 5
SEED = 42
FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}


def _load(agg: str):
    """Returns {(family, T): {task: auc}} where family ∈ {'TopK', 'BatchTopK', 'agentic_TopK'}."""
    out: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
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
            # Classify
            is_batchtopk = arch.endswith("_batchtopk")
            core = arch.removesuffix("_batchtopk") if is_batchtopk else arch
            if core.startswith("txcdr_t") and core[len("txcdr_t"):].isdigit():
                T = int(core[len("txcdr_t"):])
                family = "BatchTopK" if is_batchtopk else "TopK"
            elif core.startswith("agentic_txc_02_t") and core[len("agentic_txc_02_t"):].isdigit():
                T = int(core[len("agentic_txc_02_t"):])
                family = "agentic_BatchTopK" if is_batchtopk else "agentic_TopK"
            elif core == "agentic_txc_02":
                T = 5
                family = "agentic_BatchTopK" if is_batchtopk else "agentic_TopK"
            else:
                continue
            if T not in T_VALUES:
                continue
            v = float(auc)
            if r.get("task_name") in FLIP_TASKS:
                v = max(v, 1.0 - v)
            out[(family, T)][r["task_name"]] = v
    return out


def _curve(data, family: str, min_n: int = 30):
    xs, ms, ss, ns = [], [], [], []
    for T in T_VALUES:
        tasks = data.get((family, T), {})
        if len(tasks) < min_n:
            continue
        vals = np.array(list(tasks.values()))
        xs.append(T)
        ms.append(float(vals.mean()))
        ss.append(float(vals.std() / np.sqrt(len(vals))))
        ns.append(len(vals))
    return np.array(xs), np.array(ms), np.array(ss), ns


def _plot():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    panels = [
        ("last_position", "TopK",      axes[0][0], "#1f77b4", "o"),
        ("last_position", "BatchTopK", axes[0][1], "#dd8452", "s"),
        ("mean_pool",     "TopK",      axes[1][0], "#1f77b4", "o"),
        ("mean_pool",     "BatchTopK", axes[1][1], "#dd8452", "s"),
    ]
    data_by_agg = {"last_position": _load("last_position"),
                   "mean_pool": _load("mean_pool")}
    for agg, family, ax, color, marker in panels:
        d = data_by_agg[agg]
        xs, ms, ss, ns = _curve(d, family)
        if len(xs) > 0:
            ax.errorbar(
                xs, ms, yerr=ss, marker=marker, capsize=3,
                color=color, label=f"vanilla TXCDR ({family})  n={max(ns) if ns else 0}",
                lw=1.8, ms=7,
            )
            for x, m in zip(xs, ms):
                ax.annotate(f"{m:.3f}", xy=(x, m), xytext=(4, 7),
                            textcoords="offset points", fontsize=7, color=color)
        # Overlay agentic T-sweep on TopK panels
        if family == "TopK":
            afamily = "agentic_TopK"
            axs, ams, ass, ans = _curve(d, afamily)
            if len(axs) > 0:
                ax.errorbar(
                    axs, ams, yerr=ass, marker="D", capsize=3,
                    color="#2ca02c",
                    label=f"agentic_txc_02 ({afamily.split('_')[1]})  n={max(ans) if ans else 0}",
                    lw=1.5, ms=6, linestyle="--",
                )
                for x, m in zip(axs, ams):
                    ax.annotate(f"{m:.3f}", xy=(x, m), xytext=(-4, -12),
                                textcoords="offset points", fontsize=7, color="#2ca02c",
                                ha="right")
        elif family == "BatchTopK":
            afamily = "agentic_BatchTopK"
            axs, ams, ass, ans = _curve(d, afamily)
            if len(axs) > 0:
                ax.errorbar(
                    axs, ams, yerr=ass, marker="D", capsize=3,
                    color="#2ca02c",
                    label=f"agentic_txc_02 (BatchTopK)  n={max(ans) if ans else 0}",
                    lw=1.5, ms=6, linestyle="--",
                )
        ax.set_title(f"{agg} × {family}")
        ax.set_xlabel("T (window size)")
        ax.set_xticks(T_VALUES)
        ax.set_xticklabels([str(t) if t in [2, 5, 10, 15, 20, 28, 36] else "" for t in T_VALUES])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0][0].set_ylabel(f"mean AUC across 36 tasks (k_feat={K_FEAT})")
    axes[1][0].set_ylabel(f"mean AUC across 36 tasks (k_feat={K_FEAT})")
    fig.suptitle(
        "TXCDR T-sweep matrix: rows = aggregation, cols = sparsity "
        f"(d_sae=18432, seed={SEED}, k_feat={K_FEAT})", y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "txcdr_t_sweep_4panel.png"
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    _plot()
