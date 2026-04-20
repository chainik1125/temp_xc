"""TXCDR T-sweep plot: AUC vs T × k × aggregation.

Motivation: the 19-arch headline (T ∈ {5, 20}) shows:
  - last_position:  txcdr_t5 (0.782) > txcdr_t20 (0.750)  — T↑ hurts
  - full_window:    txcdr_t5 (0.726) < txcdr_t20 (0.752)  — T↑ helps

With only two T points the story is under-supported. This plot fills
T ∈ {2, 3, 5, 8, 10, 15, 20} and reports mean AUC over the 36 probing
tasks at each k, separately for the two aggregations.

Mechanism hypotheses:
  - last_position: T↑ dilutes the shared window latent across more
    positions → less per-token signal at the final slot.
  - full_window:  pool size K × d_sae where K = tail − T + 1 (tail=20).
    Larger T → fewer slides → smaller pool → less overfitting of the
    top-k-class-sep selector at small k. At T=tail, K=1 and
    full_window == last_position trivially.

Reads probing_results.jsonl, writes plots/txcdr_t_sweep_{metric}.png.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.save_figure import save_figure


REPO = Path("/workspace/temp_xc")
RESULTS_DIR = REPO / "experiments/phase5_downstream_utility/results"
JSONL = RESULTS_DIR / "probing_results.jsonl"
PLOTS_DIR = RESULTS_DIR / "plots"

T_VALUES = [2, 3, 5, 8, 10, 15, 20]
K_VALUES = [1, 2, 5, 20]
AGGREGATIONS = ["last_position", "full_window"]
FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}


def _load_records() -> list[dict]:
    records: list[dict] = []
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def _aggregate(records, metric: str):
    """Returns {(T, k, aggregation): {task: value}}."""
    key = "test_auc" if metric == "auc" else "test_acc"
    out: dict[tuple[int, int, str], dict[str, float]] = defaultdict(dict)
    for r in records:
        if r.get("error") or r.get(key) is None:
            continue
        arch = r.get("arch", "")
        if not (arch.startswith("txcdr_t") and arch.split("_t", 1)[1].isdigit()):
            continue
        T = int(arch.split("_t", 1)[1])
        if T not in T_VALUES:
            continue
        k = r.get("k_feat")
        if k not in K_VALUES:
            continue
        agg = r.get("aggregation")
        if agg not in AGGREGATIONS:
            continue
        task = r.get("task_name")
        v = float(r[key])
        if task in FLIP_TASKS:
            v = max(v, 1.0 - v)
        out[(T, k, agg)][task] = v
    return out


def _curve(agg_data, k: int, aggregation: str):
    """Return (T_xs, means, stds, ns) sorted by T."""
    xs, ms, ss, ns = [], [], [], []
    for T in T_VALUES:
        tasks = agg_data.get((T, k, aggregation), {})
        if not tasks:
            continue
        vals = np.array(list(tasks.values()))
        xs.append(T)
        ms.append(float(vals.mean()))
        ss.append(float(vals.std() / np.sqrt(len(vals))))  # stderr
        ns.append(len(vals))
    return np.array(xs), np.array(ms), np.array(ss), np.array(ns)


def _plot(metric: str):
    records = _load_records()
    agg_data = _aggregate(records, metric)

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(4.2 * len(K_VALUES), 4.2),
                              sharey=True)
    colors = {"last_position": "#1f77b4", "full_window": "#d62728"}

    for ax, k in zip(axes, K_VALUES):
        for aggregation in AGGREGATIONS:
            xs, ms, ss, ns = _curve(agg_data, k, aggregation)
            if len(xs) == 0:
                continue
            label = f"{aggregation} (n={int(ns.max())})"
            ax.errorbar(
                xs, ms, yerr=ss, marker="o", capsize=3,
                color=colors[aggregation], label=label,
            )
        ax.set_title(f"k = {k}")
        ax.set_xlabel("T (window size)")
        ax.set_xticks(T_VALUES)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel(f"Mean {metric.upper()} across probing tasks")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"TXCDR T-sweep — {metric.upper()} vs window size T "
        f"(tail=20, d_sae=18432, seed=42)"
    )
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"txcdr_t_sweep_{metric}.png"
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Also write a summary JSON so the numbers land in results/.
    summary = {
        metric: {
            f"T{T}_k{k}_{aggregation}": {
                "mean": float(np.mean(list(tasks.values()))),
                "std": float(np.std(list(tasks.values()))),
                "n": len(tasks),
            }
            for (T, k, aggregation), tasks in agg_data.items()
        }
    }
    (RESULTS_DIR / f"txcdr_t_sweep_summary_{metric}.json").write_text(
        json.dumps(summary, indent=2)
    )


def main():
    _plot("auc")
    _plot("acc")


if __name__ == "__main__":
    main()
