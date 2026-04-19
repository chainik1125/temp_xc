"""Aggregate probing_results.jsonl into the Phase 5 headline bar chart.

Output:
    results/plots/headline_bar_k5.png + .thumb.png
    results/plots/per_task_k5.png + .thumb.png
    results/plots/headline_summary.json    (mean AUC per arch across tasks)

Usage:
    PYTHONPATH=/workspace/temp_xc \\
      .venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py
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
HEADLINE_K = 5


def _load_records() -> list[dict]:
    if not JSONL.exists():
        raise FileNotFoundError(f"{JSONL} not found — run probing first")
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


def aggregate(records: list[dict]) -> dict:
    """Return a nested dict: {arch: {task: auc_at_k5}}."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        if r.get("error"):
            continue
        if r.get("test_auc") is None:
            continue
        k = r.get("k_feat")
        if k is not None and k != HEADLINE_K:
            continue
        arch = r.get("arch")
        task = r.get("task_name")
        out[arch][task] = float(r["test_auc"])
    return out


def write_summary(agg: dict) -> dict:
    rows = {}
    for arch, tasks in agg.items():
        vals = list(tasks.values())
        if not vals:
            continue
        rows[arch] = {
            "mean_auc": float(np.mean(vals)),
            "std_auc": float(np.std(vals)),
            "n_tasks": len(vals),
            "per_task": tasks,
        }
    return rows


def headline_bar(summary: dict, out_path: Path) -> None:
    """Bar chart — mean AUC per arch, baselines shown as lines."""
    ordered_archs = [
        a for a in [
            "topk_sae", "stacked_t5", "stacked_t20",
            "shared_perpos_t5", "txcdr_t5", "txcdr_t20",
            "matryoshka_t5", "mlc", "tfa", "tfa_pos",
        ] if a in summary
    ]
    baseline_archs = [
        a for a in [
            "baseline_last_token_lr", "baseline_attn_pool",
        ] if a in summary
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ordered_archs))
    means = [summary[a]["mean_auc"] for a in ordered_archs]
    stds = [summary[a]["std_auc"] for a in ordered_archs]
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=["C0"] * len(ordered_archs))
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_archs, rotation=30, ha="right")
    ax.set_ylabel(f"mean AUC (k={HEADLINE_K})")
    ax.set_title("Phase 5 headline: sparse-probing AUC by architecture")
    ax.set_ylim(0.5, 1.0)

    # Baselines as horizontal lines
    colors = {"baseline_last_token_lr": "red",
              "baseline_attn_pool": "purple"}
    for ba in baseline_archs:
        m = summary[ba]["mean_auc"]
        ax.axhline(m, color=colors.get(ba, "black"), ls="--",
                   label=f"{ba.replace('baseline_', '')} ({m:.3f})")
    ax.legend(loc="lower right")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.3f}", ha="center", fontsize=8)

    save_figure(fig, str(out_path))
    plt.close(fig)


def per_task_heatmap(agg: dict, out_path: Path) -> None:
    ordered_archs = [
        a for a in [
            "topk_sae", "stacked_t5", "stacked_t20",
            "shared_perpos_t5", "txcdr_t5", "txcdr_t20",
            "matryoshka_t5", "mlc", "tfa", "tfa_pos",
        ] if a in agg
    ]
    tasks = sorted({t for a in ordered_archs for t in agg[a].keys()})
    if not tasks:
        return
    mat = np.zeros((len(ordered_archs), len(tasks)))
    for i, a in enumerate(ordered_archs):
        for j, t in enumerate(tasks):
            mat[i, j] = agg[a].get(t, np.nan)

    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 0.3), len(ordered_archs) * 0.4 + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ordered_archs)))
    ax.set_yticklabels(ordered_archs)
    fig.colorbar(im, ax=ax, label=f"test AUC (k={HEADLINE_K})")
    ax.set_title("Per-task sparse-probing AUC")
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    records = _load_records()
    agg = aggregate(records)
    summary = write_summary(agg)

    out_json = RESULTS_DIR / "headline_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"{len(summary)} archs, summary written to {out_json}")
    for arch, row in sorted(summary.items(), key=lambda x: -x[1]["mean_auc"]):
        print(f"  {arch:30s} mean={row['mean_auc']:.4f} (n={row['n_tasks']})")

    headline_bar(summary, PLOTS_DIR / "headline_bar_k5.png")
    per_task_heatmap(agg, PLOTS_DIR / "per_task_k5.png")


if __name__ == "__main__":
    main()
