"""Aggregate probing_results.jsonl into Phase 5 headline bar charts.

Generates one bar + one heatmap per (aggregation, metric) combo:
    headline_{bar,per_task}_k5_{aggregation}_{metric}.png

where aggregation ∈ {last_position, full_window} and metric ∈ {auc, acc}.

Also writes `headline_summary.json` at the top level for the default
pair (last_position × auc) for historical compatibility.

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

import os
REPO = Path(os.environ.get("PHASE5_REPO", Path(__file__).resolve().parents[3]))
RESULTS_DIR = REPO / "experiments/phase5_downstream_utility/results"
JSONL = RESULTS_DIR / "probing_results.jsonl"
PLOTS_DIR = RESULTS_DIR / "plots"
HEADLINE_K = 5

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}

# Task-filter sets for comparison against prior work.
# Aniket's 8-dataset SAEBench set (ag_news, amazon_reviews, amazon_reviews_sentiment,
# bias_in_bios_set{1,2,3}, europarl, github_code) — crosses out our 2
# cross-token additions.
ANIKET_DATASET_KEYS = {
    "ag_news", "amazon_reviews", "amazon_reviews_sentiment",
    "bias_in_bios_set1", "bias_in_bios_set2", "bias_in_bios_set3",
    "europarl", "github_code",
}

ORDERED_ARCHS = [
    "topk_sae", "stacked_t5", "stacked_t20",
    "shared_perpos_t5", "txcdr_t5", "txcdr_t20",
    "txcdr_shared_dec_t5", "txcdr_shared_enc_t5",
    "txcdr_tied_t5", "txcdr_pos_t5", "txcdr_causal_t5",
    "matryoshka_t5", "temporal_contrastive",
    "mlc", "tfa", "tfa_pos", "tfa_small", "tfa_pos_small",
]
BASELINE_ARCHS = ["baseline_last_token_lr", "baseline_attn_pool"]


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


def aggregate(
    records: list[dict], aggregation: str, metric: str,
    dataset_filter: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    """{arch: {task: value}} filtered by aggregation and metric.

    metric: "auc" -> uses test_auc (with FLIP_TASKS polarity fix);
            "acc" -> uses test_acc (same flip convention for fairness).
    dataset_filter: optional set of dataset_key values to keep.
    """
    assert aggregation in ("last_position", "full_window")
    assert metric in ("auc", "acc")
    out: dict[str, dict[str, float]] = defaultdict(dict)
    key = "test_auc" if metric == "auc" else "test_acc"
    for r in records:
        if r.get("error"):
            continue
        if r.get(key) is None:
            continue
        if r.get("aggregation") != aggregation:
            continue
        if dataset_filter is not None and r.get("dataset_key") not in dataset_filter:
            continue
        k = r.get("k_feat")
        if k is not None and k != HEADLINE_K:
            continue
        arch = r.get("arch")
        task = r.get("task_name")
        v = float(r[key])
        if task in FLIP_TASKS:
            v = max(v, 1.0 - v)
        out[arch][task] = v
    return out


def write_summary(agg: dict, metric: str) -> dict:
    key_mean = f"mean_{metric}"
    key_std = f"std_{metric}"
    rows = {}
    for arch, tasks in agg.items():
        vals = list(tasks.values())
        if not vals:
            continue
        rows[arch] = {
            key_mean: float(np.mean(vals)),
            key_std: float(np.std(vals)),
            "n_tasks": len(vals),
            "per_task": tasks,
        }
    return rows


def _headline_bar(
    summary: dict, out_path: Path,
    metric: str, aggregation: str,
) -> None:
    ordered = [a for a in ORDERED_ARCHS if a in summary]
    baselines = [a for a in BASELINE_ARCHS if a in summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ordered))
    means = [summary[a][f"mean_{metric}"] for a in ordered]
    stds = [summary[a][f"std_{metric}"] for a in ordered]
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=["C0"] * len(ordered))
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right")
    ax.set_ylabel(f"mean {metric.upper()} (k={HEADLINE_K})")
    ax.set_title(
        f"Phase 5: sparse-probing {metric.upper()} by arch "
        f"[{aggregation}]"
    )
    ax.set_ylim(0.5, 1.0)

    colors = {"baseline_last_token_lr": "red",
              "baseline_attn_pool": "purple"}
    for ba in baselines:
        m = summary[ba][f"mean_{metric}"]
        ax.axhline(m, color=colors.get(ba, "black"), ls="--",
                   label=f"{ba.replace('baseline_', '')} ({m:.3f})")
    ax.legend(loc="lower right")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.3f}", ha="center", fontsize=8)

    save_figure(fig, str(out_path))
    plt.close(fig)


def _per_task_heatmap(
    agg: dict, out_path: Path,
    metric: str, aggregation: str,
) -> None:
    ordered = [a for a in ORDERED_ARCHS if a in agg]
    tasks = sorted({t for a in ordered for t in agg[a].keys()})
    if not tasks or not ordered:
        return
    mat = np.zeros((len(ordered), len(tasks)))
    for i, a in enumerate(ordered):
        for j, t in enumerate(tasks):
            mat[i, j] = agg[a].get(t, np.nan)

    fig, ax = plt.subplots(
        figsize=(max(10, len(tasks) * 0.3), len(ordered) * 0.4 + 2),
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered)
    fig.colorbar(im, ax=ax, label=f"test {metric.upper()} (k={HEADLINE_K})")
    ax.set_title(f"Per-task {metric.upper()} [{aggregation}]")
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    records = _load_records()
    print(f"{len(records)} records loaded")

    # Backwards-compat: preserve historical `headline_summary.json` for the
    # default (last_position × auc) slice.
    default_agg_dict = aggregate(records, "last_position", "auc")
    default_sum = write_summary(default_agg_dict, "auc")
    # Rename to legacy keys for the default slice (mean_auc, std_auc).
    legacy_sum = {}
    for a, row in default_sum.items():
        legacy_sum[a] = {
            "mean_auc": row["mean_auc"],
            "std_auc": row["std_auc"],
            "n_tasks": row["n_tasks"],
            "per_task": row["per_task"],
        }
    (RESULTS_DIR / "headline_summary.json").write_text(
        json.dumps(legacy_sum, indent=2)
    )

    for task_set_name, dataset_filter in (
        ("full", None),
        ("aniket", ANIKET_DATASET_KEYS),
    ):
        for aggregation in ("last_position", "full_window"):
            for metric in ("auc", "acc"):
                agg_dict = aggregate(
                    records, aggregation, metric,
                    dataset_filter=dataset_filter,
                )
                if not agg_dict:
                    print(
                        f"  [skip] {task_set_name} × {aggregation} × {metric}:"
                        f" no records"
                    )
                    continue
                summ = write_summary(agg_dict, metric)
                slug = f"k{HEADLINE_K}_{aggregation}_{metric}_{task_set_name}"
                print(
                    f"\n--- {task_set_name} × {aggregation} × {metric}:"
                    f" {len(summ)} archs ---"
                )
                for a, row in sorted(
                    summ.items(),
                    key=lambda x: -x[1][f"mean_{metric}"],
                ):
                    print(
                        f"  {a:30s} mean={row[f'mean_{metric}']:.4f} "
                        f"(n={row['n_tasks']})"
                    )
                (
                    RESULTS_DIR
                    / f"headline_summary_{aggregation}_{metric}_{task_set_name}.json"
                ).write_text(json.dumps(summ, indent=2))
                _headline_bar(
                    summ, PLOTS_DIR / f"headline_bar_{slug}.png",
                    metric, aggregation,
                )
                _per_task_heatmap(
                    agg_dict, PLOTS_DIR / f"per_task_{slug}.png",
                    metric, aggregation,
                )

    # Back-compat aliases for summary.md's default links (full task set,
    # last_position, AUC).
    import shutil
    for stem_src, stem_dst in [
        (f"headline_bar_k{HEADLINE_K}_last_position_auc_full",
         f"headline_bar_k{HEADLINE_K}"),
        (f"per_task_k{HEADLINE_K}_last_position_auc_full",
         f"per_task_k{HEADLINE_K}"),
    ]:
        for suffix in (".png", ".thumb.png"):
            src = PLOTS_DIR / (stem_src + suffix)
            dst = PLOTS_DIR / (stem_dst + suffix)
            if src.exists():
                shutil.copy(src, dst)


if __name__ == "__main__":
    main()
