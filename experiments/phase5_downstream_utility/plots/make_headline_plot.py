"""Aggregate probing_results.jsonl into Phase 5 headline bar charts.

Generates one bar + one heatmap per (aggregation, metric) combo:
    headline_{bar,per_task}_k5_{aggregation}_{metric}.png

where aggregation ∈ {last_position, mean_pool} and metric ∈ {auc, acc}.

`full_window` is deprecated as of 2026-04-20 (redundant with mean_pool;
mean_pool matches SAEBench/Kantamneni convention). JSONL rows are kept
for reproducibility but are no longer plotted.

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


REPO = Path("/workspace/temp_xc")
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
    "txcdr_t5", "txcdr_t20",
    "txcdr_shared_dec_t5", "txcdr_shared_enc_t5",
    "txcdr_tied_t5", "txcdr_pos_t5", "txcdr_causal_t5",
    "txcdr_block_sparse_t5", "txcdr_lowrank_dec_t5",
    "txcdr_rank_k_dec_t5",
    "matryoshka_t5", "temporal_contrastive",
    "time_layer_crosscoder_t5",
    "mlc", "mlc_contrastive",
    # Phase 5.7 Part-B α sweep winners
    "matryoshka_txcdr_contrastive_t5_alpha100",
    "mlc_contrastive_alpha100",
    # Phase 5.7 agentic multi-scale winners
    "agentic_txc_02", "agentic_mlc_08",
    # TFA dual probing: z_novel vs z_novel+z_pred
    "tfa_small", "tfa_small_full",
    "tfa_pos_small", "tfa_pos_small_full",
    # Phase 5.7 experiment (i): full-size TFA (d_sae=18432, seq_len=128)
    "tfa_big", "tfa_big_full",
    "tfa_pos_big", "tfa_pos_big_full",
    # Phase 5.7 experiment (ii): BatchTopK apples-to-apples
    "txcdr_t5_batchtopk", "mlc_batchtopk",
    "agentic_txc_02_batchtopk", "agentic_mlc_08_batchtopk",
    # Phase 5.7 BatchTopK extended scope (2026-04-23)
    "topk_sae_batchtopk",
    "matryoshka_t5_batchtopk",
    "matryoshka_txcdr_contrastive_t5_alpha100_batchtopk",
    "mlc_contrastive_batchtopk",
    "mlc_contrastive_alpha100_batchtopk",
    "time_layer_crosscoder_t5_batchtopk",
    "stacked_t5_batchtopk", "stacked_t20_batchtopk",
    "txcdr_t2_batchtopk", "txcdr_t3_batchtopk",
    "txcdr_t8_batchtopk", "txcdr_t10_batchtopk",
    "txcdr_t15_batchtopk", "txcdr_t20_batchtopk",
    "agentic_txc_02_t2_batchtopk",
    "agentic_txc_02_t3_batchtopk",
    "agentic_txc_02_t8_batchtopk",
    # Phase 5.7 experiment (iii): T-sweep on agentic_txc_02
    "agentic_txc_02_t2", "agentic_txc_02_t3",
    "agentic_txc_02_t6", "agentic_txc_02_t7",
    "agentic_txc_02_t8", "agentic_txc_02_t10",
    "agentic_txc_02_t15", "agentic_txc_02_t20",
    # Phase 5.7 detailed T-sweep (2026-04-24): T=6, 7
    "txcdr_t6", "txcdr_t7", "txcdr_t6_batchtopk", "txcdr_t7_batchtopk",
    # Part B H1 (conv encoder — fails T-scaling)
    "conv_txcdr_t5", "conv_txcdr_t10", "conv_txcdr_t15",
    "conv_txcdr_t20", "conv_txcdr_t30",
    # Part B H7 (anti-dead + multi-scale InfoNCE — TXC leader)
    "phase57_partB_h7_bare_multiscale",
    # Part B H8 (anti-dead + multi-distance InfoNCE — NEW CHAMPION at mp)
    "phase57_partB_h8_bare_multidistance",
    # Part B H8 T-sweep (paper-critical T-scaling test)
    "phase57_partB_h8_bare_multidistance_t6",
    "phase57_partB_h8_bare_multidistance_t7",
    "phase57_partB_h8_bare_multidistance_t8",
    "phase57_partB_h8_bare_multidistance_t10",
    "phase57_partB_h8_bare_multidistance_t15",
    "phase57_partB_h8_bare_multidistance_t20",
    "phase57_partB_h8_bare_multidistance_t30",
    # Part B H8a shift ablation
    "phase57_partB_h8a_shifts1",
    "phase57_partB_h8a_shifts123",
    "phase57_partB_h8a_shifts1234",
    "phase57_partB_h8a_shifts124",
    "phase57_partB_h8a_shifts2",
    "phase57_partB_h8a_shifts4",
    "phase57_partB_h8a_shifts123_uniform",
    # Long-range shift ablation (user-requested):
    "phase57_partB_h8a_shifts5",
    "phase57_partB_h8a_shifts_10",
    "phase57_partB_h8a_shifts_20",
    "phase57_partB_h8a_shifts1_5",
    "phase57_partB_h8a_shifts1_5_10",
    "phase57_partB_h8a_shifts1_10",
    "phase57_partB_h8a_shifts1_2_5_10",
    # Part B H9 (feature-nested matryoshka)
    "feature_nested_matryoshka_t5",
    "feature_nested_matryoshka_t5_contrastive",
    # Part B H10/H12 encoder ablations
    "txc_shared_relu_sum_pos_t5",
    "txc_shared_relu_sum_nopos_t5",
    "txc_shared_concat_two_layer_t5",
    # Part B H3 log-matryoshka T-sweep
    "log_matryoshka_t5", "log_matryoshka_t10", "log_matryoshka_t15",
    "log_matryoshka_t20", "log_matryoshka_t30",
    # P0c MLC + anti-dead (fairness counterparts)
    "mlc_bare_antidead",
    "mlc_bare_matryoshka_contrastive_antidead",
    "mlc_bare_multiscale_antidead",
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
    assert aggregation in ("last_position", "full_window", "mean_pool")
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


FLAVOR_COLORS = {
    "txc":        "#1f77b4",   # blue        — TXCDR family (incl. matryoshka + agentic_txc)
    "mlc":        "#2ca02c",   # green       — MLC family (incl. agentic_mlc)
    "tfa":        "#d62728",   # red         — TFA (novel and novel+pred)
    "topk_sae":   "#ff7f0e",   # orange      — single-token TopK SAE
    "stacked":    "#9467bd",   # purple      — per-position stacked-t
    "time_layer": "#8c564b",   # brown       — layer-axis crosscoder
    "temporal":   "#e377c2",   # pink        — temporal_contrastive (single-token contrastive SAE)
}


def flavor_of(arch: str) -> str:
    """Map an arch name to one of the flavor keys in FLAVOR_COLORS."""
    if arch == "topk_sae":
        return "topk_sae"
    if arch.startswith("stacked_"):
        return "stacked"
    if arch.startswith("tfa_"):
        return "tfa"
    if arch.startswith("time_layer_"):
        return "time_layer"
    if arch == "temporal_contrastive":
        return "temporal"
    if arch == "mlc" or arch.startswith("mlc_") or arch.startswith("agentic_mlc_"):
        return "mlc"
    # Everything else that's TXCDR-shaped: vanilla TXCDR variants, matryoshka
    # variants (incl. matryoshka_txcdr_*), and agentic TXC cycles.
    if arch.startswith("txcdr_") or arch.startswith("matryoshka_") or arch.startswith("agentic_txc_"):
        return "txc"
    return "txc"   # safe default for any un-mapped arch


def _headline_bar(
    summary: dict, out_path: Path,
    metric: str, aggregation: str,
) -> None:
    candidates = [a for a in ORDERED_ARCHS if a in summary]
    baselines = [a for a in BASELINE_ARCHS if a in summary]

    # Sort by mean metric descending so the chart is visually a ranking.
    candidates.sort(key=lambda a: -summary[a][f"mean_{metric}"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(candidates))
    means = [summary[a][f"mean_{metric}"] for a in candidates]
    stds = [summary[a][f"std_{metric}"] for a in candidates]
    flavors = [flavor_of(a) for a in candidates]
    colors = [FLAVOR_COLORS[f] for f in flavors]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="#333", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(candidates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"mean {metric.upper()} (k={HEADLINE_K})")
    ax.set_title(
        f"Phase 5: sparse-probing {metric.upper()} by arch "
        f"[{aggregation}] — sorted by score, coloured by family"
    )
    ax.set_ylim(0.5, 1.0)

    # Two legends: (1) flavor swatches in the upper-right; (2) attn-pool /
    # last-token LR dashed baselines in lower-right. Keep them distinct so
    # viewers can read both without overlap.
    used_flavors = []
    for f in flavors:
        if f not in used_flavors:
            used_flavors.append(f)
    from matplotlib.patches import Patch
    flavor_handles = [Patch(facecolor=FLAVOR_COLORS[f], label=f) for f in used_flavors]
    leg_flavor = ax.legend(handles=flavor_handles, title="family",
                           loc="upper right", fontsize=8, title_fontsize=8,
                           framealpha=0.9)
    ax.add_artist(leg_flavor)

    baseline_colors = {"baseline_last_token_lr": "black",
                       "baseline_attn_pool": "gray"}
    baseline_handles = []
    for ba in baselines:
        m = summary[ba][f"mean_{metric}"]
        bc = baseline_colors.get(ba, "black")
        line = ax.axhline(m, color=bc, ls="--", lw=1.2,
                          label=f"{ba.replace('baseline_', '')} ({m:.3f})")
        baseline_handles.append(line)
    if baseline_handles:
        ax.legend(handles=baseline_handles, loc="lower right", fontsize=8,
                  framealpha=0.9)

    # Value labels on top of each bar
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.3f}", ha="center", fontsize=7, rotation=90,
                va="bottom")

    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def _per_task_heatmap(
    agg: dict, out_path: Path,
    metric: str, aggregation: str,
) -> None:
    ordered = [a for a in ORDERED_ARCHS if a in agg]
    # Sort heatmap rows by mean across tasks (descending), matching the
    # bar-chart ordering.
    ordered.sort(key=lambda a: -float(np.mean(list(agg[a].values()))))
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
        # full_window deprecated 2026-04-20 — JSONL rows retained for reproducibility.
        for aggregation in ("last_position", "mean_pool"):
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
