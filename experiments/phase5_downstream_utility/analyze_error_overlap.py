"""Error-overlap analysis between the top-performing Phase-5 archs.

Reads per-example predictions from `results/predictions/*.npz` (emitted by
`run_probing.py --save-predictions`). For each pair of archs (at the same
aggregation + k) computes:

  - McNemar's χ² p-value from the discordant 2×2 table.
  - Jaccard of error sets.
  - Per-task "A right, B wrong" fraction.

Emits three plots under `results/plots/` plus a JSON summary.

Usage:
    .venv/bin/python experiments/phase5_downstream_utility/analyze_error_overlap.py \\
        [--aggregation last_position] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from src.plotting.save_figure import save_figure


REPO = Path("/workspace/temp_xc")
RESULTS_DIR = REPO / "experiments/phase5_downstream_utility/results"
PRED_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"

TOP_ARCHS = [
    "mlc__seed42",
    "time_layer_crosscoder_t5__seed42",
    "txcdr_rank_k_dec_t5__seed42",
    "txcdr_t5__seed42",
    "txcdr_tied_t5__seed42",
    "topk_sae__seed42",
    "mlc_contrastive__seed42",
]

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}

FILENAME_RE = re.compile(
    r"^(?P<run_id>.+?)__(?P<agg>last_position|full_window|mean_pool)__(?P<task>.+?)__k(?P<k>\d+)\.npz$"
)


def load_predictions(aggregation: str, k: int) -> dict[str, dict[str, dict]]:
    """{run_id: {task_name: {y_true, y_pred, decision_score}}}"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    if not PRED_DIR.exists():
        return out
    for p in PRED_DIR.iterdir():
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        if m.group("agg") != aggregation or int(m.group("k")) != k:
            continue
        d = np.load(p)
        out[m.group("run_id")][m.group("task")] = {
            "y_true": d["y_true"].astype(np.int32),
            "y_pred": d["y_pred"].astype(np.int32),
            "decision_score": d["decision_score"],
        }
    return out


def _apply_polarity_flip(pred_dict: dict) -> dict:
    """For FLIP tasks, if mean AUC < 0.5 the learnt polarity was inverted —
    flip y_pred + negate decision_score so error sets are comparable across
    runs that happened to pick different polarities on a chance-level task.
    """
    return pred_dict  # No-op for now; we keep raw predictions. FLIP_TASKS
    # correction is a plotting-time concern, not error-set semantics.


def mcnemar_chi2(y_true, y_pred_a, y_pred_b) -> tuple[float, int, int]:
    """Return (p_value, wins_a_over_b, wins_b_over_a).

    Discordant 2x2 counts discordant pairs: b01 = A right & B wrong,
    b10 = A wrong & B right. Use χ² = (|b01 - b10| - 1)² / (b01 + b10)
    with Yates correction (standard).
    """
    a_right = (y_pred_a == y_true).astype(np.int8)
    b_right = (y_pred_b == y_true).astype(np.int8)
    b01 = int(((a_right == 1) & (b_right == 0)).sum())
    b10 = int(((a_right == 0) & (b_right == 1)).sum())
    disc = b01 + b10
    if disc == 0:
        return 1.0, b01, b10
    chi2_stat = (abs(b01 - b10) - 1) ** 2 / disc
    p = 1.0 - chi2.cdf(chi2_stat, df=1)
    return float(p), b01, b10


def error_jaccard(y_true, y_pred_a, y_pred_b) -> float:
    err_a = y_pred_a != y_true
    err_b = y_pred_b != y_true
    union = (err_a | err_b).sum()
    if union == 0:
        return 1.0  # both perfect — trivially identical
    inter = (err_a & err_b).sum()
    return float(inter) / float(union)


def run(aggregation: str, k: int) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(aggregation, k)
    available = [a for a in TOP_ARCHS if a in preds]
    if len(available) < 2:
        print(f"[error_overlap] not enough archs with predictions: {available}")
        return
    # Intersection of task names (each arch must have same task set).
    task_sets = [set(preds[a].keys()) for a in available]
    common_tasks = sorted(set.intersection(*task_sets))
    print(
        f"[error_overlap] {len(available)} archs × {len(common_tasks)} tasks "
        f"@ {aggregation} × k={k}"
    )

    n = len(available)
    jaccard_mat = np.full((n, n), np.nan)
    winsloss_mat = np.full((n, n), np.nan)  # "row wins vs column"
    mcnemar_p_mat = np.full((n, n), np.nan)

    pair_summary: dict[str, dict] = {}

    for i, a in enumerate(available):
        jaccard_mat[i, i] = 1.0
        winsloss_mat[i, i] = 0.0
        mcnemar_p_mat[i, i] = 1.0

    for (i, a), (j, b) in combinations(enumerate(available), 2):
        jac_per_task = []
        winsA_list = []
        winsB_list = []
        mcnemar_ps = []
        for task in common_tasks:
            pa = preds[a][task]
            pb = preds[b][task]
            y = pa["y_true"]
            assert np.array_equal(y, pb["y_true"]), (
                f"y_true mismatch for {a} vs {b} on {task}"
            )
            jac_per_task.append(error_jaccard(y, pa["y_pred"], pb["y_pred"]))
            pval, winsA, winsB = mcnemar_chi2(y, pa["y_pred"], pb["y_pred"])
            winsA_list.append(winsA / max(1, len(y)))
            winsB_list.append(winsB / max(1, len(y)))
            mcnemar_ps.append(pval)

        jmean = float(np.mean(jac_per_task))
        wA = float(np.mean(winsA_list))
        wB = float(np.mean(winsB_list))
        pmin = float(np.min(mcnemar_ps))
        pmedian = float(np.median(mcnemar_ps))
        n_sig_05 = int(sum(1 for p in mcnemar_ps if p < 0.05))
        n_sig_bonf = int(sum(
            1 for p in mcnemar_ps if p < 0.05 / len(common_tasks)
        ))

        jaccard_mat[i, j] = jmean
        jaccard_mat[j, i] = jmean
        winsloss_mat[i, j] = wA
        winsloss_mat[j, i] = wB
        mcnemar_p_mat[i, j] = pmedian
        mcnemar_p_mat[j, i] = pmedian

        pair_summary[f"{a}__VS__{b}"] = {
            "mean_jaccard_errors": jmean,
            "mean_winsA_over_B": wA,
            "mean_winsB_over_A": wB,
            "mcnemar_pval_median": pmedian,
            "mcnemar_pval_min": pmin,
            "n_tasks_significant_at_0.05": n_sig_05,
            "n_tasks_significant_at_bonferroni": n_sig_bonf,
            "n_tasks": len(common_tasks),
        }

    # ── Plot 1: Jaccard heatmap
    _heatmap(
        jaccard_mat, available, available,
        f"Error-set Jaccard [{aggregation}, k={k}] — 1=identical errors",
        PLOTS_DIR / f"error_overlap_jaccard_k{k}_{aggregation}.png",
        vmin=0.0, vmax=1.0, cmap="magma",
    )
    # ── Plot 2: wins/loss heatmap (row wins over column)
    _heatmap(
        winsloss_mat, available, available,
        f"Fraction (row right, column wrong) [{aggregation}, k={k}]",
        PLOTS_DIR / f"error_overlap_winsloss_k{k}_{aggregation}.png",
        vmin=0.0, vmax=max(0.01, float(np.nanmax(winsloss_mat))),
        cmap="viridis",
    )
    # ── Plot 3: per-task wins/loss for mlc vs txcdr_t5
    _plot_mlc_vs_txcdr_per_task(
        preds, common_tasks,
        PLOTS_DIR / f"error_overlap_per_task_mlc_vs_txcdr_t5_k{k}_{aggregation}.png",
        aggregation=aggregation, k=k,
    )

    out = {
        "aggregation": aggregation,
        "k_feat": k,
        "archs": available,
        "n_tasks": len(common_tasks),
        "tasks": common_tasks,
        "pairs": pair_summary,
    }
    out_path = RESULTS_DIR / f"error_overlap_summary_{aggregation}_k{k}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[error_overlap] wrote {out_path}")


def _heatmap(mat, row_labels, col_labels, title, out_path, *,
             vmin=None, vmax=None, cmap="viridis"):
    n_rows, n_cols = mat.shape
    fig, ax = plt.subplots(figsize=(1.2 * n_cols + 2, 1.0 * n_rows + 2))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def _plot_mlc_vs_txcdr_per_task(preds, tasks, out_path, *, aggregation, k):
    A, B = "mlc__seed42", "txcdr_t5__seed42"
    if A not in preds or B not in preds:
        print(f"[error_overlap] skip per-task mlc vs t5 plot — missing {A} or {B}")
        return
    winsA = []
    winsB = []
    for t in tasks:
        pa, pb = preds[A][t], preds[B][t]
        y = pa["y_true"]
        a_right = (pa["y_pred"] == y)
        b_right = (pb["y_pred"] == y)
        winsA.append(float(((a_right) & (~b_right)).mean()))
        winsB.append(float(((~a_right) & (b_right)).mean()))
    idx = np.argsort([-abs(a - b) for a, b in zip(winsA, winsB)])
    tasks_s = [tasks[i] for i in idx]
    winsA_s = [winsA[i] for i in idx]
    winsB_s = [winsB[i] for i in idx]

    fig, ax = plt.subplots(figsize=(14, max(4, 0.28 * len(tasks))))
    y = np.arange(len(tasks))
    ax.barh(y, winsA_s, color="tab:blue", label=f"{A} right, {B} wrong")
    ax.barh(y, [-w for w in winsB_s], color="tab:orange",
            label=f"{B} right, {A} wrong")
    ax.set_yticks(y)
    ax.set_yticklabels(tasks_s, fontsize=7)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("fraction of test examples")
    ax.set_title(f"mlc vs txcdr_t5 — per-task asymmetric errors [{aggregation}, k={k}]")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregation", default="last_position",
                    choices=["last_position", "mean_pool", "full_window"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    run(args.aggregation, args.k)


if __name__ == "__main__":
    main()
