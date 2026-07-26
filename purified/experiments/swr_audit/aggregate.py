"""Audit and aggregate SWR JSONL rows from persisted component scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def conservative_swr(row: dict) -> float:
    """Ordered PR-AUC minus the strongest exact invariant/single-token null."""
    baseline = max(
        row["mean_pool_param_matched"]["pr_auc"],
        row["mean_pool_same_rank"]["pr_auc"],
        row["best_token"]["pr_auc"],
    )
    return float(row["ordered"]["pr_auc"] - baseline)


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    mean = float(array.mean())
    if len(array) > 1:
        critical = student_t.ppf(0.975, len(array) - 1)
        half_width = float(critical * std / np.sqrt(len(array)))
        grouped_fold_t_95: list[float] | None = [mean - half_width, mean + half_width]
    else:
        grouped_fold_t_95 = None
    return {
        "fold_values": array.tolist(),
        "mean": mean,
        "median": float(np.median(array)),
        "std_sample": std,
        "n_folds": int(len(array)),
        "grouped_fold_t_95": grouped_fold_t_95,
    }


def audited_summary(rows: list[dict]) -> dict:
    """Recompute every derived field rather than trusting emitted SWR values."""
    if not rows:
        raise ValueError("no SWR fold rows supplied")
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["window"],
            row["normalization"],
            row["pca_dim_actual"],
            row["rank"],
        )
        groups.setdefault(key, []).append(row)

    summaries = []
    for (window, normalization, pca_dim, rank), group in sorted(groups.items()):
        group.sort(key=lambda row: row["fold"])
        conservative = [conservative_swr(row) for row in group]
        summaries.append(
            {
                "window": window,
                "normalization": normalization,
                "pca_dim": pca_dim,
                "rank": rank,
                "folds": [int(row["fold"]) for row in group],
                "ordered_pr_auc": _stats(
                    [float(row["ordered"]["pr_auc"]) for row in group]
                ),
                "mean_pool_param_matched_pr_auc": _stats(
                    [float(row["mean_pool_param_matched"]["pr_auc"]) for row in group]
                ),
                "mean_pool_same_rank_pr_auc": _stats(
                    [float(row["mean_pool_same_rank"]["pr_auc"]) for row in group]
                ),
                "best_token_pr_auc": _stats(
                    [float(row["best_token"]["pr_auc"]) for row in group]
                ),
                "swr_pr_auc_conservative": {
                    **_stats(conservative),
                    "n_above_0_02": int(np.sum(np.asarray(conservative) > 0.02)),
                },
                "order_gap_pr_auc": _stats(
                    [float(row["order_gap_pr_auc"]) for row in group]
                ),
            }
        )
    return {
        "schema_version": "1.0.0",
        "interpretation": (
            "supervised residual upper-bound; not evidence that an unsupervised "
            "SAE/TXC dictionary recovered the mechanism"
        ),
        "swr_definition": (
            "ordered PR-AUC minus max(param-matched exact invariant, same-rank "
            "exact invariant, best single offset)"
        ),
        "interval_definition": (
            "two-sided Student-t interval across question-grouped outer-fold values; "
            "this is not an example-level or prompt bootstrap"
        ),
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    rows = [record for record in records if record.get("record_type") != "metadata"]
    summary = audited_summary(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
