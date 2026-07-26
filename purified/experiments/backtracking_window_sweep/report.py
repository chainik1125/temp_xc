"""Incremental Markdown table and plot for completed sweep cells."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def _selected_probe(cell: dict, name: str) -> dict:
    candidates = cell["probes"][name]
    return max(candidates, key=lambda row: int(row["n_features"]))


def compact_row(cell: dict) -> dict:
    txc = _selected_probe(cell, "txc")
    positional = _selected_probe(cell, "sae_positional")
    invariant = _selected_probe(cell, "sae_invariant")
    last = _selected_probe(cell, "sae_last_token")
    txc_control_values = [
        float(txc["control_pr_auc"][name]["mean"])
        for name in ("shuffle", "reverse", "circular")
    ]
    comparisons = cell["grouped_question_bootstrap"]["comparisons"]
    positional_gap = comparisons["txc_minus_sae_positional"]
    strongest_gap = comparisons["txc_minus_strongest_learned_control"]
    return {
        "window": int(cell["window"]),
        "seed": int(cell["seed"]),
        "n_features": int(txc["n_features"]),
        "txc_ordered": float(txc["ordered_pr_auc"]["mean"]),
        "txc_strongest_order_control": max(txc_control_values),
        "txc_order_gap_conservative": (
            float(txc["ordered_pr_auc"]["mean"]) - max(txc_control_values)
        ),
        "txc_minus_sae_positional": float(positional_gap["point_estimate"]),
        "txc_minus_sae_positional_ci": [
            float(positional_gap["lower_95"]),
            float(positional_gap["upper_95"]),
        ],
        "txc_minus_strongest_learned": float(
            strongest_gap["point_estimate"]
        ),
        "txc_minus_strongest_learned_ci": [
            float(strongest_gap["lower_95"]),
            float(strongest_gap["upper_95"]),
        ],
        "sae_positional": float(positional["ordered_pr_auc"]["mean"]),
        "sae_invariant": float(invariant["ordered_pr_auc"]["mean"]),
        "sae_last_token": float(last["ordered_pr_auc"]["mean"]),
        "residual_ordered": float(cell["residual"]["ordered_pr_auc"]["mean"]),
        "residual_invariant": float(
            cell["residual"]["invariant_mean_pr_auc"]["mean"]
        ),
    }


def load_completed(output_root: Path) -> list[dict]:
    rows = []
    for path in sorted((output_root / "cells").glob("T*_seed*/result.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete":
            rows.append(compact_row(payload))
    return sorted(rows, key=lambda row: (row["window"], row["seed"]))


def _mean_std(rows: list[dict], field: str) -> tuple[float, float]:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    )


def _markdown(rows: list[dict], figure_exists: bool) -> str:
    lines = [
        "# Backtracking window-size sweep",
        "",
        "This report is regenerated after every completed cell. Values are means "
        "over question-grouped outer folds at the largest registered sparse-probe "
        "feature budget. A row is not a three-seed result until all seeds are present.",
        "",
        "| T | Seed | TXC ordered | Strongest TXC order control | TXC − positional SAE [95% question CI] | TXC − strongest learned control [95% question CI] | SAE invariant | Last-token SAE | Residual ordered |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['window']} | {row['seed']} | "
            f"{row['txc_ordered']:.4f} | "
            f"{row['txc_strongest_order_control']:.4f} | "
            f"{row['txc_minus_sae_positional']:+.4f} "
            f"[{row['txc_minus_sae_positional_ci'][0]:+.4f}, "
            f"{row['txc_minus_sae_positional_ci'][1]:+.4f}] | "
            f"{row['txc_minus_strongest_learned']:+.4f} "
            f"[{row['txc_minus_strongest_learned_ci'][0]:+.4f}, "
            f"{row['txc_minus_strongest_learned_ci'][1]:+.4f}] | "
            f"{row['sae_invariant']:.4f} | "
            f"{row['sae_last_token']:.4f} | "
            f"{row['residual_ordered']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Seed aggregation",
            "",
            "| T | Seeds complete | TXC ordered mean ± SD | SAE positional mean ± SD | Conservative order gap mean ± SD |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for window in sorted({row["window"] for row in rows}):
        group = [row for row in rows if row["window"] == window]
        txc = _mean_std(group, "txc_ordered")
        sae = _mean_std(group, "sae_positional")
        gap = _mean_std(group, "txc_order_gap_conservative")
        lines.append(
            f"| {window} | {len(group)} | {txc[0]:.4f} ± {txc[1]:.4f} | "
            f"{sae[0]:.4f} ± {sae[1]:.4f} | "
            f"{gap[0]:+.4f} ± {gap[1]:.4f} |"
        )
    if figure_exists:
        lines.extend(
            [
                "",
                "![Backtracking detection versus window size](window_curve.png)",
            ]
        )
    lines.extend(
        [
            "",
            "The fixed-probe order perturbations measure sensitivity to the learned "
            "ordered representation under covariate shift. The SAE positional stack "
            "and the train-fold-only residual control are the stronger tests of "
            "whether TXC adds value beyond multi-token support.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(rows: list[dict], path: Path) -> bool:
    if not rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    windows = sorted({row["window"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fields = [
        ("txc_ordered", "TXC ordered", "o"),
        ("sae_positional", "SAE positional", "s"),
        ("sae_invariant", "SAE invariant", "^"),
        ("sae_last_token", "Last-token SAE", "D"),
    ]
    for field, label, marker in fields:
        means, errors = [], []
        for window in windows:
            group = [row for row in rows if row["window"] == window]
            mean, std = _mean_std(group, field)
            means.append(mean)
            errors.append(std)
        ax.errorbar(
            windows,
            means,
            yerr=errors,
            marker=marker,
            linewidth=1.5,
            capsize=3,
            label=label,
        )
    ax.set_xlabel("Trailing pre-sentence window T")
    ax.set_ylabel("Question-grouped detection PR-AUC")
    ax.set_xticks(windows)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, path)
    return True


def write_report(output_root: Path) -> dict:
    rows = load_completed(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_path = output_root / "window_curve.png"
    figure_exists = _plot(rows, figure_path)
    markdown = _markdown(rows, figure_exists)
    temporary = output_root / "summary.tmp.md"
    temporary.write_text(markdown)
    os.replace(temporary, output_root / "summary.md")
    payload = {"n_cells_complete": len(rows), "rows": rows}
    temporary_json = output_root / "summary.tmp.json"
    temporary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_json, output_root / "summary.json")
    return payload
