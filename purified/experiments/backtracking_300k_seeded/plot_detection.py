"""Build the compact reviewer package for corrected 300K C7 detection runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S_GRID = (1, 2, 4, 8, 16, 32)
HEADLINE_S = 8
SEEDS = (1, 2, 42)
TRAIN_KEYS = {
    1: "a300c63374c3597e",
    2: "27078b0d7700ae05",
    42: "8787f8fe527218ad",
}
TRAINING_PROTOCOL = "c7-300k-seeded-v1"
DETECTION_PROTOCOL = "c7-detection-seeded-v1"
HISTORICAL_COMMIT = "284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3"
SENTENCE_ARTIFACT_SHA256 = (
    "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
)
SENTENCE_ARTIFACT_BYTES = 1_137_333_114

# Three-decimal, table-transcribed seed-42 values from
# purified/docs/aniket/appendix.tex. These are contextual references, not
# members of the corrected-seed replication.
SUBMITTED = {
    "SAE 32k (submitted)": {
        "pr_auc": (0.130, 0.132, 0.137, 0.175, 0.196, 0.229),
        "roc_auc": (0.507, 0.508, 0.518, 0.566, 0.589, 0.626),
    },
    "T-SAE 32k (submitted)": {
        "pr_auc": (0.158, 0.164, 0.169, 0.196, 0.213, 0.245),
        "roc_auc": (0.591, 0.603, 0.617, 0.640, 0.655, 0.683),
    },
    "TXC-base 32k (submitted)": {
        "pr_auc": (0.143, 0.168, 0.177, 0.201, 0.217, 0.250),
        "roc_auc": (0.536, 0.593, 0.609, 0.628, 0.647, 0.679),
    },
}

COLORS = {
    "txc": "#3366CC",
    "txc_submitted": "#7F9FD1",
    "tsae16": "#CC7A00",
    "tsae32": "#8E44AD",
    "sae": "#59636E",
    "grid": "#D9DEE7",
    "text": "#273142",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root containing cells/txc_base_d32768_seed*/detection.json.",
    )
    parser.add_argument(
        "--tsae16-json",
        type=Path,
        required=True,
        help="Detection JSON for the new 16,384-width T-SAE sensitivity cell.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _require_fields(
    payload: dict[str, Any], expected: dict[str, Any], path: Path
) -> None:
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: provenance mismatch: {mismatches}")


def _require_detection(
    payload: dict[str, Any], expected: dict[str, Any], path: Path
) -> None:
    _require_fields(payload, expected, path)
    if tuple(int(value) for value in payload.get("S_grid", ())) != S_GRID:
        raise ValueError(f"{path}: unexpected S grid {payload.get('S_grid')}")
    for metric in ("pr_auc", "roc_auc"):
        if set(payload.get(metric, {})) != {str(value) for value in S_GRID}:
            raise ValueError(f"{path}: incomplete {metric}")


def _load(root: Path, tsae16_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = root / "source_artifact.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _require_fields(
        source,
        {
            "status": "verified",
            "sha256": SENTENCE_ARTIFACT_SHA256,
            "size_bytes": SENTENCE_ARTIFACT_BYTES,
            "shape": [25_204, 6, 4_096],
        },
        source_path,
    )
    cohort = {
        "n_sentences": source["n_sentences"],
        "n_positive": source["n_positive"],
        "positive_fraction": source["positive_fraction"],
        "sentence_window": 6,
        "probe_random_state": 42,
    }
    txc = []
    for seed in SEEDS:
        path = root / "cells" / f"txc_base_d32768_seed{seed}" / "detection.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require_detection(
            payload,
            {
                "status": "complete",
                "detection_protocol": DETECTION_PROTOCOL,
                "training_protocol": TRAINING_PROTOCOL,
                "historical_commit": HISTORICAL_COMMIT,
                "train_key": TRAIN_KEYS[seed],
                "arch": "txc_base",
                "d_sae": 32_768,
                "seed": seed,
                "arch_window": 5,
                **cohort,
            },
            path,
        )
        config_path = path.parent / "checkpoint" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        _require_fields(
            config,
            {
                "status": "complete",
                "protocol_version": TRAINING_PROTOCOL,
                "historical_commit": HISTORICAL_COMMIT,
                "train_key": TRAIN_KEYS[seed],
                "arch": "txc_base",
                "d_sae": 32_768,
                "seed": seed,
                "n_steps_completed": 300_000,
            },
            config_path,
        )
        txc.append(payload)

    tsae16 = json.loads(tsae16_path.read_text(encoding="utf-8"))
    _require_detection(
        tsae16,
        {
            "status": "complete",
            "detection_protocol": DETECTION_PROTOCOL,
            "training_protocol": TRAINING_PROTOCOL,
            "historical_commit": HISTORICAL_COMMIT,
            "train_key": "b97e3c00153a5271",
            "arch": "tsae_paper",
            "d_sae": 16_384,
            "seed": 42,
            "arch_window": 1,
            **cohort,
        },
        tsae16_path,
    )
    tsae16_config_path = tsae16_path.parent / "checkpoint" / "config.json"
    tsae16_config = json.loads(tsae16_config_path.read_text(encoding="utf-8"))
    _require_fields(
        tsae16_config,
        {
            "status": "complete",
            "protocol_version": TRAINING_PROTOCOL,
            "historical_commit": HISTORICAL_COMMIT,
            "train_key": "b97e3c00153a5271",
            "arch": "tsae_paper",
            "d_sae": 16_384,
            "seed": 42,
            "n_steps_completed": 300_000,
        },
        tsae16_config_path,
    )
    return txc, tsae16


def _stats(txc: list[dict[str, Any]], metric: str, s: int) -> tuple[float, float]:
    values = np.asarray(
        [float(payload[metric][str(s)]) for payload in txc], dtype=np.float64
    )
    return float(values.mean()), float(values.std(ddof=1))


def _configure() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(
    txc: list[dict[str, Any]], tsae16: dict[str, Any], output_dir: Path
) -> None:
    means, sds = zip(*[_stats(txc, "pr_auc", s) for s in S_GRID])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for payload in txc:
        ax.plot(
            S_GRID,
            [payload["pr_auc"][str(s)] for s in S_GRID],
            color=COLORS["txc"],
            alpha=0.22,
            linewidth=1.0,
        )
    ax.errorbar(
        S_GRID,
        means,
        yerr=sds,
        color=COLORS["txc"],
        marker="o",
        capsize=3,
        linewidth=2.0,
        label="TXC-base 32k, corrected (3-seed mean ± SD)",
    )
    ax.plot(
        S_GRID,
        SUBMITTED["TXC-base 32k (submitted)"]["pr_auc"],
        color=COLORS["txc_submitted"],
        marker="o",
        linestyle="--",
        label="TXC-base 32k, submitted seed 42",
    )
    ax.plot(
        S_GRID,
        [tsae16["pr_auc"][str(s)] for s in S_GRID],
        color=COLORS["tsae16"],
        marker="^",
        linestyle="--",
        label="T-SAE 16k, new seed 42",
    )
    ax.plot(
        S_GRID,
        SUBMITTED["T-SAE 32k (submitted)"]["pr_auc"],
        color=COLORS["tsae32"],
        marker="D",
        linestyle="-.",
        label="T-SAE 32k, submitted seed 42",
    )
    ax.plot(
        S_GRID,
        SUBMITTED["SAE 32k (submitted)"]["pr_auc"],
        color=COLORS["sae"],
        marker="s",
        linestyle=":",
        label="SAE 32k, submitted seed 42",
    )
    ax.axhline(
        float(txc[0]["positive_fraction"]),
        color="#999999",
        linewidth=0.9,
        linestyle=(0, (3, 3)),
        label="Class prior",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(S_GRID, labels=[str(s) for s in S_GRID])
    ax.set_xlabel("Sparse-probe feature budget $S$")
    ax.set_ylabel("Backtracking detection PR-AUC")
    ax.set_title("Backtracking detection after 300k dictionary-training steps")
    ax.legend(
        frameon=False,
        fontsize=7.3,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    _save(fig, output_dir, "backtracking_detection_300k_curve")


def _plot_headline(
    txc: list[dict[str, Any]], tsae16: dict[str, Any], output_dir: Path
) -> None:
    txc_values = np.asarray(
        [float(payload["pr_auc"][str(HEADLINE_S)]) for payload in txc],
        dtype=np.float64,
    )
    labels = (
        "SAE\n32k",
        "T-SAE\n16k",
        "T-SAE\n32k",
        "TXC-base\nsubmitted",
        "TXC-base\ncorrected",
    )
    values = (
        SUBMITTED["SAE 32k (submitted)"]["pr_auc"][S_GRID.index(HEADLINE_S)],
        float(tsae16["pr_auc"][str(HEADLINE_S)]),
        SUBMITTED["T-SAE 32k (submitted)"]["pr_auc"][S_GRID.index(HEADLINE_S)],
        SUBMITTED["TXC-base 32k (submitted)"]["pr_auc"][S_GRID.index(HEADLINE_S)],
        float(txc_values.mean()),
    )
    colors = (
        COLORS["sae"],
        COLORS["tsae16"],
        COLORS["tsae32"],
        COLORS["txc_submitted"],
        COLORS["txc"],
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors, width=0.62, alpha=0.92)
    ax.errorbar(
        positions[-1],
        values[-1],
        yerr=float(txc_values.std(ddof=1)),
        color=COLORS["text"],
        capsize=4,
        linewidth=1.2,
    )
    jitter = np.asarray((-0.08, 0.0, 0.08))
    ax.scatter(
        positions[-1] + jitter,
        txc_values,
        color="white",
        edgecolor=COLORS["txc"],
        linewidth=1.1,
        zorder=3,
    )
    ax.axhline(
        float(txc[0]["positive_fraction"]),
        color="#999999",
        linewidth=0.9,
        linestyle=(0, (3, 3)),
    )
    for x, value in zip(positions, values):
        ax.text(x, value + 0.006, f"{value:.3f}", ha="center", va="bottom")
    ax.set_xticks(positions, labels)
    ax.set_ylabel(f"PR-AUC at $S={HEADLINE_S}$")
    ax.set_title("Backtracking detection at the submitted probe budget")
    ax.text(
        0.01,
        -0.22,
        "Dashed line: class prior. Submitted: rounded seed-42 table values. "
        "Corrected: mean ± SD over seeds 1, 2, 42.",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#59636E",
    )
    _save(fig, output_dir, "backtracking_detection_300k_s8")


def _write_tables(
    txc: list[dict[str, Any]], tsae16: dict[str, Any], output_dir: Path
) -> None:
    raw_rows: list[dict[str, Any]] = []
    for payload in txc:
        for s in S_GRID:
            raw_rows.append(
                {
                    "source": "corrected_replication",
                    "architecture": "TXC-base 32k",
                    "seed": payload["seed"],
                    "S": s,
                    "pr_auc": payload["pr_auc"][str(s)],
                    "roc_auc": payload["roc_auc"][str(s)],
                }
            )
    for s in S_GRID:
        raw_rows.append(
            {
                "source": "new_width_sensitivity",
                "architecture": "T-SAE 16k",
                "seed": 42,
                "S": s,
                "pr_auc": tsae16["pr_auc"][str(s)],
                "roc_auc": tsae16["roc_auc"][str(s)],
            }
        )
        for name, metrics in SUBMITTED.items():
            raw_rows.append(
                {
                    "source": "submitted_rounded_table_reference",
                    "architecture": name.replace(" (submitted)", ""),
                    "seed": 42,
                    "S": s,
                    "pr_auc": metrics["pr_auc"][S_GRID.index(s)],
                    "roc_auc": metrics["roc_auc"][S_GRID.index(s)],
                }
            )

    with (output_dir / "raw_detection_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(raw_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for s in S_GRID:
        pr_mean, pr_sd = _stats(txc, "pr_auc", s)
        roc_mean, roc_sd = _stats(txc, "roc_auc", s)
        summary_rows.append(
            {
                "S": s,
                "txc_pr_auc_mean": pr_mean,
                "txc_pr_auc_sd": pr_sd,
                "txc_roc_auc_mean": roc_mean,
                "txc_roc_auc_sd": roc_sd,
                "tsae16_seed42_pr_auc": tsae16["pr_auc"][str(s)],
                "submitted_tsae32_seed42_pr_auc": SUBMITTED["T-SAE 32k (submitted)"][
                    "pr_auc"
                ][S_GRID.index(s)],
                "submitted_sae32_seed42_pr_auc": SUBMITTED["SAE 32k (submitted)"][
                    "pr_auc"
                ][S_GRID.index(s)],
                "submitted_txc_base_seed42_pr_auc": SUBMITTED[
                    "TXC-base 32k (submitted)"
                ]["pr_auc"][S_GRID.index(s)],
            }
        )
    with (output_dir / "summary_detection_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(summary_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    headline_mean, headline_sd = _stats(txc, "pr_auc", HEADLINE_S)
    payload = {
        "status": "complete",
        "training_protocol": TRAINING_PROTOCOL,
        "detection_protocol": DETECTION_PROTOCOL,
        "S_grid": list(S_GRID),
        "txc_seeds": list(SEEDS),
        "headline_S": HEADLINE_S,
        "txc_headline_pr_auc_mean": headline_mean,
        "txc_headline_pr_auc_sd": headline_sd,
        "txc_headline_pr_auc_by_seed": {
            str(row["seed"]): row["pr_auc"][str(HEADLINE_S)] for row in txc
        },
        "tsae16_headline_pr_auc_seed42": tsae16["pr_auc"][str(HEADLINE_S)],
        "submitted_tsae32_headline_pr_auc_seed42": SUBMITTED["T-SAE 32k (submitted)"][
            "pr_auc"
        ][S_GRID.index(HEADLINE_S)],
        "submitted_sae32_headline_pr_auc_seed42": SUBMITTED["SAE 32k (submitted)"][
            "pr_auc"
        ][S_GRID.index(HEADLINE_S)],
        "submitted_txc_base_headline_pr_auc_seed42": SUBMITTED[
            "TXC-base 32k (submitted)"
        ]["pr_auc"][S_GRID.index(HEADLINE_S)],
        "positive_fraction": txc[0]["positive_fraction"],
        "sentence_artifact_sha256": SENTENCE_ARTIFACT_SHA256,
        "caveat": (
            "Submitted values are rounded, table-transcribed seed-42 "
            "references and are not pooled with the corrected-seed TXC-base "
            "replication. This package does not replicate TXC-pro or steering."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    seed_text = ", ".join(
        f"{row['seed']}: {float(row['pr_auc'][str(HEADLINE_S)]):.4f}" for row in txc
    )
    markdown = f"""# Corrected 300K Backtracking detection

The paper-faithful TXC-base architecture was retrained for 300,000 steps with
fully seeded Python, NumPy, CPU-Torch, and CUDA RNGs. At the submitted probe
budget $S={HEADLINE_S}$, the three TXC-base seeds score
**{headline_mean:.4f} ± {headline_sd:.4f}
PR-AUC** (sample SD; {seed_text}). The positive-class prior is
{float(txc[0]['positive_fraction']):.4f}.

| Architecture | Width | Seeds | PR-AUC at S={HEADLINE_S} | Status |
|:--|--:|:--|--:|:--|
| SAE | 32,768 | 42 | {SUBMITTED['SAE 32k (submitted)']['pr_auc'][S_GRID.index(HEADLINE_S)]:.4f} | submitted rounded table reference |
| T-SAE | 16,384 | 42 | {float(tsae16['pr_auc'][str(HEADLINE_S)]):.4f} | new width sensitivity |
| T-SAE | 32,768 | 42 | {SUBMITTED['T-SAE 32k (submitted)']['pr_auc'][S_GRID.index(HEADLINE_S)]:.4f} | submitted rounded table reference |
| TXC-base | 32,768 | 42 | {SUBMITTED['TXC-base 32k (submitted)']['pr_auc'][S_GRID.index(HEADLINE_S)]:.4f} | submitted rounded table reference |
| TXC-base | 32,768 | 1, 2, 42 | {headline_mean:.4f} ± {headline_sd:.4f} | corrected 300K replication |

![Full sparse-probe curve](backtracking_detection_300k_curve.png)

![Submitted-budget comparison](backtracking_detection_300k_s8.png)

The submitted SAE, 32k T-SAE, and TXC-base rows are rounded historical
seed-42 table references from the paper, while the corrected TXC-base
statistics are a new seeded replication. They are shown together for context
but are not treated as one shared multi-seed experiment. This package evaluates
detection for TXC-base, the submitted steering winner. The submitted detection
winner was TXC-pro, which is not rerun here. This package also does not provide
new multi-seed steering measurements.
"""
    (output_dir / "reviewer_summary.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    txc, tsae16 = _load(args.root.resolve(), args.tsae16_json.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure()
    _plot_curve(txc, tsae16, output_dir)
    _plot_headline(txc, tsae16, output_dir)
    _write_tables(txc, tsae16, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
