"""Plot the controlled Shamir/HMM suite and the denoising frequency replay.

The controlled runner writes append-only JSONL, so this script keeps only the
latest row for each cell.  It refuses incomplete full runs and preserves the
important distinction between raw balanced accuracy and chance-normalized
recovery in the Shamir panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = POWER_ROOT / "configs" / "controlled_frequency_suite.json"
DEFAULT_CONTROLLED_RESULTS = (
    POWER_ROOT / "results" / "controlled_frequency_suite_remote" / "results.jsonl"
)
DEFAULT_FROZEN_CONFIG = (
    POWER_ROOT / "results" / "controlled_frequency_suite_remote" / "frozen_config.json"
)
DEFAULT_CONTROLLED_SUMMARY = (
    POWER_ROOT / "results" / "controlled_frequency_suite_remote" / "summary.json"
)
DEFAULT_DENOISING_RESULT = (
    POWER_ROOT / "results" / "denoising_frequency_usage_remote" / "result.json"
)
DEFAULT_FIGURES_DIR = POWER_ROOT / "figures"
DEFAULT_AGGREGATE_JSON = POWER_ROOT / "results" / "controlled_frequency_analysis.json"
DEFAULT_AGGREGATE_CSV = POWER_ROOT / "results" / "controlled_frequency_analysis.csv"

MODEL_ORDER = ("sae", "txc", "spectral_v1", "spectral_mat_global")
MODEL_LABELS = {
    "sae": "BatchTopK SAE",
    "txc": "TXC-post",
    "spectral_v1": "Spectral v1",
    "spectral_mat_global": "Spectral Matryoshka",
}
MODEL_COLORS = {
    "sae": "#6B6B6B",
    "txc": "#8C6D31",
    "spectral_v1": "#0072B2",
    "spectral_mat_global": "#009E73",
}
MODEL_MARKERS = {
    "sae": "o",
    "txc": "s",
    "spectral_v1": "^",
    "spectral_mat_global": "D",
}
REFERENCE_LABELS = {
    "chance": "Chance (reviewer h=2)",
    "sae_best_k": "SAE best-k (reviewer h=2)",
    "txc_k1": "TXC k=1 (reviewer h=2)",
    "txc_k2": "TXC k=2 (reviewer h=2)",
    "txc_k5": "TXC k=5 (reviewer h=2)",
}
REFERENCE_STYLES = {
    "chance": {"color": "#A0A0A0", "linestyle": ":", "marker": None},
    "sae_best_k": {"color": "#595959", "linestyle": "--", "marker": "o"},
    "txc_k1": {"color": "#8C6D31", "linestyle": ":", "marker": "s"},
    "txc_k2": {"color": "#B08B4F", "linestyle": "--", "marker": "P"},
    "txc_k5": {"color": "#D2AE6D", "linestyle": "-.", "marker": "X"},
}
HMM_TASK_ORDER = ("hmm_slow", "hmm_alternating", "hmm_mixed")
HMM_TASK_LABELS = {
    "hmm_slow": "Slow\n$\\lambda=+0.9$",
    "hmm_alternating": "Alternating\n$\\lambda=-0.9$",
    "hmm_mixed": "Mixed frequencies",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _latest_controlled_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "cell_id" not in row:
            raise ValueError(f"{path}:{line_number}: expected a result row with cell_id")
        latest[str(row["cell_id"])] = row
    return [
        row
        for row in latest.values()
        if row.get("status") == "ok" and not bool(row.get("smoke"))
    ]


def _validate_controlled_run(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if bool(summary.get("smoke")):
        raise RuntimeError("refusing to plot a smoke-run summary")
    if not bool(summary.get("complete")):
        raise RuntimeError(
            "refusing to plot an incomplete controlled run: "
            f"{summary.get('ok_cells', 0)}/{summary.get('expected_cells', '?')} cells are OK"
        )
    expected = int(summary["expected_cells"])
    if len(rows) != expected:
        raise RuntimeError(
            "results.jsonl and summary.json disagree: "
            f"found {len(rows)} latest successful full rows, expected {expected}"
        )
    summary_keys = {
        (str(row["task"]), str(row["model"]))
        for row in summary.get("aggregates", [])
    }
    row_keys = {(str(row["task"]), str(row["model"])) for row in rows}
    if summary_keys != row_keys:
        raise RuntimeError("results.jsonl and summary.json have different task/model cells")


def _sample_stats(values: Iterable[float]) -> dict[str, Any]:
    values_list = [float(value) for value in values]
    if not values_list:
        raise ValueError("cannot aggregate an empty value set")
    return {
        "n": len(values_list),
        "mean": statistics.fmean(values_list),
        "std": statistics.stdev(values_list) if len(values_list) > 1 else 0.0,
        "min": min(values_list),
        "max": max(values_list),
        "values": values_list,
    }


def _group_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["model"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["seed"]))
    return dict(grouped)


def _task_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(task["name"]): task for task in config["tasks"]}


def _portable_input_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(POWER_ROOT))
    except ValueError:
        return str(path)


def _shamir_records(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    tasks: dict[str, dict[str, Any]],
    *,
    group: str,
    models: Sequence[str] = MODEL_ORDER,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_name, task in tasks.items():
        if task.get("group") != group:
            continue
        for model in models:
            selected = grouped.get((task_name, model), [])
            if not selected:
                continue
            chances = [float(row["metrics"]["secret_chance"]) for row in selected]
            if not np.allclose(chances, chances[0], atol=1e-12, rtol=0):
                raise RuntimeError(f"{task_name}/{model}: inconsistent chance levels")
            records.append(
                {
                    "task": task_name,
                    "model": model,
                    "window": int(task["window"]),
                    "h": int(task["h"]),
                    "q": int(task["q"]),
                    "chance": chances[0],
                    "raw_balanced_accuracy": _sample_stats(
                        row["metrics"]["secret_balanced_accuracy"] for row in selected
                    ),
                    "normalized_recovery": _sample_stats(
                        row["metrics"]["secret_recovery"] for row in selected
                    ),
                }
            )
    return sorted(records, key=lambda record: (record["window"], MODEL_ORDER.index(record["model"])))


def _reviewer_reference(config: dict[str, Any]) -> list[dict[str, Any]]:
    reference = config["reviewer_response_reference"]
    windows = [int(value) for value in reference["windows"]]
    chance = [float(value) for value in reference["chance"]]
    if len(windows) != len(chance):
        raise ValueError("reviewer reference windows and chance arrays differ in length")
    records: list[dict[str, Any]] = []
    for key in REFERENCE_LABELS:
        accuracy = [float(value) for value in reference[key]]
        if len(accuracy) != len(windows):
            raise ValueError(f"reviewer reference {key} has the wrong length")
        normalized = [
            (value - chance_value) / (1.0 - chance_value)
            for value, chance_value in zip(accuracy, chance, strict=True)
        ]
        records.append(
            {
                "series": key,
                "label": REFERENCE_LABELS[key],
                "windows": windows,
                "raw_balanced_accuracy": accuracy,
                "normalized_recovery": normalized,
                "chance": chance,
            }
        )
    return records


def _hmm_latent_records(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in HMM_TASK_ORDER:
        for model in MODEL_ORDER:
            selected = grouped.get((task, model), [])
            if selected:
                records.append(
                    {
                        "task": task,
                        "model": model,
                        "latent_r2": _sample_stats(
                            row["metrics"]["latent_r2"] for row in selected
                        ),
                    }
                )
    return records


def _same_nested_values(values: Sequence[Any], *, name: str) -> Any:
    if not values:
        raise ValueError(f"{name} has no values")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"{name} differs across seeds")
    return first


def _mixed_hmm_band_records(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model in ("spectral_v1", "spectral_mat_global"):
        selected = grouped.get(("hmm_mixed", model), [])
        if not selected:
            continue
        matrices = np.asarray(
            [row["metrics"]["band_latent_r2_per_source"] for row in selected],
            dtype=np.float64,
        )
        expected_energy = np.asarray(
            [
                row["metrics"]["expected_dct_band_energy_per_source"]
                for row in selected
            ],
            dtype=np.float64,
        )
        expected_bands = _same_nested_values(
            [row["metrics"]["expected_band_per_source"] for row in selected],
            name=f"{model} expected bands",
        )
        bands = _same_nested_values(
            [row["metrics"]["spectral_usage"]["bands"] for row in selected],
            name=f"{model} DCT bands",
        )
        lambdas = _same_nested_values(
            [row["metrics"]["lambdas"] for row in selected],
            name=f"{model} HMM lambdas",
        )
        records.append(
            {
                "model": model,
                "n": len(selected),
                "bands": bands,
                "lambdas": lambdas,
                "r2_mean": matrices.mean(axis=0).tolist(),
                "r2_std": (
                    matrices.std(axis=0, ddof=1)
                    if matrices.shape[0] > 1
                    else np.zeros_like(matrices[0])
                ).tolist(),
                "r2_seed_values": matrices.tolist(),
                "expected_energy_mean": expected_energy.mean(axis=0).tolist(),
                "expected_band_per_source": [int(value) for value in expected_bands],
            }
        )
    return records


def _spectral_hmm_diagnostics(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in HMM_TASK_ORDER:
        for model in ("spectral_v1", "spectral_mat_global"):
            selected = grouped.get((task, model), [])
            if not selected:
                continue
            activation = np.asarray(
                [
                    row["metrics"]["spectral_usage"]["activation_energy_share"]
                    for row in selected
                ],
                dtype=np.float64,
            )
            selections = np.asarray(
                [
                    row["metrics"]["spectral_usage"]["selection_event_share"]
                    for row in selected
                ],
                dtype=np.float64,
            )
            records.append(
                {
                    "task": task,
                    "model": model,
                    "bands": _same_nested_values(
                        [
                            row["metrics"]["spectral_usage"]["bands"]
                            for row in selected
                        ],
                        name=f"{task}/{model} DCT bands",
                    ),
                    "band_localization_accuracy": _sample_stats(
                        row["metrics"]["band_localization_accuracy"]
                        for row in selected
                    ),
                    "time_shuffled_latent_r2": _sample_stats(
                        row["metrics"]["time_shuffled_latent_r2"]
                        for row in selected
                    ),
                    "activation_energy_share_mean": activation.mean(axis=0).tolist(),
                    "activation_energy_share_seed_values": activation.tolist(),
                    "selection_event_share_mean": selections.mean(axis=0).tolist(),
                    "selection_event_share_seed_values": selections.tolist(),
                    "expected_band_per_source_seed_values": [
                        row["metrics"]["expected_band_per_source"]
                        for row in selected
                    ],
                    "recovered_band_per_source_seed_values": [
                        row["metrics"]["recovered_band_per_source"]
                        for row in selected
                    ],
                }
            )
    return records


def _denoising_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("experiment") != "denoising_frequency_usage":
        raise ValueError("denoising input has the wrong experiment name")
    rows = sorted(payload["rows"], key=lambda row: int(row["seed"]))
    energy_specs = (
        ("decoded", ("decoded_reconstruction_energy",), "Decoded output\n(bias excluded)"),
        (
            "decoder_coefficient",
            ("activation_weighted_decoder_coefficient_energy",),
            "Activation-weighted\ndecoder weights",
        ),
        (
            "observed_activation",
            ("true_dct_power", "observed_activation"),
            "Observed\nactivation",
        ),
        (
            "hidden_support",
            ("true_dct_power", "hidden_support"),
            "Hidden-state\nsupport",
        ),
    )
    energy: list[dict[str, Any]] = []
    for name, path, label in energy_specs:
        values = [row["analysis"] for row in rows]
        for key in path:
            values = [value[key] for value in values]
        energy.append(
            {
                "name": name,
                "label": label,
                "dc_share": _sample_stats(value["dc_share"] for value in values),
                "ac_share": _sample_stats(value["ac_share"] for value in values),
            }
        )
    ridge = []
    for component in ("full", "dc", "ac"):
        ridge.append(
            {
                "component": component,
                "mean_r2": _sample_stats(
                    row["analysis"]["hidden_state_ridge_r2"][component]["mean_r2"]
                    for row in rows
                ),
            }
        )
    return {
        "n": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "energy": energy,
        "ridge": ridge,
    }


def build_analysis(
    *,
    run_config: dict[str, Any],
    reference_config: dict[str, Any],
    summary: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    denoising: dict[str, Any],
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    grouped = _group_rows(rows)
    tasks = _task_map(run_config)
    return {
        "schema_version": 1,
        "inputs": {
            key: _portable_input_path(path) for key, path in input_paths.items()
        },
        "controlled_run": {
            "run_name": summary["run_name"],
            "generated_at": summary.get("generated_at"),
            "expected_cells": int(summary["expected_cells"]),
            "ok_cells": int(summary["ok_cells"]),
            "complete": bool(summary["complete"]),
        },
        "shamir": {
            "h1_fresh": _shamir_records(grouped, tasks, group="shamir_h1"),
            "h2_new_spectral": _shamir_records(
                grouped,
                tasks,
                group="shamir_h2",
                models=("spectral_v1", "spectral_mat_global"),
            ),
            "h2_reviewer_reference": _reviewer_reference(reference_config),
            "normalization": "(balanced_accuracy - chance) / (1 - chance)",
        },
        "hmm": {
            "latent_r2": _hmm_latent_records(grouped),
            "mixed_band_r2": _mixed_hmm_band_records(grouped),
            "spectral_diagnostics": _spectral_hmm_diagnostics(grouped),
        },
        "denoising": _denoising_analysis(denoising),
    }


def _csv_row(
    *,
    section: str,
    metric: str,
    stats: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    row = {
        "section": section,
        "task": "",
        "model": "",
        "window": "",
        "source": "",
        "band": "",
        "metric": metric,
        "n": "",
        "mean": "",
        "std": "",
        "values": "",
        "expected_band": "",
        "expected_energy": "",
        "provenance": "fresh",
    }
    row.update(fields)
    if stats is not None:
        row.update(
            {
                "n": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "values": json.dumps(stats["values"]),
            }
        )
    return row


def flatten_analysis(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name in ("h1_fresh", "h2_new_spectral"):
        for record in analysis["shamir"][group_name]:
            common = {
                "task": record["task"],
                "model": record["model"],
                "window": record["window"],
            }
            rows.append(
                _csv_row(
                    section=f"shamir_{group_name}",
                    metric="secret_balanced_accuracy",
                    stats=record["raw_balanced_accuracy"],
                    **common,
                )
            )
            rows.append(
                _csv_row(
                    section=f"shamir_{group_name}",
                    metric="secret_recovery",
                    stats=record["normalized_recovery"],
                    **common,
                )
            )
    for record in analysis["shamir"]["h2_reviewer_reference"]:
        for window, raw, normalized in zip(
            record["windows"],
            record["raw_balanced_accuracy"],
            record["normalized_recovery"],
            strict=True,
        ):
            common = {
                "model": record["series"],
                "window": window,
                "n": 1,
                "std": 0.0,
                "provenance": "reviewer_response_reference",
            }
            rows.append(
                _csv_row(
                    section="shamir_h2_reference",
                    metric="secret_balanced_accuracy",
                    mean=raw,
                    values=json.dumps([raw]),
                    **common,
                )
            )
            rows.append(
                _csv_row(
                    section="shamir_h2_reference",
                    metric="secret_recovery",
                    mean=normalized,
                    values=json.dumps([normalized]),
                    **common,
                )
            )
    for record in analysis["hmm"]["latent_r2"]:
        rows.append(
            _csv_row(
                section="hmm_latent",
                task=record["task"],
                model=record["model"],
                metric="latent_r2",
                stats=record["latent_r2"],
            )
        )
    for record in analysis["hmm"]["mixed_band_r2"]:
        seed_values = np.asarray(record["r2_seed_values"], dtype=np.float64)
        expected_energy = np.asarray(record["expected_energy_mean"], dtype=np.float64)
        for source in range(seed_values.shape[1]):
            for band in range(seed_values.shape[2]):
                stats = _sample_stats(seed_values[:, source, band])
                rows.append(
                    _csv_row(
                        section="hmm_mixed_band",
                        task="hmm_mixed",
                        model=record["model"],
                        source=source,
                        band=band,
                        metric="band_latent_r2",
                        stats=stats,
                        expected_band=int(
                            band == record["expected_band_per_source"][source]
                        ),
                        expected_energy=float(expected_energy[source, band]),
                    )
                )
    for record in analysis["hmm"]["spectral_diagnostics"]:
        common = {
            "task": record["task"],
            "model": record["model"],
        }
        rows.append(
            _csv_row(
                section="hmm_spectral_diagnostics",
                metric="band_localization_accuracy",
                stats=record["band_localization_accuracy"],
                **common,
            )
        )
        rows.append(
            _csv_row(
                section="hmm_spectral_diagnostics",
                metric="time_shuffled_latent_r2",
                stats=record["time_shuffled_latent_r2"],
                **common,
            )
        )
        for band, values in enumerate(
            np.asarray(
                record["selection_event_share_seed_values"],
                dtype=np.float64,
            ).T
        ):
            rows.append(
                _csv_row(
                    section="hmm_spectral_diagnostics",
                    band=band,
                    metric="selection_event_share",
                    stats=_sample_stats(values),
                    **common,
                )
            )
    for record in analysis["denoising"]["energy"]:
        for component in ("dc", "ac"):
            rows.append(
                _csv_row(
                    section="denoising_energy",
                    task="denoising",
                    model="spectral_v1",
                    metric=f"{record['name']}_{component}_share",
                    stats=record[f"{component}_share"],
                )
            )
    for record in analysis["denoising"]["ridge"]:
        rows.append(
            _csv_row(
                section="denoising_ridge",
                task="denoising",
                model="spectral_v1",
                metric=f"ridge_{record['component']}_mean_r2",
                stats=record["mean_r2"],
            )
        )
    return rows


def _save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> tuple[Path, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def _draw_fresh_shamir(
    axis: plt.Axes,
    records: Sequence[dict[str, Any]],
    *,
    metric: str,
) -> None:
    for model in MODEL_ORDER:
        selected = [record for record in records if record["model"] == model]
        if not selected:
            continue
        x = np.asarray([record["window"] for record in selected], dtype=float)
        means = np.asarray([record[metric]["mean"] for record in selected])
        errors = np.asarray([record[metric]["std"] for record in selected])
        axis.errorbar(
            x,
            means,
            yerr=errors,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.6,
            capsize=2.5,
            label=MODEL_LABELS[model],
            zorder=3,
        )


def _draw_reference_shamir(
    axis: plt.Axes,
    records: Sequence[dict[str, Any]],
    *,
    metric: str,
) -> None:
    for record in records:
        style = REFERENCE_STYLES[record["series"]]
        axis.plot(
            record["windows"],
            record[metric],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.5,
            linewidth=1.1,
            alpha=0.9,
            label=record["label"],
            zorder=1,
        )


def _finish_shamir_axis(
    axis: plt.Axes,
    *,
    h: int,
    raw: bool,
    windows: Sequence[int],
    chance: float | None = None,
) -> None:
    axis.axvline(
        h + 1,
        color="#333333",
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        alpha=0.7,
        zorder=0,
    )
    axis.text(
        h + 1,
        0.985,
        "$W=h+1$",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.5,
        color="#333333",
    )
    axis.set_xticks(sorted(set(int(value) for value in windows)))
    axis.set_xlabel("Window length $W$")
    axis.set_ylim((-0.14, 1.03) if not raw else (0.0, 1.03))
    axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.35)
    if not raw:
        axis.axhline(0, color="#888888", linewidth=0.7, zorder=0)
    elif chance is not None:
        axis.axhline(
            chance,
            color="#A0A0A0",
            linestyle=":",
            linewidth=1.1,
            zorder=0,
        )
        axis.text(
            0.99,
            chance,
            "chance $=1/q$",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color="#666666",
        )


def plot_shamir(analysis: dict[str, Any], figures_dir: Path) -> tuple[Path, Path]:
    fresh_h1 = analysis["shamir"]["h1_fresh"]
    fresh_h2 = analysis["shamir"]["h2_new_spectral"]
    reference = analysis["shamir"]["h2_reviewer_reference"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), sharey="row")
    for column, (h, fresh) in enumerate(((1, fresh_h1), (2, fresh_h2))):
        raw_axis = axes[0, column]
        normalized_axis = axes[1, column]
        if h == 2:
            _draw_reference_shamir(
                raw_axis,
                reference,
                metric="raw_balanced_accuracy",
            )
            _draw_reference_shamir(
                normalized_axis,
                reference,
                metric="normalized_recovery",
            )
        _draw_fresh_shamir(raw_axis, fresh, metric="raw_balanced_accuracy")
        _draw_fresh_shamir(normalized_axis, fresh, metric="normalized_recovery")
        fresh_windows = [int(record["window"]) for record in fresh]
        reference_windows = reference[0]["windows"] if h == 2 else []
        all_windows = [*fresh_windows, *reference_windows]
        _finish_shamir_axis(
            raw_axis,
            h=h,
            raw=True,
            windows=all_windows,
            chance=float(fresh[0]["chance"]) if h == 1 else None,
        )
        _finish_shamir_axis(normalized_axis, h=h, raw=False, windows=all_windows)
        raw_axis.set_title(
            "Fresh h=1 controlled suite"
            if h == 1
            else "h=2 reviewer curves + fresh spectral points"
        )
    axes[0, 0].set_ylabel("Secret balanced accuracy\n(raw; chance is not zero)")
    axes[1, 0].set_ylabel(
        "Secret recovery\n$(\\mathrm{accuracy}-\\mathrm{chance})/"
        "(1-\\mathrm{chance})$"
    )
    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes.flat:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels, strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Polynomial-clock secret recovery: raw and chance-normalized views",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.96))
    return _save_figure(fig, figures_dir, "controlled_shamir_recovery")


def _band_label(band: Sequence[int]) -> str:
    values = [int(value) for value in band]
    if len(values) == 1:
        return f"$k={values[0]}$"
    if values == list(range(values[0], values[-1] + 1)):
        return f"$k={values[0]}$–${values[-1]}$"
    return "$k=$" + ",".join(str(value) for value in values)


def _heatmap_norm(records: Sequence[dict[str, Any]]) -> Normalize:
    values = np.concatenate(
        [np.asarray(record["r2_mean"], dtype=np.float64).ravel() for record in records]
    )
    minimum = min(float(values.min()), -0.02)
    maximum = max(float(values.max()), 0.02)
    return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)


def _annotation_color(color: Sequence[float]) -> str:
    red, green, blue = color[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.58 else "white"


def plot_hmm(analysis: dict[str, Any], figures_dir: Path) -> tuple[Path, Path]:
    latent = analysis["hmm"]["latent_r2"]
    band_records = analysis["hmm"]["mixed_band_r2"]
    if not band_records:
        raise RuntimeError("no mixed-HMM spectral band records were found")
    n_heatmaps = len(band_records)
    fig = plt.figure(figsize=(11.8, 7.9))
    grid = fig.add_gridspec(
        2,
        n_heatmaps + 1,
        height_ratios=(1.0, 1.35),
        width_ratios=(*([1.0] * n_heatmaps), 0.045),
        hspace=0.4,
        wspace=0.24,
    )
    latent_axis = fig.add_subplot(grid[0, :n_heatmaps])
    x = np.arange(len(HMM_TASK_ORDER), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(MODEL_ORDER))
    for offset, model in zip(offsets, MODEL_ORDER, strict=True):
        selected = {
            record["task"]: record for record in latent if record["model"] == model
        }
        if not selected:
            continue
        means = np.asarray(
            [selected[task]["latent_r2"]["mean"] for task in HMM_TASK_ORDER]
        )
        errors = np.asarray(
            [selected[task]["latent_r2"]["std"] for task in HMM_TASK_ORDER]
        )
        latent_axis.errorbar(
            x + offset,
            means,
            yerr=errors,
            linestyle="none",
            marker=MODEL_MARKERS[model],
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color=MODEL_COLORS[model],
            capsize=2.5,
            label=MODEL_LABELS[model],
            zorder=3,
        )
    latent_axis.axhline(0, color="#888888", linewidth=0.7, zorder=0)
    latent_axis.set_xticks(x, [HMM_TASK_LABELS[task] for task in HMM_TASK_ORDER])
    latent_axis.set_ylabel("Held-out latent-state $R^2$")
    latent_axis.set_title("Linear recovery of controlled HMM factors")
    latent_axis.grid(axis="y", linestyle=":", alpha=0.35)
    latent_axis.legend(loc="best", ncol=2, frameon=False)

    norm = _heatmap_norm(band_records)
    cmap = plt.get_cmap("RdBu")
    images = []
    for column, record in enumerate(band_records):
        axis = fig.add_subplot(grid[1, column])
        values = np.asarray(record["r2_mean"], dtype=np.float64)
        image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        images.append(image)
        expected = record["expected_band_per_source"]
        for source in range(values.shape[0]):
            for band in range(values.shape[1]):
                marker = " ★" if band == int(expected[source]) else ""
                color = _annotation_color(cmap(norm(values[source, band])))
                axis.text(
                    band,
                    source,
                    f"{values[source, band]:.2f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold" if marker else "normal",
                )
        axis.set_xticks(
            np.arange(values.shape[1]),
            [_band_label(band) for band in record["bands"]],
            rotation=25,
            ha="right",
        )
        axis.set_yticks(
            np.arange(values.shape[0]),
            [f"source {index + 1}  ($\\lambda={value:+.1f}$)" for index, value in enumerate(record["lambdas"])],
        )
        axis.set_xlabel("Assigned DCT band")
        axis.set_title(MODEL_LABELS[record["model"]])
        if column == 0:
            axis.set_ylabel("Mixed-HMM latent source")
        else:
            axis.tick_params(labelleft=False)
    colorbar_axis = fig.add_subplot(grid[1, n_heatmaps])
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("Band-only latent $R^2$ (seed mean)")
    fig.legend(
        handles=[Patch(facecolor="none", edgecolor="none", label="★ expected maximum-power band")],
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("Frequency-selective recovery on controlled HMMs", y=0.995, fontsize=13)
    fig.subplots_adjust(top=0.92, bottom=0.12)
    return _save_figure(fig, figures_dir, "controlled_hmm_frequency_localization")


def plot_denoising(analysis: dict[str, Any], figures_dir: Path) -> tuple[Path, Path]:
    denoising = analysis["denoising"]
    energy = denoising["energy"]
    ridge = denoising["ridge"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))

    energy_axis = axes[0]
    x = np.arange(len(energy), dtype=float)
    dc = np.asarray([record["dc_share"]["mean"] for record in energy])
    ac = np.asarray([record["ac_share"]["mean"] for record in energy])
    dc_errors = np.asarray([record["dc_share"]["std"] for record in energy])
    energy_axis.bar(x, dc, color="#0072B2", width=0.72, label="DC ($k=0$)")
    energy_axis.bar(
        x,
        ac,
        bottom=dc,
        color="#E69F00",
        width=0.72,
        label="AC ($k>0$)",
    )
    energy_axis.errorbar(
        x,
        dc,
        yerr=dc_errors,
        linestyle="none",
        color="#202020",
        linewidth=0.8,
        capsize=2.5,
        zorder=4,
    )
    for index, record in enumerate(energy):
        values = np.asarray(record["dc_share"]["values"], dtype=float)
        jitter = np.linspace(-0.09, 0.09, len(values))
        energy_axis.scatter(
            index + jitter,
            values,
            s=12,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.6,
            zorder=5,
        )
    energy_axis.axvline(1.5, color="#B0B0B0", linestyle=":", linewidth=0.8)
    energy_axis.set_xticks(x, [record["label"] for record in energy])
    energy_axis.set_ylim(0, 1)
    energy_axis.set_ylabel("DCT energy share")
    energy_axis.set_title("Where the denoising model puts energy", pad=32)
    energy_axis.text(
        0.5,
        1.005,
        "learned representation",
        transform=energy_axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    energy_axis.text(
        2.5,
        1.005,
        "data reference",
        transform=energy_axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    energy_axis.legend(loc="lower right", frameon=False)
    energy_axis.grid(axis="y", linestyle=":", alpha=0.3)

    ridge_axis = axes[1]
    ridge_x = np.arange(len(ridge), dtype=float)
    ridge_means = np.asarray([record["mean_r2"]["mean"] for record in ridge])
    ridge_errors = np.asarray([record["mean_r2"]["std"] for record in ridge])
    ridge_colors = ("#595959", "#0072B2", "#E69F00")
    ridge_axis.bar(
        ridge_x,
        ridge_means,
        yerr=ridge_errors,
        color=ridge_colors,
        width=0.68,
        error_kw={"capsize": 3, "elinewidth": 0.9, "ecolor": "#202020"},
    )
    for index, record in enumerate(ridge):
        values = np.asarray(record["mean_r2"]["values"], dtype=float)
        jitter = np.linspace(-0.07, 0.07, len(values))
        ridge_axis.scatter(
            index + jitter,
            values,
            s=18,
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.7,
            zorder=4,
        )
    ridge_axis.axhline(0, color="#777777", linewidth=0.7)
    ridge_axis.set_xticks(ridge_x, ["Full code", "DC features", "AC features"])
    lower = min(-0.025, float(ridge_means.min() - 2 * ridge_errors.max()))
    upper = max(0.05, float(ridge_means.max() + 2 * ridge_errors.max()))
    ridge_axis.set_ylim(lower, upper * 1.08)
    ridge_axis.set_ylabel("Hidden-state Ridge $R^2$")
    ridge_axis.set_title("Which frequency block carries the signal")
    ridge_axis.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle(
        "Denoising replay: DC/AC energy and task-relevant code usage",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_figure(fig, figures_dir, "denoising_frequency_usage")


def _write_analysis(
    analysis: dict[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    flattened = flatten_analysis(analysis)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "task",
        "model",
        "window",
        "source",
        "band",
        "metric",
        "n",
        "mean",
        "std",
        "values",
        "expected_band",
        "expected_energy",
        "provenance",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render controlled frequency results after the full JSONL run is fetched."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=DEFAULT_FROZEN_CONFIG,
        help="Config frozen with the fetched 70-cell run; used for task metadata.",
    )
    parser.add_argument(
        "--controlled-results",
        type=Path,
        default=DEFAULT_CONTROLLED_RESULTS,
    )
    parser.add_argument(
        "--controlled-summary",
        type=Path,
        default=DEFAULT_CONTROLLED_SUMMARY,
    )
    parser.add_argument(
        "--denoising-result",
        type=Path,
        default=DEFAULT_DENOISING_RESULT,
    )
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--aggregate-json", type=Path, default=DEFAULT_AGGREGATE_JSON)
    parser.add_argument("--aggregate-csv", type=Path, default=DEFAULT_AGGREGATE_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_config = _read_json(args.config)
    run_config = _read_json(args.frozen_config)
    summary = _read_json(args.controlled_summary)
    rows = _latest_controlled_rows(args.controlled_results)
    _validate_controlled_run(rows, summary)
    denoising = _read_json(args.denoising_result)
    input_paths = {
        "reviewer_reference_config": args.config,
        "frozen_run_config": args.frozen_config,
        "controlled_results": args.controlled_results,
        "controlled_summary": args.controlled_summary,
        "denoising_result": args.denoising_result,
    }
    analysis = build_analysis(
        run_config=run_config,
        reference_config=reference_config,
        summary=summary,
        rows=rows,
        denoising=denoising,
        input_paths=input_paths,
    )
    _configure_style()
    outputs = [
        *plot_shamir(analysis, args.figures_dir),
        *plot_hmm(analysis, args.figures_dir),
        *plot_denoising(analysis, args.figures_dir),
    ]
    _write_analysis(
        analysis,
        json_path=args.aggregate_json,
        csv_path=args.aggregate_csv,
    )
    outputs.extend((args.aggregate_json, args.aggregate_csv))
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
