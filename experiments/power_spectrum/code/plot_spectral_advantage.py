"""Aggregate and plot the matched-parameter spectral-advantage experiment.

The input is the append-only ``results.jsonl`` emitted by
``run_controlled_frequency_suite``.  This module deliberately does not depend
on a completed run, frozen config, or summary file: it can be tested before the
GPU job exists and can analyze a complete JSONL file once it arrives.

Two figures keep three frequency diagnostics semantically separate:

* expected DCT-band power is a property of the synthetic latent data;
* selection-event share is observed model behavior;
* adaptive difficulty weight is learned loss emphasis, not an estimate of
  spectral power.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = POWER_ROOT / "results" / "spectral_advantage_remote" / "results.jsonl"
DEFAULT_JSON = POWER_ROOT / "results" / "spectral_advantage_analysis.json"
DEFAULT_CSV = POWER_ROOT / "results" / "spectral_advantage_analysis.csv"
DEFAULT_FIGURES_DIR = POWER_ROOT / "figures"

MODEL_ORDER = (
    "txc",
    "sae",
    "spectral_global",
    "spectral_adaptive",
    "spectral_dct_global",
    "spectral_fourier_global",
    "spectral_fourier_adaptive",
    "spectral_fourier_routed",
)
MODEL_LABELS = {
    "txc": "TXC-post",
    "sae": "BatchTopK SAE",
    "spectral_global": "Spectral global",
    "spectral_adaptive": "Spectral adaptive weights",
    "spectral_dct_global": "Spectral DCT global",
    "spectral_fourier_global": "Spectral Fourier global",
    "spectral_fourier_adaptive": "Fourier learned loss",
    "spectral_fourier_routed": "Fourier learned routing",
}
MODEL_COLORS = {
    "txc": "#8C6D31",
    "sae": "#D55E00",
    "spectral_global": "#0072B2",
    "spectral_adaptive": "#009E73",
    "spectral_dct_global": "#0072B2",
    "spectral_fourier_global": "#56B4E9",
    "spectral_fourier_adaptive": "#009E73",
    "spectral_fourier_routed": "#CC79A7",
}
MODEL_MARKERS = {
    "txc": "s",
    "sae": "v",
    "spectral_global": "^",
    "spectral_adaptive": "D",
    "spectral_dct_global": "^",
    "spectral_fourier_global": "o",
    "spectral_fourier_adaptive": "D",
    "spectral_fourier_routed": "P",
}
TASK_ORDER = (
    "hmm_mixed_t8_matched_params",
    "narrowband_offgrid_t8_matched_params",
    "narrowband_offgrid_t16_matched_params",
    "hmm_slow_t8_routed_screen",
    "hmm_alternating_t8_routed_screen",
    "narrowband_sparse_balanced_t8",
    "narrowband_sparse_high_crowded_t8",
)
TASK_LABELS = {
    "hmm_mixed_t8_matched_params": "Mixed HMM\n$T=8$",
    "narrowband_offgrid_t8_matched_params": "Off-grid narrowband\n$T=8$",
    "narrowband_offgrid_t16_matched_params": "Off-grid narrowband\n$T=16$",
    "hmm_slow_t8_routed_screen": "Slow HMM\n$T=8$",
    "hmm_alternating_t8_routed_screen": "Alternating HMM\n$T=8$",
    "narrowband_sparse_balanced_t8": "Sparse NB\nbalanced",
    "narrowband_sparse_high_crowded_t8": "Sparse NB\nhigh crowded",
}
ORIGINAL_MODELS = ("txc", "spectral_global", "spectral_adaptive")
ROUTED_HMM_MODELS = (
    "txc",
    "spectral_fourier_global",
    "spectral_fourier_adaptive",
    "spectral_fourier_routed",
)
ROUTED_SPARSE_MODELS = (
    "txc",
    "spectral_dct_global",
    "spectral_fourier_global",
    "spectral_fourier_adaptive",
    "spectral_fourier_routed",
)
EXPECTED_MODELS_BY_TASK = (
    {task: ORIGINAL_MODELS for task in TASK_ORDER[:3]}
    | {task: ROUTED_HMM_MODELS for task in TASK_ORDER[3:5]}
    | {task: ROUTED_SPARSE_MODELS for task in TASK_ORDER[5:]}
)
LEARNED_DIFFICULTY_MODELS = {
    "spectral_adaptive",
    "spectral_fourier_adaptive",
    "spectral_fourier_routed",
}
ROUTED_MODELS = {"spectral_fourier_routed"}
OPTIONAL_BASELINES = {"sae"}

EXPECTED_POWER_SEMANTICS = (
    "Expected latent DCT-band power: a data property, not a learned model weight."
)
SELECTION_SEMANTICS = (
    "Selection-event share: the fraction of nonzero code selections observed in each band."
)
DIFFICULTY_WEIGHT_SEMANTICS = (
    "Adaptive difficulty weight: learned adversarial loss emphasis, not expected power."
)
ROUTING_SCALE_SEMANTICS = (
    "Routing scale: detached score multiplier used for support competition, "
    "not probability or spectral power."
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _task_label(task: str) -> str:
    if task in TASK_LABELS:
        return TASK_LABELS[task]
    return task.replace("_matched_params", "").replace("_", " ").title()


def _task_sort_key(task: str) -> tuple[int, str]:
    return (
        TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER),
        task,
    )


def _band_label(band: Sequence[int], *, basis: str) -> str:
    values = [int(value) for value in band]
    if values == [0]:
        return "DC"
    prefix = "Fourier" if basis == "fourier" else "DCT"
    if len(values) == 1:
        return f"{prefix} {values[0]}"
    if values == list(range(values[0], values[-1] + 1)):
        return f"{prefix} {values[0]}–{values[-1]}"
    return f"{prefix} " + ",".join(map(str, values))


def load_latest_ok_rows(path: Path) -> list[dict[str, Any]]:
    """Read the latest successful non-smoke row for every cell."""

    if not path.is_file():
        raise FileNotFoundError(f"results JSONL is missing: {path}")
    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or "cell_id" not in row:
            raise ValueError(f"{path}:{line_number}: expected a JSON object with cell_id")
        latest[str(row["cell_id"])] = row
    rows = [
        row for row in latest.values() if row.get("status") == "ok" and not bool(row.get("smoke"))
    ]
    if not rows:
        raise RuntimeError(f"{path} has no successful non-smoke result rows")
    return rows


def _sample_stats(values: Iterable[float]) -> dict[str, Any]:
    sequence = [float(value) for value in values]
    if not sequence:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "n": len(sequence),
        "mean": statistics.fmean(sequence),
        "std": statistics.stdev(sequence) if len(sequence) > 1 else 0.0,
        "min": min(sequence),
        "max": max(sequence),
        "values": sequence,
    }


def _vector_stats(values: Sequence[Sequence[float]]) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"expected a non-empty matrix; got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("frequency diagnostic contains a non-finite value")
    return {
        "n": int(matrix.shape[0]),
        "mean": matrix.mean(axis=0).tolist(),
        "std": (
            matrix.std(axis=0, ddof=1)
            if matrix.shape[0] > 1
            else np.zeros(matrix.shape[1], dtype=np.float64)
        ).tolist(),
        "values": matrix.tolist(),
    }


def _constant(values: Sequence[Any], *, name: str) -> Any:
    if not values:
        raise ValueError(f"{name} is empty")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"{name} differs across matched rows")
    return first


def _group_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        task = str(row["task"])
        model = str(row["model"])
        seed = int(row["seed"])
        key = (task, model, seed)
        if key in seen:
            raise RuntimeError(f"multiple latest cells exist for task/model/seed {key}")
        seen.add(key)
        grouped[(task, model)].append(row)
    for selected in grouped.values():
        selected.sort(key=lambda row: int(row["seed"]))
    return dict(grouped)


def _validate_panel(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    allow_incomplete: bool,
) -> list[str]:
    tasks = sorted({task for task, _ in grouped}, key=_task_sort_key)
    for task in tasks:
        present = {model for candidate, model in grouped if candidate == task}
        unknown = present - set(MODEL_ORDER)
        if unknown:
            raise ValueError(f"{task}: unknown models {sorted(unknown)}")
        expected = set(EXPECTED_MODELS_BY_TASK.get(task, present))
        unexpected = present - expected - OPTIONAL_BASELINES
        if unexpected:
            raise RuntimeError(f"{task}: models {sorted(unexpected)} are outside its task panel")
        if not allow_incomplete:
            missing = expected - present
            if missing:
                raise RuntimeError(f"{task}: missing models {sorted(missing)}")
        seed_sets = {
            model: {int(row["seed"]) for row in grouped[(task, model)]} for model in present
        }
        if "txc" not in seed_sets:
            if not allow_incomplete:
                raise RuntimeError(f"{task}: TXC is required for paired deltas")
            continue
        txc_seeds = seed_sets["txc"]
        for model, seeds in seed_sets.items():
            if model != "txc" and seeds != txc_seeds:
                raise RuntimeError(
                    f"{task}/{model}: seeds {sorted(seeds)} do not match "
                    f"TXC seeds {sorted(txc_seeds)}"
                )
    return tasks


def _metric_stats(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    present = [metric in row.get("metrics", {}) for row in rows]
    if required and not all(present):
        raise KeyError(f"metric {metric!r} is missing from one or more rows")
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"metric {metric!r} is only present for some seeds")
    return _sample_stats(row["metrics"][metric] for row in rows)


def _performance_records(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    tasks: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        for model in MODEL_ORDER:
            selected = grouped.get((task, model), [])
            if not selected:
                continue
            seeds = [int(row["seed"]) for row in selected]
            counts = [int(row.get("training", {}).get("parameter_count", -1)) for row in selected]
            if any(value < 0 for value in counts):
                raise KeyError(f"{task}/{model}: training.parameter_count is missing")
            record = {
                "task": task,
                "task_label": _task_label(task),
                "family": _constant(
                    [str(row["family"]) for row in selected],
                    name=f"{task}/{model} family",
                ),
                "window": int(
                    _constant(
                        [int(row["window"]) for row in selected],
                        name=f"{task}/{model} window",
                    )
                ),
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seeds": seeds,
                "d_sae": int(
                    _constant(
                        [int(row["d_sae"]) for row in selected],
                        name=f"{task}/{model} d_sae",
                    )
                ),
                "parameter_count": int(_constant(counts, name=f"{task}/{model} parameter count")),
                "direct_latent_r2": _metric_stats(selected, "direct_latent_r2"),
                "direct_latent_r2_active": _metric_stats(
                    selected,
                    "direct_latent_r2_active",
                    required=False,
                ),
                "inactive_to_active_energy_ratio": _metric_stats(
                    selected,
                    "inactive_to_active_energy_ratio",
                    required=False,
                ),
                "band_localization_accuracy": _metric_stats(
                    selected,
                    "band_localization_accuracy",
                    required=False,
                ),
                "latent_r2": _metric_stats(selected, "latent_r2", required=False),
                "raw_projection_r2": _metric_stats(selected, "raw_projection_r2"),
                "nmse": _metric_stats(selected, "nmse"),
                "l0_per_window": _metric_stats(selected, "l0_per_window"),
            }
            records.append(record)
    return records


def _paired_delta_records(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    tasks: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        baseline = {
            int(row["seed"]): float(row["metrics"]["direct_latent_r2"])
            for row in grouped.get((task, "txc"), [])
        }
        if not baseline:
            continue
        for model in MODEL_ORDER[1:]:
            selected = {
                int(row["seed"]): float(row["metrics"]["direct_latent_r2"])
                for row in grouped.get((task, model), [])
            }
            seeds = sorted(set(baseline) & set(selected))
            if not seeds:
                continue
            deltas = [selected[seed] - baseline[seed] for seed in seeds]
            records.append(
                {
                    "task": task,
                    "task_label": _task_label(task),
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "baseline": "txc",
                    "baseline_label": MODEL_LABELS["txc"],
                    "seeds": seeds,
                    "delta_direct_latent_r2": _sample_stats(deltas),
                    "seed_values": {
                        str(seed): delta for seed, delta in zip(seeds, deltas, strict=True)
                    },
                }
            )
    return records


def _normalize_share(values: Sequence[float], *, name: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or (array < -1e-9).any():
        raise ValueError(f"{name} is not a finite non-negative vector")
    total = float(array.sum())
    if total <= 0:
        raise ValueError(f"{name} has zero total mass")
    return (array / total).tolist()


def _temporal_basis(row: dict[str, Any]) -> str:
    metrics = row["metrics"]
    usage = metrics["spectral_usage"]
    basis = str(metrics.get("temporal_basis", usage.get("temporal_basis", "dct"))).lower()
    if basis not in {"dct", "fourier"}:
        raise ValueError(f"{row['task']}/{row['model']}: unknown temporal basis {basis!r}")
    return basis


def _expected_basis_matrix(row: dict[str, Any]) -> np.ndarray:
    metrics = row["metrics"]
    key = (
        "expected_basis_band_energy_per_source"
        if "expected_basis_band_energy_per_source" in metrics
        else "expected_dct_band_energy_per_source"
    )
    if key not in metrics:
        raise KeyError(f"{row['task']}/{row['model']}: no expected band-energy metric")
    return np.asarray(metrics[key], dtype=np.float64)


def _frequency_diagnostics(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    tasks: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        spectral_rows = {
            model: grouped.get((task, model), [])
            for model in MODEL_ORDER
            if model not in {"txc", *OPTIONAL_BASELINES}
        }
        present = {model: rows for model, rows in spectral_rows.items() if rows}
        if not present:
            continue
        basis_groups: defaultdict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        for model, selected in present.items():
            bases = {_temporal_basis(row) for row in selected}
            if len(bases) != 1:
                raise RuntimeError(f"{task}/{model}: temporal basis differs across seeds")
            basis_groups[bases.pop()][model] = selected

        for basis in ("dct", "fourier"):
            basis_models = basis_groups.get(basis)
            if not basis_models:
                continue
            all_rows = [row for selected in basis_models.values() for row in selected]
            bands = _constant(
                [row["metrics"]["spectral_usage"]["bands"] for row in all_rows],
                name=f"{task}/{basis} basis-row bands",
            )
            frequency_bin_bands = _constant(
                [
                    row["metrics"]["spectral_usage"].get(
                        "frequency_bin_bands",
                        row["metrics"]["spectral_usage"]["bands"],
                    )
                    for row in all_rows
                ],
                name=f"{task}/{basis} physical-frequency bands",
            )
            band_labels = [_band_label(band, basis=basis) for band in frequency_bin_bands]

            # Expected power is a data diagnostic duplicated by each model.
            # Retain one copy per seed after checking model agreement within
            # the same temporal basis.
            expected_by_seed: dict[int, np.ndarray] = {}
            for row in all_rows:
                seed = int(row["seed"])
                matrix = _expected_basis_matrix(row)
                if matrix.ndim != 2 or matrix.shape[1] != len(bands):
                    raise ValueError(
                        f"{task}/{basis}: expected-power matrix has shape "
                        f"{matrix.shape}, expected (*, {len(bands)})"
                    )
                if seed in expected_by_seed and not np.allclose(
                    expected_by_seed[seed],
                    matrix,
                    atol=1e-10,
                    rtol=1e-8,
                ):
                    raise RuntimeError(
                        f"{task}/{basis}: expected latent power differs by model for seed {seed}"
                    )
                expected_by_seed[seed] = matrix
            expected_source_seed_values = [
                expected_by_seed[seed] for seed in sorted(expected_by_seed)
            ]
            expected_band_seed_values = [
                _normalize_share(
                    matrix.mean(axis=0),
                    name=f"{task}/{basis} expected power seed {seed}",
                )
                for seed, matrix in zip(
                    sorted(expected_by_seed),
                    expected_source_seed_values,
                    strict=True,
                )
            ]

            selection: dict[str, Any] = {}
            for model, selected in basis_models.items():
                values = [
                    _normalize_share(
                        row["metrics"]["spectral_usage"]["selection_event_share"],
                        name=f"{task}/{model} selection share",
                    )
                    for row in selected
                ]
                selection[model] = _vector_stats(values)

            difficulty_models: dict[str, Any] = {}
            routing_models: dict[str, Any] = {}
            for model, selected in basis_models.items():
                if model in LEARNED_DIFFICULTY_MODELS:
                    weights = [
                        _normalize_share(
                            row["metrics"]["spectral_usage"]["learned_frequency_weight"],
                            name=f"{task}/{model} difficulty weight",
                        )
                        for row in selected
                    ]
                    priors = [
                        _normalize_share(
                            row["metrics"]["spectral_usage"]["frequency_weight_prior"],
                            name=f"{task}/{model} frequency-weight prior",
                        )
                        for row in selected
                    ]
                    prior = np.asarray(priors[0])
                    if any(
                        not np.allclose(
                            prior,
                            candidate,
                            atol=1e-10,
                            rtol=1e-8,
                        )
                        for candidate in priors[1:]
                    ):
                        raise RuntimeError(
                            f"{task}/{model}: frequency-weight prior differs by seed"
                        )
                    difficulty_models[model] = {
                        **_vector_stats(weights),
                        "prior": prior.tolist(),
                    }
                if model in ROUTED_MODELS:
                    scales = [
                        row["metrics"]["spectral_usage"]["frequency_routing_scale"]
                        for row in selected
                    ]
                    routing_models[model] = _vector_stats(scales)

            expected_stats = _vector_stats(expected_band_seed_values)
            expected_stats["per_source_seed_values"] = [
                matrix.tolist() for matrix in expected_source_seed_values
            ]
            legacy_adaptive = difficulty_models.get("spectral_adaptive")
            if legacy_adaptive is not None:
                legacy_adaptive = {
                    "semantics": DIFFICULTY_WEIGHT_SEMANTICS,
                    **legacy_adaptive,
                }
            multiple_bases = len(basis_groups) > 1
            basis_label = "Fourier" if basis == "fourier" else "DCT"
            records.append(
                {
                    "task": task,
                    "task_label": _task_label(task),
                    "row_label": (
                        f"{_task_label(task).replace(chr(10), ' ')} — {basis_label}"
                        if multiple_bases
                        else _task_label(task).replace("\n", " ")
                    ),
                    "temporal_basis": basis,
                    "basis_label": basis_label,
                    "models": [model for model in MODEL_ORDER if model in basis_models],
                    "bands": [[int(value) for value in band] for band in bands],
                    "frequency_bin_bands": [
                        [int(value) for value in band] for band in frequency_bin_bands
                    ],
                    "band_labels": band_labels,
                    "expected_power": {
                        "semantics": EXPECTED_POWER_SEMANTICS,
                        "metric_key": ("expected_basis_band_energy_per_source"),
                        **expected_stats,
                    },
                    "selection_event_share": {
                        "semantics": SELECTION_SEMANTICS,
                        "models": selection,
                    },
                    "learned_difficulty_weights": {
                        "semantics": DIFFICULTY_WEIGHT_SEMANTICS,
                        "models": difficulty_models,
                    },
                    "routing_scales": {
                        "semantics": ROUTING_SCALE_SEMANTICS,
                        "models": routing_models,
                    },
                    # Backward-compatible alias used by the original
                    # spectral-advantage analysis/tests.
                    "adaptive_difficulty_weight": legacy_adaptive,
                }
            )
    return records


def _routing_trajectories(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    tasks: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate learned routed-score multipliers over training."""

    records: list[dict[str, Any]] = []
    for task in tasks:
        for model in sorted(ROUTED_MODELS):
            selected = grouped.get((task, model), [])
            if not selected:
                continue
            histories = [
                row.get("training", {}).get("metric_history", []) for row in selected
            ]
            has_routing = [
                bool(history)
                and any(
                    key.startswith("frequency_routing_scale_")
                    for key in history[0]
                )
                for history in histories
            ]
            if not any(has_routing):
                continue
            if not all(has_routing):
                raise RuntimeError(f"{task}/{model}: routing history is missing for some seeds")

            usage = selected[0]["metrics"]["spectral_usage"]
            physical_bands = usage.get("frequency_bin_bands", usage["bands"])
            band_labels = [
                _band_label(band, basis=_temporal_basis(selected[0]))
                for band in physical_bands
            ]
            band_count = len(band_labels)
            steps_by_seed = [
                {
                    int(item["step"]): [
                        float(item[f"frequency_routing_scale_{band}"])
                        for band in range(band_count)
                    ]
                    for item in history
                    if all(
                        f"frequency_routing_scale_{band}" in item
                        for band in range(band_count)
                    )
                }
                for history in histories
            ]
            common_steps = sorted(
                set.intersection(*(set(values) for values in steps_by_seed))
            )
            if not common_steps:
                raise RuntimeError(f"{task}/{model}: no common routing-history steps")
            values = np.asarray(
                [
                    [step_values[step] for step in common_steps]
                    for step_values in steps_by_seed
                ],
                dtype=np.float64,
            )
            records.append(
                {
                    "task": task,
                    "task_label": _task_label(task),
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "seeds": [int(row["seed"]) for row in selected],
                    "steps": common_steps,
                    "band_labels": band_labels,
                    "routing_scale": {
                        "semantics": ROUTING_SCALE_SEMANTICS,
                        "mean": values.mean(axis=0).tolist(),
                        "std": (
                            values.std(axis=0, ddof=1)
                            if values.shape[0] > 1
                            else np.zeros_like(values[0])
                        ).tolist(),
                        "values": values.tolist(),
                    },
                }
            )
    return records


def build_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    source: str = "results.jsonl",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate and aggregate result rows into a plot-ready JSON object."""

    grouped = _group_rows(rows)
    tasks = _validate_panel(grouped, allow_incomplete=allow_incomplete)
    performance = _performance_records(grouped, tasks)
    paired = _paired_delta_records(grouped, tasks)
    diagnostics = _frequency_diagnostics(grouped, tasks)
    routing_trajectories = _routing_trajectories(grouped, tasks)
    present_models = {str(record["model"]) for record in performance}
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "source": source,
        "models": [
            {"name": model, "label": MODEL_LABELS[model]}
            for model in MODEL_ORDER
            if model in present_models
        ],
        "tasks": [{"name": task, "label": _task_label(task)} for task in tasks],
        "performance": performance,
        "paired_deltas_vs_txc": paired,
        "frequency_diagnostics": diagnostics,
        "routing_trajectories": routing_trajectories,
        "semantic_guardrail": {
            "expected_power": EXPECTED_POWER_SEMANTICS,
            "selection": SELECTION_SEMANTICS,
            "adaptive_weight": DIFFICULTY_WEIGHT_SEMANTICS,
            "routing_scale": ROUTING_SCALE_SEMANTICS,
        },
    }


def _diagnostics_by_model(
    analysis: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(record["task"]), str(model)): record
        for record in analysis["frequency_diagnostics"]
        for model in record["models"]
    }


def _delta_map(
    analysis: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(record["task"]), str(record["model"])): record
        for record in analysis["paired_deltas_vs_txc"]
    }


def write_aggregate_csv(analysis: dict[str, Any], path: Path) -> Path:
    """Write one flat task/model row while preserving vector diagnostics."""

    diagnostics = _diagnostics_by_model(analysis)
    deltas = _delta_map(analysis)
    fields = [
        "task",
        "task_label",
        "family",
        "window",
        "model",
        "model_label",
        "n",
        "seeds",
        "d_sae",
        "parameter_count",
        "direct_latent_r2_mean",
        "direct_latent_r2_std",
        "direct_latent_r2_active_mean",
        "direct_latent_r2_active_std",
        "inactive_to_active_energy_ratio_mean",
        "inactive_to_active_energy_ratio_std",
        "band_localization_accuracy_mean",
        "band_localization_accuracy_std",
        "paired_delta_vs_txc_mean",
        "paired_delta_vs_txc_std",
        "raw_projection_r2_mean",
        "raw_projection_r2_std",
        "latent_r2_mean",
        "latent_r2_std",
        "nmse_mean",
        "nmse_std",
        "l0_per_window_mean",
        "l0_per_window_std",
        "temporal_basis",
        "bands",
        "frequency_bin_bands",
        "expected_power_by_band",
        "selection_event_share",
        "adaptive_difficulty_weight",
        "difficulty_weight_prior",
        "frequency_routing_scale",
    ]
    flattened: list[dict[str, Any]] = []
    for record in analysis["performance"]:
        task = str(record["task"])
        model = str(record["model"])
        diagnostic = diagnostics.get((task, model))
        delta = deltas.get((task, model))
        selection = None
        if diagnostic is not None:
            selection = diagnostic["selection_event_share"]["models"].get(model)
        difficulty = (
            diagnostic["learned_difficulty_weights"]["models"].get(model)
            if diagnostic is not None
            else None
        )
        routing = (
            diagnostic["routing_scales"]["models"].get(model) if diagnostic is not None else None
        )
        flattened.append(
            {
                "task": task,
                "task_label": str(record["task_label"]).replace("\n", " "),
                "family": record["family"],
                "window": record["window"],
                "model": model,
                "model_label": record["model_label"],
                "n": record["direct_latent_r2"]["n"],
                "seeds": json.dumps(record["seeds"], separators=(",", ":")),
                "d_sae": record["d_sae"],
                "parameter_count": record["parameter_count"],
                "direct_latent_r2_mean": record["direct_latent_r2"]["mean"],
                "direct_latent_r2_std": record["direct_latent_r2"]["std"],
                "direct_latent_r2_active_mean": (
                    record["direct_latent_r2_active"]["mean"]
                    if record["direct_latent_r2_active"]
                    else ""
                ),
                "direct_latent_r2_active_std": (
                    record["direct_latent_r2_active"]["std"]
                    if record["direct_latent_r2_active"]
                    else ""
                ),
                "inactive_to_active_energy_ratio_mean": (
                    record["inactive_to_active_energy_ratio"]["mean"]
                    if record["inactive_to_active_energy_ratio"]
                    else ""
                ),
                "inactive_to_active_energy_ratio_std": (
                    record["inactive_to_active_energy_ratio"]["std"]
                    if record["inactive_to_active_energy_ratio"]
                    else ""
                ),
                "band_localization_accuracy_mean": (
                    record["band_localization_accuracy"]["mean"]
                    if record["band_localization_accuracy"]
                    else ""
                ),
                "band_localization_accuracy_std": (
                    record["band_localization_accuracy"]["std"]
                    if record["band_localization_accuracy"]
                    else ""
                ),
                "paired_delta_vs_txc_mean": (
                    0.0
                    if model == "txc"
                    else delta["delta_direct_latent_r2"]["mean"]
                    if delta
                    else ""
                ),
                "paired_delta_vs_txc_std": (
                    0.0
                    if model == "txc"
                    else delta["delta_direct_latent_r2"]["std"]
                    if delta
                    else ""
                ),
                "raw_projection_r2_mean": record["raw_projection_r2"]["mean"],
                "raw_projection_r2_std": record["raw_projection_r2"]["std"],
                "latent_r2_mean": (record["latent_r2"]["mean"] if record["latent_r2"] else ""),
                "latent_r2_std": (record["latent_r2"]["std"] if record["latent_r2"] else ""),
                "nmse_mean": record["nmse"]["mean"],
                "nmse_std": record["nmse"]["std"],
                "l0_per_window_mean": record["l0_per_window"]["mean"],
                "l0_per_window_std": record["l0_per_window"]["std"],
                "temporal_basis": (diagnostic["temporal_basis"] if diagnostic else ""),
                "bands": (
                    json.dumps(diagnostic["bands"], separators=(",", ":")) if diagnostic else ""
                ),
                "frequency_bin_bands": (
                    json.dumps(
                        diagnostic["frequency_bin_bands"],
                        separators=(",", ":"),
                    )
                    if diagnostic
                    else ""
                ),
                "expected_power_by_band": (
                    json.dumps(
                        diagnostic["expected_power"]["mean"],
                        separators=(",", ":"),
                    )
                    if diagnostic
                    else ""
                ),
                "selection_event_share": (
                    json.dumps(selection["mean"], separators=(",", ":")) if selection else ""
                ),
                "adaptive_difficulty_weight": (
                    json.dumps(difficulty["mean"], separators=(",", ":")) if difficulty else ""
                ),
                "difficulty_weight_prior": (
                    json.dumps(difficulty["prior"], separators=(",", ":")) if difficulty else ""
                ),
                "frequency_routing_scale": (
                    json.dumps(routing["mean"], separators=(",", ":")) if routing else ""
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)
    os.replace(temporary, path)
    return path


def write_aggregate_json(analysis: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def _save_figure(
    figure: plt.Figure,
    figures_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def plot_performance(
    analysis: dict[str, Any],
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Plot direct latent R² and seed-paired improvements over TXC."""

    _configure_style()
    tasks = [str(item["name"]) for item in analysis["tasks"]]
    performance = {
        (str(record["task"]), str(record["model"])): record for record in analysis["performance"]
    }
    deltas = _delta_map(analysis)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(10.8, 3.2 * len(tasks)), 4.1),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    x = np.arange(len(tasks), dtype=np.float64)
    legend_seen: set[str] = set()
    for index, task in enumerate(tasks):
        models = [model for model in MODEL_ORDER if (task, model) in performance]
        offsets = np.linspace(-0.30, 0.30, len(models)) if len(models) > 1 else np.zeros(1)
        for offset, model in zip(offsets, models, strict=True):
            record = performance[(task, model)]
            stats = record["direct_latent_r2"]
            position = x[index] + offset
            axes[0].errorbar(
                position,
                stats["mean"],
                yerr=stats["std"],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=2.5,
                linewidth=1.4,
                label=(MODEL_LABELS[model] if model not in legend_seen else None),
                zorder=3,
            )
            legend_seen.add(model)
            values = np.asarray(stats["values"], dtype=np.float64)
            jitter = np.linspace(-0.035, 0.035, len(values))
            axes[0].scatter(
                np.full(len(values), position) + jitter,
                values,
                s=15,
                facecolors="none",
                edgecolors=MODEL_COLORS[model],
                linewidths=0.8,
                zorder=4,
            )
    axes[0].axhline(0.0, color="#B0B0B0", linewidth=0.8, zorder=0)
    axes[0].set_ylabel("Direct latent reconstruction $R^2$")
    axes[0].set_xticks(x, [_task_label(task) for task in tasks])
    axes[0].set_title("(a) Known-emission reconstruction")
    axes[0].grid(axis="y", color="#E2E2E2", linewidth=0.6)

    for index, task in enumerate(tasks):
        models = [model for model in MODEL_ORDER[1:] if (task, model) in deltas]
        offsets = np.linspace(-0.27, 0.27, len(models)) if len(models) > 1 else np.zeros(1)
        for offset, model in zip(offsets, models, strict=True):
            record = deltas[(task, model)]
            stats = record["delta_direct_latent_r2"]
            position = x[index] + offset
            axes[1].errorbar(
                position,
                stats["mean"],
                yerr=stats["std"],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=2.5,
                linewidth=1.4,
                zorder=3,
            )
            values = np.asarray(stats["values"], dtype=np.float64)
            jitter = np.linspace(-0.025, 0.025, len(values))
            axes[1].scatter(
                np.full(len(values), position) + jitter,
                values,
                s=15,
                facecolors="none",
                edgecolors=MODEL_COLORS[model],
                linewidths=0.8,
                zorder=4,
            )
    axes[1].axhline(0.0, color="#505050", linewidth=0.9, zorder=0)
    axes[1].set_ylabel("$\\Delta R^2$ versus TXC-post (paired seed)")
    axes[1].set_xticks(x, [_task_label(task) for task in tasks])
    axes[1].set_title("(b) Matched-seed spectral advantage")
    axes[1].grid(axis="y", color="#E2E2E2", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles, strict=True))
    labels = [
        MODEL_LABELS[model] for model in MODEL_ORDER if MODEL_LABELS[model] in handle_by_label
    ]
    handles = [handle_by_label[label] for label in labels]
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(5, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86 if len(handles) > 4 else 0.92))
    return _save_figure(
        figure,
        figures_dir,
        "spectral_advantage_direct_latent_r2",
    )


def plot_frequency_diagnostics(
    analysis: dict[str, Any],
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Separate expected power, observed selections, and learned difficulty."""

    _configure_style()
    diagnostics = analysis["frequency_diagnostics"]
    if not diagnostics:
        raise RuntimeError("no spectral frequency diagnostics are available")
    figure, axes = plt.subplots(
        len(diagnostics),
        3,
        figsize=(11.4, 2.45 * len(diagnostics) + 0.8),
        squeeze=False,
        sharey=True,
    )
    for row_index, record in enumerate(diagnostics):
        labels = record["band_labels"]
        positions = np.arange(len(labels), dtype=np.float64)

        expected = np.asarray(record["expected_power"]["mean"], dtype=np.float64)
        axes[row_index, 0].bar(
            positions,
            expected,
            width=0.68,
            color="#777777",
            alpha=0.78,
            label="Expected latent power",
        )

        selections = record["selection_event_share"]["models"]
        selection_models = [model for model in MODEL_ORDER[1:] if model in selections]
        width = min(0.60, 0.78 / max(1, len(selection_models)))
        selection_offsets = width * (
            np.arange(len(selection_models)) - (len(selection_models) - 1) / 2
        )
        for offset, model in zip(selection_offsets, selection_models, strict=True):
            axes[row_index, 1].bar(
                positions + offset,
                selections[model]["mean"],
                width=width,
                color=MODEL_COLORS[model],
                alpha=0.82,
                label=MODEL_LABELS[model],
            )
        axes[row_index, 1].legend(
            loc="upper right",
            frameon=False,
            fontsize=6.8,
        )

        difficulty_models = record["learned_difficulty_weights"]["models"]
        if difficulty_models:
            ordered_difficulty = [model for model in MODEL_ORDER[1:] if model in difficulty_models]
            prior = difficulty_models[ordered_difficulty[0]]["prior"]
            for model in ordered_difficulty[1:]:
                if not np.allclose(
                    prior,
                    difficulty_models[model]["prior"],
                    atol=1e-10,
                    rtol=1e-8,
                ):
                    raise RuntimeError(
                        f"{record['task']}/{record['temporal_basis']}: "
                        "difficulty-weight priors differ by model"
                    )
            difficulty_series = ["prior", *ordered_difficulty]
            difficulty_width = min(0.60, 0.78 / len(difficulty_series))
            difficulty_offsets = difficulty_width * (
                np.arange(len(difficulty_series)) - (len(difficulty_series) - 1) / 2
            )
            axes[row_index, 2].bar(
                positions + difficulty_offsets[0],
                prior,
                width=difficulty_width,
                color="#A0A0A0",
                alpha=0.72,
                label="Bandwidth prior",
            )
            for offset, model in zip(
                difficulty_offsets[1:],
                ordered_difficulty,
                strict=True,
            ):
                axes[row_index, 2].bar(
                    positions + offset,
                    difficulty_models[model]["mean"],
                    width=difficulty_width,
                    color=MODEL_COLORS[model],
                    alpha=0.86,
                    label=MODEL_LABELS[model],
                )
            axes[row_index, 2].legend(
                loc="upper right",
                frameon=False,
                fontsize=6.8,
            )
        else:
            axes[row_index, 2].text(
                0.5,
                0.5,
                "No adaptive-weight rows",
                ha="center",
                va="center",
                transform=axes[row_index, 2].transAxes,
                color="#606060",
            )

        for column in range(3):
            axis = axes[row_index, column]
            axis.set_xticks(positions, labels, rotation=24, ha="right")
            axis.set_ylim(0, 1)
            axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
        axes[row_index, 0].set_ylabel("Share")
        axes[row_index, 0].text(
            0.02,
            0.93,
            record["row_label"],
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    axes[0, 0].set_title("Expected latent power\n(data; not learned)")
    axes[0, 1].set_title("Selected code events\n(observed model behavior)")
    axes[0, 2].set_title("Learned difficulty weights\n(loss emphasis; not power)")
    figure.tight_layout()
    return _save_figure(
        figure,
        figures_dir,
        "spectral_advantage_frequency_diagnostics",
    )


def plot_routing_trajectories(
    analysis: dict[str, Any],
    figures_dir: Path,
) -> tuple[Path, Path] | None:
    """Plot the transient learned score multipliers used by routed BatchTopK."""

    records = analysis.get("routing_trajectories", [])
    if not records:
        return None
    _configure_style()
    columns = min(2, len(records))
    rows = (len(records) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.6 * columns, 3.05 * rows),
        squeeze=False,
        sharey=True,
    )
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    for axis, record in zip(axes.flat, records, strict=False):
        steps = np.asarray(record["steps"], dtype=np.float64)
        mean = np.asarray(record["routing_scale"]["mean"], dtype=np.float64)
        std = np.asarray(record["routing_scale"]["std"], dtype=np.float64)
        for band, label in enumerate(record["band_labels"]):
            color = colors[band % len(colors)]
            axis.plot(
                steps,
                mean[:, band],
                color=color,
                linewidth=1.7,
                label=label,
            )
            axis.fill_between(
                steps,
                mean[:, band] - std[:, band],
                mean[:, band] + std[:, band],
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        axis.axhline(1.0, color="#606060", linewidth=0.8, linestyle="--")
        axis.set_title(str(record["task_label"]).replace("\n", " "))
        axis.set_xlabel("Optimizer step")
        axis.grid(axis="y", color="#E2E2E2", linewidth=0.6)
        axis.legend(frameon=False, ncol=2, fontsize=7)
    for axis in axes.flat[len(records) :]:
        axis.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel("Learned routing score multiplier")
    figure.suptitle(
        "Spectral Matryoshka routing dynamics (1 = bandwidth prior)",
        y=1.01,
        fontsize=11,
    )
    figure.tight_layout()
    return _save_figure(
        figure,
        figures_dir,
        "spectral_matryoshka_routing_trajectories",
    )


def analyze(
    results_path: Path | Sequence[Path],
    *,
    output_json: Path,
    output_csv: Path,
    figures_dir: Path,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    paths = [results_path] if isinstance(results_path, Path) else list(results_path)
    if not paths:
        raise ValueError("at least one results JSONL path is required")
    rows = [row for path in paths for row in load_latest_ok_rows(path)]
    analysis = build_analysis(
        rows,
        source=";".join(str(path) for path in paths),
        allow_incomplete=allow_incomplete,
    )
    write_aggregate_json(analysis, output_json)
    write_aggregate_csv(analysis, output_csv)
    performance_paths = plot_performance(analysis, figures_dir)
    frequency_paths = plot_frequency_diagnostics(analysis, figures_dir)
    routing_paths = plot_routing_trajectories(analysis, figures_dir)
    outputs = {
        "json": str(output_json),
        "csv": str(output_csv),
        "performance_png": str(performance_paths[0]),
        "performance_pdf": str(performance_paths[1]),
        "frequency_png": str(frequency_paths[0]),
        "frequency_pdf": str(frequency_paths[1]),
    }
    if routing_paths is not None:
        outputs.update(
            {
                "routing_png": str(routing_paths[0]),
                "routing_pdf": str(routing_paths[1]),
            }
        )
    return {"analysis": analysis, "outputs": outputs}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot the matched-parameter spectral-advantage run."
    )
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        help=(
            "Results JSONL to aggregate. Repeat for independent run directories; "
            f"default: {DEFAULT_RESULTS}"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Plot available task/model cells instead of requiring a matched panel.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = analyze(
        args.results or [DEFAULT_RESULTS],
        output_json=args.output_json,
        output_csv=args.output_csv,
        figures_dir=args.figures_dir,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
