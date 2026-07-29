"""Validate and aggregate the completed matched synthetic benchmark."""

from __future__ import annotations

import argparse
import copy
import importlib
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = POWER_ROOT / "configs" / "overnight.json"
DEFAULT_RESULTS = POWER_ROOT / "results" / "overnight_remote"
BASELINES = ("txc_pre", "txc_post", "spectral_v1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def latest_by_cell(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        cell_id = row.get("cell_id")
        if cell_id:
            latest[str(cell_id)] = row
    return latest


def correct_colored_zero_candidate_chance(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Correct chance adjustment when a hard DC mask creates zero candidates.

    The repository evaluator counts every flattened decoder slice when drawing
    its random-dictionary chance floor. A DC-removed spectral model retains
    zeroed DC parameters for shape compatibility, so that count is too large.
    Raw squared-cosine recovery is unaffected; only the chance adjustment is
    recomputed with the number of nonzero candidate slices.
    """
    from temp_bench.core.config import load_datasource
    from temp_bench.evals.colored_recovery import (
        _candidate_directions,
        _empirical_chance,
    )

    model_specs = {str(model["name"]): model for model in config["models"]}
    task_specs = {str(task["name"]): task for task in config["tasks"]}
    receipts: list[dict[str, Any]] = []
    for model_name, model_spec in model_specs.items():
        hparams = model_spec.get("hparams", {})
        if hparams.get("dc_mode") != "remove":
            continue
        relevant = [
            row
            for row in rows
            if row.get("phase") == "full"
            and row.get("status") == "ok"
            and row.get("model") == model_name
            and "colored_rec_sq" in row.get("metrics", {})
        ]
        if not relevant:
            continue
        task_name = str(relevant[0]["task"])
        task = task_specs[task_name]
        data_spec = load_datasource(str(task["datasource"]))
        d_in = int((data_spec.params or {})["d_in"])
        n_sources = int((data_spec.params or {})["N"])
        module_name, class_name = str(model_spec["class_path"]).split(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        model = cls(
            d_in=d_in,
            d_sae=int(task["d_sae"]),
            T=int(task["T"]),
            k_pos=int(task["k_pos"]),
            **hparams,
        )
        candidates = _candidate_directions(model)
        effective_candidates = int((candidates.norm(dim=1) > 1e-8).sum().item())
        total_candidates = int(candidates.shape[0])
        if effective_candidates == total_candidates:
            continue
        chance, chance_std = _empirical_chance(
            effective_candidates,
            d_in,
            n_sources,
        )
        for row in relevant:
            metrics = row["metrics"]
            reported = float(metrics["colored_rec_adj"])
            rec_sq = float(metrics["colored_rec_sq"])
            corrected = (rec_sq - chance) / max(1.0 - chance, 1e-9)
            metrics["colored_rec_adj_reported"] = reported
            metrics["colored_chance_reported"] = float(metrics["colored_chance"])
            metrics["colored_chance_std_reported"] = float(
                metrics["colored_chance_std"]
            )
            metrics["colored_rec_adj"] = corrected
            metrics["colored_chance"] = chance
            metrics["colored_chance_std"] = chance_std
        receipts.append(
            {
                "task": task_name,
                "model": model_name,
                "rows_corrected": len(relevant),
                "total_candidates": total_candidates,
                "effective_nonzero_candidates": effective_candidates,
                "corrected_chance": chance,
                "reason": "hard DC mask leaves zero decoder slices",
            }
        )
    return receipts


def expected_full_keys(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    phase = config["phases"]["full"]
    all_models = [str(model["name"]) for model in config["models"]]
    all_tasks = [str(task["name"]) for task in config["tasks"]]
    models = all_models if phase.get("models", "all") == "all" else list(phase["models"])
    tasks = all_tasks if phase.get("tasks", "all") == "all" else list(phase["tasks"])
    seeds = [int(seed) for seed in phase["seeds"]]
    return {(task, model, seed) for task in tasks for model in models for seed in seeds}


def full_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in latest_by_cell(rows).values()
        if row.get("phase") == "full"
    ]


def validate_panel(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    panel = full_rows(rows)
    observed: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_keys: list[tuple[str, str, int]] = []
    failed: list[dict[str, Any]] = []
    fairness_errors: list[str] = []
    training_identities: dict[str, tuple[str, str, int]] = {}
    parameter_counts: dict[tuple[str, str], set[int]] = {}

    task_specs = {str(task["name"]): task for task in config["tasks"]}
    expected_batch_tokens = int(config["training"]["batch_tokens"])
    for row in panel:
        key = (str(row["task"]), str(row["model"]), int(row["seed"]))
        if key in observed:
            duplicate_keys.append(key)
        observed[key] = row
        if row.get("status") != "ok":
            failed.append(
                {
                    "task": key[0],
                    "model": key[1],
                    "seed": key[2],
                    "status": row.get("status"),
                    "error": row.get("error"),
                }
            )
            continue
        task = task_specs[key[0]]
        expected = {
            "T": int(task["T"]),
            "d_sae": int(task["d_sae"]),
            "k_pos": int(task["k_pos"]),
            "target_steps": int(task["n_steps"]),
        }
        for field, value in expected.items():
            if int(row[field]) != value:
                fairness_errors.append(f"{key}: {field}={row[field]} expected {value}")
        batch_tokens = int(row["training"]["batch_tokens"])
        if batch_tokens != expected_batch_tokens:
            fairness_errors.append(
                f"{key}: batch_tokens={batch_tokens} expected {expected_batch_tokens}"
            )
        end_step = int(row["training"]["end_step"])
        if end_step != int(row["target_steps"]):
            fairness_errors.append(
                f"{key}: end_step={end_step} expected {row['target_steps']}"
            )
        training_id = str(row["training_id"])
        prior_identity = training_identities.setdefault(training_id, key)
        if prior_identity != key:
            fairness_errors.append(
                f"training_id {training_id} maps to both {prior_identity} and {key}"
            )
        parameter_count = int(row["training"]["parameter_count"])
        if parameter_count <= 0:
            fairness_errors.append(f"{key}: invalid parameter_count={parameter_count}")
        parameter_counts.setdefault((key[0], key[1]), set()).add(parameter_count)
        primary = str(row["primary_metric"])
        for metric in (primary, "nmse", "l0_per_window", "l0_per_token"):
            if metric not in row["metrics"]:
                fairness_errors.append(f"{key}: missing {metric}")
                continue
            value = float(row["metrics"][metric])
            if not math.isfinite(value):
                fairness_errors.append(f"{key}: non-finite {metric}")
        if float(row["metrics"].get("l0_per_window", 0.0)) <= 0:
            fairness_errors.append(f"{key}: non-positive l0_per_window")

    for key, values in parameter_counts.items():
        if len(values) != 1:
            fairness_errors.append(
                f"{key}: parameter_count varies across seeds: {sorted(values)}"
            )

    expected_keys = expected_full_keys(config)
    observed_keys = set(observed)
    missing = sorted(expected_keys - observed_keys)
    unexpected = sorted(observed_keys - expected_keys)
    complete = not missing and not unexpected and not failed and not fairness_errors
    cell_counts = Counter(str(row.get("cell_id")) for row in rows if row.get("cell_id"))
    return {
        "complete": complete,
        "expected_rows": len(expected_keys),
        "observed_rows": len(observed_keys),
        "missing": [list(key) for key in missing],
        "unexpected": [list(key) for key in unexpected],
        "duplicate_task_model_seed": [list(key) for key in sorted(set(duplicate_keys))],
        "failed": failed,
        "fairness_errors": fairness_errors,
        "raw_row_count": len(rows),
        "unique_cell_count": len(latest_by_cell(rows)),
        "retried_cell_ids": {
            cell_id: count for cell_id, count in sorted(cell_counts.items()) if count > 1
        },
        "unique_training_id_count": len(training_identities),
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _paired_values(
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    *,
    task: str,
    model: str,
    baseline: str,
    metric: str,
) -> list[float]:
    seeds = sorted(
        seed
        for row_task, row_model, seed in by_key
        if row_task == task and row_model == model
    )
    deltas: list[float] = []
    for seed in seeds:
        model_row = by_key.get((task, model, seed))
        baseline_row = by_key.get((task, baseline, seed))
        if model_row is None or baseline_row is None:
            continue
        deltas.append(
            float(model_row["metrics"][metric])
            - float(baseline_row["metrics"][metric])
        )
    return deltas


def aggregate(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    panel = [row for row in full_rows(rows) if row.get("status") == "ok"]
    by_key = {
        (str(row["task"]), str(row["model"]), int(row["seed"])): row
        for row in panel
    }
    entries: list[dict[str, Any]] = []
    for task_spec in config["tasks"]:
        task = str(task_spec["name"])
        metric = str(task_spec["primary_metric"])
        for model_spec in config["models"]:
            model = str(model_spec["name"])
            selected = sorted(
                (
                    row
                    for (row_task, row_model, _seed), row in by_key.items()
                    if row_task == task and row_model == model
                ),
                key=lambda row: int(row["seed"]),
            )
            if not selected:
                continue
            values = [float(row["metrics"][metric]) for row in selected]
            fresh = [
                float(row["metrics"][metric])
                for row in selected
                if int(row["training"]["start_step"]) == 0
            ]
            entry: dict[str, Any] = {
                "task": task,
                "metric": metric,
                "model": model,
                "n": len(values),
                "mean": _mean(values),
                "std": _std(values),
                "seed_values": {
                    str(int(row["seed"])): float(row["metrics"][metric])
                    for row in selected
                },
                "fresh_n": len(fresh),
                "fresh_mean": _mean(fresh) if fresh else None,
                "mean_l0_per_window": _mean(
                    [float(row["metrics"]["l0_per_window"]) for row in selected]
                ),
                "std_l0_per_window": _std(
                    [float(row["metrics"]["l0_per_window"]) for row in selected]
                ),
                "mean_l0_per_token": _mean(
                    [float(row["metrics"]["l0_per_token"]) for row in selected]
                ),
                "mean_nmse": _mean([float(row["metrics"]["nmse"]) for row in selected]),
                "std_nmse": _std([float(row["metrics"]["nmse"]) for row in selected]),
                "parameter_count": int(selected[0]["training"]["parameter_count"]),
                "fairness_role": selected[0].get("fairness_role"),
            }
            for baseline in BASELINES:
                deltas = _paired_values(
                    by_key,
                    task=task,
                    model=model,
                    baseline=baseline,
                    metric=metric,
                )
                entry[f"delta_vs_{baseline}"] = _mean(deltas) if deltas else None
                entry[f"delta_vs_{baseline}_std"] = _std(deltas) if deltas else None
                entry[f"wins_vs_{baseline}"] = sum(delta > 0 for delta in deltas)
                entry[f"ties_vs_{baseline}"] = sum(delta == 0 for delta in deltas)
                entry[f"paired_n_vs_{baseline}"] = len(deltas)
            entries.append(entry)
    return entries


def l0_match_report(
    aggregates: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Report realized support ratios for arms declared equal-window-support."""
    output: list[dict[str, Any]] = []
    for task in (str(task["name"]) for task in config["tasks"]):
        task_rows = {row["model"]: row for row in aggregates if row["task"] == task}
        baseline = task_rows.get("txc_pre")
        if baseline is None:
            continue
        baseline_l0 = float(baseline["mean_l0_per_window"])
        for model_spec in config["models"]:
            role = str(model_spec.get("fairness_role", ""))
            if not (
                role.startswith("equal_window_support")
                or role == "matched_monolithic_window_support"
            ):
                continue
            model = str(model_spec["name"])
            row = task_rows.get(model)
            if row is None:
                continue
            ratio = float(row["mean_l0_per_window"]) / max(baseline_l0, 1e-12)
            output.append(
                {
                    "task": task,
                    "model": model,
                    "l0_per_window": row["mean_l0_per_window"],
                    "txc_pre_l0_per_window": baseline_l0,
                    "ratio_vs_txc_pre": ratio,
                    "within_25_percent": 0.75 <= ratio <= 1.25,
                }
            )
    return output


def seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(
        (row for row in full_rows(rows) if row.get("status") == "ok"),
        key=lambda row: (str(row["task"]), str(row["model"]), int(row["seed"])),
    ):
        metric = str(row["primary_metric"])
        output.append(
            {
                "task": row["task"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "metric": metric,
                "primary_value": float(row["metrics"][metric]),
                "nmse": float(row["metrics"]["nmse"]),
                "l0_per_window": float(row["metrics"]["l0_per_window"]),
                "parameter_count": int(row["training"]["parameter_count"]),
                "start_step": int(row["training"]["start_step"]),
                "end_step": int(row["training"]["end_step"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "training_id": row["training_id"],
                "cell_id": row["cell_id"],
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    scalar_rows = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fieldnames = list(scalar_rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(scalar_rows)


def run_analysis(
    config_path: Path,
    results_path: Path,
    output_dir: Path,
    *,
    require_complete: bool = True,
    control_config_path: Path | None = None,
    control_results_path: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    raw_rows = read_jsonl(results_path)
    metric_corrections = correct_colored_zero_candidate_chance(raw_rows, config)
    main_integrity = validate_panel(raw_rows, config)
    if require_complete and not main_integrity["complete"]:
        raise RuntimeError(
            "benchmark panel is incomplete or invalid: "
            f"missing={len(main_integrity['missing'])}, "
            f"failed={len(main_integrity['failed'])}, "
            f"fairness_errors={len(main_integrity['fairness_errors'])}"
        )
    if (control_config_path is None) != (control_results_path is None):
        raise ValueError("control config and results must be supplied together")

    combined_config = copy.deepcopy(config)
    combined_rows = list(raw_rows)
    control_integrity: dict[str, Any] | None = None
    if control_config_path is not None and control_results_path is not None:
        control_config = json.loads(control_config_path.read_text())
        control_rows = read_jsonl(control_results_path)
        metric_corrections.extend(
            correct_colored_zero_candidate_chance(control_rows, control_config)
        )
        control_integrity = validate_panel(control_rows, control_config)
        if require_complete and not control_integrity["complete"]:
            raise RuntimeError(
                "matched-control panel is incomplete or invalid: "
                f"missing={len(control_integrity['missing'])}, "
                f"failed={len(control_integrity['failed'])}, "
                f"fairness_errors={len(control_integrity['fairness_errors'])}"
            )
        selected = control_config["phases"]["full"].get("models", "all")
        if selected == "all":
            selected_control_models = {
                str(model["name"]) for model in control_config["models"]
            }
        else:
            selected_control_models = {str(model) for model in selected}
        known_models = {
            str(model["name"]) for model in combined_config["models"]
        }
        combined_config["models"].extend(
            copy.deepcopy(model)
            for model in control_config["models"]
            if str(model["name"]) in selected_control_models
            and str(model["name"]) not in known_models
        )
        combined_rows.extend(control_rows)

    aggregates = aggregate(combined_rows, combined_config)
    seeds = seed_rows(combined_rows)
    l0_matches = l0_match_report(aggregates, combined_config)
    task_winners = {}
    for task in (str(task["name"]) for task in combined_config["tasks"]):
        candidates = [row for row in aggregates if row["task"] == task]
        if candidates:
            best = max(candidates, key=lambda row: float(row["mean"]))
            task_winners[task] = {
                "model": best["model"],
                "mean": best["mean"],
                "metric": best["metric"],
            }
    all_complete = main_integrity["complete"] and (
        control_integrity is None or control_integrity["complete"]
    )
    integrity: dict[str, Any]
    if control_integrity is None:
        integrity = main_integrity
    else:
        integrity = {
            "complete": all_complete,
            "main": main_integrity,
            "matched_control": control_integrity,
        }
    payload = {
        "schema_version": 1,
        "run_name": config["run_name"],
        "integrity": integrity,
        "task_winners": task_winners,
        "aggregates": aggregates,
        "l0_match_report": l0_matches,
        "metric_corrections": metric_corrections,
        "resume_sensitivity": {
            "definition": "fresh_mean includes only rows with training.start_step == 0",
            "resumed_start_steps": dict(
                sorted(Counter(row["start_step"] for row in seeds).items())
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "benchmark_aggregate.csv", aggregates)
    _write_csv(output_dir / "benchmark_seed_rows.csv", seeds)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS / "results.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--control-config", type=Path)
    parser.add_argument("--control-results", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(
        args.config,
        args.results,
        args.output_dir,
        require_complete=not args.allow_incomplete,
        control_config_path=args.control_config,
        control_results_path=args.control_results,
    )
    print(json.dumps(payload["integrity"], indent=2, sort_keys=True))
    for task, winner in payload["task_winners"].items():
        print(f"{task}: {winner['model']} {winner['metric']}={winner['mean']:.4f}")


if __name__ == "__main__":
    main()
