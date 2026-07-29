from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.power_spectrum.code import analyze_benchmark


def _config() -> dict:
    return {
        "run_name": "test",
        "training": {"batch_tokens": 16},
        "models": [
            {"name": "txc_pre"},
            {"name": "txc_post"},
            {"name": "spectral_v1"},
            {"name": "candidate"},
        ],
        "tasks": [
            {
                "name": "tone",
                "primary_metric": "recovery",
                "T": 4,
                "d_sae": 8,
                "k_pos": 1,
                "n_steps": 10,
            }
        ],
        "phases": {"full": {"seeds": [1, 2]}},
    }


def _row(model: str, seed: int, value: float, *, start_step: int = 0) -> dict:
    return {
        "cell_id": f"{model}-{seed}",
        "training_id": f"train-{model}-{seed}",
        "phase": "full",
        "status": "ok",
        "task": "tone",
        "model": model,
        "seed": seed,
        "T": 4,
        "d_sae": 8,
        "k_pos": 1,
        "target_steps": 10,
        "primary_metric": "recovery",
        "metrics": {
            "recovery": value,
            "nmse": 0.5,
            "l0_per_window": 4.0,
            "l0_per_token": 1.0,
        },
        "training": {
            "batch_tokens": 16,
            "parameter_count": 100,
            "start_step": start_step,
            "end_step": 10,
        },
        "fairness_role": "equal_window_support",
        "elapsed_seconds": 1.0,
    }


def _complete_rows() -> list[dict]:
    values = {
        "txc_pre": (0.2, 0.3),
        "txc_post": (0.4, 0.5),
        "spectral_v1": (0.6, 0.7),
        "candidate": (0.8, 0.9),
    }
    return [
        _row(model, seed, value, start_step=5 if seed == 2 else 0)
        for model, model_values in values.items()
        for seed, value in zip((1, 2), model_values, strict=True)
    ]


def test_analysis_checks_panel_and_computes_paired_deltas(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    results_path = tmp_path / "results.jsonl"
    config_path.write_text(json.dumps(_config()))
    results_path.write_text("\n".join(json.dumps(row) for row in _complete_rows()) + "\n")

    payload = analyze_benchmark.run_analysis(config_path, results_path, tmp_path / "out")

    assert payload["integrity"]["complete"]
    candidate = next(
        row for row in payload["aggregates"] if row["model"] == "candidate"
    )
    assert candidate["mean"] == pytest.approx(0.85)
    assert candidate["fresh_mean"] == pytest.approx(0.8)
    assert candidate["delta_vs_txc_pre"] == pytest.approx(0.6)
    assert candidate["delta_vs_spectral_v1"] == pytest.approx(0.2)
    assert candidate["wins_vs_spectral_v1"] == 2
    assert (tmp_path / "out" / "benchmark_seed_rows.csv").is_file()


def test_analysis_rejects_incomplete_panel(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    results_path = tmp_path / "results.jsonl"
    config_path.write_text(json.dumps(_config()))
    results_path.write_text(json.dumps(_complete_rows()[0]) + "\n")

    with pytest.raises(RuntimeError, match="incomplete"):
        analyze_benchmark.run_analysis(config_path, results_path, tmp_path / "out")
