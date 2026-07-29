from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.power_spectrum.code import analyze_benchmark


OVERNIGHT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "overnight.json"
)


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


def test_analysis_merges_matched_control_and_pairs_against_main(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    results_path = tmp_path / "results.jsonl"
    control_config_path = tmp_path / "control-config.json"
    control_results_path = tmp_path / "control-results.jsonl"
    config_path.write_text(json.dumps(_config()))
    results_path.write_text("\n".join(json.dumps(row) for row in _complete_rows()) + "\n")
    control_config = {
        **_config(),
        "run_name": "control",
        "models": [
            {
                "name": "full_control",
                "fairness_role": "matched_monolithic_window_support",
            }
        ],
        "phases": {"full": {"models": ["full_control"], "seeds": [1, 2]}},
    }
    control_config_path.write_text(json.dumps(control_config))
    control_rows = [
        {**_row("full_control", seed, value), "fairness_role": "matched_monolithic_window_support"}
        for seed, value in ((1, 0.7), (2, 0.8))
    ]
    control_results_path.write_text(
        "\n".join(json.dumps(row) for row in control_rows) + "\n"
    )

    payload = analyze_benchmark.run_analysis(
        config_path,
        results_path,
        tmp_path / "out",
        control_config_path=control_config_path,
        control_results_path=control_results_path,
    )

    assert payload["integrity"]["complete"]
    control = next(
        row for row in payload["aggregates"] if row["model"] == "full_control"
    )
    assert control["delta_vs_txc_pre"] == pytest.approx(0.5)
    assert control["delta_vs_spectral_v1"] == pytest.approx(0.1)


def test_colored_chance_correction_excludes_zeroed_dc_candidates() -> None:
    config = json.loads(OVERNIGHT_CONFIG.read_text())
    row = {
        "phase": "full",
        "status": "ok",
        "task": "colored",
        "model": "v2_remove_dc",
        "metrics": {
            "colored_rec_sq": 0.5,
            "colored_rec_adj": 0.25,
            "colored_chance": 1.0 / 3.0,
            "colored_chance_std": 0.01,
        },
    }

    receipts = analyze_benchmark.correct_colored_zero_candidate_chance(
        [row], config
    )

    assert receipts[0]["total_candidates"] == 256
    assert receipts[0]["effective_nonzero_candidates"] == 192
    assert row["metrics"]["colored_rec_adj_reported"] == 0.25
    assert row["metrics"]["colored_rec_adj"] != 0.25


def test_csv_outputs_use_repository_lf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    analyze_benchmark._write_csv(path, [{"name": "candidate", "score": 0.5}])

    assert b"\r\n" not in path.read_bytes()
