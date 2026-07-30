from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.power_spectrum.code import plot_spectral_advantage


def _synthetic_rows() -> list[dict]:
    tasks = (
        ("hmm_mixed_t8_matched_params", "factorial_hmm", 8),
        ("narrowband_offgrid_t8_matched_params", "narrowband_sources", 8),
        ("narrowband_offgrid_t16_matched_params", "narrowband_sources", 16),
    )
    models = ("txc", "spectral_global", "spectral_adaptive")
    gains = {"txc": 0.0, "spectral_global": 0.04, "spectral_adaptive": 0.07}
    widths = {
        (8, "txc"): 64,
        (8, "spectral_global"): 254,
        (8, "spectral_adaptive"): 253,
        (16, "txc"): 64,
        (16, "spectral_global"): 255,
        (16, "spectral_adaptive"): 255,
    }
    counts = {
        (8, "txc"): 16576,
        (8, "spectral_global"): 16606,
        (8, "spectral_adaptive"): 16541,
        (16, "txc"): 33088,
        (16, "spectral_global"): 33119,
        (16, "spectral_adaptive"): 33119,
    }
    rows = []
    for task_index, (task, family, window) in enumerate(tasks):
        bands = (
            [[0], [1, 2], [3, 4], [5, 6, 7]]
            if window == 8
            else [
                [0],
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
            ]
        )
        prior = [len(band) / window for band in bands]
        expected = [
            [0.42, 0.28, 0.19, 0.11],
            [0.12, 0.47, 0.29, 0.12],
            [0.09, 0.19, 0.48, 0.24],
            [0.06, 0.12, 0.24, 0.58],
        ]
        for model in models:
            for seed in (1, 2):
                base = 0.31 + 0.05 * task_index + 0.01 * seed
                metrics = {
                    "direct_latent_r2": base + gains[model],
                    "latent_r2": base + gains[model] + 0.08,
                    "raw_projection_r2": 0.47 + 0.02 * task_index,
                    "nmse": 0.72 - gains[model],
                    "l0_per_window": window + 0.02 * (seed - 1),
                }
                if model != "txc":
                    learned = [0.10, 0.20, 0.30, 0.40] if model == "spectral_adaptive" else prior
                    selection = (
                        [0.12, 0.23, 0.29, 0.36]
                        if model == "spectral_adaptive"
                        else [0.16, 0.25, 0.28, 0.31]
                    )
                    metrics.update(
                        {
                            "spectral_usage": {
                                "bands": bands,
                                "selection_event_share": selection,
                                "learned_frequency_weight": learned,
                                "frequency_weight_prior": prior,
                            },
                            "expected_dct_band_energy_per_source": expected,
                            "expected_band_per_source": [0, 1, 2, 3],
                        }
                    )
                rows.append(
                    {
                        "schema_version": 1,
                        "cell_id": f"{task}-{model}-{seed}",
                        "task": task,
                        "family": family,
                        "model": model,
                        "seed": seed,
                        "window": window,
                        "d_sae": widths[(window, model)],
                        "k_pos": window if model == "txc" else 1,
                        "n_steps": 4000,
                        "smoke": False,
                        "status": "ok",
                        "metrics": metrics,
                        "primary_metric": "direct_latent_r2",
                        "primary_value": metrics["direct_latent_r2"],
                        "training": {"parameter_count": counts[(window, model)]},
                    }
                )
    return rows


def _routed_synthetic_rows() -> list[dict]:
    tasks = (
        (
            "hmm_slow_t8_routed_screen",
            "factorial_hmm",
            plot_spectral_advantage.ROUTED_HMM_MODELS,
        ),
        (
            "hmm_alternating_t8_routed_screen",
            "factorial_hmm",
            plot_spectral_advantage.ROUTED_HMM_MODELS,
        ),
        (
            "narrowband_sparse_balanced_t8",
            "narrowband_sources",
            plot_spectral_advantage.ROUTED_SPARSE_MODELS,
        ),
        (
            "narrowband_sparse_high_crowded_t8",
            "narrowband_sources",
            plot_spectral_advantage.ROUTED_SPARSE_MODELS,
        ),
    )
    dct_bands = [[0], [1, 2], [3, 4], [5, 6, 7]]
    fourier_rows = [[0], [1, 2], [3, 4], [5, 6, 7]]
    fourier_bins = [[0], [1], [2], [3, 4]]
    expected_dct = [[0.40, 0.30, 0.20, 0.10]] * 4
    expected_fourier = [[0.10, 0.20, 0.30, 0.40]] * 4
    prior = [0.125, 0.25, 0.25, 0.375]
    gains = {
        "txc": 0.0,
        "spectral_dct_global": 0.01,
        "spectral_fourier_global": 0.03,
        "spectral_fourier_adaptive": 0.05,
        "spectral_fourier_routed": 0.07,
    }
    rows = []
    for task_index, (task, family, models) in enumerate(tasks):
        for model in models:
            for seed in (1, 2):
                direct = 0.25 + 0.04 * task_index + gains[model] + 0.005 * seed
                metrics = {
                    "direct_latent_r2": direct,
                    "latent_r2": direct + 0.05,
                    "raw_projection_r2": 0.48,
                    "nmse": 0.70 - gains[model],
                    "l0_per_window": 8.0,
                }
                if model != "txc":
                    basis = "dct" if model == "spectral_dct_global" else "fourier"
                    bands = dct_bands if basis == "dct" else fourier_rows
                    physical = dct_bands if basis == "dct" else fourier_bins
                    expected = expected_dct if basis == "dct" else expected_fourier
                    learned = {
                        "spectral_fourier_adaptive": [0.08, 0.18, 0.29, 0.45],
                        "spectral_fourier_routed": [0.06, 0.16, 0.28, 0.50],
                    }.get(model, prior)
                    selection = {"spectral_fourier_routed": [0.07, 0.17, 0.28, 0.48]}.get(
                        model, [0.12, 0.23, 0.28, 0.37]
                    )
                    metrics.update(
                        {
                            "temporal_basis": basis,
                            "spectral_usage": {
                                "bands": bands,
                                "temporal_basis": basis,
                                "frequency_bin_bands": physical,
                                "selection_event_share": selection,
                                "learned_frequency_weight": learned,
                                "frequency_weight_prior": prior,
                                "frequency_routing_scale": (
                                    [0.7, 0.9, 1.1, 1.5]
                                    if model == "spectral_fourier_routed"
                                    else [1.0, 1.0, 1.0, 1.0]
                                ),
                            },
                            # The basis-aware key takes precedence over this
                            # legacy DCT key for Fourier rows.
                            "expected_dct_band_energy_per_source": expected_dct,
                            "expected_basis_band_energy_per_source": expected,
                        }
                    )
                rows.append(
                    {
                        "schema_version": 1,
                        "cell_id": f"{task}-{model}-{seed}",
                        "task": task,
                        "family": family,
                        "model": model,
                        "seed": seed,
                        "window": 8,
                        "d_sae": 64 if model == "txc" else 255,
                        "k_pos": 8 if model == "txc" else 1,
                        "n_steps": 3500,
                        "smoke": False,
                        "status": "ok",
                        "metrics": metrics,
                        "primary_metric": "direct_latent_r2",
                        "primary_value": direct,
                        "training": {"parameter_count": (16576 if model == "txc" else 16671)},
                    }
                )
    return rows


def test_analysis_pairs_seeds_and_keeps_frequency_semantics_distinct() -> None:
    analysis = plot_spectral_advantage.build_analysis(_synthetic_rows())
    delta = next(
        record
        for record in analysis["paired_deltas_vs_txc"]
        if record["task"] == "hmm_mixed_t8_matched_params"
        and record["model"] == "spectral_adaptive"
    )
    assert delta["seeds"] == [1, 2]
    assert delta["delta_direct_latent_r2"]["mean"] == pytest.approx(0.07)

    diagnostic = analysis["frequency_diagnostics"][0]
    expected = diagnostic["expected_power"]["mean"]
    difficulty = diagnostic["adaptive_difficulty_weight"]["mean"]
    assert expected != pytest.approx(difficulty)
    assert "data property" in diagnostic["expected_power"]["semantics"]
    assert "not expected power" in diagnostic["adaptive_difficulty_weight"]["semantics"]
    assert diagnostic["selection_event_share"]["models"]["spectral_global"]["n"] == 2


def test_routed_analysis_supports_basis_specific_task_panels() -> None:
    analysis = plot_spectral_advantage.build_analysis(_routed_synthetic_rows())
    assert len(analysis["performance"]) == 18
    assert len(analysis["frequency_diagnostics"]) == 6

    slow_hmm = next(
        record
        for record in analysis["frequency_diagnostics"]
        if record["task"] == "hmm_slow_t8_routed_screen"
    )
    assert set(slow_hmm["learned_difficulty_weights"]["models"]) == {
        "spectral_fourier_adaptive",
        "spectral_fourier_routed",
    }

    sparse_fourier = next(
        record
        for record in analysis["frequency_diagnostics"]
        if record["task"] == "narrowband_sparse_high_crowded_t8"
        and record["temporal_basis"] == "fourier"
    )
    assert sparse_fourier["band_labels"] == [
        "DC",
        "Fourier 1",
        "Fourier 2",
        "Fourier 3–4",
    ]
    assert sparse_fourier["expected_power"]["mean"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    learned = sparse_fourier["learned_difficulty_weights"]["models"]
    assert set(learned) == {
        "spectral_fourier_adaptive",
        "spectral_fourier_routed",
    }
    assert learned["spectral_fourier_adaptive"]["mean"] != pytest.approx(
        learned["spectral_fourier_routed"]["mean"]
    )
    routing = sparse_fourier["routing_scales"]["models"]
    assert routing["spectral_fourier_routed"]["mean"] == pytest.approx([0.7, 0.9, 1.1, 1.5])


def test_incomplete_mode_never_relaxes_seed_matching() -> None:
    rows = _routed_synthetic_rows()
    missing_model = [
        row
        for row in rows
        if not (
            row["task"] == "narrowband_sparse_balanced_t8" and row["model"] == "spectral_dct_global"
        )
    ]
    with pytest.raises(RuntimeError, match="missing models"):
        plot_spectral_advantage.build_analysis(missing_model)
    plot_spectral_advantage.build_analysis(
        missing_model,
        allow_incomplete=True,
    )

    mismatched_seed = [
        row
        for row in rows
        if not (
            row["task"] == "hmm_slow_t8_routed_screen"
            and row["model"] == "spectral_fourier_routed"
            and row["seed"] == 2
        )
    ]
    with pytest.raises(RuntimeError, match="do not match TXC seeds"):
        plot_spectral_advantage.build_analysis(
            mismatched_seed,
            allow_incomplete=True,
        )


def test_two_seed_jsonl_writes_aggregates_and_both_figures(tmp_path) -> None:
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(row) for row in _synthetic_rows()) + "\n")
    output_json = tmp_path / "analysis.json"
    output_csv = tmp_path / "analysis.csv"
    figures = tmp_path / "figures"

    result = plot_spectral_advantage.analyze(
        results,
        output_json=output_json,
        output_csv=output_csv,
        figures_dir=figures,
    )

    assert output_json.stat().st_size > 0
    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    adaptive = next(
        row
        for row in rows
        if row["task"] == "narrowband_offgrid_t16_matched_params"
        and row["model"] == "spectral_adaptive"
    )
    assert adaptive["model_label"] == "Spectral adaptive weights"
    assert json.loads(adaptive["adaptive_difficulty_weight"]) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    for key in (
        "performance_png",
        "performance_pdf",
        "frequency_png",
        "frequency_pdf",
    ):
        path = Path(result["outputs"][key])
        assert figures in path.parents
        assert path.stat().st_size > 0


def test_routed_rows_write_dynamic_figures_and_model_specific_csv(
    tmp_path,
) -> None:
    results = tmp_path / "routed-results.jsonl"
    results.write_text("\n".join(json.dumps(row) for row in _routed_synthetic_rows()) + "\n")
    output_json = tmp_path / "routed-analysis.json"
    output_csv = tmp_path / "routed-analysis.csv"
    figures = tmp_path / "routed-figures"
    result = plot_spectral_advantage.analyze(
        results,
        output_json=output_json,
        output_csv=output_csv,
        figures_dir=figures,
    )

    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 18
    routed = next(
        row
        for row in rows
        if row["task"] == "narrowband_sparse_balanced_t8"
        and row["model"] == "spectral_fourier_routed"
    )
    assert routed["temporal_basis"] == "fourier"
    assert json.loads(routed["adaptive_difficulty_weight"]) == pytest.approx(
        [0.06, 0.16, 0.28, 0.50]
    )
    assert json.loads(routed["frequency_routing_scale"]) == pytest.approx([0.7, 0.9, 1.1, 1.5])
    for key in (
        "performance_png",
        "performance_pdf",
        "frequency_png",
        "frequency_pdf",
    ):
        assert Path(result["outputs"][key]).stat().st_size > 0
