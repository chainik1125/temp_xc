from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pytest

from experiments.power_spectrum.code import plot_backtracking_fourier as plotter


def _frequency_bands(window: int) -> list[list[int]]:
    if window == 1:
        return [[0]]
    if window == 2:
        return [[0], [1]]
    if window == 4:
        return [[0], [1], [2]]
    if window == 6:
        return [[0], [1], [2], [3]]
    return [[0], [1], [2, 3], [4, 5]]


def _write_fake_panel(
    root: Path,
    *,
    declared_mismatch: bool = True,
    duplicate_s32: bool = False,
    inconsistent_band: bool = False,
) -> None:
    for window in plotter.WINDOWS:
        bands = _frequency_bands(window)
        for seed in plotter.SEEDS:
            value = 0.18 + 0.005 * window + 0.0001 * seed
            fold_values = [value + offset for offset in (-0.002, -0.001, 0.0, 0.001, 0.002)]
            s32 = {
                "n_features": 32,
                "ordered_pr_auc": {
                    "fold_values": fold_values,
                    "mean": statistics.fmean(fold_values),
                },
                "folds": [
                    {
                        "n_features": 32,
                        "n_features_actual": 32,
                    }
                    for _ in range(5)
                ],
            }
            probes = [
                {
                    "n_features": 8,
                    "ordered_pr_auc": {
                        "fold_values": [0.99] * 5,
                        "mean": 0.99,
                    },
                },
                s32,
            ]
            if duplicate_s32 and window == 1 and seed == 1:
                probes.append(dict(s32))
            local_bands = [list(values) for values in bands]
            if inconsistent_band and window == 6 and seed == 42:
                local_bands[-1] = [99]
            raw = [index + 1 for index in range(len(local_bands))]
            total = float(sum(raw))
            provenance = (
                {
                    "schema_version": "power-spectrum.backtracking-recovery-artifact.v1",
                    "matches_reference_cohort": False,
                    "provenance_warning": "Synthetic recovered artifact is not the reference cohort.",
                }
                if declared_mismatch
                else None
            )
            payload = {
                "status": "complete",
                "window": window,
                "seed": seed,
                "artifact_sha256": "recovered-artifact-sha",
                "cohort_sha256": "recovered-cohort-sha",
                "artifact_provenance": provenance,
                "reference_commit": "d9c7fc7b2",
                "reference_protocol_version": plotter.REFERENCE_PROTOCOL,
                "effective_l0": {
                    "ordered": {
                        "nominal_l0": 20 * window,
                        "effective_l0_mean": 20.0 * window,
                    }
                },
                "ordered_band_usage": [
                    {
                        "band": index,
                        "frequencies": frequencies,
                        "activation_mass_share": raw[index] / total,
                    }
                    for index, frequencies in enumerate(local_bands)
                ],
                "probes": {"fourier": probes},
            }
            path = root / "cells" / f"T{window}_seed{seed}" / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))


def test_extracts_only_s32_and_aggregates_across_seeds(tmp_path: Path) -> None:
    _write_fake_panel(tmp_path)
    cells = plotter.load_cells(tmp_path)
    summary = plotter.build_summary(cells)

    assert len(cells) == 15
    t4 = next(row for row in summary["fourier"] if row["window"] == 4)
    expected = [0.18 + 0.005 * 4 + 0.0001 * seed for seed in plotter.SEEDS]
    assert t4["mean"] == pytest.approx(statistics.fmean(expected))
    assert t4["std_sample"] == pytest.approx(statistics.stdev(expected))
    assert t4["mean"] != pytest.approx(0.99)
    assert summary["probe_extraction"]["n_features"] == 32
    assert not summary["provenance"]["comparable_to_aniket_reference"]
    assert summary["provenance"]["comparison_kind"] == "recovered-artifact-sensitivity"


def test_analyze_writes_plot_json_and_long_csv(tmp_path: Path) -> None:
    root = tmp_path / "results"
    output = tmp_path / "publication"
    _write_fake_panel(root)
    summary = plotter.analyze(root, output_dir=output, stem="fourier")

    for path in map(Path, summary["outputs"].values()):
        assert path.is_file()
        assert path.stat().st_size > 0
    with Path(summary["outputs"]["csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    performance = [row for row in rows if row["record_type"] == "performance"]
    assert len(performance) == 25
    marker = [row for row in rows if row["record_type"] == "context_marker"]
    assert len(marker) == 1
    assert marker[0]["series"] == "tsae_t1_300k_seed42"
    assert "not a T curve" in marker[0]["source"]
    bands = [row for row in rows if row["record_type"] == "band_usage"]
    assert bands
    assert all(row["comparable_to_aniket_reference"] == "False" for row in bands)


def test_duplicate_s32_probe_is_rejected(tmp_path: Path) -> None:
    _write_fake_panel(tmp_path, duplicate_s32=True)
    with pytest.raises(ValueError, match="exactly one Fourier S=32"):
        plotter.load_cells(tmp_path)


def test_unmarked_artifact_mismatch_is_rejected(tmp_path: Path) -> None:
    _write_fake_panel(tmp_path, declared_mismatch=False)
    cells = plotter.load_cells(tmp_path)
    with pytest.raises(ValueError, match="no artifact_provenance"):
        plotter.build_summary(cells)


def test_inconsistent_band_definitions_are_rejected(tmp_path: Path) -> None:
    _write_fake_panel(tmp_path, inconsistent_band=True)
    cells = plotter.load_cells(tmp_path)
    with pytest.raises(ValueError, match="band definitions"):
        plotter.aggregate_band_usage(cells)


def test_wrong_reference_protocol_is_rejected(tmp_path: Path) -> None:
    _write_fake_panel(tmp_path)
    path = tmp_path / "cells" / "T1_seed1" / "result.json"
    payload = json.loads(path.read_text())
    payload["reference_protocol_version"] = "wrong"
    path.write_text(json.dumps(payload))
    cells = plotter.load_cells(tmp_path)
    with pytest.raises(ValueError, match="reference protocol"):
        plotter.build_summary(cells)


def test_reference_statistics_reproduce_published_curve() -> None:
    reference = plotter.reference_series()
    ordered = {row["window"]: row for row in reference["txc_ordered"]}
    assert ordered[1]["mean"] == pytest.approx(0.2177600805709575)
    assert ordered[10]["mean"] == pytest.approx(0.2548184874129588)
    assert ordered[10]["std_sample"] == pytest.approx(0.008453839938130753)
