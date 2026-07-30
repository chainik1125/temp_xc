"""Render the Fourier backtracking sensitivity result against pinned references.

The scientific comparison is deliberately narrow:

* every Fourier point is the unique ``probes["fourier"]`` summary with
  ``n_features == 32``;
* each seed-level point is the unweighted mean of five question-grouped outer
  fold average-precision scores;
* plotted summaries are mean +/- sample SD over seeds ``(1, 2, 42)``;
* Aniket's reference values are frozen from commit
  ``d9c7fc7b22352394b6d1b91897cdb82d0b128f0e``.

Recovered artifacts are useful sensitivity analyses, but they are not
replications on Aniket's exact cohort.  The renderer therefore refuses an
unmarked artifact/cohort mismatch and makes a declared mismatch prominent in
the figure and machine-readable outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WINDOWS = (1, 2, 4, 6, 10)
SEEDS = (1, 2, 42)
PROBE_FEATURES = 32
REFERENCE_COMMIT = "d9c7fc7b22352394b6d1b91897cdb82d0b128f0e"
REFERENCE_PROTOCOL = "2026-07-26.t16.1"
REFERENCE_ARTIFACT_SHA256 = "a816928d6e0520e027d0ef06925a43e2baec1e62e2dc1ff85b26cff5f1b6fc1a"
REFERENCE_COHORT_SHA256 = "f397f4caf6212825bd98b1b82be932ae634f01a716fd7e3642fd3d7640b27c0b"

# Exact rows from the pinned publication/window_sweep_seed_metrics.csv.  Keeping
# seed values rather than only rounded figure summaries makes the reference
# aggregation independently checkable.
REFERENCE_SEED_VALUES: dict[str, dict[int, dict[int, float]]] = {
    "txc_ordered": {
        1: {1: 0.2225705087906432, 2: 0.21272051016770127, 42: 0.21798922275452806},
        2: {1: 0.22493309538897183, 2: 0.22543836783282134, 42: 0.23626933001783446},
        4: {1: 0.25382563442937667, 2: 0.24634574449390523, 42: 0.2394986866528357},
        6: {1: 0.24553747141042298, 2: 0.2509032769353158, 42: 0.2572137374644074},
        10: {1: 0.25045286117076315, 2: 0.2494399917620031, 42: 0.2645626093051201},
    },
    "sae_positional": {
        1: {1: 0.22753160261269895, 2: 0.23370185458226142, 42: 0.2031028741304593},
        2: {1: 0.1869833879015828, 2: 0.20612340342164046, 42: 0.19369916679600094},
        4: {1: 0.20314612542144134, 2: 0.1896698673989865, 42: 0.1903676072483251},
        6: {1: 0.2019095307928675, 2: 0.17857005923039343, 42: 0.16144003385868688},
        10: {1: 0.17753940819167105, 2: 0.16939043784691052, 42: 0.1661015477720333},
    },
    "sae_invariant": {
        1: {1: 0.22753160261269895, 2: 0.23370185458226142, 42: 0.2031028741304593},
        2: {1: 0.21242295116688234, 2: 0.2184010905196117, 42: 0.20288997145110152},
        4: {1: 0.22500432335325887, 2: 0.21317498442526323, 42: 0.218748251429705},
        6: {1: 0.22758623434594227, 2: 0.21279516854114983, 42: 0.22020336663291307},
        10: {1: 0.22830653261068284, 2: 0.21579586425528058, 42: 0.2259604743339803},
    },
    "sae_last_token": {
        1: {1: 0.22753160261269895, 2: 0.23370185458226142, 42: 0.2031028741304593},
        2: {1: 0.20354124461907022, 2: 0.2111041274894844, 42: 0.20820707761977908},
        4: {1: 0.22832436490835145, 2: 0.19751908589938638, 42: 0.20394903314179355},
        6: {1: 0.22018881091417036, 2: 0.20484309019792302, 42: 0.19620358059846849},
        10: {1: 0.22188154929434512, 2: 0.20764213429568995, 42: 0.212610016499986},
    },
}

REFERENCE_LABELS = {
    "txc_ordered": "TXC ordered (Aniket, exact cohort)",
    "sae_positional": "Positional SAE (same sweep)",
    "sae_invariant": "Invariant SAE (same sweep)",
    "sae_last_token": "Last-token SAE (same sweep)",
}
TSAE_PUBLISHED_VALUE = 0.245
TSAE_UNROUNDED_VALUE = 0.24481534796544918


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sample_stats(seed_values: dict[int, float]) -> dict[str, Any]:
    values = [float(seed_values[seed]) for seed in sorted(seed_values)]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("statistics require finite seed values")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std_sample": statistics.stdev(values) if len(values) > 1 else 0.0,
        "seed_values": {str(seed): float(seed_values[seed]) for seed in sorted(seed_values)},
    }


def reference_series() -> dict[str, list[dict[str, Any]]]:
    """Return exact reference seed summaries in publication order."""

    return {
        name: [
            {
                "window": window,
                **_sample_stats(REFERENCE_SEED_VALUES[name][window]),
            }
            for window in WINDOWS
        ]
        for name in REFERENCE_SEED_VALUES
    }


def _unique_s32_probe(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        candidates = payload["probes"]["fourier"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing probes['fourier']") from exc
    matches = [
        row
        for row in candidates
        if isinstance(row, dict) and int(row.get("n_features", -1)) == PROBE_FEATURES
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected exactly one Fourier S={PROBE_FEATURES} probe; found {len(matches)}"
        )
    selected = matches[0]
    ordered = selected.get("ordered_pr_auc")
    if not isinstance(ordered, dict):
        raise ValueError(f"{path}: S=32 probe lacks ordered_pr_auc")
    fold_values = [float(value) for value in ordered.get("fold_values", [])]
    if len(fold_values) != 5 or not all(math.isfinite(value) for value in fold_values):
        raise ValueError(f"{path}: ordered PR-AUC must contain five finite fold values")
    recorded_mean = float(ordered.get("mean", math.nan))
    calculated_mean = statistics.fmean(fold_values)
    if not math.isfinite(recorded_mean) or not math.isclose(
        recorded_mean, calculated_mean, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"{path}: ordered_pr_auc.mean {recorded_mean} does not equal "
            f"the five-fold mean {calculated_mean}"
        )
    folds = selected.get("folds", [])
    if folds and (
        len(folds) != 5
        or any(
            int(row.get("n_features", -1)) != PROBE_FEATURES
            or int(row.get("n_features_actual", -1)) != PROBE_FEATURES
            for row in folds
        )
    ):
        raise ValueError(f"{path}: S=32 fold metadata is inconsistent")
    return selected


def _band_usage(payload: dict[str, Any], path: Path) -> list[dict[str, Any]] | None:
    raw = payload.get("ordered_band_usage")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: ordered_band_usage must be a non-empty list")
    rows = []
    for expected_band, row in enumerate(raw):
        if not isinstance(row, dict) or int(row.get("band", -1)) != expected_band:
            raise ValueError(f"{path}: ordered_band_usage bands must be sequential")
        frequencies = tuple(int(value) for value in row.get("frequencies", []))
        share = float(row.get("activation_mass_share", math.nan))
        if not frequencies or not math.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ValueError(f"{path}: invalid activation-mass share for band {expected_band}")
        rows.append(
            {
                "band": expected_band,
                "frequencies": frequencies,
                "activation_mass_share": share,
            }
        )
    if not math.isclose(
        sum(row["activation_mass_share"] for row in rows),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise ValueError(f"{path}: activation-mass shares do not sum to one")
    return rows


def load_cells(results_root: Path) -> list[dict[str, Any]]:
    """Load and validate the exact five-window, three-seed result panel."""

    cells = []
    for window in WINDOWS:
        for seed in SEEDS:
            path = results_root / "cells" / f"T{window}_seed{seed}" / "result.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing expected result: {path}")
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                raise ValueError(f"{path}: status is not complete")
            if int(payload.get("window", -1)) != window or int(payload.get("seed", -1)) != seed:
                raise ValueError(f"{path}: path and payload window/seed disagree")
            probe = _unique_s32_probe(payload, path)
            effective_l0 = payload.get("effective_l0", {}).get("ordered")
            cells.append(
                {
                    "window": window,
                    "seed": seed,
                    "ordered_pr_auc": float(probe["ordered_pr_auc"]["mean"]),
                    "fold_values": [
                        float(value) for value in probe["ordered_pr_auc"]["fold_values"]
                    ],
                    "artifact_sha256": str(payload.get("artifact_sha256", "")),
                    "cohort_sha256": str(payload.get("cohort_sha256", "")),
                    "artifact_provenance": payload.get("artifact_provenance"),
                    "reference_commit": str(payload.get("reference_commit", "")),
                    "reference_protocol_version": str(
                        payload.get("reference_protocol_version", "")
                    ),
                    "effective_l0": effective_l0,
                    "ordered_band_usage": _band_usage(payload, path),
                    "path": str(path),
                }
            )
    return cells


def _constant(values: Sequence[Any], *, label: str) -> Any:
    if not values:
        raise ValueError(f"{label} is empty")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{label} differs across cells")
    return first


def provenance_summary(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reference_commit = _constant(
        [cell["reference_commit"] for cell in cells],
        label="reference commit",
    )
    reference_protocol = _constant(
        [cell["reference_protocol_version"] for cell in cells],
        label="reference protocol",
    )
    if not reference_commit or not REFERENCE_COMMIT.startswith(reference_commit):
        raise ValueError(
            f"results cite reference commit {reference_commit!r}, not pinned {REFERENCE_COMMIT}"
        )
    if reference_protocol != REFERENCE_PROTOCOL:
        raise ValueError(
            f"results cite reference protocol {reference_protocol!r}, not {REFERENCE_PROTOCOL!r}"
        )
    artifact = _constant(
        [cell["artifact_sha256"] for cell in cells],
        label="artifact SHA-256",
    )
    cohort = _constant(
        [cell["cohort_sha256"] for cell in cells],
        label="cohort SHA-256",
    )
    declared = _constant(
        [cell["artifact_provenance"] for cell in cells],
        label="artifact provenance",
    )
    exact = (
        artifact == REFERENCE_ARTIFACT_SHA256
        and cohort == REFERENCE_COHORT_SHA256
        and declared is None
    )
    if exact:
        return {
            "comparison_kind": "exact-reference-artifact",
            "comparable_to_aniket_reference": True,
            "artifact_sha256": artifact,
            "cohort_sha256": cohort,
            "reference_commit": reference_commit,
            "reference_protocol_version": reference_protocol,
            "warning": None,
            "declared_provenance": None,
        }
    if not isinstance(declared, dict):
        raise ValueError(
            "artifact/cohort differs from Aniket's pinned reference but no "
            "artifact_provenance mismatch declaration is present"
        )
    if declared.get("matches_reference_cohort") is not False:
        raise ValueError("mismatched artifact must declare matches_reference_cohort=false")
    warning = str(declared.get("provenance_warning", "")).strip()
    if not warning:
        raise ValueError("mismatched artifact provenance must include a warning")
    return {
        "comparison_kind": "recovered-artifact-sensitivity",
        "comparable_to_aniket_reference": False,
        "artifact_sha256": artifact,
        "cohort_sha256": cohort,
        "reference_commit": reference_commit,
        "reference_protocol_version": reference_protocol,
        "warning": warning,
        "declared_provenance": declared,
    }


def aggregate_fourier(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for window in WINDOWS:
        local = [cell for cell in cells if cell["window"] == window]
        seeds = {int(cell["seed"]) for cell in local}
        if seeds != set(SEEDS):
            raise ValueError(f"T={window}: seeds {sorted(seeds)} do not match {list(SEEDS)}")
        seed_values = {int(cell["seed"]): float(cell["ordered_pr_auc"]) for cell in local}
        l0_values = {
            str(cell["seed"]): cell["effective_l0"]
            for cell in sorted(local, key=lambda x: x["seed"])
        }
        rows.append(
            {
                "window": window,
                **_sample_stats(seed_values),
                "effective_l0_by_seed": l0_values,
            }
        )
    return rows


def aggregate_band_usage(
    cells: Sequence[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    present = [cell["ordered_band_usage"] is not None for cell in cells]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("ordered_band_usage is present for only some cells")
    output = []
    for window in WINDOWS:
        local = sorted(
            [cell for cell in cells if cell["window"] == window],
            key=lambda cell: cell["seed"],
        )
        definitions = [
            tuple(tuple(row["frequencies"]) for row in cell["ordered_band_usage"]) for cell in local
        ]
        frequencies = _constant(definitions, label=f"T={window} band definitions")
        for band, frequency_tuple in enumerate(frequencies):
            seed_values = {
                int(cell["seed"]): float(cell["ordered_band_usage"][band]["activation_mass_share"])
                for cell in local
            }
            output.append(
                {
                    "window": window,
                    "band": band,
                    "frequencies": list(frequency_tuple),
                    **_sample_stats(seed_values),
                }
            )
    return output


def build_summary(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "probe_extraction": {
            "representation": "fourier",
            "n_features": PROBE_FEATURES,
            "selection": "unique probes['fourier'] row with n_features == 32",
            "seed_statistic": (
                "unweighted mean of five question-grouped outer-fold average_precision_score values"
            ),
            "cross_seed_statistic": "mean +/- sample SD over seeds 1, 2, and 42",
        },
        "provenance": provenance_summary(cells),
        "fourier": aggregate_fourier(cells),
        "reference": {
            "commit": REFERENCE_COMMIT,
            "protocol_version": REFERENCE_PROTOCOL,
            "artifact_sha256": REFERENCE_ARTIFACT_SHA256,
            "cohort_sha256": REFERENCE_COHORT_SHA256,
            "series": reference_series(),
            "tsae_t1_marker": {
                "window": 1,
                "published_value": TSAE_PUBLISHED_VALUE,
                "unrounded_value": TSAE_UNROUNDED_VALUE,
                "dictionary_seed": 42,
                "training_steps": 300_000,
                "n_features": 32,
                "note": "single contextual marker; never a temporal-window curve",
            },
        },
        "fourier_band_usage": aggregate_band_usage(cells),
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_errorbar(
    ax: plt.Axes,
    rows: Sequence[dict[str, Any]],
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str | tuple,
    linewidth: float,
    zorder: int,
) -> None:
    ax.errorbar(
        [row["window"] for row in rows],
        [row["mean"] for row in rows],
        yerr=[row["std_sample"] for row in rows],
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=5.0,
        capsize=2.5,
        capthick=0.8,
        elinewidth=0.8,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label=label,
        zorder=zorder,
    )


def plot_summary(summary: dict[str, Any], output_stem: Path) -> tuple[Path, Path]:
    _configure_style()
    has_bands = bool(summary["fourier_band_usage"])
    figure, axes = plt.subplots(
        1,
        2 if has_bands else 1,
        figsize=(11.4 if has_bands else 6.2, 5.35),
        squeeze=False,
    )
    performance = axes[0, 0]
    provenance = summary["provenance"]
    recovered = not provenance["comparable_to_aniket_reference"]
    fourier_label = (
        "Fourier XC (RECOVERED artifact sensitivity)"
        if recovered
        else "Fourier XC (exact reference artifact)"
    )
    _plot_errorbar(
        performance,
        summary["fourier"],
        label=fourier_label,
        color="#B2182B",
        marker="o",
        linestyle="-",
        linewidth=2.2,
        zorder=7,
    )
    reference = summary["reference"]["series"]
    _plot_errorbar(
        performance,
        reference["txc_ordered"],
        label=REFERENCE_LABELS["txc_ordered"],
        color="#2166AC",
        marker="s",
        linestyle="-",
        linewidth=1.8,
        zorder=6,
    )
    contextual = (
        ("sae_positional", "#4D4D4D", "^", (0, (4, 2))),
        ("sae_invariant", "#1B9E77", "v", (0, (2, 2))),
        ("sae_last_token", "#D95F02", "P", (0, (1, 2))),
    )
    for name, color, marker, linestyle in contextual:
        _plot_errorbar(
            performance,
            reference[name],
            label=REFERENCE_LABELS[name],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.15,
            zorder=4,
        )
    performance.scatter(
        [1],
        [TSAE_PUBLISHED_VALUE],
        marker="D",
        s=38,
        color="#7570B3",
        edgecolors="white",
        linewidths=0.6,
        label="T-SAE T=1 only (300k, seed 42; not a curve)",
        zorder=8,
    )
    performance.set_title("S=32 question-grouped ordered detection")
    performance.set_xlabel("Window length T")
    performance.set_ylabel("Backtracking detection PR-AUC")
    performance.set_xticks(WINDOWS)
    performance.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    performance.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
    )

    if has_bands:
        band_axis = axes[0, 1]
        band_rows = summary["fourier_band_usage"]
        maximum_bands = max(row["band"] for row in band_rows) + 1
        colors = ("#2166AC", "#67A9CF", "#FDDBC7", "#B2182B")
        bottoms = np.zeros(len(WINDOWS), dtype=np.float64)
        for band in range(maximum_bands):
            local = {row["window"]: row for row in band_rows if row["band"] == band}
            means = np.asarray(
                [local[window]["mean"] if window in local else 0.0 for window in WINDOWS]
            )
            band_axis.bar(
                WINDOWS,
                means,
                bottom=bottoms,
                width=0.68,
                color=colors[band % len(colors)],
                edgecolor="white",
                linewidth=0.6,
                label=("DC" if band == 0 else f"AC band {band}"),
            )
            for index, window in enumerate(WINDOWS):
                if window not in local:
                    continue
                std = float(local[window]["std_sample"])
                if std:
                    band_axis.errorbar(
                        [window],
                        [bottoms[index] + means[index] / 2.0],
                        yerr=[std],
                        color="#2B2B2B",
                        linewidth=0.7,
                        capsize=1.8,
                        zorder=6,
                    )
            bottoms += means
        band_axis.set_title("Fourier code activation-mass allocation")
        band_axis.set_xlabel("Window length T")
        band_axis.set_ylabel("Activation-mass share")
        band_axis.set_xticks(WINDOWS)
        band_axis.set_ylim(0.0, 1.0)
        band_axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        band_axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=2,
            frameon=False,
            title="Band edges vary with T; exact bins are in JSON/CSV",
        )

    figure.suptitle("Fourier backtracking recovery against Aniket's pinned reference", y=0.985)
    if recovered:
        subtitle = (
            "RECOVERED-ARTIFACT SENSITIVITY ONLY — cohort/artifact differs from "
            "Aniket d9c7fc7b; curves are not a direct replication"
        )
        subtitle_color = "#B2182B"
    else:
        subtitle = (
            f"Exact reference artifact and cohort; Aniket reference commit {REFERENCE_COMMIT[:9]}"
        )
        subtitle_color = "#333333"
    figure.text(
        0.5,
        0.935,
        subtitle,
        ha="center",
        va="top",
        color=subtitle_color,
        fontsize=9.0,
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.01,
        (
            "Points: mean +/- sample SD across seeds 1, 2, 42. "
            "Reference curves are frozen from the pinned publication CSV."
        ),
        ha="left",
        va="bottom",
        color="#555555",
        fontsize=7.2,
    )
    if has_bands:
        figure.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.285,
            top=0.78,
            wspace=0.20,
        )
    else:
        figure.subplots_adjust(
            left=0.13,
            right=0.98,
            bottom=0.285,
            top=0.78,
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    temporary_png = png.with_name(f".{png.name}.tmp")
    temporary_pdf = pdf.with_name(f".{pdf.name}.tmp")
    figure.savefig(temporary_png, dpi=300, format="png")
    figure.savefig(temporary_pdf, format="pdf")
    plt.close(figure)
    os.replace(temporary_png, png)
    os.replace(temporary_pdf, pdf)
    return png, pdf


def write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = (
        "record_type",
        "series",
        "window",
        "band",
        "frequencies",
        "n",
        "mean",
        "std_sample",
        "seed_values",
        "source",
        "comparable_to_aniket_reference",
        "artifact_sha256",
        "cohort_sha256",
    )
    provenance = summary["provenance"]
    rows = []
    rows.extend(
        {
            "record_type": "performance",
            "series": "fourier",
            "window": row["window"],
            "band": "",
            "frequencies": "",
            "n": row["n"],
            "mean": row["mean"],
            "std_sample": row["std_sample"],
            "seed_values": json.dumps(row["seed_values"], sort_keys=True),
            "source": provenance["comparison_kind"],
            "comparable_to_aniket_reference": provenance["comparable_to_aniket_reference"],
            "artifact_sha256": provenance["artifact_sha256"],
            "cohort_sha256": provenance["cohort_sha256"],
        }
        for row in summary["fourier"]
    )
    for name, series in summary["reference"]["series"].items():
        rows.extend(
            {
                "record_type": "performance",
                "series": name,
                "window": row["window"],
                "band": "",
                "frequencies": "",
                "n": row["n"],
                "mean": row["mean"],
                "std_sample": row["std_sample"],
                "seed_values": json.dumps(row["seed_values"], sort_keys=True),
                "source": f"Aniket commit {REFERENCE_COMMIT}",
                "comparable_to_aniket_reference": True,
                "artifact_sha256": REFERENCE_ARTIFACT_SHA256,
                "cohort_sha256": REFERENCE_COHORT_SHA256,
            }
            for row in series
        )
    rows.append(
        {
            "record_type": "context_marker",
            "series": "tsae_t1_300k_seed42",
            "window": 1,
            "band": "",
            "frequencies": "",
            "n": 1,
            "mean": TSAE_PUBLISHED_VALUE,
            "std_sample": "",
            "seed_values": json.dumps({"42": TSAE_UNROUNDED_VALUE}),
            "source": "submitted-paper S=32 marker; not a T curve",
            "comparable_to_aniket_reference": True,
            "artifact_sha256": "",
            "cohort_sha256": "",
        }
    )
    for row in summary["fourier_band_usage"] or []:
        rows.append(
            {
                "record_type": "band_usage",
                "series": "fourier_activation_mass_share",
                "window": row["window"],
                "band": row["band"],
                "frequencies": json.dumps(row["frequencies"]),
                "n": row["n"],
                "mean": row["mean"],
                "std_sample": row["std_sample"],
                "seed_values": json.dumps(row["seed_values"], sort_keys=True),
                "source": provenance["comparison_kind"],
                "comparable_to_aniket_reference": provenance["comparable_to_aniket_reference"],
                "artifact_sha256": provenance["artifact_sha256"],
                "cohort_sha256": provenance["cohort_sha256"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def analyze(
    results_root: Path,
    *,
    output_dir: Path,
    stem: str = "backtracking_fourier_summary",
) -> dict[str, Any]:
    cells = load_cells(results_root)
    summary = build_summary(cells)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / stem
    png, pdf = plot_summary(summary, output_stem)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    summary["outputs"] = {
        "png": str(png),
        "pdf": str(pdf),
        "json": str(json_path),
        "csv": str(csv_path),
    }
    temporary_json = json_path.with_name(f".{json_path.name}.tmp")
    temporary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, json_path)
    write_csv(summary, csv_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Fourier backtracking S=32 ordered PR-AUC sensitivity "
            "against Aniket's pinned reference curves."
        )
    )
    parser.add_argument(
        "results_root",
        type=Path,
        help="result root containing cells/T{window}_seed{seed}/result.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: RESULTS_ROOT/publication)",
    )
    parser.add_argument("--stem", default="backtracking_fourier_summary")
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_root / "publication"
    summary = analyze(args.results_root, output_dir=output_dir, stem=args.stem)
    print(json.dumps(summary["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
