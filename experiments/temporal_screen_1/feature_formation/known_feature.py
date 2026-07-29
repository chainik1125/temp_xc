"""Measurements for the formation of a known scalar feature.

The primary use is Ward-style backtracking directions.  A direction is fixed
before this analysis; at each event-relative token we project event and
same-rollout neutral residuals onto it.  This avoids fitting a new readout at
every position and therefore separates "is this known object present?" from
"is some position-specific information decodable?".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


Array = np.ndarray


@dataclass(frozen=True)
class ScalarCurvePoint:
    """Paired event-versus-neutral statistics at one relative token."""

    offset: int
    event_mean: float
    neutral_mean: float
    paired_difference: float
    paired_difference_se: float
    paired_effect_dz: float
    auc: float
    event_nonzero_fraction: float
    neutral_nonzero_fraction: float


def _validate_scalar_panel(
    event: Array,
    neutral: Array,
    offsets: Iterable[int],
) -> tuple[Array, Array, Array]:
    event_array = np.asarray(event, dtype=np.float64)
    neutral_array = np.asarray(neutral, dtype=np.float64)
    offsets_array = np.asarray(list(offsets), dtype=np.int64)
    if event_array.shape != neutral_array.shape:
        raise ValueError("event and neutral values must have the same shape")
    if event_array.ndim != 2:
        raise ValueError("expected scalar arrays with shape [pairs, time]")
    if event_array.shape[1] != len(offsets_array):
        raise ValueError("time dimension does not match offsets")
    if event_array.shape[0] < 2:
        raise ValueError("at least two pairs are required")
    if not np.all(np.isfinite(event_array)) or not np.all(
        np.isfinite(neutral_array)
    ):
        raise ValueError("scalar panel contains non-finite values")
    return event_array, neutral_array, offsets_array


def paired_scalar_curve(
    event: Array,
    neutral: Array,
    offsets: Iterable[int],
) -> list[ScalarCurvePoint]:
    """Measure a prespecified scalar feature through event-relative time."""

    event_array, neutral_array, offsets_array = _validate_scalar_panel(
        event,
        neutral,
        offsets,
    )
    n_pairs = event_array.shape[0]
    labels = np.concatenate(
        [
            np.ones(n_pairs, dtype=np.int64),
            np.zeros(n_pairs, dtype=np.int64),
        ]
    )
    points = []
    for time_index, offset in enumerate(offsets_array):
        event_time = event_array[:, time_index]
        neutral_time = neutral_array[:, time_index]
        difference = event_time - neutral_time
        difference_std = float(np.std(difference, ddof=1))
        scores = np.concatenate([event_time, neutral_time])
        points.append(
            ScalarCurvePoint(
                offset=int(offset),
                event_mean=float(np.mean(event_time)),
                neutral_mean=float(np.mean(neutral_time)),
                paired_difference=float(np.mean(difference)),
                paired_difference_se=float(
                    difference_std / np.sqrt(n_pairs)
                ),
                paired_effect_dz=float(
                    np.mean(difference) / max(difference_std, 1e-12)
                ),
                auc=float(roc_auc_score(labels, scores)),
                event_nonzero_fraction=float(np.mean(event_time != 0)),
                neutral_nonzero_fraction=float(np.mean(neutral_time != 0)),
            )
        )
    return points


def project_direction_panel(
    panel: Array,
    direction: Array,
    offsets: Iterable[int],
) -> list[ScalarCurvePoint]:
    """Project a [pairs, 2, time, d] panel onto one unit direction."""

    values = np.asarray(panel, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("expected panel with shape [pairs, 2, time, d]")
    if vector.shape != (values.shape[-1],):
        raise ValueError("direction dimension does not match panel")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction must have finite nonzero norm")
    projection = np.einsum(
        "patd,d->pat",
        values,
        vector / norm,
        optimize=True,
    )
    return paired_scalar_curve(
        projection[:, 0],
        projection[:, 1],
        offsets,
    )


def curve_summary(
    points: list[ScalarCurvePoint],
    *,
    bands: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """Summarize peaks and fixed temporal bands without assuming monotonicity."""

    if not points:
        raise ValueError("curve cannot be empty")
    default_bands = {
        "far": (-10_000, -33),
        "pre": (-32, -14),
        "ward": (-13, -8),
        "immediate": (-7, 0),
        "post": (1, 16),
    }
    bands = default_bands if bands is None else bands
    peak_auc = max(points, key=lambda row: row.auc)
    peak_effect = max(points, key=lambda row: row.paired_effect_dz)
    peak_excess = max(peak_auc.auc - 0.5, 0.0)
    threshold = 0.5 + 0.5 * peak_excess
    onset = next(
        (row.offset for row in points if row.auc >= threshold),
        None,
    )
    band_rows = {}
    for name, (lo, hi) in bands.items():
        selected = [row for row in points if lo <= row.offset <= hi]
        if not selected:
            continue
        band_rows[name] = {
            "n_offsets": len(selected),
            "mean_auc": float(np.mean([row.auc for row in selected])),
            "max_auc": float(np.max([row.auc for row in selected])),
            "mean_paired_difference": float(
                np.mean([row.paired_difference for row in selected])
            ),
            "mean_paired_effect_dz": float(
                np.mean([row.paired_effect_dz for row in selected])
            ),
        }
    return {
        "peak_auc": float(peak_auc.auc),
        "peak_auc_offset": int(peak_auc.offset),
        "peak_paired_effect_dz": float(peak_effect.paired_effect_dz),
        "peak_effect_offset": int(peak_effect.offset),
        "half_peak_auc_onset_offset": (
            None if onset is None else int(onset)
        ),
        "bands": band_rows,
    }


def curve_to_dict(points: list[ScalarCurvePoint]) -> list[dict]:
    """JSON-ready form of a known-feature curve."""

    return [asdict(row) for row in points]
