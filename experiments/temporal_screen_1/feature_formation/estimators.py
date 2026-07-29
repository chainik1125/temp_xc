"""SAE-free estimators for event-aligned feature-formation curves.

The basic input is a paired panel with shape ``[pairs, 2, time, features]``.
Index 0 in the second axis is an event-aligned trajectory and index 1 is a
same-trajectory neutral anchor.  Cross-validation always holds out complete
pairs, so a classifier cannot recognize a rollout and use that recognition to
distinguish its two anchors.

Two curves are deliberately kept separate:

``positionwise``
    Fits a new local or trailing-window linear readout at every relative time.
    This measures whether *some* linearly decodable state has formed.

``transported``
    Fits one readout in a predeclared discovery band and evaluates that same
    readout at every relative time.  This is the closer operational analogue
    of one feature forming through time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


Array = np.ndarray


@dataclass(frozen=True)
class CurvePoint:
    """Held-out discrimination at one relative token offset."""

    offset: int
    auc: float
    log_loss_gain_nats: float


@dataclass(frozen=True)
class FormationSummary:
    """Threshold summaries of a non-negative formation curve."""

    peak_offset: int
    peak_value: float
    onset_10_offset: int | None
    midpoint_50_offset: int | None
    completion_90_offset: int | None
    rise_time_tokens: int | None
    positive_area: float


def paired_design(panel: Array, time_index: int, width: int = 1) -> tuple[Array, Array, Array]:
    """Return a paired binary design for one causal trailing window.

    Args:
        panel: ``[n_pairs, 2, n_time, d]``; arm 0 is event, arm 1 neutral.
        time_index: Index of the final token in the trailing window.
        width: Number of tokens in the flattened trailing window.
    """

    values = np.asarray(panel, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("panel must have shape [pairs, 2, time, features]")
    if width < 1 or time_index - width + 1 < 0:
        raise ValueError("the requested trailing window is out of bounds")
    block = values[:, :, time_index - width + 1 : time_index + 1]
    x = block.reshape(values.shape[0] * 2, -1)
    y = np.tile(np.asarray([1, 0], dtype=np.int64), values.shape[0])
    groups = np.repeat(np.arange(values.shape[0]), 2)
    return x, y, groups


def _crossfit_predictions(
    x: Array,
    y: Array,
    groups: Array,
    *,
    n_splits: int,
    regularization: float,
    seed: int,
) -> Array:
    """Fit standardized ridge-logistic probes while holding out whole pairs."""

    unique_groups = np.unique(groups)
    splits = min(int(n_splits), len(unique_groups))
    if splits < 2:
        raise ValueError("at least two pairs are required")
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    cv = GroupKFold(n_splits=splits)
    for fold, (train, test) in enumerate(cv.split(x, y, groups)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(regularization),
                solver="lbfgs",
                max_iter=2_000,
                random_state=int(seed + fold),
            ),
        )
        model.fit(x[train], y[train])
        prediction[test] = model.predict_proba(x[test])[:, 1]
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("cross-fitting left non-finite predictions")
    return prediction


def _score_predictions(y: Array, prediction: Array) -> tuple[float, float]:
    """Return AUC and held-out log-loss improvement over the class prior."""

    probability = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1 - 1e-6)
    prevalence = float(np.mean(y))
    baseline = np.full(len(y), prevalence, dtype=np.float64)
    gain = float(log_loss(y, baseline) - log_loss(y, probability))
    return float(roc_auc_score(y, probability)), gain


def positionwise_curve(
    panel: Array,
    offsets: Iterable[int],
    *,
    width: int = 1,
    n_splits: int = 5,
    regularization: float = 0.1,
    seed: int = 0,
) -> list[CurvePoint]:
    """Cross-fit an independently optimized probe at every valid offset."""

    offset_values = [int(value) for value in offsets]
    if len(offset_values) != np.asarray(panel).shape[2]:
        raise ValueError("offsets must match the panel time axis")
    rows = []
    for time_index in range(width - 1, len(offset_values)):
        x, y, groups = paired_design(panel, time_index, width)
        prediction = _crossfit_predictions(
            x,
            y,
            groups,
            n_splits=n_splits,
            regularization=regularization,
            seed=seed + 10_007 * time_index,
        )
        auc, gain = _score_predictions(y, prediction)
        rows.append(
            CurvePoint(
                offset=offset_values[time_index],
                auc=auc,
                log_loss_gain_nats=gain,
            )
        )
    return rows


def transported_curve(
    panel: Array,
    offsets: Iterable[int],
    discovery_band: tuple[int, int],
    *,
    width: int = 1,
    n_splits: int = 5,
    regularization: float = 0.1,
    seed: int = 0,
) -> list[CurvePoint]:
    """Learn one precursor readout and transport it over the whole curve.

    The discovery examples concatenate all valid positions in the inclusive
    band.  The fold-specific scaler and linear direction are then reused
    unchanged at every evaluation offset.
    """

    values = np.asarray(panel, dtype=np.float32)
    offset_values = np.asarray([int(value) for value in offsets])
    if values.shape[2] != len(offset_values):
        raise ValueError("offsets must match the panel time axis")
    lo, hi = (int(discovery_band[0]), int(discovery_band[1]))
    discovery = np.flatnonzero((offset_values >= lo) & (offset_values <= hi))
    discovery = discovery[discovery >= width - 1]
    if not len(discovery):
        raise ValueError("discovery band contains no valid trailing windows")

    n_pairs = values.shape[0]
    groups_by_pair = np.arange(n_pairs)
    cv = GroupKFold(n_splits=min(n_splits, n_pairs))
    predictions = {
        index: np.full(n_pairs * 2, np.nan, dtype=np.float64)
        for index in range(width - 1, len(offset_values))
    }
    labels = np.tile(np.asarray([1, 0], dtype=np.int64), n_pairs)

    for fold, (train_pairs, test_pairs) in enumerate(
        cv.split(groups_by_pair, groups=groups_by_pair)
    ):
        train_blocks = []
        train_labels = []
        for time_index in discovery:
            x_time, y_time, _ = paired_design(values[train_pairs], int(time_index), width)
            train_blocks.append(x_time)
            train_labels.append(y_time)
        x_train = np.concatenate(train_blocks, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(regularization),
                solver="lbfgs",
                max_iter=2_000,
                random_state=int(seed + fold),
            ),
        )
        model.fit(x_train, y_train)
        flat_test = np.concatenate(
            [np.asarray([2 * pair, 2 * pair + 1]) for pair in test_pairs]
        )
        for time_index in predictions:
            x_test, _, _ = paired_design(values[test_pairs], time_index, width)
            predictions[time_index][flat_test] = model.predict_proba(x_test)[:, 1]

    rows = []
    for time_index, prediction in predictions.items():
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("transported cross-fitting left missing predictions")
        auc, gain = _score_predictions(labels, prediction)
        rows.append(
            CurvePoint(
                offset=int(offset_values[time_index]),
                auc=auc,
                log_loss_gain_nats=gain,
            )
        )
    return rows


def summarize_curve(points: Iterable[CurvePoint], *, field: str = "log_loss_gain_nats") -> FormationSummary:
    """Summarize a curve without assuming it is monotone.

    Threshold offsets are the first crossings of fractions of the observed
    positive peak.  They are descriptive; callers should also inspect the
    entire curve and uncertainty.
    """

    rows = list(points)
    if not rows:
        raise ValueError("cannot summarize an empty curve")
    offsets = np.asarray([row.offset for row in rows], dtype=np.int64)
    raw = np.asarray([float(getattr(row, field)) for row in rows])
    values = np.maximum(raw, 0.0)
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])

    def first_crossing(fraction: float) -> int | None:
        if peak <= 0:
            return None
        hits = np.flatnonzero(values >= fraction * peak)
        return int(offsets[hits[0]]) if len(hits) else None

    onset = first_crossing(0.1)
    midpoint = first_crossing(0.5)
    completion = first_crossing(0.9)
    rise = None if onset is None or completion is None else completion - onset
    area = float(np.trapezoid(values, offsets)) if len(rows) > 1 else 0.0
    return FormationSummary(
        peak_offset=int(offsets[peak_index]),
        peak_value=peak,
        onset_10_offset=onset,
        midpoint_50_offset=midpoint,
        completion_90_offset=completion,
        rise_time_tokens=rise,
        positive_area=area,
    )


def curve_to_dict(points: Iterable[CurvePoint]) -> list[dict]:
    """JSON-safe serialization helper."""

    return [asdict(point) for point in points]

