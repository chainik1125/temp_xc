"""Cross-fitted formation curves in a conventional SAE feature basis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.temporal_screen_1.feature_formation.estimators import CurvePoint


Array = np.ndarray


@dataclass(frozen=True)
class SAEFeatureCurve:
    """A held-out curve plus fold-wise feature-selection provenance."""

    points: list[CurvePoint]
    selected_feature_counts: dict[int, int]
    selected_per_fold: list[list[int]]


def dense_sae_design(
    indices: Array,
    values: Array,
    time_index: int,
    *,
    d_sae: int,
    feature_ids: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Densify one sparse TopK token for paired event/neutral examples."""

    index_values = np.asarray(indices)
    activation_values = np.asarray(values, dtype=np.float32)
    if index_values.shape != activation_values.shape:
        raise ValueError("indices and values must have the same shape")
    if index_values.ndim != 4 or index_values.shape[1] != 2:
        raise ValueError("expected [pairs, 2, time, k] sparse arrays")
    n_pairs = index_values.shape[0]
    flat_indices = index_values[:, :, time_index].reshape(n_pairs * 2, -1)
    flat_values = activation_values[:, :, time_index].reshape(n_pairs * 2, -1)
    if feature_ids is None:
        features = np.arange(d_sae, dtype=np.int64)
    else:
        features = np.asarray(feature_ids, dtype=np.int64)
    lookup = np.full(d_sae, -1, dtype=np.int64)
    lookup[features] = np.arange(len(features))
    columns = lookup[flat_indices]
    x = np.zeros((n_pairs * 2, len(features)), dtype=np.float32)
    rows = np.repeat(np.arange(n_pairs * 2), flat_indices.shape[1])
    valid = columns.reshape(-1) >= 0
    np.add.at(
        x,
        (rows[valid], columns.reshape(-1)[valid]),
        flat_values.reshape(-1)[valid],
    )
    y = np.tile(np.asarray([1, 0], dtype=np.int64), n_pairs)
    groups = np.repeat(np.arange(n_pairs), 2)
    return x, y, groups


def _select_features(
    x: Array,
    y: Array,
    pair_indices: Array,
    *,
    top_n: int,
) -> Array:
    """Select features by paired standardized event-minus-neutral effect."""

    event = x[2 * pair_indices]
    neutral = x[2 * pair_indices + 1]
    difference = event - neutral
    scale = np.std(difference, axis=0, ddof=1)
    score = np.mean(difference, axis=0) / np.maximum(scale, 1e-5)
    order = np.argsort(np.abs(score), kind="stable")
    return order[-min(int(top_n), x.shape[1]) :][::-1].astype(np.int64)


def _fit_model(x: Array, y: Array, *, regularization: float, seed: int):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(regularization),
            solver="lbfgs",
            max_iter=2_000,
            random_state=int(seed),
        ),
    )
    model.fit(x, y)
    return model


def _points_from_predictions(
    offsets: Array,
    predictions: dict[int, Array],
    labels: Array,
) -> list[CurvePoint]:
    baseline = np.full(len(labels), float(np.mean(labels)))
    baseline_loss = float(log_loss(labels, baseline))
    rows = []
    for time_index in sorted(predictions):
        probability = np.clip(predictions[time_index], 1e-6, 1 - 1e-6)
        rows.append(
            CurvePoint(
                offset=int(offsets[time_index]),
                auc=float(roc_auc_score(labels, probability)),
                log_loss_gain_nats=float(
                    baseline_loss - log_loss(labels, probability)
                ),
            )
        )
    return rows


def transported_sae_curve(
    indices: Array,
    values: Array,
    offsets: Iterable[int],
    discovery_band: tuple[int, int],
    *,
    d_sae: int,
    top_n: int = 16,
    n_splits: int = 5,
    regularization: float = 0.1,
    seed: int = 0,
) -> SAEFeatureCurve:
    """Select SAE features in one band and transport the readout through time."""

    offsets_array = np.asarray(list(offsets), dtype=np.int64)
    n_pairs = np.asarray(indices).shape[0]
    lo, hi = discovery_band
    discovery = np.flatnonzero((offsets_array >= lo) & (offsets_array <= hi))
    if not len(discovery):
        raise ValueError("empty discovery band")
    all_x = [
        dense_sae_design(indices, values, t, d_sae=d_sae)[0]
        for t in range(len(offsets_array))
    ]
    labels = np.tile(np.asarray([1, 0], dtype=np.int64), n_pairs)
    predictions = {
        t: np.full(n_pairs * 2, np.nan, dtype=np.float64)
        for t in range(len(offsets_array))
    }
    selected_per_fold = []
    cv = GroupKFold(n_splits=min(n_splits, n_pairs))
    pair_ids = np.arange(n_pairs)
    for fold, (train_pairs, test_pairs) in enumerate(
        cv.split(pair_ids, groups=pair_ids)
    ):
        discovery_x = np.mean(
            np.stack([all_x[int(t)] for t in discovery], axis=0),
            axis=0,
        )
        selected = _select_features(
            discovery_x,
            labels,
            train_pairs,
            top_n=top_n,
        )
        selected_per_fold.append(selected.tolist())
        train_rows = np.concatenate(
            [np.asarray([2 * pair, 2 * pair + 1]) for pair in train_pairs]
        )
        test_rows = np.concatenate(
            [np.asarray([2 * pair, 2 * pair + 1]) for pair in test_pairs]
        )
        model = _fit_model(
            discovery_x[train_rows][:, selected],
            labels[train_rows],
            regularization=regularization,
            seed=seed + fold,
        )
        for t, x_time in enumerate(all_x):
            predictions[t][test_rows] = model.predict_proba(
                x_time[test_rows][:, selected]
            )[:, 1]
    if any(not np.all(np.isfinite(row)) for row in predictions.values()):
        raise RuntimeError("SAE transported cross-fit left missing predictions")
    counts = Counter(feature for fold in selected_per_fold for feature in fold)
    return SAEFeatureCurve(
        points=_points_from_predictions(offsets_array, predictions, labels),
        selected_feature_counts=dict(sorted(counts.items())),
        selected_per_fold=selected_per_fold,
    )


def positionwise_sae_curve(
    indices: Array,
    values: Array,
    offsets: Iterable[int],
    *,
    d_sae: int,
    top_n: int = 16,
    n_splits: int = 5,
    regularization: float = 0.1,
    seed: int = 0,
) -> SAEFeatureCurve:
    """Select and fit conventional SAE features independently at each time."""

    offsets_array = np.asarray(list(offsets), dtype=np.int64)
    n_pairs = np.asarray(indices).shape[0]
    labels = np.tile(np.asarray([1, 0], dtype=np.int64), n_pairs)
    predictions = {
        t: np.full(n_pairs * 2, np.nan, dtype=np.float64)
        for t in range(len(offsets_array))
    }
    selected_per_fold: list[list[int]] = []
    cv = GroupKFold(n_splits=min(n_splits, n_pairs))
    pair_ids = np.arange(n_pairs)
    splits = list(cv.split(pair_ids, groups=pair_ids))
    for t in range(len(offsets_array)):
        x_time, _, _ = dense_sae_design(
            indices,
            values,
            t,
            d_sae=d_sae,
        )
        for fold, (train_pairs, test_pairs) in enumerate(splits):
            selected = _select_features(
                x_time,
                labels,
                train_pairs,
                top_n=top_n,
            )
            selected_per_fold.append(selected.tolist())
            train_rows = np.concatenate(
                [
                    np.asarray([2 * pair, 2 * pair + 1])
                    for pair in train_pairs
                ]
            )
            test_rows = np.concatenate(
                [
                    np.asarray([2 * pair, 2 * pair + 1])
                    for pair in test_pairs
                ]
            )
            model = _fit_model(
                x_time[train_rows][:, selected],
                labels[train_rows],
                regularization=regularization,
                seed=seed + 10_007 * t + fold,
            )
            predictions[t][test_rows] = model.predict_proba(
                x_time[test_rows][:, selected]
            )[:, 1]
    if any(not np.all(np.isfinite(row)) for row in predictions.values()):
        raise RuntimeError("SAE positionwise cross-fit left missing predictions")
    counts = Counter(feature for fold in selected_per_fold for feature in fold)
    return SAEFeatureCurve(
        points=_points_from_predictions(offsets_array, predictions, labels),
        selected_feature_counts=dict(sorted(counts.items())),
        selected_per_fold=selected_per_fold,
    )

