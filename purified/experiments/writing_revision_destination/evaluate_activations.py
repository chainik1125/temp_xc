"""Evaluate temporal signal in cached KLiCKe layer-10 activation windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .extract_activations import (
    EXTRACTION_PROTOCOL_VERSION,
    ExtractionConfig,
    validate_shard_tensors,
    validate_token_cohort,
)
from .klicke import sha256_file


EVALUATION_PROTOCOL_VERSION = "klicke-deletion-raw-activation-gate-v1"
TARGET_COLUMNS = ("capped_token_label", "lexical_label")


@dataclass(frozen=True)
class ActivationDataset:
    activations: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    event_hashes: np.ndarray
    target_name: str
    provenance: dict[str, object]


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_activation_dataset(
    *,
    cohort_path: str | Path,
    cohort_manifest_path: str | Path,
    cache_dir: str | Path,
    target: str,
) -> ActivationDataset:
    """Load a complete cache and revalidate it against the token cohort."""

    if target not in TARGET_COLUMNS:
        raise ValueError(f"target must be one of {TARGET_COLUMNS}")
    cohort_path = Path(cohort_path)
    cohort_manifest_path = Path(cohort_manifest_path)
    cache_dir = Path(cache_dir)
    request_path = cache_dir / "request.json"
    runtime_path = cache_dir / "runtime.json"
    complete_path = cache_dir / "complete.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if request.get("protocol_version") != EXTRACTION_PROTOCOL_VERSION:
        raise ValueError("activation request protocol version drifted")
    if complete.get("status") != "complete":
        raise ValueError("activation cache is not marked complete")
    if complete.get("protocol_version") != EXTRACTION_PROTOCOL_VERSION:
        raise ValueError("activation completion protocol version drifted")
    if complete.get("request_sha256") != sha256_file(request_path):
        raise ValueError("activation request checksum failed")
    if complete.get("runtime_sha256") != sha256_file(runtime_path):
        raise ValueError("activation runtime checksum failed")

    config = ExtractionConfig(**request["config"])
    cohort, _manifest = validate_token_cohort(
        cohort_path,
        cohort_manifest_path,
        config,
    )
    limit = request.get("limit")
    if limit is not None:
        cohort = cohort.iloc[: int(limit)].reset_index(drop=True)
    if len(cohort) != int(complete["rows"]):
        raise ValueError("activation completion row count drifted")
    rows = len(cohort)
    window_tokens = int(complete["window_tokens"])
    hidden_size = int(complete["hidden_size"])
    activations = np.empty(
        (rows, window_tokens, hidden_size),
        dtype=np.float16,
    )
    coverage = np.zeros(rows, dtype=bool)
    for record in complete["shards"]:
        path = cache_dir / str(record["path"])
        sidecar_path = cache_dir / str(record["sidecar"])
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"activation shard checksum failed: {path.name}")
        if record.get("sidecar_sha256") != sha256_file(sidecar_path):
            raise ValueError(
                f"activation shard sidecar checksum failed: {path.name}"
            )
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("tensor_sha256") != record.get("sha256"):
            raise ValueError(f"activation sidecar tensor hash failed: {path.name}")
        if sidecar.get("request_sha256") != complete["request_sha256"]:
            raise ValueError(f"activation sidecar request hash failed: {path.name}")
        start = int(record["start"])
        end = int(record["end"])
        if not 0 <= start < end <= rows or coverage[start:end].any():
            raise ValueError("activation shard coverage overlaps or is invalid")
        tensors = load_file(str(path), device="cpu")
        validate_shard_tensors(
            tensors,
            cohort.iloc[start:end],
            start=start,
            window_tokens=window_tokens,
            hidden_size=hidden_size,
        )
        activations[start:end] = tensors["activations"].numpy()
        coverage[start:end] = True
    if not coverage.all() or not np.isfinite(activations).all():
        raise ValueError("activation cache does not cover finite cohort rows")

    return ActivationDataset(
        activations=activations,
        target=cohort[target].to_numpy(dtype=int),
        groups=cohort["writer_hash"].astype(str).to_numpy(),
        event_hashes=cohort["event_hash"].astype(str).to_numpy(),
        target_name=target,
        provenance={
            "cohort_sha256": sha256_file(cohort_path),
            "cohort_manifest_sha256": sha256_file(cohort_manifest_path),
            "request_sha256": sha256_file(request_path),
            "runtime_sha256": sha256_file(runtime_path),
            "complete_sha256": sha256_file(complete_path),
            "model": request["config"]["model"],
            "model_revision_requested": request["config"]["revision"],
            "model_revision_observed": runtime["model_revision_observed"],
            "layer": request["config"]["layer"],
            "hookpoint": request["hookpoint"],
            "hook_semantics": request["hook_semantics"],
        },
    )


def stable_permutation(
    event_hash: str,
    window_tokens: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return a deterministic nonidentity permutation when T is above one."""

    if window_tokens < 1:
        raise ValueError("window_tokens must be positive")
    if window_tokens == 1:
        return np.asarray([0], dtype=int)
    digest = hashlib.sha256(
        f"{seed}:{window_tokens}:{event_hash}".encode("ascii")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    order = rng.permutation(window_tokens)
    if np.array_equal(order, np.arange(window_tokens)):
        order = np.roll(order, 1)
    return order


def shuffled_windows(
    windows: np.ndarray,
    event_hashes: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    if windows.ndim != 3 or len(windows) != len(event_hashes):
        raise ValueError("windows and event hashes disagree")
    result = np.empty_like(windows)
    for row, event_hash in enumerate(event_hashes):
        result[row] = windows[
            row,
            stable_permutation(
                str(event_hash),
                windows.shape[1],
                seed=seed,
            ),
        ]
    return result


def select_hidden_coordinates(
    activations: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    *,
    dimensions: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Select shared residual coordinates by outer-train-only ANOVA score."""

    if activations.ndim != 3 or len(activations) != len(target):
        raise ValueError("activation and target shapes disagree")
    hidden_size = activations.shape[-1]
    if not 1 <= dimensions <= hidden_size:
        raise ValueError("dimensions must lie within the hidden size")
    labels = np.unique(target[train_indices])
    if len(labels) < 2:
        raise ValueError("coordinate selection requires multiple labels")

    sums = np.zeros((len(labels), hidden_size), dtype=np.float64)
    sums_squared = np.zeros_like(sums)
    counts = np.zeros(len(labels), dtype=np.int64)
    for label_index, label in enumerate(labels):
        indices = train_indices[target[train_indices] == label]
        for start in range(0, len(indices), batch_size):
            batch = activations[indices[start : start + batch_size]].astype(
                np.float32
            )
            flat = batch.reshape(-1, hidden_size)
            sums[label_index] += flat.sum(axis=0, dtype=np.float64)
            sums_squared[label_index] += np.square(
                flat,
                dtype=np.float32,
            ).sum(axis=0, dtype=np.float64)
            counts[label_index] += len(flat)
    total_count = int(counts.sum())
    means = sums / counts[:, None]
    grand_mean = sums.sum(axis=0) / total_count
    between = (
        counts[:, None] * np.square(means - grand_mean[None, :])
    ).sum(axis=0)
    within = (
        sums_squared - np.square(sums) / counts[:, None]
    ).sum(axis=0)
    numerator = between / max(len(labels) - 1, 1)
    denominator = within / max(total_count - len(labels), 1)
    scores = numerator / np.maximum(denominator, 1e-12)
    scores[~np.isfinite(scores)] = -np.inf
    candidates = np.argpartition(scores, -dimensions)[-dimensions:]
    selected = sorted(
        (int(index) for index in candidates),
        key=lambda index: (-scores[index], index),
    )
    return np.asarray(selected, dtype=int)


def activation_view(windows: np.ndarray, view: str) -> np.ndarray:
    """Construct one probe view from oldest-to-newest activation windows."""

    if windows.ndim != 3 or windows.shape[1] < 1:
        raise ValueError("windows must have shape [row, token, feature]")
    endpoint = windows[:, -1, :]
    if view == "endpoint":
        return endpoint
    if view == "invariant_mean_std_max":
        return np.column_stack(
            [
                windows.mean(axis=1),
                windows.std(axis=1),
                windows.max(axis=1),
            ]
        )
    if view == "first_difference":
        if windows.shape[1] < 2:
            raise ValueError("first difference requires T >= 2")
        difference = endpoint - windows[:, -2, :]
        return np.column_stack([endpoint, difference])
    if view == "second_difference":
        if windows.shape[1] < 3:
            raise ValueError("second difference requires T >= 3")
        first = endpoint - windows[:, -2, :]
        second = endpoint - 2 * windows[:, -2, :] + windows[:, -3, :]
        return np.column_stack([endpoint, first, second])
    if view == "trajectory_residual":
        if windows.shape[1] < 4:
            raise ValueError("trajectory residual requires T >= 4")
        extrapolated = (
            -(2.0 / 3.0) * windows[:, -4, :]
            + (1.0 / 3.0) * windows[:, -3, :]
            + (4.0 / 3.0) * windows[:, -2, :]
        )
        return np.column_stack([endpoint, endpoint - extrapolated])
    if view == "ordered":
        return windows.reshape(len(windows), -1)
    if view.startswith("offset_"):
        offset = int(view.removeprefix("offset_"))
        if not 0 <= offset < windows.shape[1]:
            raise ValueError("offset lies outside the activation window")
        return windows[:, -1 - offset, :]
    raise ValueError(f"unknown activation view: {view}")


def _grouped_splits(
    target: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.unique(target)
    groups_per_label = [
        len(np.unique(groups[target == label])) for label in labels
    ]
    effective = min(folds, len(np.unique(groups)), min(groups_per_label))
    if effective < 2:
        raise ValueError("each label needs at least two writer groups")
    splitter = StratifiedGroupKFold(
        n_splits=effective,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(np.zeros(len(target)), target, groups))
    expected_labels = set(int(value) for value in labels)
    for train, test in splits:
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError("writer leakage across evaluation folds")
        if set(int(value) for value in np.unique(target[train])) != expected_labels:
            raise ValueError("a grouped training fold omits an observed label")
    return splits


def _model(*, c_value: float, max_iter: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            solver="lbfgs",
        ),
    )


def _aligned_probabilities(
    model,
    features: np.ndarray,
    labels: tuple[int, ...],
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = model.named_steps["logisticregression"].classes_
    aligned = np.zeros((len(features), len(labels)), dtype=np.float32)
    for source, label in enumerate(classes):
        aligned[:, labels.index(int(label))] = probabilities[:, source]
    return aligned


def select_best_offsets(
    projected: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    *,
    labels: tuple[int, ...],
    folds: int,
    c_value: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """Inner-writer-CV every offset once; prefixes reuse those scores."""

    local_target = target[outer_train]
    local_groups = groups[outer_train]
    inner_splits = _grouped_splits(
        local_target,
        local_groups,
        folds=folds,
        seed=seed,
    )
    offset_scores = []
    for offset in range(projected.shape[1]):
        features = projected[:, -1 - offset, :]
        losses = []
        for inner_train, inner_valid in inner_splits:
            train = outer_train[inner_train]
            valid = outer_train[inner_valid]
            model = _model(c_value=c_value, max_iter=max_iter)
            model.fit(features[train], target[train])
            probabilities = _aligned_probabilities(
                model,
                features[valid],
                labels,
            )
            losses.append(
                log_loss(target[valid], probabilities, labels=labels)
            )
        offset_scores.append(float(np.mean(losses)))
    best_by_window = np.asarray(
        [
            int(np.argmin(offset_scores[:window_tokens]))
            for window_tokens in range(1, projected.shape[1] + 1)
        ],
        dtype=int,
    )
    return best_by_window, offset_scores


def _views(window_tokens: int) -> tuple[str, ...]:
    names = [
        "prior",
        "endpoint",
        "best_offset",
        "invariant_mean_std_max",
    ]
    if window_tokens >= 2:
        names.append("first_difference")
    if window_tokens >= 3:
        names.append("second_difference")
    if window_tokens >= 4:
        names.append("trajectory_residual")
    names.extend(
        [
            "ordered",
            "ordered_fixed_reverse",
            "ordered_fixed_shuffle",
            "ordered_retrained_shuffle",
        ]
    )
    return tuple(names)


def _writer_contrasts(
    *,
    probabilities: dict[str, np.ndarray],
    target: np.ndarray,
    labels: tuple[int, ...],
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    true_columns = np.asarray([labels.index(int(value)) for value in target])
    rows = np.arange(len(target))
    losses = {
        name: -np.log(
            np.clip(values[rows, true_columns], 1e-12, 1.0)
        )
        for name, values in probabilities.items()
    }
    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in np.unique(groups)
    }
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float | int | None]] = {}
    for competitor in probabilities:
        if competitor == "ordered":
            continue
        differences = np.asarray(
            [
                float(
                    (
                        losses[competitor][indices]
                        - losses["ordered"][indices]
                    ).mean()
                )
                for indices in group_indices.values()
            ]
        )
        if draws:
            samples = rng.choice(
                differences,
                size=(draws, len(differences)),
                replace=True,
            ).mean(axis=1)
            lower, upper = np.quantile(samples, [0.025, 0.975])
        else:
            lower = upper = None
        result[f"{competitor}_minus_ordered"] = {
            "equal_writer_mean_log_loss_difference": float(
                differences.mean()
            ),
            "ci95_lower": None if lower is None else float(lower),
            "ci95_upper": None if upper is None else float(upper),
            "writers_positive": int((differences > 0).sum()),
            "writers_total": len(differences),
        }
    return result


def evaluate_sweep(
    dataset: ActivationDataset,
    *,
    window_sizes: Sequence[int],
    projection_dimensions: int = 64,
    outer_folds: int = 5,
    inner_folds: int = 3,
    c_value: float = 0.01,
    max_iter: int = 1_000,
    shuffle_seed: int = 20_260_726,
    bootstrap_draws: int = 2_000,
) -> dict[str, object]:
    """Run the full writer-grouped sweep with fold-local feature selection."""

    activations = dataset.activations
    max_window = activations.shape[1]
    requested = sorted(set(int(value) for value in window_sizes))
    if not requested or requested[0] < 1 or requested[-1] > max_window:
        raise ValueError("window sizes lie outside the activation cache")
    labels = tuple(int(value) for value in sorted(np.unique(dataset.target)))
    if len(labels) < 2:
        raise ValueError("activation target has fewer than two labels")
    splits = _grouped_splits(
        dataset.target,
        dataset.groups,
        folds=outer_folds,
        seed=shuffle_seed,
    )
    predictions = {
        window: {
            view: np.full(
                (len(dataset.target), len(labels)),
                np.nan,
                dtype=np.float32,
            )
            for view in _views(window)
        }
        for window in requested
    }
    fold_audit = []

    for fold, (train, test) in enumerate(splits):
        selected = select_hidden_coordinates(
            activations,
            dataset.target,
            train,
            dimensions=projection_dimensions,
        )
        projected = activations[:, :, selected].astype(np.float32)
        best_offsets, offset_scores = select_best_offsets(
            projected,
            dataset.target,
            dataset.groups,
            train,
            labels=labels,
            folds=inner_folds,
            c_value=c_value,
            max_iter=max_iter,
            seed=shuffle_seed + fold + 1,
        )
        selected_sha = hashlib.sha256(
            selected.astype(np.int32).tobytes()
        ).hexdigest()
        fold_audit.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_writers": len(np.unique(dataset.groups[train])),
                "test_writers": len(np.unique(dataset.groups[test])),
                "selected_coordinates": selected.tolist(),
                "selected_coordinates_sha256": selected_sha,
                "offset_inner_log_loss": offset_scores,
                "best_offset_by_window": {
                    str(window): int(best_offsets[window - 1])
                    for window in requested
                },
            }
        )

        prior = np.asarray(
            [np.sum(dataset.target[train] == label) for label in labels],
            dtype=float,
        )
        prior /= prior.sum()
        for window_tokens in requested:
            windows = projected[:, -window_tokens:, :]
            output = predictions[window_tokens]
            output["prior"][test] = prior
            ordinary_views = [
                "endpoint",
                "invariant_mean_std_max",
            ]
            if window_tokens >= 2:
                ordinary_views.append("first_difference")
            if window_tokens >= 3:
                ordinary_views.append("second_difference")
            if window_tokens >= 4:
                ordinary_views.append("trajectory_residual")
            for view in ordinary_views:
                features = activation_view(windows, view)
                model = _model(c_value=c_value, max_iter=max_iter)
                model.fit(features[train], dataset.target[train])
                output[view][test] = _aligned_probabilities(
                    model,
                    features[test],
                    labels,
                )

            best_offset = int(best_offsets[window_tokens - 1])
            best_features = activation_view(
                windows,
                f"offset_{best_offset}",
            )
            best_model = _model(c_value=c_value, max_iter=max_iter)
            best_model.fit(best_features[train], dataset.target[train])
            output["best_offset"][test] = _aligned_probabilities(
                best_model,
                best_features[test],
                labels,
            )

            ordered = activation_view(windows, "ordered")
            ordered_model = _model(c_value=c_value, max_iter=max_iter)
            ordered_model.fit(ordered[train], dataset.target[train])
            output["ordered"][test] = _aligned_probabilities(
                ordered_model,
                ordered[test],
                labels,
            )
            reversed_test = activation_view(
                windows[test, ::-1, :],
                "ordered",
            )
            output["ordered_fixed_reverse"][test] = _aligned_probabilities(
                ordered_model,
                reversed_test,
                labels,
            )
            shuffled_train_windows = shuffled_windows(
                windows[train],
                dataset.event_hashes[train],
                seed=shuffle_seed,
            )
            shuffled_test_windows = shuffled_windows(
                windows[test],
                dataset.event_hashes[test],
                seed=shuffle_seed,
            )
            shuffled_train = activation_view(
                shuffled_train_windows,
                "ordered",
            )
            shuffled_test = activation_view(
                shuffled_test_windows,
                "ordered",
            )
            output["ordered_fixed_shuffle"][test] = _aligned_probabilities(
                ordered_model,
                shuffled_test,
                labels,
            )
            shuffled_model = _model(c_value=c_value, max_iter=max_iter)
            shuffled_model.fit(
                shuffled_train,
                dataset.target[train],
            )
            output["ordered_retrained_shuffle"][
                test
            ] = _aligned_probabilities(
                shuffled_model,
                shuffled_test,
                labels,
            )

    results = {}
    summary = []
    for window_tokens in requested:
        probabilities = predictions[window_tokens]
        if any(not np.isfinite(values).all() for values in probabilities.values()):
            raise RuntimeError("a raw activation view has missing predictions")
        if window_tokens == 1:
            for control in (
                "best_offset",
                "ordered",
                "ordered_fixed_reverse",
                "ordered_fixed_shuffle",
                "ordered_retrained_shuffle",
            ):
                if not np.allclose(
                    probabilities["endpoint"],
                    probabilities[control],
                    rtol=1e-6,
                    atol=1e-7,
                ):
                    raise AssertionError(
                        f"T=1 endpoint/ordered limit failed for {control}"
                    )
        metrics = {}
        for view, values in probabilities.items():
            predicted = np.asarray(labels)[np.argmax(values, axis=1)]
            metrics[view] = {
                "log_loss": float(
                    log_loss(dataset.target, values, labels=labels)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(dataset.target, predicted)
                ),
            }
        contrasts = _writer_contrasts(
            probabilities=probabilities,
            target=dataset.target,
            labels=labels,
            groups=dataset.groups,
            draws=bootstrap_draws,
            seed=shuffle_seed + window_tokens,
        )
        results[str(window_tokens)] = {
            "window_tokens": window_tokens,
            "metrics": metrics,
            "equal_writer_bootstrap": contrasts,
            "feature_dimensions": {
                view: (
                    0
                    if view == "prior"
                    else (
                        projection_dimensions
                        if view == "best_offset"
                        else (
                            activation_view(
                                np.zeros(
                                    (
                                        1,
                                        window_tokens,
                                        projection_dimensions,
                                    ),
                                    dtype=np.float32,
                                ),
                                (
                                    "ordered"
                                    if view.startswith("ordered_")
                                    else view
                                ),
                            ).shape[1]
                        )
                    )
                )
                for view in metrics
            },
        }
        row: dict[str, object] = {"window_tokens": window_tokens}
        for view, values in metrics.items():
            row[f"{view}_log_loss"] = values["log_loss"]
            row[f"{view}_balanced_accuracy"] = values[
                "balanced_accuracy"
            ]
        summary.append(row)

    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "target": dataset.target_name,
        "labels_observed": list(labels),
        "rows": len(dataset.target),
        "writers": len(np.unique(dataset.groups)),
        "split": "stratified writer-grouped outer folds",
        "dimensionality_control": {
            "method": (
                "outer-train-only ANOVA selection of shared hidden "
                "coordinates, applied identically at every temporal position"
            ),
            "coordinates_per_position": projection_dimensions,
        },
        "best_offset_control": (
            "offset selected using inner writer-grouped cross-validation "
            "inside each outer training fold"
        ),
        "shuffle_control": (
            "stable per-event nonidentity temporal permutation; report both "
            "fixed ordered-probe evaluation and a probe retrained on shuffled "
            "training windows"
        ),
        "trajectory_residual": (
            "endpoint plus the Barenholtz three-preceding-state residual: "
            "x[-1]+2/3*x[-4]-1/3*x[-3]-4/3*x[-2]"
        ),
        "configuration": {
            "window_sizes": requested,
            "outer_folds_requested": outer_folds,
            "outer_folds_effective": len(splits),
            "inner_folds_requested": inner_folds,
            "c_value": c_value,
            "max_iter": max_iter,
            "shuffle_seed": shuffle_seed,
            "bootstrap_draws": bootstrap_draws,
        },
        "activation_provenance": dataset.provenance,
        "implementation_sha256": sha256_file(Path(__file__)),
        "fold_audit": fold_audit,
        "summary": summary,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=TARGET_COLUMNS,
        default="capped_token_label",
    )
    parser.add_argument("--window-sizes", type=int, nargs="+")
    parser.add_argument("--projection-dimensions", type=int, default=64)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--c-value", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--shuffle-seed", type=int, default=20_260_726)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()

    dataset = load_activation_dataset(
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        cache_dir=args.cache_dir,
        target=args.target,
    )
    window_sizes = (
        args.window_sizes
        if args.window_sizes is not None
        else list(range(1, dataset.activations.shape[1] + 1))
    )
    result = evaluate_sweep(
        dataset,
        window_sizes=window_sizes,
        projection_dimensions=args.projection_dimensions,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        c_value=args.c_value,
        max_iter=args.max_iter,
        shuffle_seed=args.shuffle_seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    _atomic_json(result, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
