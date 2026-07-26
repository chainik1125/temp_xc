"""Grouped detection evaluation for one trained ``(T, seed)`` cell."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments.swr_audit.dictionary import (
    _sparse_from_topk,
    encode_txc_batch,
)
from experiments.swr_audit.matched_filter import run_fold as run_residual_fold
from experiments.swr_audit.run import _shuffle_rows, c7_groups, trailing_window

from .protocol import (
    ARTIFACT_OFFSETS,
    ORDER_CONTROLS,
    PROTOCOL_VERSION,
    atomic_json,
    sha256,
    whole_group_subsample,
)
from .train import load_dictionary


def _atomic_sparse(matrix: sparse.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    sparse.save_npz(temporary, matrix)
    os.replace(temporary, path)


def _atomic_predictions(payload: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _load_sparse(path: Path, n_rows: int) -> sparse.csr_matrix:
    matrix = sparse.load_npz(path).tocsr()
    if matrix.shape[0] != n_rows:
        raise ValueError(
            f"cached code row mismatch at {path}: {matrix.shape[0]} != {n_rows}"
        )
    return matrix


def sparse_effective_l0(
    matrix: sparse.csr_matrix, *, nominal_l0: int
) -> dict[str, float | int]:
    """Summarize actual nonzeros after the implementation's TopK then ReLU.

    A nominal TopK budget is only an upper bound because selected negative
    preactivations are zeroed by ReLU. CSR row counts expose that underfill
    directly without changing the trained architecture.
    """

    if nominal_l0 < 1:
        raise ValueError("nominal_l0 must be positive")
    counts = np.diff(matrix.indptr).astype(np.float64, copy=False)
    if len(counts) == 0:
        raise ValueError("cannot summarize effective L0 for an empty matrix")
    mean = float(counts.mean())
    return {
        "nominal_l0": int(nominal_l0),
        "effective_l0_mean": mean,
        "effective_l0_std_sample": (
            float(counts.std(ddof=1)) if len(counts) > 1 else 0.0
        ),
        "effective_l0_min": int(counts.min()),
        "effective_l0_max": int(counts.max()),
        "fill_fraction_mean": mean / nominal_l0,
        "underfilled_row_fraction": float(np.mean(counts < nominal_l0)),
        "zero_row_fraction": float(np.mean(counts == 0)),
    }


def effective_l0_diagnostics(
    matrices: dict[str, dict[str, sparse.csr_matrix]],
    *,
    window: int,
    txc_k_pos: int,
    sae_k_pos: int,
    txc_d_sae: int,
    sae_d_sae: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Return per-representation, per-condition effective-L0 summaries."""

    nominal = {
        "txc": min(txc_k_pos * window, txc_d_sae),
        "sae_positional": min(sae_k_pos, sae_d_sae) * window,
        "sae_invariant": min(sae_k_pos * window, sae_d_sae),
        "sae_last_token": min(sae_k_pos, sae_d_sae),
    }
    return {
        representation: {
            condition: sparse_effective_l0(
                matrix,
                nominal_l0=nominal[representation],
            )
            for condition, matrix in variants.items()
        }
        for representation, variants in matrices.items()
    }


@torch.no_grad()
def _encode_sae_positional_batch(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    k_pos: int,
) -> sparse.csr_matrix:
    """Concatenate one shared SAE's codes in explicit relative-position blocks."""

    weight, bias, decoder_bias = state["W_enc"], state["b_enc"], state["b_dec"]
    chunks = []
    for position in range(x.shape[1]):
        token = x[:, position].to(weight.dtype)
        pre = (token - decoder_bias) @ weight.T + bias
        values, indices = pre.topk(min(k_pos, weight.shape[0]), dim=-1)
        chunks.append(
            _sparse_from_topk(torch.relu(values), indices, weight.shape[0])
        )
    return sparse.hstack(chunks, format="csr")


def encode_sae_positional(
    x: np.ndarray,
    *,
    state: dict[str, torch.Tensor],
    k_pos: int,
    batch_size: int,
    device: str,
) -> sparse.csr_matrix:
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(
            x[start : start + batch_size].astype(np.float32, copy=False)
        ).to(device)
        chunks.append(
            _encode_sae_positional_batch(batch, state, k_pos=k_pos)
        )
    return sparse.vstack(chunks, format="csr")


@torch.no_grad()
def _encode_sae_token_batch(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    k_pos: int,
) -> sparse.csr_matrix:
    weight, bias, decoder_bias = state["W_enc"], state["b_enc"], state["b_dec"]
    token = x.to(weight.dtype)
    pre = (token - decoder_bias) @ weight.T + bias
    values, indices = pre.topk(min(k_pos, weight.shape[0]), dim=-1)
    return _sparse_from_topk(torch.relu(values), indices, weight.shape[0])


def encode_sae_token(
    x: np.ndarray,
    *,
    state: dict[str, torch.Tensor],
    k_pos: int,
    batch_size: int,
    device: str,
) -> sparse.csr_matrix:
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(
            x[start : start + batch_size].astype(np.float32, copy=False)
        ).to(device)
        chunks.append(_encode_sae_token_batch(batch, state, k_pos=k_pos))
    return sparse.vstack(chunks, format="csr")


def encode_txc(
    x: np.ndarray,
    *,
    state: dict[str, torch.Tensor],
    k_pos: int,
    batch_size: int,
    device: str,
) -> sparse.csr_matrix:
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(
            x[start : start + batch_size].astype(np.float32, copy=False)
        ).to(device)
        chunks.append(encode_txc_batch(batch, state, k_pos=k_pos))
    return sparse.vstack(chunks, format="csr")


@torch.no_grad()
def _encode_sae_invariant_batch(
    x: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    k_pos: int,
) -> sparse.csr_matrix:
    weight, bias, decoder_bias = state["W_enc"], state["b_enc"], state["b_dec"]
    batch, window, width = x.shape
    flat = x.reshape(batch * window, width).to(weight.dtype)
    pre = (flat - decoder_bias) @ weight.T + bias
    values, indices = pre.topk(min(k_pos, weight.shape[0]), dim=-1)
    dense = torch.zeros_like(pre)
    dense.scatter_(1, indices, torch.relu(values))
    pooled = dense.reshape(batch, window, -1).amax(dim=1)
    pooled_values, pooled_indices = pooled.topk(
        min(k_pos * window, weight.shape[0]), dim=-1
    )
    return _sparse_from_topk(
        pooled_values, pooled_indices, weight.shape[0]
    )


def encode_sae_invariant(
    x: np.ndarray,
    *,
    state: dict[str, torch.Tensor],
    k_pos: int,
    batch_size: int,
    device: str,
) -> sparse.csr_matrix:
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(
            x[start : start + batch_size].astype(np.float32, copy=False)
        ).to(device)
        chunks.append(
            _encode_sae_invariant_batch(batch, state, k_pos=k_pos)
        )
    return sparse.vstack(chunks, format="csr")


def _condition(
    x: np.ndarray, name: str, *, seed: int, control_index: int
) -> np.ndarray:
    if name == "ordered":
        return x
    return _shuffle_rows(x, seed + 10_000 + control_index, name)


def _codes(
    *,
    x: np.ndarray,
    txc_checkpoint: Path,
    sae_checkpoint: Path,
    code_dir: Path,
    batch_size: int,
    device: str,
    seed: int,
    code_fingerprint: dict,
) -> dict[str, dict[str, sparse.csr_matrix]]:
    code_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = code_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing != code_fingerprint:
            raise ValueError(
                f"code cache provenance mismatch at {metadata_path}; "
                "move the old cache rather than overwriting it"
            )
    else:
        atomic_json(code_fingerprint, metadata_path)

    txc, txc_config = load_dictionary(txc_checkpoint, device=device)
    txc_state = {"W_enc": txc.W_enc, "b_enc": txc.b_enc}
    matrices: dict[str, dict[str, sparse.csr_matrix]] = {
        "txc": {},
        "sae_positional": {},
        "sae_invariant": {},
        "sae_last_token": {},
    }
    for control_index, name in enumerate(("ordered", *ORDER_CONTROLS)):
        path = code_dir / f"txc_{name}.npz"
        if path.exists():
            matrices["txc"][name] = _load_sparse(path, len(x))
        else:
            conditioned = _condition(
                x, name, seed=seed, control_index=control_index
            )
            matrix = encode_txc(
                conditioned,
                state=txc_state,
                k_pos=txc_config.k_pos,
                batch_size=batch_size,
                device=device,
            )
            _atomic_sparse(matrix, path)
            matrices["txc"][name] = matrix
    del txc, txc_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sae, sae_config = load_dictionary(sae_checkpoint, device=device)
    sae_state = {
        "W_enc": sae.W_enc,
        "b_enc": sae.b_enc,
        "b_dec": sae.b_dec,
    }
    for control_index, name in enumerate(("ordered", *ORDER_CONTROLS)):
        conditioned = _condition(
            x, name, seed=seed, control_index=control_index
        )
        positional_path = code_dir / f"sae_positional_{name}.npz"
        if positional_path.exists():
            positional = _load_sparse(positional_path, len(x))
        else:
            positional = encode_sae_positional(
                conditioned,
                state=sae_state,
                k_pos=sae_config.k_pos,
                batch_size=batch_size,
                device=device,
            )
            _atomic_sparse(positional, positional_path)
        matrices["sae_positional"][name] = positional

        last_path = code_dir / f"sae_last_token_{name}.npz"
        if last_path.exists():
            last = _load_sparse(last_path, len(x))
        else:
            last = encode_sae_token(
                conditioned[:, -1],
                state=sae_state,
                k_pos=sae_config.k_pos,
                batch_size=batch_size,
                device=device,
            )
            _atomic_sparse(last, last_path)
        matrices["sae_last_token"][name] = last

    invariant_path = code_dir / "sae_invariant_ordered.npz"
    if invariant_path.exists():
        invariant = _load_sparse(invariant_path, len(x))
    else:
        invariant = encode_sae_invariant(
            x,
            state=sae_state,
            k_pos=sae_config.k_pos,
            batch_size=batch_size,
            device=device,
        )
        _atomic_sparse(invariant, invariant_path)
    matrices["sae_invariant"] = {
        "ordered": invariant,
        **{name: invariant for name in ORDER_CONTROLS},
    }
    del sae, sae_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return matrices


def _mean(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "fold_values": array.tolist(),
        "mean": float(array.mean()),
        "std_sample": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def grouped_fixed_probe_with_predictions(
    ordered: sparse.csr_matrix,
    controls: dict[str, sparse.csr_matrix],
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    s_grid: tuple[int, ...],
    seed: int,
    prediction_dir: Path,
) -> list[dict]:
    """Fit the ordered sparse probe and persist group-bootstrap inputs."""

    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(ordered, y, groups)
    ):
        pos = ordered[train_idx[y[train_idx] == 1]].mean(axis=0).A1
        neg = ordered[train_idx[y[train_idx] == 0]].mean(axis=0).A1
        ranking = np.argsort(np.abs(pos - neg))
        for requested_features in s_grid:
            selected = ranking[-min(requested_features, ordered.shape[1]) :]
            classifier = LogisticRegression(
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=2_000,
                random_state=seed + fold,
            ).fit(ordered[train_idx][:, selected].toarray(), y[train_idx])
            ordered_probability = classifier.predict_proba(
                ordered[test_idx][:, selected].toarray()
            )[:, 1]
            control_probability = {
                name: classifier.predict_proba(
                    matrix[test_idx][:, selected].toarray()
                )[:, 1]
                for name, matrix in controls.items()
            }
            ordered_score = _metrics(y[test_idx], ordered_probability)
            control_scores = {
                name: _metrics(y[test_idx], probability)
                for name, probability in control_probability.items()
            }
            prediction_path = (
                prediction_dir
                / f"S{requested_features}_fold{fold}.npz"
            )
            _atomic_predictions(
                {
                    "test_indices": test_idx.astype(np.int64),
                    "y": y[test_idx].astype(np.int8),
                    "groups": groups[test_idx].astype(str),
                    "ordered": ordered_probability.astype(np.float32),
                    **{
                        f"control_{name}": probability.astype(np.float32)
                        for name, probability in control_probability.items()
                    },
                },
                prediction_path,
            )
            rows.append(
                {
                    "fold": fold,
                    "n_features": int(requested_features),
                    "n_features_actual": int(len(selected)),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_positive_rate": float(y[test_idx].mean()),
                    "ordered": ordered_score,
                    "controls": control_scores,
                    "fixed_probe_order_gap_pr_auc": {
                        name: float(
                            ordered_score["pr_auc"] - score["pr_auc"]
                        )
                        for name, score in control_scores.items()
                    },
                    "prediction_path": str(prediction_path),
                }
            )
    return rows


def summarize_probe(rows: list[dict]) -> list[dict]:
    by_features: dict[int, list[dict]] = {}
    for row in rows:
        by_features.setdefault(int(row["n_features"]), []).append(row)
    summaries = []
    for n_features, group in sorted(by_features.items()):
        group.sort(key=lambda row: int(row["fold"]))
        summaries.append(
            {
                "n_features": n_features,
                "ordered_pr_auc": _mean(
                    [float(row["ordered"]["pr_auc"]) for row in group]
                ),
                "control_pr_auc": {
                    name: _mean(
                        [
                            float(row["controls"][name]["pr_auc"])
                            for row in group
                        ]
                    )
                    for name in ORDER_CONTROLS
                },
                "order_gap_pr_auc": {
                    name: _mean(
                        [
                            float(row["fixed_probe_order_gap_pr_auc"][name])
                            for row in group
                        ]
                    )
                    for name in ORDER_CONTROLS
                },
                "folds": group,
            }
        )
    return summaries


def _load_prediction(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _selected_prediction_folds(
    probes: dict[str, list[dict]], name: str
) -> list[dict[str, np.ndarray]]:
    selected = max(
        probes[name], key=lambda summary: int(summary["n_features"])
    )
    return [
        _load_prediction(row["prediction_path"])
        for row in sorted(selected["folds"], key=lambda row: int(row["fold"]))
    ]


def _paired_fold_gaps(
    *,
    txc: dict[str, np.ndarray],
    sae_positional: dict[str, np.ndarray],
    sae_invariant: dict[str, np.ndarray],
    sae_last_token: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, float] | None:
    y = txc["y"][indices]
    if len(np.unique(y)) < 2:
        return None
    txc_ap = float(average_precision_score(y, txc["ordered"][indices]))
    order_control_aps = [
        float(average_precision_score(y, txc[f"control_{name}"][indices]))
        for name in ORDER_CONTROLS
    ]
    sae_positional_ap = float(
        average_precision_score(y, sae_positional["ordered"][indices])
    )
    sae_invariant_ap = float(
        average_precision_score(y, sae_invariant["ordered"][indices])
    )
    sae_last_ap = float(
        average_precision_score(y, sae_last_token["ordered"][indices])
    )
    residual_ap = float(
        average_precision_score(y, residual["ordered"][indices])
    )
    learned_controls = [
        *order_control_aps,
        sae_positional_ap,
        sae_invariant_ap,
        sae_last_ap,
    ]
    return {
        "txc_minus_sae_positional": txc_ap - sae_positional_ap,
        "txc_minus_strongest_learned_control": (
            txc_ap - max(learned_controls)
        ),
        "txc_minus_strongest_control_including_residual": (
            txc_ap - max(*learned_controls, residual_ap)
        ),
    }


def grouped_question_bootstrap(
    probes: dict[str, list[dict]],
    residual: dict,
    *,
    repeats: int,
    seed: int,
) -> dict:
    """Paired question-cluster bootstrap over the fixed outer-fold predictions."""

    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    payloads = {
        name: _selected_prediction_folds(probes, name)
        for name in ("txc", "sae_positional", "sae_invariant", "sae_last_token")
    }
    payloads["residual"] = [
        _load_prediction(row["prediction_path"])
        for row in sorted(residual["folds"], key=lambda row: int(row["fold"]))
    ]
    n_folds = len(payloads["txc"])
    if any(len(values) != n_folds for values in payloads.values()):
        raise ValueError("prediction fold count mismatch")

    original: dict[str, list[float]] = {}
    for fold in range(n_folds):
        reference = payloads["txc"][fold]
        for name, values in payloads.items():
            candidate = values[fold]
            for field in ("test_indices", "y", "groups"):
                if not np.array_equal(reference[field], candidate[field]):
                    raise ValueError(
                        f"prediction alignment mismatch: {name}/fold={fold}/{field}"
                    )
        gaps = _paired_fold_gaps(
            txc=reference,
            sae_positional=payloads["sae_positional"][fold],
            sae_invariant=payloads["sae_invariant"][fold],
            sae_last_token=payloads["sae_last_token"][fold],
            residual=payloads["residual"][fold],
            indices=np.arange(len(reference["y"]), dtype=np.int64),
        )
        if gaps is None:
            raise ValueError(f"original fold {fold} is single-class")
        for name, value in gaps.items():
            original.setdefault(name, []).append(value)

    rng = np.random.default_rng(seed)
    bootstrap: dict[str, list[float]] = {name: [] for name in original}
    for _ in range(repeats):
        replicate: dict[str, list[float]] = {name: [] for name in original}
        for fold in range(n_folds):
            reference = payloads["txc"][fold]
            groups = reference["groups"]
            unique_groups = np.unique(groups)
            sampled = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            indices = np.concatenate(
                [np.flatnonzero(groups == group) for group in sampled]
            )
            gaps = _paired_fold_gaps(
                txc=reference,
                sae_positional=payloads["sae_positional"][fold],
                sae_invariant=payloads["sae_invariant"][fold],
                sae_last_token=payloads["sae_last_token"][fold],
                residual=payloads["residual"][fold],
                indices=indices,
            )
            if gaps is None:
                continue
            for name, value in gaps.items():
                replicate[name].append(value)
        if all(replicate[name] for name in replicate):
            for name, values in replicate.items():
                bootstrap[name].append(float(np.mean(values)))
    if any(not values for values in bootstrap.values()):
        raise RuntimeError("all question bootstrap replicates were single-class")

    return {
        "unit": "question group, resampled within each fixed outer test fold",
        "aggregation": "mean paired PR-AUC gap across outer folds",
        "repeats_requested": repeats,
        "comparisons": {
            name: {
                "point_estimate": float(np.mean(original[name])),
                "fold_values": original[name],
                "repeats": len(bootstrap[name]),
                "lower_95": float(np.quantile(bootstrap[name], 0.025)),
                "median": float(np.median(bootstrap[name])),
                "upper_95": float(np.quantile(bootstrap[name], 0.975)),
            }
            for name in original
        },
    }


def _run_residual_control(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    window: int,
    seed: int,
    folds: int,
    pca_dim: int,
    prediction_dir: Path,
    artifact_offsets: tuple[int, ...] = ARTIFACT_OFFSETS,
) -> dict:
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups)):
        row, predictions = run_residual_fold(
            x,
            y,
            groups,
            train_idx,
            test_idx,
            fold=fold,
            window=window,
            artifact_offsets=artifact_offsets,
            normalization="raw",
            pca_dim=pca_dim,
            pca_sample_tokens=50_000,
            seed=seed,
        )
        _atomic_predictions(
            {
                "test_indices": test_idx.astype(np.int64),
                "y": predictions["y"].astype(np.int8),
                "groups": predictions["groups"].astype(str),
                "ordered": predictions["ordered"].astype(np.float32),
                "invariant_mean": predictions["invariant_mean"].astype(
                    np.float32
                ),
                "best_token": predictions["best_token"].astype(np.float32),
                **{
                    f"control_{name}": predictions[f"control_{name}"].astype(
                        np.float32
                    )
                    for name in ORDER_CONTROLS
                },
            },
            prediction_dir / f"fold{fold}.npz",
        )
        row["prediction_path"] = str(prediction_dir / f"fold{fold}.npz")
        rows.append(row)
    return {
        "interpretation": (
            "supervised raw-residual upper bound; all preprocessing is fit "
            "inside question-grouped outer folds"
        ),
        "ordered_pr_auc": _mean(
            [float(row["ordered"]["pr_auc"]) for row in rows]
        ),
        "invariant_mean_pr_auc": _mean(
            [float(row["invariant_mean"]["pr_auc"]) for row in rows]
        ),
        "best_token_pr_auc": _mean(
            [float(row["best_token"]["pr_auc"]) for row in rows]
        ),
        "g_order_pr_auc": _mean(
            [float(row["g_order_pr_auc"]) for row in rows]
        ),
        "order_gap_pr_auc": {
            name: _mean(
                [float(row["order_gap_pr_auc"][name]) for row in rows]
            )
            for name in ORDER_CONTROLS
        },
        "folds": rows,
    }


def evaluate_cell(
    *,
    artifact: Path,
    artifact_sha256: str,
    txc_checkpoint: Path,
    sae_checkpoint: Path,
    output_dir: Path,
    window: int,
    seed: int,
    folds: int,
    s_grid: tuple[int, ...],
    max_rows: int | None,
    batch_size: int,
    pca_dim: int,
    device: str,
    bootstrap_repeats: int,
    protocol_version: str = PROTOCOL_VERSION,
    artifact_offsets: tuple[int, ...] = ARTIFACT_OFFSETS,
    include_effective_l0: bool = False,
    cohort_sha256: str | None = None,
) -> dict:
    result_path = output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        checks = {
            "status": result.get("status") == "complete",
            "protocol": result.get("protocol_version") == protocol_version,
            "window": result.get("window") == window,
            "seed": result.get("seed") == seed,
        }
        if cohort_sha256 is not None:
            checks["cohort"] = result.get("cohort_sha256") == cohort_sha256
        if not all(checks.values()):
            raise ValueError(f"invalid completed-cell marker at {result_path}")
        return result

    with np.load(artifact, allow_pickle=True) as payload:
        x = payload["X"]
        y = payload["is_bt"].astype(np.int64, copy=False)
        groups = c7_groups(payload["keys"])
    x = trailing_window(x, window)
    keep = whole_group_subsample(groups, max_rows, seed)
    x, y, groups = x[keep], y[keep], groups[keep]

    txc_model_path = txc_checkpoint / "model.safetensors"
    sae_model_path = sae_checkpoint / "model.safetensors"
    fingerprint = {
        "protocol_version": protocol_version,
        "artifact_sha256": artifact_sha256,
        "window": window,
        "window_offsets": list(artifact_offsets[-window:]),
        "seed": seed,
        "row_indices_sha256": sha256_array(keep),
        "txc_checkpoint_sha256": sha256(txc_model_path),
        "sae_checkpoint_sha256": sha256(sae_model_path),
    }
    if cohort_sha256 is not None:
        fingerprint["cohort_sha256"] = cohort_sha256
    matrices = _codes(
        x=x,
        txc_checkpoint=txc_checkpoint,
        sae_checkpoint=sae_checkpoint,
        code_dir=output_dir / "codes",
        batch_size=batch_size,
        device=device,
        seed=seed,
        code_fingerprint=fingerprint,
    )
    code_effective_l0 = None
    if include_effective_l0:
        txc_config = json.loads(
            (txc_checkpoint / "config.json").read_text()
        )
        sae_config = json.loads(
            (sae_checkpoint / "config.json").read_text()
        )
        code_effective_l0 = effective_l0_diagnostics(
            matrices,
            window=window,
            txc_k_pos=int(txc_config["k_pos"]),
            sae_k_pos=int(sae_config["k_pos"]),
            txc_d_sae=int(txc_config["d_sae"]),
            sae_d_sae=int(sae_config["d_sae"]),
        )

    probes = {}
    for name, variants in matrices.items():
        ordered = variants["ordered"]
        controls = {control: variants[control] for control in ORDER_CONTROLS}
        rows = grouped_fixed_probe_with_predictions(
            ordered,
            controls,
            y,
            groups,
            folds=folds,
            s_grid=s_grid,
            seed=seed,
            prediction_dir=output_dir / "predictions" / name,
        )
        probes[name] = summarize_probe(rows)
    del matrices
    gc.collect()

    residual = _run_residual_control(
        x,
        y,
        groups,
        window=window,
        seed=seed,
        folds=folds,
        pca_dim=pca_dim,
        prediction_dir=output_dir / "predictions" / "residual",
        artifact_offsets=artifact_offsets,
    )
    grouped_bootstrap = grouped_question_bootstrap(
        probes,
        residual,
        repeats=bootstrap_repeats,
        seed=seed + 700_000,
    )
    result = {
        "status": "complete",
        "protocol_version": protocol_version,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": artifact_sha256,
        "window": window,
        "window_offsets": list(artifact_offsets[-window:]),
        "seed": seed,
        "n_rows": int(len(x)),
        "n_groups": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "folds": folds,
        "s_grid": list(s_grid),
        "estimand": (
            "question-grouped sentence-level backtracking detection PR-AUC; "
            "the sparse probe is fit on ordered codes then held fixed under "
            "shuffle, reversal, and non-zero circular shifts"
        ),
        "controls": {
            "sae_positional": (
                "one shared per-token SAE with relative-position-specific "
                "feature blocks; a conservative multi-token ordered baseline"
            ),
            "sae_invariant": "max pool of the same per-token SAE codes",
            "sae_last_token": "same SAE restricted to the endpoint at offset -8",
            "residual": "train-fold-only covariance-whitened raw-activation upper bound",
        },
        "probes": probes,
        "residual": residual,
        "grouped_question_bootstrap": grouped_bootstrap,
        "code_fingerprint": fingerprint,
    }
    if cohort_sha256 is not None:
        result["cohort_sha256"] = cohort_sha256
    if code_effective_l0 is not None:
        result["effective_l0"] = {
            "definition": (
                "actual nonzero count after TopK selection and ReLU; nominal "
                "TopK is an upper bound because negative selected values become zero"
            ),
            "codes": code_effective_l0,
        }
    atomic_json(result, result_path)
    return result


def sha256_array(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()
