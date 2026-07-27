"""Evaluate frozen submitted T=5 dictionaries on deletion destination.

The ordered TXC probe is held fixed for the shuffled and reversed controls.
Matched TopK-SAE baselines receive the same examples, writer folds, and sparse
feature budgets. The primary metric is equal-writer multiclass log loss.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from experiments.backtracking_window_sweep.evaluate import (
    encode_sae_invariant,
    encode_sae_positional,
    encode_sae_token,
    encode_txc,
    sparse_effective_l0,
)
from experiments.swr_audit.dictionary import _load_encoder

from .evaluate_activations import (
    ActivationDataset,
    load_activation_dataset,
    shuffled_windows,
)
from .klicke import sha256_file


PROTOCOL_VERSION = "klicke-deletion-frozen-dictionary-t5-v1"
MODEL_REPO = "han1823123123/temp-bench-models"
MODEL_REVISION = "1e8d750f61b82cabf1981b098070cb334e5d7fb9"
WINDOW_TOKENS = 5
K_POS = 20
PRIMARY_BUDGET = 32
DEFAULT_S_GRID = (8, 16, 32, 64, 128)
DEFAULT_BOOTSTRAP_SEED = 20_260_726
DEFAULT_GATE_MARGIN = 0.02

EXPECTED_DATASET = {
    "rows": 6_224,
    "writers": 2_510,
    "cache_window_tokens": 10,
    "hidden_size": 4_096,
    "target": "capped_token_label",
    "label_counts": {2: 1_218, 3: 1_634, 4: 1_058, 5: 692, 6: 1_622},
    "cohort_sha256": "051deb240c285496213f5ac14f153f8a141f27068e05ec54416b183c20f8b6c2",
    "cohort_manifest_sha256": (
        "8cd98199a06f7e3f03318724bd2ae81d46042baaf22e83f7f3b4ada840af526f"
    ),
    "request_sha256": "9401b237a85273a30194fe480e50ce4ff56dbe7d1dc40a22ed6098e53d20d20f",
    "runtime_sha256": "0200183aedadea1784d3f8d555a19efbcf7fc906591ca5364d3bf9c473935210",
    "complete_sha256": "f55bce8884b95496375d2266561a186265219f29d7105d90ff7c9851138aad89",
    "model": "NousResearch/Meta-Llama-3.1-8B",
    "model_revision_observed": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
    "layer": 10,
    "hook_semantics": "resid_post",
}


@dataclass(frozen=True)
class CheckpointSpec:
    train_key: str
    arch: str
    sha256: str
    size_bytes: int


CHECKPOINTS = {
    "txc": CheckpointSpec(
        train_key="08fe3af07682fab4",
        arch="txc_base",
        sha256="ed2ecf4670f889fd97e82c53a949f963c36a292f05944cd678f943a87f1f9cb1",
        size_bytes=2_684_723_624,
    ),
    "sae": CheckpointSpec(
        train_key="f437e623fabc37ec",
        arch="topk_sae",
        sha256="2efc15ad39603bcb554730c900e0c4b69035ac5be07d925aa489adbd15533d40",
        size_bytes=1_073_889_608,
    ),
}

METHODS = (
    "txc_ordered",
    "txc_fixed_shuffle",
    "txc_fixed_reverse",
    "sae_positional",
    "sae_invariant",
    "sae_last_token",
)
SAE_METHODS = ("sae_positional", "sae_invariant", "sae_last_token")
EXPECTED_CODE_COLUMNS = {
    "txc_ordered": 32_768,
    "txc_fixed_shuffle": 32_768,
    "txc_fixed_reverse": 32_768,
    "sae_positional": WINDOW_TOKENS * 32_768,
    "sae_invariant": 32_768,
    "sae_last_token": 32_768,
}


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


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


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def download_checkpoints(checkpoint_root: Path) -> dict[str, str]:
    """Download only the two frozen encoder configs and model tensors."""

    from huggingface_hub import hf_hub_download

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for name, spec in CHECKPOINTS.items():
        for filename in ("config.json", "model.safetensors"):
            path = Path(
                hf_hub_download(
                    MODEL_REPO,
                    filename=f"{spec.train_key}/{filename}",
                    revision=MODEL_REVISION,
                    local_dir=checkpoint_root,
                )
            )
            resolved[f"{name}_{filename}"] = str(path)
    validate_checkpoints(checkpoint_root)
    return resolved


def validate_checkpoints(checkpoint_root: Path) -> dict[str, dict[str, object]]:
    """Fail closed on checkpoint revision, config, byte size, and checksum."""

    records: dict[str, dict[str, object]] = {}
    for name, spec in CHECKPOINTS.items():
        directory = checkpoint_root / spec.train_key
        config_path = directory / "config.json"
        model_path = directory / "model.safetensors"
        if not config_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"missing frozen {name} checkpoint under {directory}"
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checks = {
            "train_key": config.get("train_key") == spec.train_key,
            "arch": config.get("arch") == spec.arch,
            "arch_version": config.get("arch_version") == "1.0.0",
            "seed": int(config.get("seed", -1)) == 42,
            "act_cache_key": config.get("act_cache_key") == "fb2a74be884e512a",
            "datasource": (
                config.get("datasource")
                == "llama_3_1_8b_base_l10_ward_nousmirror"
            ),
            "size_bytes": model_path.stat().st_size == spec.size_bytes,
            "sha256": sha256_file(model_path) == spec.sha256,
        }
        if not all(checks.values()):
            raise ValueError(
                f"frozen checkpoint provenance mismatch for {spec.train_key}: "
                f"{checks}"
            )
        records[name] = {
            "train_key": spec.train_key,
            "arch": spec.arch,
            "hub_repo": MODEL_REPO,
            "hub_revision": MODEL_REVISION,
            "model_sha256": spec.sha256,
            "model_size_bytes": spec.size_bytes,
            "config_sha256": sha256_file(config_path),
            "training_act_cache_key": config["act_cache_key"],
            "training_datasource": config["datasource"],
            "k_pos": K_POS,
        }
    return records


def validate_deletion_dataset(dataset: ActivationDataset) -> None:
    """Require the exact audited deletion cohort and extraction artifact."""

    label_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(dataset.target, return_counts=True))
    }
    observed = {
        "rows": len(dataset.target),
        "writers": len(np.unique(dataset.groups)),
        "cache_window_tokens": int(dataset.activations.shape[1]),
        "hidden_size": int(dataset.activations.shape[2]),
        "target": dataset.target_name,
        "label_counts": label_counts,
        **{
            name: dataset.provenance.get(name)
            for name in (
                "cohort_sha256",
                "cohort_manifest_sha256",
                "request_sha256",
                "runtime_sha256",
                "complete_sha256",
                "model",
                "model_revision_observed",
                "layer",
                "hook_semantics",
            )
        },
    }
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in EXPECTED_DATASET.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"deletion activation cohort drifted: {mismatches}")
    if len(np.unique(dataset.event_hashes)) != len(dataset.event_hashes):
        raise ValueError("deletion event hashes are not globally unique")


def _load_cached_code(path: Path, rows: int, columns: int) -> sparse.csr_matrix:
    matrix = sparse.load_npz(path).tocsr()
    if matrix.shape != (rows, columns):
        raise ValueError(
            f"cached code shape mismatch at {path}: "
            f"{matrix.shape} != {(rows, columns)}"
        )
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"cached code contains nonfinite values: {path}")
    return matrix


def _code_fingerprint(
    dataset: ActivationDataset,
    checkpoint_records: dict[str, dict[str, object]],
    *,
    seed: int,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "rows": len(dataset.target),
        "window_tokens": WINDOW_TOKENS,
        "hidden_size": int(dataset.activations.shape[-1]),
        "event_hashes_sha256": _hash_strings(dataset.event_hashes),
        "activation_provenance": dataset.provenance,
        "checkpoint_provenance": checkpoint_records,
        "k_pos": K_POS,
        "conditions": {
            "ordered": "trailing five states, oldest to newest",
            "fixed_shuffle": (
                "stable per-event nonidentity permutation"
            ),
            "fixed_reverse": "all five positions reversed at test",
        },
        "condition_seed": seed,
    }


def encode_code_matrices(
    dataset: ActivationDataset,
    *,
    checkpoint_root: Path,
    code_dir: Path,
    checkpoint_records: dict[str, dict[str, object]],
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, object]]:
    """Encode and atomically cache every representation needed by the gate."""

    code_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _code_fingerprint(dataset, checkpoint_records, seed=seed)
    metadata_path = code_dir / "metadata.json"
    if metadata_path.exists():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed != fingerprint:
            raise ValueError(
                f"code cache provenance mismatch at {metadata_path}; "
                "use a new code directory"
            )
    else:
        _atomic_json(fingerprint, metadata_path)

    rows = len(dataset.target)
    paths = {name: code_dir / f"{name}.npz" for name in METHODS}
    matrices: dict[str, sparse.csr_matrix] = {}
    for name, path in paths.items():
        if path.exists():
            matrices[name] = _load_cached_code(
                path, rows, EXPECTED_CODE_COLUMNS[name]
            )

    windows = np.asarray(
        dataset.activations[:, -WINDOW_TOKENS:, :],
        dtype=np.float16,
    )
    txc_missing = [
        name
        for name in ("txc_ordered", "txc_fixed_shuffle", "txc_fixed_reverse")
        if name not in matrices
    ]
    if txc_missing:
        txc_loaded = _load_encoder(
            checkpoint_root / CHECKPOINTS["txc"].train_key,
            CHECKPOINTS["txc"].train_key,
            device,
        )
        txc_state = txc_loaded["state"]
        if tuple(txc_state["W_enc"].shape) != (WINDOW_TOKENS, 4_096, 32_768):
            raise ValueError("TXC encoder shape does not match submitted T=5 model")
        conditions = {
            "txc_ordered": windows,
            "txc_fixed_reverse": windows[:, ::-1, :],
        }
        if "txc_fixed_shuffle" in txc_missing:
            conditions["txc_fixed_shuffle"] = shuffled_windows(
                windows,
                dataset.event_hashes,
                seed=seed,
            )
        for name in txc_missing:
            matrix = encode_txc(
                conditions[name],
                state=txc_state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            )
            _atomic_sparse(matrix, paths[name])
            matrices[name] = matrix
        del txc_loaded, txc_state, conditions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sae_missing = [name for name in SAE_METHODS if name not in matrices]
    if sae_missing:
        sae_loaded = _load_encoder(
            checkpoint_root / CHECKPOINTS["sae"].train_key,
            CHECKPOINTS["sae"].train_key,
            device,
        )
        sae_state = sae_loaded["state"]
        expected_sae = {
            "W_enc": (32_768, 4_096),
            "b_enc": (32_768,),
            "b_dec": (4_096,),
        }
        if {
            name: tuple(value.shape) for name, value in sae_state.items()
        } != expected_sae:
            raise ValueError("SAE encoder shape does not match submitted checkpoint")
        encoders = {
            "sae_positional": lambda: encode_sae_positional(
                windows,
                state=sae_state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            ),
            "sae_invariant": lambda: encode_sae_invariant(
                windows,
                state=sae_state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            ),
            "sae_last_token": lambda: encode_sae_token(
                windows[:, -1, :],
                state=sae_state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            ),
        }
        for name in sae_missing:
            matrix = encoders[name]()
            _atomic_sparse(matrix, paths[name])
            matrices[name] = matrix
        del sae_loaded, sae_state, encoders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matrices = {
        name: _load_cached_code(
            paths[name], rows, EXPECTED_CODE_COLUMNS[name]
        )
        for name in METHODS
    }
    nominal_l0 = {
        "txc_ordered": WINDOW_TOKENS * K_POS,
        "txc_fixed_shuffle": WINDOW_TOKENS * K_POS,
        "txc_fixed_reverse": WINDOW_TOKENS * K_POS,
        "sae_positional": WINDOW_TOKENS * K_POS,
        "sae_invariant": WINDOW_TOKENS * K_POS,
        "sae_last_token": K_POS,
    }
    diagnostics = {
        name: sparse_effective_l0(
            matrix,
            nominal_l0=nominal_l0[name],
        )
        for name, matrix in matrices.items()
    }
    return matrices, {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "files": {
            name: {
                "path": str(paths[name]),
                "sha256": sha256_file(paths[name]),
                "shape": list(matrices[name].shape),
                "nnz": int(matrices[name].nnz),
            }
            for name in METHODS
        },
        "effective_l0": diagnostics,
    }


def writer_grouped_splits(
    target: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = set(int(value) for value in np.unique(target))
    groups_per_label = [
        len(np.unique(groups[target == label])) for label in labels
    ]
    effective = min(folds, len(np.unique(groups)), min(groups_per_label))
    if effective < 2:
        raise ValueError("each target class needs at least two writer groups")
    splitter = StratifiedGroupKFold(
        n_splits=effective,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(np.zeros(len(target)), target, groups))
    for train, test in splits:
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError("writer leakage across outer folds")
        if set(int(value) for value in np.unique(target[train])) != labels:
            raise ValueError("an outer training fold omits a target class")
    return splits


def multiclass_sparse_anova_ranking(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
) -> np.ndarray:
    """Rank sparse coordinates by multiclass ANOVA using training rows only."""

    matrix = matrix.tocsr()
    target = np.asarray(target)
    if matrix.shape[0] != len(target):
        raise ValueError("feature rows and targets disagree")
    labels = np.unique(target)
    if len(labels) < 2:
        raise ValueError("ANOVA ranking requires multiple classes")
    sums = []
    sums_squared = []
    counts = []
    for label in labels:
        subset = matrix[target == label]
        sums.append(np.asarray(subset.sum(axis=0)).ravel())
        sums_squared.append(
            np.asarray(subset.multiply(subset).sum(axis=0)).ravel()
        )
        counts.append(subset.shape[0])
    sums_array = np.asarray(sums, dtype=np.float64)
    squared_array = np.asarray(sums_squared, dtype=np.float64)
    counts_array = np.asarray(counts, dtype=np.float64)
    means = sums_array / counts_array[:, None]
    grand_mean = sums_array.sum(axis=0) / counts_array.sum()
    between = (
        counts_array[:, None] * np.square(means - grand_mean)
    ).sum(axis=0)
    within = (
        squared_array - np.square(sums_array) / counts_array[:, None]
    ).sum(axis=0)
    numerator = between / (len(labels) - 1)
    denominator = within / max(matrix.shape[0] - len(labels), 1)
    scores = numerator / np.maximum(denominator, 1e-12)
    scores[~np.isfinite(scores)] = -np.inf
    indices = np.arange(matrix.shape[1], dtype=np.int64)
    return np.lexsort((indices, -scores))


def _fit_probe(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    train: np.ndarray,
    selected: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    train_features = scaler.fit_transform(
        matrix[train][:, selected].toarray()
    )
    classifier = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    classifier.fit(train_features, target[train])
    return scaler, classifier


def _predict_probe(
    matrix: sparse.csr_matrix,
    indices: np.ndarray,
    selected: np.ndarray,
    scaler: StandardScaler,
    classifier: LogisticRegression,
    labels: tuple[int, ...],
) -> np.ndarray:
    raw = classifier.predict_proba(
        scaler.transform(matrix[indices][:, selected].toarray())
    )
    aligned = np.zeros((len(indices), len(labels)), dtype=np.float32)
    for source, label in enumerate(classifier.classes_):
        aligned[:, labels.index(int(label))] = raw[:, source]
    return aligned


def _true_row_losses(
    probabilities: np.ndarray,
    target: np.ndarray,
    labels: tuple[int, ...],
) -> np.ndarray:
    columns = np.asarray([labels.index(int(value)) for value in target])
    values = probabilities[np.arange(len(target)), columns]
    return -np.log(np.clip(values, 1e-12, 1.0))


def _equal_writer_mean_loss(
    probabilities: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    labels: tuple[int, ...],
) -> float:
    row_losses = _true_row_losses(probabilities, target, labels)
    return float(
        np.mean(
            [
                row_losses[groups == writer].mean()
                for writer in np.unique(groups)
            ]
        )
    )


def summarize_equal_writer(
    probabilities: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    labels: tuple[int, ...],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Paired bootstrap after reducing every writer to one mean loss."""

    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    writers = np.unique(groups)
    writer_indices = [np.flatnonzero(groups == writer) for writer in writers]
    row_losses = {
        name: _true_row_losses(values, target, labels)
        for name, values in probabilities.items()
    }
    writer_losses = {
        name: np.asarray(
            [float(losses[indices].mean()) for indices in writer_indices],
            dtype=np.float64,
        )
        for name, losses in row_losses.items()
    }
    point_losses = {
        name: float(values.mean()) for name, values in writer_losses.items()
    }
    competitors = tuple(name for name in METHODS if name != "txc_ordered")
    deltas = {
        name: np.empty(draws, dtype=np.float64) for name in competitors
    }
    strongest_deltas = np.empty(draws, dtype=np.float64)
    strongest_choices = np.empty(draws, dtype=np.int8)
    rng = np.random.default_rng(seed)
    batch_size = min(128, draws)
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        sampled = rng.integers(
            0,
            len(writers),
            size=(stop - start, len(writers)),
        )
        sampled_means = {
            name: values[sampled].mean(axis=1)
            for name, values in writer_losses.items()
        }
        for name in competitors:
            deltas[name][start:stop] = (
                sampled_means[name] - sampled_means["txc_ordered"]
            )
        sae_samples = np.column_stack(
            [sampled_means[name] for name in SAE_METHODS]
        )
        strongest_choices[start:stop] = np.argmin(sae_samples, axis=1)
        strongest_deltas[start:stop] = (
            sae_samples.min(axis=1) - sampled_means["txc_ordered"]
        )

    contrasts: dict[str, dict[str, object]] = {}
    for name in competitors:
        writer_difference = writer_losses[name] - writer_losses["txc_ordered"]
        contrasts[f"{name}_minus_txc_ordered"] = {
            "equal_writer_mean_log_loss_difference": float(
                writer_difference.mean()
            ),
            "ci95_lower": float(np.quantile(deltas[name], 0.025)),
            "ci95_median": float(np.median(deltas[name])),
            "ci95_upper": float(np.quantile(deltas[name], 0.975)),
            "writers_positive": int((writer_difference > 0).sum()),
            "writers_total": len(writers),
        }

    strongest_name = min(SAE_METHODS, key=point_losses.__getitem__)
    strongest_point = (
        point_losses[strongest_name] - point_losses["txc_ordered"]
    )
    strongest = {
        "selection_rule": (
            "minimum equal-writer SAE loss; each bootstrap replicate "
            "reselects the minimum among positional, invariant, and last-token"
        ),
        "point_selected_method": strongest_name,
        "equal_writer_mean_log_loss_difference": float(strongest_point),
        "ci95_lower": float(np.quantile(strongest_deltas, 0.025)),
        "ci95_median": float(np.median(strongest_deltas)),
        "ci95_upper": float(np.quantile(strongest_deltas, 0.975)),
        "bootstrap_selection_counts": {
            name: int((strongest_choices == index).sum())
            for index, name in enumerate(SAE_METHODS)
        },
    }
    return {
        "unit": "writer",
        "writers": len(writers),
        "draws": draws,
        "seed": seed,
        "writer_hashes_sha256": _hash_strings(writers),
        "method_equal_writer_log_loss": point_losses,
        "contrasts": contrasts,
        "strongest_sae_minus_txc_ordered": strongest,
    }


def primary_gate(
    equal_writer: dict[str, object],
    *,
    margin: float,
) -> dict[str, object]:
    contrasts = equal_writer["contrasts"]
    records = {
        "fixed_shuffle": contrasts[
            "txc_fixed_shuffle_minus_txc_ordered"
        ],
        "fixed_reverse": contrasts[
            "txc_fixed_reverse_minus_txc_ordered"
        ],
        "strongest_sae": equal_writer["strongest_sae_minus_txc_ordered"],
    }
    components = {}
    for name, record in records.items():
        point = float(record["equal_writer_mean_log_loss_difference"])
        lower = float(record["ci95_lower"])
        components[name] = {
            "point_estimate": point,
            "ci95_lower": lower,
            "margin_pass": point >= margin,
            "confidence_pass": lower > 0.0,
            "passed": point >= margin and lower > 0.0,
        }
    return {
        "passed": all(component["passed"] for component in components.values()),
        "required_margin": margin,
        "components": components,
    }


def evaluate_code_matrices(
    matrices: dict[str, sparse.csr_matrix],
    target: np.ndarray,
    groups: np.ndarray,
    event_hashes: np.ndarray,
    *,
    budgets: Sequence[int],
    primary_budget: int,
    folds: int,
    c_value: float,
    max_iter: int,
    bootstrap_draws: int,
    seed: int,
    gate_margin: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Cross-fit sparse probes and return metrics plus held-out predictions."""

    if set(matrices) != set(METHODS):
        raise ValueError(
            f"code methods drifted: expected={set(METHODS)}, "
            f"observed={set(matrices)}"
        )
    n_rows = len(target)
    if (
        len(groups) != n_rows
        or len(event_hashes) != n_rows
        or any(matrix.shape[0] != n_rows for matrix in matrices.values())
    ):
        raise ValueError("code matrices, targets, groups, and event hashes disagree")
    requested = tuple(sorted(set(int(value) for value in budgets)))
    if not requested or requested[0] < 1 or primary_budget not in requested:
        raise ValueError("budgets must be positive and include the primary budget")
    labels = tuple(int(value) for value in sorted(np.unique(target)))
    splits = writer_grouped_splits(target, groups, folds=folds, seed=seed)
    predictions = {
        budget: {
            name: np.full((n_rows, len(labels)), np.nan, dtype=np.float32)
            for name in METHODS
        }
        for budget in requested
    }
    fold_audit = []

    for fold, (train, test) in enumerate(splits):
        rankings = {
            "txc": multiclass_sparse_anova_ranking(
                matrices["txc_ordered"][train], target[train]
            ),
            **{
                name: multiclass_sparse_anova_ranking(
                    matrices[name][train], target[train]
                )
                for name in SAE_METHODS
            },
        }
        audit = {
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_writers": len(np.unique(groups[train])),
            "test_writers": len(np.unique(groups[test])),
            "selected": {},
        }
        for budget in requested:
            txc_selected = rankings["txc"][
                : min(budget, matrices["txc_ordered"].shape[1])
            ]
            txc_scaler, txc_classifier = _fit_probe(
                matrices["txc_ordered"],
                target,
                train,
                txc_selected,
                c_value=c_value,
                max_iter=max_iter,
                seed=seed + 100 * fold + budget,
            )
            for name in (
                "txc_ordered",
                "txc_fixed_shuffle",
                "txc_fixed_reverse",
            ):
                predictions[budget][name][test] = _predict_probe(
                    matrices[name],
                    test,
                    txc_selected,
                    txc_scaler,
                    txc_classifier,
                    labels,
                )
            selected_audit = {
                "txc": {
                    "count": len(txc_selected),
                    "sha256": hashlib.sha256(
                        txc_selected.astype(np.int64).tobytes()
                    ).hexdigest(),
                    "classifier_n_iter_max": int(
                        np.max(txc_classifier.n_iter_)
                    ),
                }
            }
            for representation_index, name in enumerate(SAE_METHODS, start=1):
                selected = rankings[name][
                    : min(budget, matrices[name].shape[1])
                ]
                scaler, classifier = _fit_probe(
                    matrices[name],
                    target,
                    train,
                    selected,
                    c_value=c_value,
                    max_iter=max_iter,
                    seed=seed
                    + 100 * fold
                    + budget
                    + 10_000 * representation_index,
                )
                predictions[budget][name][test] = _predict_probe(
                    matrices[name],
                    test,
                    selected,
                    scaler,
                    classifier,
                    labels,
                )
                selected_audit[name] = {
                    "count": len(selected),
                    "sha256": hashlib.sha256(
                        selected.astype(np.int64).tobytes()
                    ).hexdigest(),
                    "classifier_n_iter_max": int(np.max(classifier.n_iter_)),
                }
            audit["selected"][str(budget)] = selected_audit
        fold_audit.append(audit)

    results = {}
    for budget in requested:
        probabilities = predictions[budget]
        if any(
            not np.isfinite(values).all()
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5)
            for values in probabilities.values()
        ):
            raise RuntimeError("held-out probability matrix is incomplete")
        metrics = {}
        for name, values in probabilities.items():
            predicted = np.asarray(labels)[np.argmax(values, axis=1)]
            metrics[name] = {
                "row_weighted_log_loss": float(
                    log_loss(target, values, labels=labels)
                ),
                "equal_writer_log_loss": _equal_writer_mean_loss(
                    values,
                    target,
                    groups,
                    labels,
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(target, predicted)
                ),
            }
        equal_writer = summarize_equal_writer(
            probabilities,
            target,
            groups,
            labels,
            draws=bootstrap_draws,
            seed=seed + budget,
        )
        results[str(budget)] = {
            "feature_budget": budget,
            "metrics": metrics,
            "equal_writer_bootstrap": equal_writer,
            "primary_gate": (
                primary_gate(equal_writer, margin=gate_margin)
                if budget == primary_budget
                else None
            ),
        }

    prediction_payload = {
        "target": np.asarray(target, dtype=np.int16),
        "groups": np.asarray(groups, dtype=str),
        "event_hashes": np.asarray(event_hashes, dtype=str),
        "labels": np.asarray(labels, dtype=np.int16),
    }
    for budget in requested:
        for name, values in predictions[budget].items():
            prediction_payload[f"S{budget}__{name}"] = values
    return {
        "labels": list(labels),
        "rows": n_rows,
        "writers": len(np.unique(groups)),
        "outer_folds_effective": len(splits),
        "results": results,
        "fold_audit": fold_audit,
    }, prediction_payload


def _render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Frozen T=5 deletion-destination dictionary gate",
        "",
        (
            f"Protocol `{result['protocol_version']}` on "
            f"{result['evaluation']['rows']:,} events from "
            f"{result['evaluation']['writers']:,} writers. Lower equal-writer "
            "log loss is better; positive control-minus-TXC gaps favor TXC."
        ),
        "",
        (
            "| S | TXC ordered | TXC shuffled | TXC reversed | "
            "Positional SAE | Invariant SAE | Last-token SAE | "
            "Strongest SAE minus TXC [95% CI] |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, record in sorted(
        result["evaluation"]["results"].items(),
        key=lambda item: int(item[0]),
    ):
        metrics = record["metrics"]
        strongest = record["equal_writer_bootstrap"][
            "strongest_sae_minus_txc_ordered"
        ]
        lines.append(
            "| {budget} | {ordered:.4f} | {shuffle:.4f} | {reverse:.4f} | "
            "{positional:.4f} | {invariant:.4f} | {last:.4f} | "
            "{gap:+.4f} [{low:+.4f}, {high:+.4f}] |".format(
                budget=budget,
                ordered=metrics["txc_ordered"]["equal_writer_log_loss"],
                shuffle=metrics["txc_fixed_shuffle"]["equal_writer_log_loss"],
                reverse=metrics["txc_fixed_reverse"]["equal_writer_log_loss"],
                positional=metrics["sae_positional"]["equal_writer_log_loss"],
                invariant=metrics["sae_invariant"]["equal_writer_log_loss"],
                last=metrics["sae_last_token"]["equal_writer_log_loss"],
                gap=strongest["equal_writer_mean_log_loss_difference"],
                low=strongest["ci95_lower"],
                high=strongest["ci95_upper"],
            )
        )
    gate = result["evaluation"]["results"][str(result["primary_budget"])][
        "primary_gate"
    ]
    lines.extend(
        [
            "",
            (
                f"Primary S={result['primary_budget']} gate: "
                f"**{'PASS' if gate['passed'] else 'FAIL'}**. It requires at "
                f"least {gate['required_margin']:.3f} log-loss improvement over "
                "fixed shuffle, fixed reverse, and the strongest matched SAE, "
                "with every paired writer-bootstrap lower bound above zero."
            ),
            "",
            "![Frozen dictionary sensitivity](frozen_dictionary.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _render_plot(result: dict[str, object], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = result["evaluation"]["results"]
    budgets = sorted(int(value) for value in records)
    styles = {
        "txc_ordered": ("Ordered TXC", "#2f6df6", "o", "-"),
        "txc_fixed_shuffle": ("Fixed shuffled TXC", "#d97706", "^", "--"),
        "txc_fixed_reverse": ("Fixed reversed TXC", "#8b5cf6", "D", ":"),
        "sae_positional": ("Positional SAE", "#1f2937", "s", "-."),
        "sae_invariant": ("Invariant SAE", "#0f8a70", "v", "-."),
        "sae_last_token": ("Last-token SAE", "#6b7280", "s", "--"),
    }
    fig, axis = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for name, (label, color, marker, linestyle) in styles.items():
        values = [
            records[str(budget)]["metrics"][name]["equal_writer_log_loss"]
            for budget in budgets
        ]
        axis.plot(
            budgets,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=6,
        )
    axis.set_title("Frozen T=5 dictionaries on human deletion destination")
    axis.set_xlabel("Train-only sparse feature budget S")
    axis.set_ylabel("Equal-writer multiclass log loss (lower is better)")
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, labels=[str(value) for value in budgets])
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    for suffix, dpi in (("png", 300), ("pdf", None)):
        fig.savefig(
            output_dir / f"frozen_dictionary.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("feature budgets must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--s-grid", type=_csv_ints, default=DEFAULT_S_GRID)
    parser.add_argument("--primary-budget", type=int, default=PRIMARY_BUDGET)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--gate-margin", type=float, default=DEFAULT_GATE_MARGIN)
    args = parser.parse_args()

    if args.batch_size < 1 or args.folds < 2:
        parser.error("batch size must be positive and folds must be at least two")
    if args.primary_budget not in args.s_grid:
        parser.error("the primary budget must appear in --s-grid")
    if args.download_checkpoints:
        download_checkpoints(args.checkpoint_root)
    checkpoint_records = validate_checkpoints(args.checkpoint_root)
    dataset = load_activation_dataset(
        cohort_path=args.cohort,
        cohort_manifest_path=args.cohort_manifest,
        cache_dir=args.activation_cache,
        target="capped_token_label",
    )
    validate_deletion_dataset(dataset)
    matrices, code_cache = encode_code_matrices(
        dataset,
        checkpoint_root=args.checkpoint_root,
        code_dir=args.code_dir,
        checkpoint_records=checkpoint_records,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    evaluation, predictions = evaluate_code_matrices(
        matrices,
        dataset.target,
        dataset.groups,
        dataset.event_hashes,
        budgets=args.s_grid,
        primary_budget=args.primary_budget,
        folds=args.folds,
        c_value=args.c_value,
        max_iter=args.max_iter,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        gate_margin=args.gate_margin,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "heldout_predictions.npz"
    _atomic_predictions(predictions, prediction_path)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "interpretation": (
            "frozen submitted unsupervised dictionaries with cross-fitted "
            "multiclass sparse probes"
        ),
        "primary_budget": args.primary_budget,
        "configuration": {
            "window_tokens": WINDOW_TOKENS,
            "k_pos": K_POS,
            "feature_budgets": list(args.s_grid),
            "folds_requested": args.folds,
            "c_value": args.c_value,
            "max_iter": args.max_iter,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
            "gate_margin": args.gate_margin,
            "fixed_control_contract": (
                "feature ranking, scaling, and classifier are fitted on "
                "ordered TXC codes and held fixed for shuffle and reverse"
            ),
            "baseline_contract": (
                "each SAE is fit independently on the identical outer "
                "training writers at the identical sparse feature budget"
            ),
        },
        "activation_provenance": dataset.provenance,
        "checkpoint_provenance": checkpoint_records,
        "code_cache": code_cache,
        "evaluation": evaluation,
        "heldout_predictions": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    _render_plot(result, args.output_dir)
    _atomic_text(
        _render_markdown(result),
        args.output_dir / "summary.md",
    )
    _atomic_json(result, args.output_dir / "result.json")
    primary = evaluation["results"][str(args.primary_budget)]
    print(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "primary_budget": args.primary_budget,
                "primary_gate": primary["primary_gate"],
                "metrics": primary["metrics"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
