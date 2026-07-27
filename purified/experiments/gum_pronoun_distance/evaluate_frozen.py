"""Evaluate frozen submitted T=5 dictionaries on GUM pronoun distance."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score, log_loss
import torch

from experiments.backtracking_window_sweep.evaluate import (
    encode_sae_invariant,
    encode_sae_positional,
    encode_sae_token,
    encode_txc,
    sparse_effective_l0,
)
from experiments.swr_audit.dictionary import _load_encoder
from experiments.writing_revision_destination.frozen_dictionary import (
    CHECKPOINTS,
    K_POS,
    MODEL_REPO,
    MODEL_REVISION,
    _atomic_json,
    _atomic_predictions,
    _atomic_sparse,
    _fit_probe,
    _hash_strings,
    _predict_probe,
    _true_row_losses,
    download_checkpoints,
    multiclass_sparse_anova_ranking,
    validate_checkpoints,
    writer_grouped_splits,
)

from .cohort import (
    EXPECTED_BALANCED_ROWS,
    EXPECTED_ROWS,
    EXPECTED_SEMANTIC_SHA256,
    WINDOW_TOKENS,
    sha256_file,
)
from .extract_activations import (
    EXTRACTION_PROTOCOL_VERSION,
    HIDDEN_SIZE,
    _validate_shard,
    load_cohort,
)

PROTOCOL_VERSION = "gum-pronoun-distance-frozen-dictionaries-t5-v1"
DEFAULT_S_GRID = (8, 16, 32, 64, 128)
PRIMARY_BUDGET = 32
DEFAULT_SEED = 20_260_726
DEFAULT_GATE_MARGIN = 0.02
EXPECTED_CONFIG_SHA256 = {
    "txc": "618cd7bffcf94d142e39e92440271e31bac938e1ee5c1712f90869fc89352542",
    "sae": "d992ae9f8370fd5c831ad58b0a648cb29bd9268a3ca8f617ab737aaef0ed00af",
}
METHODS = (
    "txc_ordered",
    "txc_fixed_shuffle_history",
    "txc_fixed_reverse_history",
    "sae_positional",
    "sae_invariant_history_endpoint",
    "sae_endpoint",
)
SAE_METHODS = (
    "sae_positional",
    "sae_invariant_history_endpoint",
    "sae_endpoint",
)
EXPECTED_CODE_COLUMNS = {
    "txc_ordered": 32_768,
    "txc_fixed_shuffle_history": 32_768,
    "txc_fixed_reverse_history": 32_768,
    "sae_positional": WINDOW_TOKENS * 32_768,
    "sae_invariant_history_endpoint": 2 * 32_768,
    "sae_endpoint": 32_768,
}
DISPLAY_NAMES = {
    "txc_ordered": "Ordered TXC",
    "txc_fixed_shuffle_history": "TXC fixed shuffled history",
    "txc_fixed_reverse_history": "TXC fixed reversed history",
    "sae_positional": "Positional SAE",
    "sae_invariant_history_endpoint": "Invariant-history + endpoint SAE",
    "sae_endpoint": "Endpoint SAE",
}


@dataclass(frozen=True)
class PronounActivationDataset:
    activations: np.ndarray
    frame: pd.DataFrame
    provenance: dict[str, object]


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "cohort": Path(__file__).with_name("cohort.py"),
        "extraction": Path(__file__).with_name("extract_activations.py"),
        "evaluation": Path(__file__),
        "encoder": Path(inspect.getsourcefile(encode_txc) or ""),
        "dictionary_loader": Path(inspect.getsourcefile(_load_encoder) or ""),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"cannot fingerprint implementation files: {paths}")
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_frozen_checkpoints(
    checkpoint_root: str | Path,
) -> dict[str, dict[str, object]]:
    records = validate_checkpoints(Path(checkpoint_root))
    for name, expected in EXPECTED_CONFIG_SHA256.items():
        config_path = (
            Path(checkpoint_root) / CHECKPOINTS[name].train_key / "config.json"
        )
        observed = sha256_file(config_path)
        if observed != expected:
            raise ValueError(f"frozen {name} config checksum drifted: {observed}")
        records[name]["config_sha256"] = observed
    return records


def load_activation_dataset(
    cohort_path: str | Path,
    manifest_path: str | Path,
    cache_dir: str | Path,
) -> PronounActivationDataset:
    frame, cohort_manifest = load_cohort(cohort_path, manifest_path)
    cache_dir = Path(cache_dir)
    request_path = cache_dir / "request.json"
    runtime_path = cache_dir / "runtime.json"
    repeatability_path = cache_dir / "repeatability.json"
    complete_path = cache_dir / "complete.json"
    for path in (
        request_path,
        runtime_path,
        repeatability_path,
        complete_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing GUM activation provenance: {path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    checks = {
        "status": complete.get("status") == "complete",
        "protocol": complete.get("protocol_version") == EXTRACTION_PROTOCOL_VERSION,
        "rows": complete.get("rows") == EXPECTED_ROWS,
        "window_tokens": complete.get("window_tokens") == WINDOW_TOKENS,
        "hidden_size": complete.get("hidden_size") == HIDDEN_SIZE,
        "request_sha256": complete.get("request_sha256") == sha256_file(request_path),
        "runtime_sha256": complete.get("runtime_sha256") == sha256_file(runtime_path),
        "repeatability_sha256": complete.get("repeatability_sha256")
        == sha256_file(repeatability_path),
        "semantic": request.get("cohort_semantic_sha256") == EXPECTED_SEMANTIC_SHA256,
        "cohort_sha256": request.get("cohort_sha256") == sha256_file(cohort_path),
        "manifest_sha256": request.get("cohort_manifest_sha256")
        == sha256_file(manifest_path),
    }
    if not all(checks.values()):
        raise ValueError(f"GUM activation-cache provenance drifted: {checks}")

    request_sha256 = sha256_file(request_path)
    runtime_sha256 = sha256_file(runtime_path)
    arrays = []
    expected_start = 0
    shard_records = complete.get("shards", [])
    if not isinstance(shard_records, list) or not shard_records:
        raise ValueError("GUM activation complete manifest has no shards")
    for record in shard_records:
        start = int(record["start"])
        stop = int(record["stop"])
        if start != expected_start or not start < stop <= len(frame):
            raise ValueError("GUM activation shards are not a contiguous partition")
        shard_path = cache_dir / f"{record['name']}.safetensors"
        sidecar_path = cache_dir / f"{record['name']}.json"
        validated = _validate_shard(
            shard_path,
            sidecar_path,
            frame,
            start=start,
            stop=stop,
            request_sha256=request_sha256,
            runtime_sha256=runtime_sha256,
        )
        if validated != record:
            raise ValueError(
                "GUM activation complete record differs from its shard sidecar"
            )
        from safetensors.torch import load_file

        arrays.append(load_file(str(shard_path), device="cpu")["activations"].numpy())
        expected_start = stop
    if expected_start != len(frame):
        raise ValueError("GUM activation shards do not cover the cohort")
    observed_total_bytes = sum(int(record["size_bytes"]) for record in shard_records)
    if complete.get("total_shard_bytes") != observed_total_bytes:
        raise ValueError("GUM activation total shard bytes drifted")
    activations = np.concatenate(arrays, axis=0)
    if activations.shape != (EXPECTED_ROWS, WINDOW_TOKENS, HIDDEN_SIZE):
        raise ValueError("GUM activation array shape drifted")
    return PronounActivationDataset(
        activations=activations,
        frame=frame,
        provenance={
            "cohort_sha256": sha256_file(cohort_path),
            "cohort_manifest_sha256": sha256_file(manifest_path),
            "cohort_semantic_sha256": EXPECTED_SEMANTIC_SHA256,
            "request_sha256": request_sha256,
            "runtime_sha256": runtime_sha256,
            "complete_sha256": sha256_file(complete_path),
            "source": cohort_manifest["source"],
            "tokenizer": cohort_manifest["tokenizer"],
            "extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
        },
    )


def _history_permutation(event_hash: str, seed: int) -> np.ndarray:
    raw = hashlib.sha256(f"{seed}:{event_hash}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(raw[:8], "little"))
    permutation = rng.permutation(WINDOW_TOKENS - 1)
    identity = np.arange(WINDOW_TOKENS - 1)
    if np.array_equal(permutation, identity):
        permutation = np.roll(permutation, 1)
    return permutation


def controlled_windows(
    windows: np.ndarray,
    event_hashes: Sequence[str],
    *,
    mode: str,
    seed: int,
) -> np.ndarray:
    """Transform four history slots while leaving the pronoun endpoint fixed."""

    if windows.ndim != 3 or windows.shape[1] != WINDOW_TOKENS:
        raise ValueError("controlled GUM windows must have shape [N, 5, d]")
    if len(windows) != len(event_hashes):
        raise ValueError("GUM event hashes and activation rows disagree")
    result = windows.copy()
    if mode == "reverse":
        result[:, : WINDOW_TOKENS - 1] = windows[:, WINDOW_TOKENS - 2 :: -1]
    elif mode == "shuffle":
        for index, event_hash in enumerate(event_hashes):
            permutation = _history_permutation(str(event_hash), seed)
            result[index, : WINDOW_TOKENS - 1] = windows[index, permutation]
    else:
        raise ValueError("GUM history control must be shuffle or reverse")
    if not np.array_equal(result[:, -1], windows[:, -1]):
        raise AssertionError("GUM control changed the pronoun endpoint")
    return result


def _load_cached_code(
    path: Path,
    *,
    method: str,
    rows: int,
    columns: int,
    metadata_sha256: str,
) -> sparse.csr_matrix:
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"incomplete GUM sparse-code cache for {method}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checks = {
        "protocol_version": sidecar.get("protocol_version") == PROTOCOL_VERSION,
        "method": sidecar.get("method") == method,
        "metadata_sha256": sidecar.get("metadata_sha256") == metadata_sha256,
        "sha256": sidecar.get("sha256") == sha256_file(path),
        "size_bytes": sidecar.get("size_bytes") == path.stat().st_size,
        "shape": sidecar.get("shape") == [rows, columns],
    }
    if not all(checks.values()):
        raise ValueError(f"cached GUM code provenance drifted for {method}: {checks}")
    matrix = sparse.load_npz(path).tocsr()
    if matrix.shape != (rows, columns):
        raise ValueError(f"cached GUM code shape drifted: {matrix.shape}")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"cached GUM code contains nonfinite values: {path}")
    if sidecar.get("nnz") != int(matrix.nnz):
        raise ValueError(f"cached GUM code nnz drifted for {method}")
    return matrix


def _write_cached_code(
    matrix: sparse.csr_matrix,
    path: Path,
    *,
    method: str,
    metadata_sha256: str,
) -> dict[str, object]:
    _atomic_sparse(matrix, path)
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "metadata_sha256": metadata_sha256,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
    }
    _atomic_json(record, path.with_suffix(".json"))
    return record


def encode_code_matrices(
    dataset: PronounActivationDataset,
    *,
    checkpoint_root: Path,
    code_dir: Path,
    checkpoint_records: dict[str, dict[str, object]],
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, object]]:
    code_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "protocol_version": PROTOCOL_VERSION,
        "rows": len(dataset.frame),
        "event_hashes_sha256": _hash_strings(dataset.frame["event_hash"].tolist()),
        "activation_provenance": dataset.provenance,
        "checkpoint_provenance": checkpoint_records,
        "implementation_sha256": _implementation_hashes(),
        "conditions": {
            "ordered": "all five states oldest to pronoun endpoint",
            "fixed_shuffle": (
                "stable nonidentity per-event permutation of positions 0..3; "
                "pronoun endpoint position 4 fixed"
            ),
            "fixed_reverse": (
                "positions 0..3 reversed; pronoun endpoint position 4 fixed"
            ),
            "sae_invariant": (
                "max pool four history SAE codes concatenated with anchored "
                "pronoun-endpoint SAE code"
            ),
        },
        "condition_seed": seed,
        "k_pos": K_POS,
    }
    metadata_path = code_dir / "metadata.json"
    complete_path = code_dir / "complete.json"
    paths = {name: code_dir / f"{name}.npz" for name in METHODS}
    sidecar_paths = {name: path.with_suffix(".json") for name, path in paths.items()}
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != fingerprint:
            raise ValueError("GUM sparse-code cache drifted; use a new directory")
    else:
        existing = [
            path
            for path in (*paths.values(), *sidecar_paths.values(), complete_path)
            if path.exists()
        ]
        if existing:
            raise ValueError(
                "GUM sparse-code files exist without metadata; use a new directory"
            )
        _atomic_json(fingerprint, metadata_path)
    metadata_sha256 = sha256_file(metadata_path)
    for name in METHODS:
        if paths[name].exists() != sidecar_paths[name].exists():
            raise ValueError(f"orphan GUM sparse-code artifact for {name}")
    if complete_path.exists() and not all(paths[name].exists() for name in METHODS):
        raise ValueError("completed GUM sparse-code cache is missing method files")
    matrices = {
        name: _load_cached_code(
            paths[name],
            method=name,
            rows=len(dataset.frame),
            columns=EXPECTED_CODE_COLUMNS[name],
            metadata_sha256=metadata_sha256,
        )
        for name in METHODS
        if paths[name].exists()
    }
    windows = np.asarray(dataset.activations, dtype=np.float16)
    event_hashes = dataset.frame["event_hash"].astype(str).to_numpy()

    txc_names = (
        "txc_ordered",
        "txc_fixed_shuffle_history",
        "txc_fixed_reverse_history",
    )
    txc_missing = [name for name in txc_names if name not in matrices]
    if txc_missing:
        loaded = _load_encoder(
            checkpoint_root / CHECKPOINTS["txc"].train_key,
            CHECKPOINTS["txc"].train_key,
            device,
        )
        state = loaded["state"]
        if tuple(state["W_enc"].shape) != (WINDOW_TOKENS, HIDDEN_SIZE, 32_768):
            raise ValueError("submitted TXC tensor shape drifted")
        conditions = {
            "txc_ordered": windows,
            "txc_fixed_shuffle_history": controlled_windows(
                windows,
                event_hashes,
                mode="shuffle",
                seed=seed,
            ),
            "txc_fixed_reverse_history": controlled_windows(
                windows,
                event_hashes,
                mode="reverse",
                seed=seed,
            ),
        }
        for name in txc_missing:
            matrix = encode_txc(
                conditions[name],
                state=state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            )
            _write_cached_code(
                matrix,
                paths[name],
                method=name,
                metadata_sha256=metadata_sha256,
            )
            matrices[name] = matrix
        del loaded, state, conditions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sae_missing = [name for name in SAE_METHODS if name not in matrices]
    if sae_missing:
        loaded = _load_encoder(
            checkpoint_root / CHECKPOINTS["sae"].train_key,
            CHECKPOINTS["sae"].train_key,
            device,
        )
        state = loaded["state"]
        if tuple(state["W_enc"].shape) != (32_768, HIDDEN_SIZE):
            raise ValueError("submitted SAE tensor shape drifted")
        if "sae_positional" in sae_missing:
            matrices["sae_positional"] = encode_sae_positional(
                windows,
                state=state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            )
            _write_cached_code(
                matrices["sae_positional"],
                paths["sae_positional"],
                method="sae_positional",
                metadata_sha256=metadata_sha256,
            )
        if "sae_invariant_history_endpoint" in sae_missing:
            history = encode_sae_invariant(
                windows[:, :-1],
                state=state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            )
            endpoint = encode_sae_token(
                windows[:, -1],
                state=state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            )
            matrices["sae_invariant_history_endpoint"] = sparse.hstack(
                [history, endpoint],
                format="csr",
            )
            _write_cached_code(
                matrices["sae_invariant_history_endpoint"],
                paths["sae_invariant_history_endpoint"],
                method="sae_invariant_history_endpoint",
                metadata_sha256=metadata_sha256,
            )
        if "sae_endpoint" in sae_missing:
            matrices["sae_endpoint"] = encode_sae_token(
                windows[:, -1],
                state=state,
                k_pos=K_POS,
                batch_size=batch_size,
                device=device,
            )
            _write_cached_code(
                matrices["sae_endpoint"],
                paths["sae_endpoint"],
                method="sae_endpoint",
                metadata_sha256=metadata_sha256,
            )
        del loaded, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    matrices = {
        name: _load_cached_code(
            paths[name],
            method=name,
            rows=len(dataset.frame),
            columns=EXPECTED_CODE_COLUMNS[name],
            metadata_sha256=metadata_sha256,
        )
        for name in METHODS
    }
    file_records = {
        name: json.loads(sidecar_paths[name].read_text(encoding="utf-8"))
        for name in METHODS
    }
    complete = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "metadata_sha256": metadata_sha256,
        "files": file_records,
    }
    if complete_path.exists():
        if json.loads(complete_path.read_text(encoding="utf-8")) != complete:
            raise ValueError("GUM sparse-code complete manifest drifted")
    else:
        _atomic_json(complete, complete_path)
    nominal_l0 = {
        "txc_ordered": WINDOW_TOKENS * K_POS,
        "txc_fixed_shuffle_history": WINDOW_TOKENS * K_POS,
        "txc_fixed_reverse_history": WINDOW_TOKENS * K_POS,
        "sae_positional": WINDOW_TOKENS * K_POS,
        "sae_invariant_history_endpoint": WINDOW_TOKENS * K_POS,
        "sae_endpoint": K_POS,
    }
    diagnostics = {
        name: sparse_effective_l0(matrix, nominal_l0=nominal_l0[name])
        for name, matrix in matrices.items()
    }
    return matrices, {
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "complete_path": str(complete_path),
        "complete_sha256": sha256_file(complete_path),
        "files": file_records,
        "effective_l0": diagnostics,
    }


def _equal_group_summary(
    probabilities: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    labels: tuple[int, ...],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    unique_groups = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique_groups]
    row_losses = {
        name: _true_row_losses(values, target, labels)
        for name, values in probabilities.items()
    }
    group_losses = {
        name: np.asarray(
            [float(losses[group_indices].mean()) for group_indices in indices]
        )
        for name, losses in row_losses.items()
    }
    point_losses = {name: float(losses.mean()) for name, losses in group_losses.items()}
    competitors = [name for name in METHODS if name != "txc_ordered"]
    deltas = {name: np.empty(draws) for name in competitors}
    strongest_deltas = np.empty(draws)
    strongest_choices = np.empty(draws, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        sampled = rng.integers(
            0,
            len(unique_groups),
            size=(stop - start, len(unique_groups)),
        )
        means = {
            name: values[sampled].mean(axis=1) for name, values in group_losses.items()
        }
        for name in competitors:
            deltas[name][start:stop] = means[name] - means["txc_ordered"]
        sae_values = np.column_stack([means[name] for name in SAE_METHODS])
        strongest_choices[start:stop] = np.argmin(sae_values, axis=1)
        strongest_deltas[start:stop] = sae_values.min(axis=1) - means["txc_ordered"]

    contrasts = {}
    for name in competitors:
        group_difference = group_losses[name] - group_losses["txc_ordered"]
        contrasts[f"{name}_minus_txc_ordered"] = {
            "equal_document_mean_log_loss_difference": float(group_difference.mean()),
            "ci95_lower": float(np.quantile(deltas[name], 0.025)),
            "ci95_median": float(np.median(deltas[name])),
            "ci95_upper": float(np.quantile(deltas[name], 0.975)),
            "documents_positive": int((group_difference > 0).sum()),
            "documents_total": len(unique_groups),
        }
    strongest_name = min(SAE_METHODS, key=point_losses.__getitem__)
    strongest = {
        "selection_rule": (
            "minimum equal-document SAE loss; reselect minimum positional, "
            "invariant-history+endpoint, or endpoint SAE in every bootstrap "
            "replicate (conservative for the SAE competitor)"
        ),
        "point_selected_method": strongest_name,
        "equal_document_mean_log_loss_difference": float(
            point_losses[strongest_name] - point_losses["txc_ordered"]
        ),
        "ci95_lower": float(np.quantile(strongest_deltas, 0.025)),
        "ci95_median": float(np.median(strongest_deltas)),
        "ci95_upper": float(np.quantile(strongest_deltas, 0.975)),
        "bootstrap_selection_counts": {
            name: int((strongest_choices == index).sum())
            for index, name in enumerate(SAE_METHODS)
        },
    }
    return {
        "unit": "document",
        "documents": len(unique_groups),
        "draws": draws,
        "seed": seed,
        "document_ids_sha256": _hash_strings(unique_groups),
        "method_equal_document_log_loss": point_losses,
        "contrasts": contrasts,
        "strongest_sae_minus_txc_ordered": strongest,
    }


def _gate(summary: dict[str, object], margin: float) -> dict[str, object]:
    contrasts = summary["contrasts"]
    records = {
        "fixed_shuffle_history": contrasts[
            "txc_fixed_shuffle_history_minus_txc_ordered"
        ],
        "fixed_reverse_history": contrasts[
            "txc_fixed_reverse_history_minus_txc_ordered"
        ],
        "strongest_sae": summary["strongest_sae_minus_txc_ordered"],
    }
    components = {}
    for name, record in records.items():
        point = float(record["equal_document_mean_log_loss_difference"])
        lower = float(record["ci95_lower"])
        components[name] = {
            "point_estimate": point,
            "ci95_lower": lower,
            "margin_pass": point >= margin,
            "confidence_pass": lower > 0,
            "passed": point >= margin and lower > 0,
        }
    return {
        "passed": all(value["passed"] for value in components.values()),
        "required_margin": margin,
        "components": components,
    }


def _evaluate_subset(
    matrices: dict[str, sparse.csr_matrix],
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    subset_name: str,
    budgets: Sequence[int],
    primary_budget: int,
    folds: int,
    c_value: float,
    max_iter: int,
    bootstrap_draws: int,
    seed: int,
    gate_margin: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    target = frame.iloc[indices]["distance"].to_numpy(dtype=np.int64)
    groups = frame.iloc[indices]["document"].astype(str).to_numpy()
    labels = tuple(int(value) for value in np.unique(target))
    if labels != (2, 3, 4):
        raise ValueError(f"{subset_name} omits a preregistered distance class")
    local_matrices = {
        name: matrix[indices].tocsr() for name, matrix in matrices.items()
    }
    splits = writer_grouped_splits(target, groups, folds=folds, seed=seed)
    results_by_budget = {}
    primary_predictions: dict[str, np.ndarray] | None = None

    for budget in budgets:
        probabilities = {
            name: np.full((len(target), len(labels)), np.nan, dtype=np.float32)
            for name in METHODS
        }
        fold_records = []
        for fold, (train, test) in enumerate(splits):
            txc_ranking = multiclass_sparse_anova_ranking(
                local_matrices["txc_ordered"][train],
                target[train],
            )
            selected = txc_ranking[:budget]
            scaler, classifier = _fit_probe(
                local_matrices["txc_ordered"],
                target,
                train,
                selected,
                c_value=c_value,
                max_iter=max_iter,
                seed=seed + fold,
            )
            for name in (
                "txc_ordered",
                "txc_fixed_shuffle_history",
                "txc_fixed_reverse_history",
            ):
                probabilities[name][test] = _predict_probe(
                    local_matrices[name],
                    test,
                    selected,
                    scaler,
                    classifier,
                    labels,
                )
            selections = {"txc_ordered": selected.tolist()}
            for name in SAE_METHODS:
                ranking = multiclass_sparse_anova_ranking(
                    local_matrices[name][train],
                    target[train],
                )
                sae_selected = ranking[:budget]
                sae_scaler, sae_classifier = _fit_probe(
                    local_matrices[name],
                    target,
                    train,
                    sae_selected,
                    c_value=c_value,
                    max_iter=max_iter,
                    seed=seed + fold,
                )
                probabilities[name][test] = _predict_probe(
                    local_matrices[name],
                    test,
                    sae_selected,
                    sae_scaler,
                    sae_classifier,
                    labels,
                )
                selections[name] = sae_selected.tolist()
            fold_records.append(
                {
                    "fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_documents": len(np.unique(groups[train])),
                    "test_documents": len(np.unique(groups[test])),
                    "selected_features": selections,
                }
            )
        for name, values in probabilities.items():
            if not np.isfinite(values).all() or not np.allclose(
                values.sum(axis=1),
                1,
                atol=1e-5,
            ):
                raise ValueError(f"{subset_name}/{name} OOF probabilities are invalid")
        equal_document = _equal_group_summary(
            probabilities,
            target,
            groups,
            labels,
            draws=bootstrap_draws,
            seed=seed + 10_000 + budget,
        )
        results_by_budget[str(budget)] = {
            "budget": budget,
            "methods": {
                name: {
                    "row_log_loss": float(log_loss(target, values, labels=labels)),
                    "row_balanced_accuracy": float(
                        balanced_accuracy_score(
                            target,
                            np.asarray(labels)[values.argmax(axis=1)],
                        )
                    ),
                    "equal_document_log_loss": equal_document[
                        "method_equal_document_log_loss"
                    ][name],
                }
                for name, values in probabilities.items()
            },
            "equal_document": equal_document,
            "folds": fold_records,
        }
        if budget == primary_budget:
            primary_predictions = probabilities

    if primary_predictions is None:
        raise ValueError("primary feature budget is absent from the sweep")
    primary = results_by_budget[str(primary_budget)]
    gate = _gate(primary["equal_document"], gate_margin)
    return {
        "name": subset_name,
        "rows": len(indices),
        "documents": len(np.unique(groups)),
        "class_counts": {int(label): int((target == label).sum()) for label in labels},
        "budgets": results_by_budget,
        "primary_budget": primary_budget,
        "gate": gate,
    }, primary_predictions


def evaluate(
    matrices: dict[str, sparse.csr_matrix],
    frame: pd.DataFrame,
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
    primary_indices = np.arange(len(frame), dtype=np.int64)
    balanced_indices = np.flatnonzero(
        frame["balanced_sensitivity"].to_numpy(dtype=np.bool_)
    )
    if (
        len(primary_indices) != EXPECTED_ROWS
        or len(balanced_indices) != EXPECTED_BALANCED_ROWS
    ):
        raise ValueError("GUM primary or balanced cohort size drifted")
    primary, primary_predictions = _evaluate_subset(
        matrices,
        frame,
        primary_indices,
        subset_name="primary",
        budgets=budgets,
        primary_budget=primary_budget,
        folds=folds,
        c_value=c_value,
        max_iter=max_iter,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
        gate_margin=gate_margin,
    )
    balanced, balanced_predictions = _evaluate_subset(
        matrices,
        frame,
        balanced_indices,
        subset_name="same_pronoun_label_balanced",
        budgets=budgets,
        primary_budget=primary_budget,
        folds=folds,
        c_value=c_value,
        max_iter=max_iter,
        bootstrap_draws=bootstrap_draws,
        seed=seed + 1_000_000,
        gate_margin=gate_margin,
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "claim": "personal-pronoun antecedent-distance decoding",
        "claim_boundary": (
            "Each T=5 window includes the target pronoun at its fixed endpoint. "
            "The experiment tests ordered contextual disambiguation, not "
            "pre-pronoun prediction."
        ),
        "primary": primary,
        "required_sensitivity": balanced,
        "preregistered_gate": {
            "passed": primary["gate"]["passed"] and balanced["gate"]["passed"],
            "rule": (
                "Both full and same-pronoun/label-balanced cohorts require "
                "ordered TXC to beat fixed shuffled history, fixed reversed "
                "history, and the strongest SAE by >= margin equal-document "
                "log loss, with paired document-bootstrap CI lower bound > 0."
            ),
            "primary_passed": primary["gate"]["passed"],
            "balanced_sensitivity_passed": balanced["gate"]["passed"],
            "margin": gate_margin,
        },
    }
    predictions = {
        "primary_indices": primary_indices,
        "primary_target": frame["distance"].to_numpy(dtype=np.int64),
        "balanced_indices": balanced_indices,
        "balanced_target": frame.iloc[balanced_indices]["distance"].to_numpy(
            dtype=np.int64
        ),
    }
    for name, values in primary_predictions.items():
        predictions[f"primary_{name}"] = values
    for name, values in balanced_predictions.items():
        predictions[f"balanced_{name}"] = values
    return result, predictions


def _report_markdown(result: dict[str, object]) -> str:
    lines = [
        "# GUM personal-pronoun antecedent-distance decoding",
        "",
        result["claim_boundary"],
        "",
        "| Cohort | Rows | Ordered TXC | Fixed shuffle | Fixed reverse | Strongest SAE | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for key in ("primary", "required_sensitivity"):
        cohort = result[key]
        budget = cohort["budgets"][str(cohort["primary_budget"])]
        losses = budget["equal_document"]["method_equal_document_log_loss"]
        strongest = budget["equal_document"]["strongest_sae_minus_txc_ordered"]
        strongest_name = strongest["point_selected_method"]
        lines.append(
            "| {name} | {rows} | {ordered:.4f} | {shuffle:.4f} | "
            "{reverse:.4f} | {sae:.4f} ({sae_name}) | {gate} |".format(
                name=cohort["name"],
                rows=cohort["rows"],
                ordered=losses["txc_ordered"],
                shuffle=losses["txc_fixed_shuffle_history"],
                reverse=losses["txc_fixed_reverse_history"],
                sae=losses[strongest_name],
                sae_name=DISPLAY_NAMES[strongest_name],
                gate="PASS" if cohort["gate"]["passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "The strongest SAE is reselected inside every bootstrap replicate, "
            "which is conservative for the SAE competitor.",
            "",
            f"Overall preregistered gate: **"
            f"{'PASS' if result['preregistered_gate']['passed'] else 'FAIL'}**.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_plot(result: dict[str, object], output_dir: Path) -> None:
    colors = {
        "txc_ordered": "#2563eb",
        "txc_fixed_shuffle_history": "#d97706",
        "txc_fixed_reverse_history": "#9333ea",
        "sae_positional": "#374151",
        "sae_invariant_history_endpoint": "#059669",
        "sae_endpoint": "#64748b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=False)
    for axis, key, title in zip(
        axes,
        ("primary", "required_sensitivity"),
        ("Full cohort", "Same-pronoun / label-balanced sensitivity"),
    ):
        cohort = result[key]
        budgets = sorted(int(value) for value in cohort["budgets"])
        for method in METHODS:
            values = [
                cohort["budgets"][str(budget)]["methods"][method][
                    "equal_document_log_loss"
                ]
                for budget in budgets
            ]
            axis.plot(
                budgets,
                values,
                marker="o",
                linewidth=2,
                color=colors[method],
                label=DISPLAY_NAMES[method],
            )
        axis.axvline(
            cohort["primary_budget"], color="#94a3b8", linestyle=":", linewidth=1
        )
        axis.set_title(title)
        axis.set_xlabel("Selected sparse-feature budget")
        axis.set_ylabel("Equal-document log loss (lower is better)")
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("GUM personal-pronoun antecedent-distance decoding", y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_dir / "gum_pronoun_distance.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "gum_pronoun_distance.pdf", bbox_inches="tight")
    plt.close(fig)


def run(
    *,
    cohort_path: Path,
    manifest_path: Path,
    activation_cache: Path,
    checkpoint_root: Path,
    code_dir: Path,
    output_dir: Path,
    download: bool,
    device: str,
    batch_size: int,
    budgets: Sequence[int],
    primary_budget: int,
    folds: int,
    c_value: float,
    max_iter: int,
    bootstrap_draws: int,
    seed: int,
    gate_margin: float,
) -> dict[str, object]:
    if download:
        download_checkpoints(checkpoint_root)
    checkpoint_records = validate_frozen_checkpoints(checkpoint_root)
    dataset = load_activation_dataset(cohort_path, manifest_path, activation_cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_request = {
        "protocol_version": PROTOCOL_VERSION,
        "cohort_sha256": sha256_file(cohort_path),
        "manifest_sha256": sha256_file(manifest_path),
        "activation_complete_sha256": sha256_file(activation_cache / "complete.json"),
        "checkpoint_provenance": checkpoint_records,
        "implementation_sha256": _implementation_hashes(),
        "configuration": {
            "budgets": list(budgets),
            "primary_budget": primary_budget,
            "folds": folds,
            "c_value": c_value,
            "max_iter": max_iter,
            "bootstrap_draws": bootstrap_draws,
            "seed": seed,
            "gate_margin": gate_margin,
        },
    }
    evaluation_request_path = output_dir / "request.json"
    if evaluation_request_path.exists():
        if (
            json.loads(evaluation_request_path.read_text(encoding="utf-8"))
            != evaluation_request
        ):
            raise ValueError("GUM evaluation request drifted; use a new directory")
    else:
        existing_outputs = [
            path
            for path in output_dir.iterdir()
            if path.name != evaluation_request_path.name
        ]
        if existing_outputs:
            raise ValueError(
                "GUM evaluation outputs exist without a request fingerprint"
            )
        _atomic_json(evaluation_request, evaluation_request_path)
    matrices, code_provenance = encode_code_matrices(
        dataset,
        checkpoint_root=checkpoint_root,
        code_dir=code_dir,
        checkpoint_records=checkpoint_records,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    evaluation, predictions = evaluate(
        matrices,
        dataset.frame,
        budgets=budgets,
        primary_budget=primary_budget,
        folds=folds,
        c_value=c_value,
        max_iter=max_iter,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
        gate_margin=gate_margin,
    )
    result = {
        **evaluation,
        "evaluation_request_sha256": sha256_file(evaluation_request_path),
        "cohort_provenance": dataset.provenance,
        "checkpoint_provenance": checkpoint_records,
        "code_provenance": code_provenance,
        "configuration": {
            "budgets": list(budgets),
            "primary_budget": primary_budget,
            "folds": folds,
            "c_value": c_value,
            "max_iter": max_iter,
            "bootstrap_draws": bootstrap_draws,
            "seed": seed,
            "gate_margin": gate_margin,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
        },
    }
    _atomic_json(result, output_dir / "results.json")
    _atomic_predictions(predictions, output_dir / "heldout_predictions.npz")
    report = _report_markdown(result)
    temporary = output_dir / "report.md.tmp"
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, output_dir / "report.md")
    _render_plot(result, output_dir)
    return result


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item) for item in value.split(",") if item)
    if not budgets or any(item < 1 for item in budgets):
        raise argparse.ArgumentTypeError("feature budgets must be positive")
    return budgets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--budgets", type=_parse_budgets, default=DEFAULT_S_GRID)
    parser.add_argument("--primary-budget", type=int, default=PRIMARY_BUDGET)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gate-margin", type=float, default=DEFAULT_GATE_MARGIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        cohort_path=args.cohort,
        manifest_path=args.manifest,
        activation_cache=args.activation_cache,
        checkpoint_root=args.checkpoint_root,
        code_dir=args.code_dir,
        output_dir=args.output_dir,
        download=args.download_checkpoints,
        device=args.device,
        batch_size=args.batch_size,
        budgets=args.budgets,
        primary_budget=args.primary_budget,
        folds=args.folds,
        c_value=args.c_value,
        max_iter=args.max_iter,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        gate_margin=args.gate_margin,
    )
    print(json.dumps(result["preregistered_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
