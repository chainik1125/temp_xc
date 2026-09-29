"""Matched C7 detection with sparse cached codes and question-held-out probes.

The production window is exactly the final five of the six frozen pre-onset
positions (-12 through -8). Ordinary and temporal SAEs use one shared encoder;
Stacked keeps distinct (position, feature) identities. No dictionary training
or language-model forward pass occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

HISTORICAL_COMMIT = "284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3"
ACT_CACHE_KEY = "fb2a74be884e512a"
SENTENCE_ACTS_SHA256 = "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
HISTORICAL_PROBE_FILE = "src/temp_bench/case_studies/backtracking.py"
HISTORICAL_PROBE_SHA256 = "5e5ef44e671616b7dfd9b7fd051f69716c119af6058664514a45ddb88d6b85b2"
STAGE_A_PROMPTS_FILE = "results/c7_backtracking/stage_a/prompts.json"
STAGE_A_PROMPTS_SHA256 = "f718d76c1be63bddb83cfb7a9fe03ebde0bf5036a02defb1addc104f8829dd6a"
PROMPT_NORMALIZATION = "Unicode NFKC, casefold, collapse whitespace; preserve mathematical punctuation"
PROTOCOL_VERSION = "c7-camera-ready-detection-v2-prompt-groups"
TRAINING_PROTOCOLS = {"c7-camera-ready-300k-v1", "c7-300k-seeded-v1"}
ARCHS = ("txc_base", "topk_sae", "tsae_paper", "stacked_sae")
S_GRID = (1, 2, 4, 8, 16, 32)
PROBE_MODES = ("historical_raw_C1", "matched_scaled_C1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def normalized_prompt_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def canonical_prompt_groups(question_ids: np.ndarray, prompts: list[dict[str, Any]], *,
                            require_all: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Different archive IDs for the same problem must share a held-out fold."""
    mapping: dict[str, str] = {}
    for row in prompts:
        qid, text = row["id"], row["prompt"]
        if not isinstance(qid, str) or not isinstance(text, str) or not text.strip() or qid in mapping:
            raise ValueError("Stage-A prompts require unique IDs and nonempty problem text")
        mapping[qid] = normalized_prompt_hash(text)
    observed = set(np.asarray(question_ids, dtype=str))
    if observed - set(mapping):
        raise ValueError(f"unknown sentence-activation question IDs: {sorted(observed - set(mapping))}")
    if require_all and observed != set(mapping):
        raise ValueError("sentence-activation questions do not cover every Stage-A prompt ID")
    by_hash: dict[str, list[str]] = {}
    for qid in sorted(observed):
        by_hash.setdefault(mapping[qid], []).append(qid)
    groups = np.asarray([mapping[qid] for qid in question_ids], dtype=str)
    return groups, {"normalization": PROMPT_NORMALIZATION, "n_question_ids": len(observed),
        "n_canonical_prompt_groups": len(by_hash),
        "question_id_to_canonical_group": {qid: mapping[qid] for qid in sorted(observed)},
        "duplicate_text_groups": {group: ids for group, ids in sorted(by_hash.items()) if len(ids) > 1}}


def grouping_leakage_audit(question_ids: np.ndarray, canonical_groups: np.ndarray, *,
                          n_folds: int = 5) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Count actual legacy train/test overlap using the cached sentence counts."""
    from sklearn.model_selection import GroupKFold

    question_ids = np.asarray(question_ids, dtype=str)
    canonical_groups = np.asarray(canonical_groups, dtype=str)
    if len(question_ids) != len(canonical_groups):
        raise ValueError("question and canonical group lengths differ")
    output, assignments = {}, {}
    for name, groups in (("canonical_prompt", canonical_groups), ("historical_question_id", question_ids)):
        fold_ids = np.full(len(groups), -1, dtype=np.int16)
        folds, exposed_questions, split_groups = [], set(), set()
        for fold, (train, test) in enumerate(GroupKFold(n_folds).split(np.zeros(len(groups)), groups=groups)):
            fold_ids[test] = fold
            overlap = np.intersect1d(canonical_groups[train], canonical_groups[test])
            affected = np.isin(canonical_groups[test], overlap)
            affected_qids = sorted(set(question_ids[test][affected]))
            exposed_questions.update(affected_qids)
            split_groups.update(overlap.tolist())
            folds.append({"fold": fold, "n_train_sentences": len(train), "n_test_sentences": len(test),
                "n_overlapping_prompt_groups": len(overlap), "overlapping_prompt_groups": overlap.tolist(),
                "n_test_sentences_with_identical_prompt_in_training": int(affected.sum()),
                "test_question_ids_with_identical_prompt_in_training": affected_qids})
        if np.any(fold_ids < 0):
            raise RuntimeError("grouping audit missed rows")
        output[name] = {"folds": folds, "n_prompt_groups_split_across_folds": len(split_groups),
            "n_question_ids_with_identical_prompt_in_training": len(exposed_questions),
            "n_test_sentences_with_identical_prompt_in_training": sum(
                fold["n_test_sentences_with_identical_prompt_in_training"] for fold in folds)}
        assignments[name] = fold_ids
    if output["canonical_prompt"]["n_prompt_groups_split_across_folds"]:
        raise RuntimeError("canonical grouping allowed duplicate problem text across folds")
    return output, assignments["canonical_prompt"], assignments["historical_question_id"]


def validate_checkpoint(
    config: dict[str, Any], *, arch: str, d_sae: int, seed: int, allow_smoke: bool = False
) -> None:
    required = {
        "arch": arch, "d_sae": d_sae, "seed": seed,
        "historical_commit": HISTORICAL_COMMIT, "act_cache_key": ACT_CACHE_KEY,
    }
    bad = {key: (value, config.get(key)) for key, value in required.items()
           if config.get(key) != value}
    if bad:
        raise ValueError(f"checkpoint provenance mismatch: {bad}")
    if config.get("protocol_version") not in TRAINING_PROTOCOLS:
        raise ValueError("checkpoint is not from an approved corrected training protocol")
    if not config.get("train_key") or not isinstance(config.get("hparams"), dict):
        raise ValueError("checkpoint is missing train_key or explicit architecture hparams")
    if not allow_smoke:
        if config.get("status") != "complete" or config.get("n_steps_completed") != 300_000:
            raise ValueError("production evaluation requires a completed 300000-step checkpoint")
        if config.get("smoke") or config.get("is_smoke"):
            raise ValueError("a smoke checkpoint cannot become a production result")
        training = config.get("training_cfg", {})
        if training.get("batch_size") != 1024 or training.get("n_steps") != 300_000:
            raise ValueError("production training must use logical batch size 1024")
        if config["hparams"].get("k_pos") != 20:
            raise ValueError("production checkpoint requires k_pos=20")
        if d_sae != 32768 and not (arch == "tsae_paper" and d_sae == 16384):
            raise ValueError("production uses width 32768, with a separately labeled T-SAE 16384 sensitivity")
        if seed not in (1, 2, 42):
            raise ValueError("production uses the frozen checkpoint seeds 1, 2 and 42")
        if arch in ("txc_base", "stacked_sae") and config["hparams"].get("T") != 5:
            raise ValueError("the matched production window must be T=5")


def _tensor_csr(z: Any) -> sparse.csr_matrix:
    """Transfer nonzeros only; dense encoder outputs live for one small batch."""
    coordinates = z.nonzero(as_tuple=False)
    rows = coordinates[:, 0].cpu().numpy()
    columns = coordinates[:, 1].cpu().numpy()
    values = z[coordinates[:, 0], coordinates[:, 1]].float().cpu().numpy()
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=tuple(z.shape))
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def code_views(encoded: Any, arch: str, *, position_aware: bool = False) -> dict[str, Any]:
    """Reduce a batch without mixing independently learned Stacked feature IDs."""
    z = encoded.abs().float()
    if arch == "txc_base":
        if z.ndim == 3 and z.shape[1] == 1:
            z = z[:, 0]
        if z.ndim != 2:
            raise ValueError("TXC-base must expose one native window code")
        return {"native": z}
    if z.ndim != 3 or z.shape[1] != 5:
        raise ValueError("per-position encoders must return (batch, 5, width)")
    if arch == "stacked_sae":
        return {"position_identity": z.reshape(z.shape[0], -1)}
    if arch not in ("topk_sae", "tsae_paper"):
        raise ValueError(f"unsupported architecture: {arch}")
    views = {"last": z[:, -1], "mean": z.mean(dim=1), "max": z.amax(dim=1)}
    if position_aware:
        views["position_identity"] = z.reshape(z.shape[0], -1)
    return views


def encode_sparse_views(
    model: Any, acts: np.ndarray, arch: str, *, batch_size: int = 128,
    position_aware: bool = False, progress_path: Path | None = None,
) -> dict[str, sparse.csr_matrix]:
    import torch

    if acts.ndim != 3 or acts.shape[1] != 6:
        raise ValueError("expected the frozen six-position pre-onset activation window")
    if batch_size < 1:
        raise ValueError("encoding batch size must be positive")
    model.eval()  # T-SAE inference uses the frozen threshold, never batch-dependent TopK.
    if arch == "tsae_paper" and float(model.threshold.item()) < 0:
        raise ValueError("T-SAE inference threshold is uninitialized")
    before = {key: value.detach().cpu().clone() for key, value in model.named_buffers()}
    parameter = next(model.parameters())
    chunks: dict[str, list[sparse.csr_matrix]] = {}
    with torch.inference_mode():
        for start in range(0, len(acts), batch_size):
            stop = min(start + batch_size, len(acts))
            x = torch.as_tensor(np.array(acts[start:stop, -5:, :], copy=True),
                                device=parameter.device, dtype=parameter.dtype)
            encoded = model.encode(x)
            for name, view in code_views(encoded, arch, position_aware=position_aware).items():
                if not torch.isfinite(view).all():
                    raise ValueError(f"non-finite encoder output in {name}")
                chunks.setdefault(name, []).append(_tensor_csr(view))
            del encoded, x
            if progress_path is not None:
                atomic_json(progress_path, {"phase": "encoding", "rows": stop, "total": len(acts)})
    after = dict(model.named_buffers())
    if any(not torch.equal(value, after[key].detach().cpu()) for key, value in before.items()):
        raise RuntimeError("read-only encoding changed a model buffer")
    return {name: sparse.vstack(parts, format="csr") for name, parts in chunks.items()}


def support_summary(features: sparse.csr_matrix) -> dict[str, Any]:
    support = np.diff(features.indptr)
    return {"candidate_features": features.shape[1], "nnz": int(features.nnz),
            "mean_l0": float(support.mean()), "median_l0": float(np.median(support)),
            "max_l0": int(support.max()), "min_l0": int(support.min()),
            "csr_bytes": int(features.data.nbytes + features.indices.nbytes + features.indptr.nbytes)}


def training_statistics(
    features: sparse.csr_matrix, labels: np.ndarray, train_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """O(nnz) training-fold moments; never materialize all candidate columns dense."""
    train = features[train_index].astype(np.float64)
    y = labels[train_index]
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("every training fold must contain both classes")
    positive = np.asarray(train[y == 1].mean(axis=0)).ravel()
    negative = np.asarray(train[y == 0].mean(axis=0)).ravel()
    mean = np.asarray(train.mean(axis=0)).ravel()
    second = np.asarray(train.multiply(train).mean(axis=0)).ravel()
    variance = np.maximum(second - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0] = 1.0
    return np.abs(positive - negative), scale


def historical_probe_replay(
    features: sparse.csr_matrix, labels: np.ndarray, groups: np.ndarray,
    sparse_raw_results: dict[str, Any], reference_helper: Any, *,
    s_grid: tuple[int, ...] = S_GRID, n_folds: int = 5,
) -> dict[str, Any]:
    """Run the pinned float32 helper separately from the float64 sparse probes.

    This diagnostic materializes just one final-token view, never all pools.
    Float32 class means can differ in the last bits from our streaming float64
    moments; retain the actual pinned result instead of promising exact replay.
    """
    from sklearn.model_selection import GroupKFold

    dense = features.toarray().astype(np.float32, copy=False)
    reference = reference_helper(dense, labels, groups, S_grid=s_grid, n_folds=n_folds,
                                 C=1.0, random_state=42)
    selected_folds = []
    mismatches = 0
    for fold, (train, _) in enumerate(GroupKFold(n_folds).split(dense, labels, groups)):
        x_train = dense[train]
        y_train = labels[train]
        difference = np.abs(x_train[y_train == 1].mean(axis=0) - x_train[y_train == 0].mean(axis=0))
        order = np.argsort(difference)
        budgets = {}
        for budget in s_grid:
            selected = order[-budget:].tolist()
            sparse_selected = sparse_raw_results["folds"][fold]["budgets"][str(budget)]["selected_columns"]
            same = selected == sparse_selected
            mismatches += int(not same)
            budgets[str(budget)] = {"selected_columns": selected, "matches_sparse_float64": same}
        selected_folds.append({"fold": fold, "budgets": budgets})
    ap = {str(s): float(reference["pr_auc"][s]) for s in s_grid}
    auc = {str(s): float(reference["roc_auc"][s]) if np.isfinite(reference["roc_auc"][s]) else None
           for s in s_grid}
    return {
        "description": "Pinned historical helper on float32 dense legacy T=1/B=1024 codes; diagnostic only.",
        "helper_source_sha256": HISTORICAL_PROBE_SHA256,
        "mean_fold_average_precision": ap, "mean_fold_roc_auc": auc,
        "selected_features_by_fold": selected_folds, "selection_mismatch_count": mismatches,
        "max_abs_ap_difference_from_sparse_float64": max(
            abs(ap[str(s)] - sparse_raw_results["metrics"][str(s)]["mean_fold_average_precision"])
            for s in s_grid),
        "one_view_dense_bytes": int(dense.nbytes),
    }


def verify_historical_runtime(historical_root: Path) -> dict[str, str]:
    """Reuse training's source hash gate, then pin the legacy probe module too."""
    spec = importlib.util.spec_from_file_location("c7_training_source_gate", Path(__file__).with_name("train.py"))
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    source_digest = training.verify_historical_source(historical_root)
    if sha256(historical_root / HISTORICAL_PROBE_FILE) != HISTORICAL_PROBE_SHA256:
        raise ValueError("historical Backtracking probe module SHA-256 mismatch")
    if sha256(historical_root / STAGE_A_PROMPTS_FILE) != STAGE_A_PROMPTS_SHA256:
        raise ValueError("Stage-A prompt artifact SHA-256 mismatch")
    return {"training_source_bundle_sha256": source_digest,
            "historical_probe_module_sha256": HISTORICAL_PROBE_SHA256,
            "stage_a_prompts_sha256": STAGE_A_PROMPTS_SHA256}


def feature_identity(indices: np.ndarray, *, width: int, position_aware: bool) -> list[Any]:
    if position_aware:
        return [{"position": int(index // width), "offset": int(index // width - 12),
                 "feature": int(index % width)} for index in indices]
    return [int(index) for index in indices]


def probe_sparse(
    features: sparse.csr_matrix, labels: np.ndarray, groups: np.ndarray, *,
    width: int, position_aware: bool = False, s_grid: tuple[int, ...] = S_GRID,
    n_folds: int = 5, random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    if features.shape[0] != len(labels) or len(labels) != len(groups):
        raise ValueError("feature, label and held-out group lengths differ")
    if not np.isfinite(features.data).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("probing requires finite features and binary labels")
    if not s_grid or min(s_grid) < 1 or max(s_grid) > features.shape[1]:
        raise ValueError("invalid selected-feature budget")
    splits = list(GroupKFold(n_splits=n_folds).split(features, labels, groups=groups))
    fold_id = np.full(len(labels), -1, dtype=np.int16)
    oof = {mode: np.full((len(labels), len(s_grid)), np.nan, dtype=np.float64)
           for mode in PROBE_MODES}
    outputs: dict[str, Any] = {
        mode: {"folds": [], "metrics": {}, "scaling": "none" if mode == PROBE_MODES[0]
               else "training-fold population std, no centering; applied before ranking and fitting"}
        for mode in PROBE_MODES
    }
    for fold, (train, test) in enumerate(splits):
        if np.intersect1d(groups[train], groups[test]).size:
            raise RuntimeError("held-out group leakage across a fold")
        fold_id[test] = fold
        differences, scale = training_statistics(features, labels, train)
        for mode in PROBE_MODES:
            scaled = mode == "matched_scaled_C1"
            rank_score = differences / scale if scaled else differences
            order = np.argsort(rank_score)  # Historical deterministic NumPy tie convention.
            fold_result: dict[str, Any] = {"fold": fold, "n_train": len(train), "n_test": len(test),
                "n_train_questions": len(np.unique(groups[train])),
                "n_test_questions": len(np.unique(groups[test])), "budgets": {}}
            for column, budget in enumerate(s_grid):
                selected = order[-budget:]
                # Only S<=32 columns become dense, at most a few MB per fold.
                x_train = features[train][:, selected].toarray().astype(np.float64)
                x_test = features[test][:, selected].toarray().astype(np.float64)
                if scaled:
                    x_train /= scale[selected]
                    x_test /= scale[selected]
                classifier = LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                                                max_iter=2000, random_state=random_state)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    classifier.fit(x_train, labels[train])
                probability = classifier.predict_proba(x_test)[:, 1]
                oof[mode][test, column] = probability
                fold_result["budgets"][str(budget)] = {
                    "average_precision": float(average_precision_score(labels[test], probability)),
                    "roc_auc": float(roc_auc_score(labels[test], probability))
                    if len(np.unique(labels[test])) == 2 else None,
                    "selected_columns": selected.tolist(),
                    "selected_features": feature_identity(selected, width=width, position_aware=position_aware),
                    "selected_scale": scale[selected].tolist() if scaled else [1.0] * budget,
                    "selection_score": rank_score[selected].tolist(),
                    "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
                    "nonzero_coefficients": int(np.count_nonzero(classifier.coef_)),
                    "n_iter": int(classifier.n_iter_[0]),
                    "warnings": [str(item.message) for item in caught
                                 if issubclass(item.category, ConvergenceWarning)],
                }
            outputs[mode]["folds"].append(fold_result)
    if np.any(fold_id < 0) or any(not np.isfinite(array).all() for array in oof.values()):
        raise RuntimeError("out-of-fold predictions are incomplete")
    for mode in PROBE_MODES:
        for column, budget in enumerate(s_grid):
            fold_metrics = [fold["budgets"][str(budget)] for fold in outputs[mode]["folds"]]
            ap = [item["average_precision"] for item in fold_metrics]
            auc = [item["roc_auc"] for item in fold_metrics if item["roc_auc"] is not None]
            outputs[mode]["metrics"][str(budget)] = {
                "mean_fold_average_precision": float(np.mean(ap)),
                "fold_average_precision": ap,
                "mean_fold_roc_auc": float(np.mean(auc)) if auc else None,
                "pooled_oof_average_precision": float(average_precision_score(labels, oof[mode][:, column])),
                "pooled_oof_roc_auc": float(roc_auc_score(labels, oof[mode][:, column])),
            }
    return outputs, oof, fold_id


def tsae_inference_diagnostic(
    model: Any, acts: np.ndarray, *, rows: int = 64, batch_sizes: tuple[int, ...] = (1, 7, 32)
) -> dict[str, Any]:
    """Expose the old train-mode dependency on batching, without updating buffers."""
    import torch

    parameter = next(model.parameters())
    before = {key: value.detach().cpu().clone() for key, value in model.named_buffers()}
    was_training = model.training
    matrices: dict[str, sparse.csr_matrix] = {}
    result: dict[str, Any] = {"n_rows": min(rows, len(acts)), "threshold": float(model.threshold.item()),
                              "batch_sizes": list(batch_sizes), "encodings": {}}
    try:
        with torch.inference_mode():
            for window in (1, 5):
                for training in (False, True):
                    mode = "batchtopk_training_mode" if training else "threshold_eval_mode"
                    for batch_size in batch_sizes:
                        model.train(training)
                        chunks = []
                        for start in range(0, result["n_rows"], batch_size):
                            stop = min(start + batch_size, result["n_rows"])
                            x = torch.as_tensor(np.array(acts[start:stop, -window:, :], copy=True),
                                                device=parameter.device, dtype=parameter.dtype)
                            encoded = model.encode(x).abs().float()
                            chunks.append(_tensor_csr(encoded.reshape(encoded.shape[0], -1)))
                        name = f"{mode}_T{window}_B{batch_size}"
                        matrices[name] = sparse.vstack(chunks, format="csr")
                        summary = support_summary(matrices[name])
                        summary["mean_l0_per_token"] = summary["mean_l0"] / window
                        reference = matrices[f"{mode}_T{window}_B{batch_sizes[0]}"]
                        difference = matrices[name] - reference
                        summary["max_abs_difference_vs_first_batch_size"] = (
                            float(np.max(np.abs(difference.data))) if difference.nnz else 0.0)
                        summary["changed_entries_vs_first_batch_size"] = int(difference.nnz)
                        result["encodings"][name] = summary
            for window in (1, 5):
                difference = (matrices[f"batchtopk_training_mode_T{window}_B{batch_sizes[-1]}"]
                              - matrices[f"threshold_eval_mode_T{window}_B{batch_sizes[-1]}"])
                result[f"train_vs_eval_T{window}_max_abs_difference"] = (
                    float(np.max(np.abs(difference.data))) if difference.nnz else 0.0)
    finally:
        model.train(was_training)
    after = dict(model.named_buffers())
    result["buffers_unchanged"] = all(
        torch.equal(value, after[key].detach().cpu()) for key, value in before.items())
    if not result["buffers_unchanged"]:
        raise RuntimeError("T-SAE diagnostic changed a model buffer")
    return result


def encode_legacy_tsae_last(model: Any, acts: np.ndarray) -> sparse.csr_matrix:
    """Separately labeled reproduction of old train-mode T=1, batch=1024 encoding."""
    import torch

    parameter = next(model.parameters())
    was_training = model.training
    chunks = []
    try:
        model.train()
        with torch.inference_mode():
            for start in range(0, len(acts), 1024):
                x = torch.as_tensor(np.array(acts[start:start + 1024, -1:, :], copy=True),
                                    device=parameter.device, dtype=parameter.dtype)
                chunks.append(_tensor_csr(model.encode(x).abs().float()[:, 0]))
    finally:
        model.train(was_training)
    return sparse.vstack(chunks, format="csr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sentence-acts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", choices=ARCHS, required=True)
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--position-aware", action="store_true",
                        help="Also retain (position, feature) identities for shared per-token SAEs.")
    parser.add_argument("--save-codes", action="store_true", help="Save compressed sparse codes remotely.")
    parser.add_argument("--historical-tsae-check", action="store_true",
                        help="Add separately labeled full T=1, training-mode, B=1024 T-SAE diagnostic.")
    parser.add_argument("--allow-smoke", action="store_true",
                        help="Explicitly permit incomplete/small checkpoints; outputs are marked smoke.")
    parser.add_argument("--max-questions", type=int, help="Smoke only: use the first N sorted question IDs.")
    parser.add_argument("--s-grid", type=int, nargs="+", default=list(S_GRID))
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing result: {output}")
    if args.batch_size < 1 or args.folds < 2:
        raise ValueError("encoding batch size must be positive and folds must be at least two")
    if args.historical_tsae_check and args.arch != "tsae_paper":
        raise ValueError("--historical-tsae-check only applies to T-SAE")
    if not args.allow_smoke and (args.max_questions is not None or args.folds != 5
                                 or tuple(args.s_grid) != S_GRID):
        raise ValueError("production uses all questions, five folds, and the fixed S grid")
    if args.max_questions is not None and args.max_questions < args.folds:
        raise ValueError("smoke question count must be at least the number of folds")
    historical_root = args.historical_root.resolve()
    if (historical_root / "HISTORICAL_COMMIT").read_text().strip() != HISTORICAL_COMMIT:
        raise ValueError("historical source marker mismatch")
    runtime_provenance = verify_historical_runtime(historical_root) if not args.allow_smoke else {
        "verification": "smoke: source marker only"}
    config_path = args.checkpoint_dir / "config.json"
    model_path = args.checkpoint_dir / "model.safetensors"
    config = json.loads(config_path.read_text())
    validate_checkpoint(config, arch=args.arch, d_sae=args.d_sae, seed=args.seed,
                        allow_smoke=args.allow_smoke)
    model_digest = sha256(model_path)
    for key in ("checkpoint_sha256", "model_sha256"):
        if config.get(key) is not None and config[key] != model_digest:
            raise ValueError(f"model weights disagree with the recorded {key}")
    data_digest = sha256(args.sentence_acts)
    if not args.allow_smoke and data_digest != SENTENCE_ACTS_SHA256:
        raise ValueError("sentence-activation SHA-256 differs from the frozen cohort")
    with np.load(args.sentence_acts, allow_pickle=True) as archive:
        acts = archive["X"]
        labels = archive["is_bt"].astype(np.int64)
        keys = archive["keys"].astype(str)
    question_ids = np.asarray([key.split("|", 1)[0] for key in keys], dtype=str)
    if acts.ndim != 3 or acts.shape[1] != 6 or len(acts) != len(keys) or len(keys) != len(labels):
        raise ValueError("activation and label cohort shapes disagree")
    if not args.allow_smoke and (acts.shape != (25204, 6, 4096) or labels.sum() != 3169
                                 or len(np.unique(question_ids)) != 300):
        raise ValueError("production cohort dimensions/counts changed")
    prompts_path = historical_root / STAGE_A_PROMPTS_FILE
    prompt_digest = sha256(prompts_path)
    if not args.allow_smoke and prompt_digest != STAGE_A_PROMPTS_SHA256:
        raise ValueError("Stage-A prompt artifact SHA-256 mismatch")
    prompts = json.loads(prompts_path.read_text())
    groups, grouping = canonical_prompt_groups(question_ids, prompts)
    if not args.allow_smoke and (len(prompts) != 300 or grouping["n_canonical_prompt_groups"] != 213):
        raise ValueError("frozen Stage-A question-to-canonical-prompt mapping changed")
    if args.max_questions is not None:
        selected = np.isin(question_ids, np.unique(question_ids)[:args.max_questions])
        acts, labels, keys, question_ids = acts[selected], labels[selected], keys[selected], question_ids[selected]
        groups, grouping = canonical_prompt_groups(question_ids, prompts, require_all=False)
    if len(np.unique(groups)) < args.folds:
        raise ValueError("too few distinct prompt texts for the requested number of folds")
    grouping_audit, canonical_fold_id, historical_fold_id = grouping_leakage_audit(
        question_ids, groups, n_folds=args.folds)

    os.environ["TEMP_BENCH_ROOT"] = str(historical_root)
    sys.path.insert(0, str(historical_root / "src"))
    import scipy
    import sklearn
    import torch
    from safetensors.torch import load_file
    from temp_bench.config import instantiate_arch, load_arch

    if not args.allow_smoke and Path(inspect.getfile(instantiate_arch)).resolve() != historical_root / "src/temp_bench/config.py":
        raise ValueError("configuration was imported from a different historical runtime")
    reference_probe = None
    if args.historical_tsae_check:
        from temp_bench.case_studies.backtracking import compute_probe_metrics_at_S

        if Path(inspect.getfile(compute_probe_metrics_at_S)).resolve() != historical_root / HISTORICAL_PROBE_FILE:
            raise ValueError("historical probe was imported from a different source checkout")
        if sha256(historical_root / HISTORICAL_PROBE_FILE) != HISTORICAL_PROBE_SHA256:
            raise ValueError("historical probe SHA-256 mismatch")
        reference_probe = compute_probe_metrics_at_S

    spec = load_arch(args.arch, component="c7")
    # Saved hparams are authoritative; never inherit changed registry defaults.
    spec = spec.model_copy(update={"hparams": config["hparams"]})
    if config["hparams"].get("d_sae") != args.d_sae:
        raise ValueError("checkpoint hparams and requested dictionary width disagree")
    model = instantiate_arch(spec, d_in=acts.shape[-1])
    model.load_state_dict(load_file(str(model_path)), strict=True)
    # The earlier corrected detector evaluated float32 weights, even for BF16 training.
    model.to(device=args.device, dtype=torch.float32)
    model.eval()
    diagnostics = {"prompt_grouping_audit": {**grouping, **grouping_audit},
        "historical_question_id": {
            "description": "Original question-ID grouping retained only as a historical diagnostic; identical problem text can cross folds.",
            "grouping": "question_id", "n_groups": len(np.unique(question_ids)), "views": {}}}
    if args.arch == "tsae_paper":
        diagnostics["tsae_inference_modes"] = tsae_inference_diagnostic(model, acts)
    progress_path = output.with_suffix(".progress.json")
    matrices = encode_sparse_views(model, acts, args.arch, batch_size=args.batch_size,
                                   position_aware=args.position_aware, progress_path=progress_path)
    legacy = encode_legacy_tsae_last(model, acts) if args.historical_tsae_check and args.arch == "tsae_paper" else None
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "running", "smoke": args.allow_smoke, "detection_protocol": PROTOCOL_VERSION,
        "arch": args.arch, "d_sae": args.d_sae, "seed": args.seed,
        "train_key": config["train_key"], "training_protocol": config["protocol_version"],
        "comparison_role": "T-SAE width sensitivity" if args.d_sae == 16384 else "matched 32K core",
        "n_steps_completed": config.get("n_steps_completed"), "checkpoint_config": config,
        "provenance": {"historical_commit": HISTORICAL_COMMIT,
            "checkpoint_sha256": model_digest, "checkpoint_config_sha256": sha256(config_path),
            "sentence_acts_sha256": data_digest, "evaluator_sha256": sha256(Path(__file__)),
            "stage_a_prompts_sha256": prompt_digest,
            "historical_runtime": runtime_provenance,
            "sentence_keys_sha256": hashlib.sha256(json.dumps(keys.tolist()).encode()).hexdigest()},
        "data": {"n_sentences": len(labels), "n_positive": int(labels.sum()),
                 "n_questions": len(np.unique(question_ids)), "n_prompt_groups": len(np.unique(groups)),
                 "positive_fraction": float(labels.mean()),
                 "cached_offsets": [-13, -12, -11, -10, -9, -8],
                 "evaluated_offsets": [-12, -11, -10, -9, -8]},
        "protocol": {"S_grid": args.s_grid, "folds": args.folds, "grouping": "normalized_prompt_sha256",
            "group_normalization": PROMPT_NORMALIZATION,
            "primary_budget": 8, "primary_metric": "mean_fold_average_precision",
            "primary_probe_mode": "matched_scaled_C1",
            "encoder_mode": "eval", "encoder_dtype": "float32", "encoding_batch_size": args.batch_size,
            "probe_modes": list(PROBE_MODES), "C": 1.0, "probe_random_state": 42,
            "historical_raw_C1_note": "Historical raw-feature probing rule with primary canonical-prompt folds. Original question-ID folds are diagnostic only.",
            "fold_count_note": "n_train_questions/n_test_questions in primary probes count unique normalized problem texts; original archive IDs remain in OOF question_ids.",
            "stacked_policy": "Keep (position, feature) identities; S counts selected coordinates total.",
            "scaling_note": "Population standard deviation fitted using training rows only, no centering.",
            "uncertainty_note": "Fold variability is not dictionary-seed uncertainty; retain paired OOF predictions."},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                        "device": str(args.device)},
        "views": {}, "diagnostics": diagnostics,
    }
    arrays: dict[str, np.ndarray] = {"labels": labels, "question_ids": question_ids,
        "canonical_group_ids": groups, "sentence_keys": keys, "fold_id": canonical_fold_id,
        "historical_question_id_fold_id": historical_fold_id,
        "S_grid": np.asarray(args.s_grid, dtype=np.int16)}
    del model, acts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    convergence_warning_count = 0
    for name, matrix in matrices.items():
        results, oof, fold_id = probe_sparse(matrix, labels, groups, width=args.d_sae,
            position_aware=name == "position_identity", s_grid=tuple(args.s_grid), n_folds=args.folds)
        view = {"support": support_summary(matrix), "probes": results}
        if args.save_codes:
            code_path = output.with_name(output.stem + f".{name}.codes.npz")
            sparse.save_npz(code_path, matrix, compressed=True)
            view["codes"] = {"path": code_path.name, "sha256": sha256(code_path)}
        payload["views"][name] = view
        for mode, probability in oof.items():
            arrays[f"{name}__{mode}"] = probability
            convergence_warning_count += sum(len(budget["warnings"]) for fold in results[mode]["folds"]
                                             for budget in fold["budgets"].values())
        if not np.array_equal(canonical_fold_id, fold_id):
            raise RuntimeError("primary views produced different canonical-prompt splits")
        historical_results, historical_oof, actual_historical_folds = probe_sparse(
            matrix, labels, question_ids, width=args.d_sae,
            position_aware=name == "position_identity", s_grid=tuple(args.s_grid), n_folds=args.folds)
        if not np.array_equal(historical_fold_id, actual_historical_folds):
            raise RuntimeError("historical views produced different question-ID splits")
        payload["diagnostics"]["historical_question_id"]["views"][name] = {
            "support": view["support"], "probes": historical_results}
        for mode, probability in historical_oof.items():
            arrays[f"historical_question_id__{name}__{mode}"] = probability
        atomic_json(progress_path, {"phase": "probing", "last_view": name, "elapsed_seconds": time.time() - started})
    if legacy is not None:
        name = "legacy_batchtopk_last"
        results, oof, fold_id = probe_sparse(legacy, labels, question_ids, width=args.d_sae,
            s_grid=tuple(args.s_grid), n_folds=args.folds)
        if not np.array_equal(historical_fold_id, fold_id):
            raise RuntimeError("legacy BatchTopK replay changed historical question-ID folds")
        payload["diagnostics"][name] = {"support": support_summary(legacy), "probes": results,
            "encoder_policy": "Historical diagnostic only: training mode, final token, batch size 1024, original 300 question-ID groups.",
            "grouping": "question_id", "n_groups": len(np.unique(question_ids)),
            "pinned_float32_probe_replay": historical_probe_replay(
                legacy, labels, question_ids, results["historical_raw_C1"], reference_probe,
                s_grid=tuple(args.s_grid), n_folds=args.folds)}
        for mode, probability in oof.items():
            arrays[f"historical_question_id__{name}__{mode}"] = probability
    oof_path = output.with_name(output.stem + ".oof.npz")
    temporary = oof_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(oof_path)
    payload["oof_predictions"] = {"path": oof_path.name, "sha256": sha256(oof_path),
        "layout": "Primary view__probe arrays use canonical_group_ids and fold_id. historical_question_id__view__probe arrays use original question_ids and historical_question_id_fold_id. All arrays follow sentence_keys rows and S_grid columns."}
    payload["convergence_warning_count"] = convergence_warning_count
    payload["status"] = "complete" if convergence_warning_count == 0 else "complete_with_warnings"
    payload["elapsed_seconds"] = time.time() - started
    atomic_json(output, payload)
    atomic_json(progress_path, {"phase": "complete", "status": payload["status"],
                               "elapsed_seconds": payload["elapsed_seconds"]})
    print(json.dumps({"status": payload["status"], "output": str(output), "smoke": args.allow_smoke,
                      "convergence_warning_count": convergence_warning_count}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
