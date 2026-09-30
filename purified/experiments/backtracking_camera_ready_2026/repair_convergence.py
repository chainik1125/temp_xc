#!/usr/bin/env python3
"""Repair only primary C7 probe fits that hit the original iteration limit.

No training or encoding occurs. Cached sparse codes, selected columns, scales,
folds, C, tolerance and random state remain fixed. The original 2000-iteration
fits must reproduce before the sole numerical change, max_iter=20000, is tried.
Convergence alone determines acceptance; test metrics do not choose settings.

Default: write verified candidates and an immutable input archive beside the
cell. --apply also installs the corrected JSON and a NEW sibling OOF filename;
the original OOF is never overwritten. --plan-only reads only detection JSON.
Run the normal aggregate and paired-analysis commands again after installation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import detect
import paired_detection
import summarize_detection as summary

REPAIR_PROTOCOL = "c7-selective-convergence-repair-v1"
ORIGINAL_MAX_ITER = 2000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checked_sibling(parent, name):
    require(isinstance(name, str) and Path(name).name == name and name not in (".", ".."),
            f"artifact path must be a sibling filename: {name!r}")
    return Path(parent) / name


def warning_slots(payload):
    slots = []
    for view, value in sorted(payload["views"].items()):
        for mode, probe in sorted(value["probes"].items()):
            for fold in probe["folds"]:
                for budget in sorted(probe["metrics"], key=int):
                    record = fold["budgets"][budget]
                    if record["warnings"]:
                        require(record["n_iter"] == ORIGINAL_MAX_ITER,
                                "only fits that reached the original iteration cap may be repaired")
                        slots.append((view, mode, int(fold["fold"]), int(budget)))
    count = sum(len(payload["views"][v]["probes"][m]["folds"][f]["budgets"][str(s)]["warnings"])
                for v, m, f, s in slots)
    require(count == payload["convergence_warning_count"], "primary warning total disagrees with fit records")
    return slots


def classifier(max_iter):
    # Same constructor as the hash-verified detector, changing only max_iter.
    return LogisticRegression(penalty="l1", C=1.0, solver="liblinear",
                              max_iter=max_iter, random_state=42)


def fitted(x_train, labels, max_iter):
    model = classifier(max_iter)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, labels)
    convergence = [str(row.message) for row in caught if issubclass(row.category, ConvergenceWarning)]
    return model, {"max_iter": max_iter, "n_iter": int(model.n_iter_[0]),
        "convergence_warnings": convergence, "elapsed_seconds": time.monotonic() - started,
        "other_warnings": [{"category": row.category.__name__, "message": str(row.message)}
                           for row in caught if not issubclass(row.category, ConvergenceWarning)]}


def validate_oof(payload, arrays):
    """Replay every primary AP, independently including the paired-analysis AP."""
    labels, groups, folds = arrays["labels"], arrays["canonical_group_ids"], arrays["fold_id"]
    require(labels.shape == groups.shape == folds.shape == (payload["data"]["n_sentences"],),
            "OOF metadata dimensions differ")
    require(np.isin(labels, (0, 1)).all() and int(labels.sum()) == payload["data"]["n_positive"],
            "OOF label cohort differs")
    require(set(np.unique(folds)) == set(range(5)), "OOF requires five folds")
    require(len(np.unique(groups)) == payload["data"]["n_prompt_groups"], "OOF prompt groups differ")
    require(len(np.unique(arrays["question_ids"])) == payload["data"]["n_questions"], "OOF question IDs differ")
    require(np.array_equal(arrays["S_grid"], payload["protocol"]["S_grid"]), "OOF budget grid differs")
    keys_hash = hashlib.sha256(json.dumps(arrays["sentence_keys"].tolist()).encode()).hexdigest()
    require(keys_hash == payload["provenance"]["sentence_keys_sha256"], "OOF sentence-key order differs")
    for fold, (train, test) in enumerate(GroupKFold(5).split(labels, labels, groups)):
        require(np.array_equal(np.flatnonzero(folds == fold), test), "saved folds differ from frozen grouping")
        require(not np.intersect1d(groups[train], groups[test]).size, "canonical prompt leakage")
    _, group_index = np.unique(groups, return_inverse=True)
    unit_counts = np.ones((1, int(group_index.max()) + 1), dtype=np.int32)
    checked = 0
    for view, value in payload["views"].items():
        for mode, probe in value["probes"].items():
            scores = arrays[f"{view}__{mode}"]
            require(scores.shape == (len(labels), len(arrays["S_grid"]))
                    and np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
                    f"invalid OOF probabilities in {view}/{mode}")
            for column, budget in enumerate(arrays["S_grid"]):
                budget = str(int(budget))
                metrics = probe["metrics"][budget]
                ap_values, auc_values = [], []
                for fold in range(5):
                    mask = folds == fold
                    ap = float(average_precision_score(labels[mask], scores[mask, column]))
                    auc = float(roc_auc_score(labels[mask], scores[mask, column])) if len(np.unique(labels[mask])) == 2 else None
                    record = probe["folds"][fold]["budgets"][budget]
                    require(np.isclose(ap, record["average_precision"], rtol=0, atol=1e-12),
                            f"OOF AP mismatch: {view}/{mode}/fold{fold}/S{budget}")
                    require(auc is None and record["roc_auc"] is None or auc is not None
                            and np.isclose(auc, record["roc_auc"], rtol=0, atol=1e-12), "OOF fold ROC-AUC mismatch")
                    prepared = paired_detection.prepare_ap(labels[mask], scores[mask, column], group_index[mask])
                    independent = float(paired_detection.weighted_ap(prepared, unit_counts)[0])
                    require(np.isclose(independent, ap, rtol=0, atol=1e-10), "paired-analysis AP replay mismatch")
                    ap_values.append(ap)
                    if auc is not None:
                        auc_values.append(auc)
                    checked += 1
                require(np.allclose(ap_values, metrics["fold_average_precision"], rtol=0, atol=1e-12), "OOF fold AP list mismatch")
                require(np.isclose(np.mean(ap_values), metrics["mean_fold_average_precision"], rtol=0, atol=1e-12), "OOF mean AP mismatch")
                require(not auc_values or np.isclose(np.mean(auc_values), metrics["mean_fold_roc_auc"], rtol=0, atol=1e-12), "OOF mean ROC-AUC mismatch")
                require(np.isclose(average_precision_score(labels, scores[:, column]), metrics["pooled_oof_average_precision"], rtol=0, atol=1e-12), "pooled OOF AP mismatch")
                require(np.isclose(roc_auc_score(labels, scores[:, column]), metrics["pooled_oof_roc_auc"], rtol=0, atol=1e-12), "pooled OOF ROC-AUC mismatch")
    return {"primary_fold_budget_AP_checks": checked, "paired_AP_replay_checks": checked,
            "canonical_folds_exactly_reproduced": True}


def recompute_metrics(payload, arrays, view, mode, budget):
    probe = payload["views"][view]["probes"][mode]
    records = [fold["budgets"][str(budget)] for fold in probe["folds"]]
    ap = [record["average_precision"] for record in records]
    auc = [record["roc_auc"] for record in records if record["roc_auc"] is not None]
    probabilities = arrays[f"{view}__{mode}"][:, list(arrays["S_grid"]).index(budget)]
    probe["metrics"][str(budget)] = {"mean_fold_average_precision": float(np.mean(ap)),
        "fold_average_precision": ap, "mean_fold_roc_auc": float(np.mean(auc)) if auc else None,
        "pooled_oof_average_precision": float(average_precision_score(arrays["labels"], probabilities)),
        "pooled_oof_roc_auc": float(roc_auc_score(arrays["labels"], probabilities))}


def verify_changes(original, corrected, before, after, slots):
    require(original["diagnostics"] == corrected["diagnostics"], "historical diagnostic was changed")
    allowed = {}
    for view, mode, fold, budget in slots:
        key = f"{view}__{mode}"
        mask = allowed.setdefault(key, np.zeros(before[key].shape, dtype=bool))
        mask[before["fold_id"] == fold, list(before["S_grid"]).index(budget)] = True
    require(set(before) == set(after), "OOF array names changed")
    for key in before:
        if key in allowed:
            require(np.array_equal(before[key][~allowed[key]], after[key][~allowed[key]]), "unaffected OOF probabilities changed")
        else:
            require(np.array_equal(before[key], after[key]), f"unaffected OOF metadata/diagnostic changed: {key}")
    targets = set(slots)
    for view, value in original["views"].items():
        for mode, probe in value["probes"].items():
            for fold in probe["folds"]:
                for budget, record in fold["budgets"].items():
                    new = corrected["views"][view]["probes"][mode]["folds"][fold["fold"]]["budgets"][budget]
                    if (view, mode, fold["fold"], int(budget)) not in targets:
                        require(record == new, "unaffected fitted probe changed")
                    for key in ("selected_columns", "selected_features", "selected_scale", "selection_score"):
                        require(record[key] == new[key], f"frozen selection/scaling changed: {key}")
    return {"unaffected_arrays_and_fit_records_preserved": True,
            "diagnostics_preserved": True, "selected_features_and_scales_preserved": True,
            "untargeted_budgets_preserved_bitwise": sorted(set(map(int, before["S_grid"])) - {s for _, _, _, s in slots})}


def repair(args):
    import scipy
    import sklearn

    source = args.detection.resolve()
    original = read(source)
    summary.validate(original)
    require("numerical_repair" not in original, "result already has a numerical-repair receipt")
    slots = warning_slots(original)
    require(slots, "no primary convergence warnings to repair")
    require(args.max_iter > ORIGINAL_MAX_ITER, "repair iteration cap must exceed 2000")
    script_sha = sha256(Path(__file__))
    input_sha = sha256(source)
    repair_id = f"{input_sha[:12]}-{script_sha[:12]}-maxiter{args.max_iter}"
    destination = source.parent / "numerical_repairs" / repair_id
    plan = {"detection": str(source), "primary_warning_slots": slots, "original_max_iter": ORIGINAL_MAX_ITER,
            "repair_max_iter": args.max_iter, "archive_and_receipt": str(destination),
            "apply": args.apply, "code_paths": [original["views"][view]["codes"]["path"] for view in sorted({s[0] for s in slots})]}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0

    import fcntl
    lock = (source.parent / ".numerical_repair.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(sha256(source) == input_sha, "detection JSON changed while acquiring lock")
    require(not destination.exists(), f"repair directory already exists: {destination}; preserve it and inspect its receipt")
    require(sha256(Path(detect.__file__)) == original["provenance"]["evaluator_sha256"],
            "repair requires the original frozen detector file")
    versions = {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    for name, version in versions.items():
        require(version == original["environment"][name], f"{name} version differs from original evaluator")
    old_params, new_params = classifier(ORIGINAL_MAX_ITER).get_params(), classifier(args.max_iter).get_params()
    require({key for key in old_params if old_params[key] != new_params[key]} == {"max_iter"},
            "repair changes a parameter other than the iteration cap")
    oof_path = checked_sibling(source.parent, original["oof_predictions"]["path"])
    require(sha256(oof_path) == original["oof_predictions"]["sha256"], "original OOF hash mismatch")
    with np.load(oof_path, allow_pickle=False) as archive:
        before = {key: archive[key] for key in archive.files}
    initial_checks = validate_oof(original, before)
    matrices, input_codes = {}, {}
    for view in sorted({slot[0] for slot in slots}):
        artifact = original["views"][view]["codes"]
        path = checked_sibling(source.parent, artifact["path"])
        require(sha256(path) == artifact["sha256"], f"cached sparse-code hash mismatch: {view}")
        matrix = sparse.load_npz(path)
        require(sparse.isspmatrix_csr(matrix) and matrix.shape ==
                (len(before["labels"]), original["views"][view]["support"]["candidate_features"])
                and np.isfinite(matrix.data).all(), f"cached code shape/format invalid: {view}")
        matrices[view] = matrix
        input_codes[view] = {"path": path.name, "sha256": artifact["sha256"]}

    (destination / "original").mkdir(parents=True)
    (destination / "corrected").mkdir()
    shutil.copyfile(source, destination / "original" / source.name)
    shutil.copyfile(oof_path, destination / "original" / oof_path.name)
    require(sha256(destination / "original" / source.name) == input_sha
            and sha256(destination / "original" / oof_path.name) == original["oof_predictions"]["sha256"],
            "input archive copy failed hash verification")
    receipt = {"protocol": REPAIR_PROTOCOL, "status": "running", "repair_script_sha256": script_sha,
        "source_evaluator_sha256": original["provenance"]["evaluator_sha256"],
        "cell": {key: original[key] for key in ("arch", "d_sae", "seed", "train_key")},
        "original_detection_sha256": input_sha, "original_oof_sha256": original["oof_predictions"]["sha256"],
        "cached_codes": input_codes, "environment": {**versions, "python": platform.python_version()},
        "original_classifier_params": old_params, "repair_classifier_params": new_params,
        "policy": "Repair every primary fit with a convergence warning by changing only max_iter. Acceptance requires convergence, never an improvement in held-out metrics.",
        "before_checks": initial_checks, "repairs": [], "started_unix": time.time()}
    receipt_path = destination / "repair_receipt.json"
    atomic_json(receipt_path, receipt)
    started = time.monotonic()
    corrected, after = copy.deepcopy(original), {key: array.copy() for key, array in before.items()}
    stats = {}
    try:
        for view, mode, fold, budget in slots:
            record = original["views"][view]["probes"][mode]["folds"][fold]["budgets"][str(budget)]
            train, test = np.flatnonzero(before["fold_id"] != fold), np.flatnonzero(before["fold_id"] == fold)
            matrix = matrices[view]
            if (view, fold) not in stats:
                stats[(view, fold)] = detect.training_statistics(matrix, before["labels"], train)
            differences, scales = stats[(view, fold)]
            scaled = mode == "matched_scaled_C1"
            rank = differences / scales if scaled else differences
            selected = np.asarray(record["selected_columns"], dtype=np.int64)
            require(np.array_equal(np.argsort(rank)[-budget:], selected), "saved feature selection does not reproduce")
            expected_scale = scales[selected] if scaled else np.ones(budget)
            require(np.allclose(expected_scale, record["selected_scale"], rtol=0, atol=1e-12), "saved training-fold scales do not reproduce")
            require(np.allclose(rank[selected], record["selection_score"], rtol=0, atol=1e-12), "saved feature-selection scores do not reproduce")
            x_train = matrix[train][:, selected].toarray().astype(np.float64)
            x_test = matrix[test][:, selected].toarray().astype(np.float64)
            if scaled:
                x_train /= np.asarray(record["selected_scale"])
                x_test /= np.asarray(record["selected_scale"])
            column = list(before["S_grid"]).index(budget)
            key = f"{view}__{mode}"
            original_probability = before[key][test, column]
            require(np.allclose(expit(x_test @ np.asarray(record["coef"]) + record["intercept"]),
                                original_probability, rtol=0, atol=1e-12), "recorded coefficients do not reproduce original OOF")
            replay, replay_log = fitted(x_train, before["labels"][train], ORIGINAL_MAX_ITER)
            require(replay_log["n_iter"] == record["n_iter"]
                    and replay_log["convergence_warnings"] == record["warnings"], "original fit stopping state did not reproduce")
            replay_probability = replay.predict_proba(x_test)[:, 1]
            require(np.allclose(replay.coef_[0], record["coef"], rtol=0, atol=1e-10)
                    and np.isclose(replay.intercept_[0], record["intercept"], rtol=0, atol=1e-10)
                    and np.allclose(replay_probability, original_probability, rtol=0, atol=1e-10),
                    "original 2000-iteration fit did not reproduce; no corrected result may be installed")
            model, repair_log = fitted(x_train, before["labels"][train], args.max_iter)
            audit = {"view": view, "probe_mode": mode, "fold": fold, "S": budget,
                "original_fit_replay": replay_log, "repair_fit": repair_log, "before_fit": record}
            receipt["repairs"].append(audit)
            atomic_json(receipt_path, receipt)
            require(not repair_log["convergence_warnings"] and repair_log["n_iter"] < args.max_iter,
                    f"{view}/{mode}/fold{fold}/S{budget} still failed to converge at {args.max_iter}; originals unchanged")
            probability = model.predict_proba(x_test)[:, 1]
            require(np.isfinite(probability).all() and np.isfinite(model.coef_).all()
                    and np.isfinite(model.intercept_).all(), "refit produced non-finite outputs")
            after[key][test, column] = probability
            updated = corrected["views"][view]["probes"][mode]["folds"][fold]["budgets"][str(budget)]
            updated.update(average_precision=float(average_precision_score(before["labels"][test], probability)),
                roc_auc=float(roc_auc_score(before["labels"][test], probability)),
                coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                nonzero_coefficients=int(np.count_nonzero(model.coef_)), n_iter=repair_log["n_iter"], warnings=[])
            audit["after_fit"] = copy.deepcopy(updated)
            audit["max_absolute_probability_change"] = float(np.max(np.abs(probability - original_probability)))
            recompute_metrics(corrected, after, view, mode, budget)
            audit["before_aggregate"] = original["views"][view]["probes"][mode]["metrics"][str(budget)]
            audit["after_aggregate"] = copy.deepcopy(corrected["views"][view]["probes"][mode]["metrics"][str(budget)])
            atomic_json(receipt_path, receipt)
            print(json.dumps({"phase": "repaired_fit", "view": view, "mode": mode, "fold": fold,
                              "S": budget, "n_iter": repair_log["n_iter"]}), flush=True)

        corrected["convergence_warning_count"] = sum(len(record["warnings"])
            for view in corrected["views"].values() for probe in view["probes"].values()
            for fold in probe["folds"] for record in fold["budgets"].values())
        require(corrected["convergence_warning_count"] == 0, "primary warnings remain after selected repairs")
        corrected["status"] = "complete"
        change_checks = verify_changes(original, corrected, before, after, slots)
        final_checks = validate_oof(corrected, after)
        new_oof_name = f"{source.stem}.repaired-{repair_id}.oof.npz"
        candidate_oof = destination / "corrected" / new_oof_name
        with candidate_oof.open("wb") as stream:
            np.savez_compressed(stream, **after)
            stream.flush()
            os.fsync(stream.fileno())
        corrected["oof_predictions"]["path"] = new_oof_name
        corrected["oof_predictions"]["sha256"] = sha256(candidate_oof)
        corrected["numerical_repair"] = {"protocol": REPAIR_PROTOCOL, "repair_script_sha256": script_sha,
            "receipt": str(receipt_path.relative_to(source.parent)), "original_detection_sha256": input_sha,
            "original_oof_sha256": original["oof_predictions"]["sha256"],
            "original_evaluator_sha256": original["provenance"]["evaluator_sha256"],
            "original_max_iter": ORIGINAL_MAX_ITER, "repair_max_iter": args.max_iter,
            "repaired_slots": [{"view": v, "probe_mode": m, "fold": f, "S": s} for v, m, f, s in slots],
            "scope": "Only warned primary probe fits and their dependent OOF/metrics; original detector, dictionaries, feature selection, scaling and diagnostics preserved."}
        summary.validate(corrected)
        candidate_json = destination / "corrected" / source.name
        atomic_json(candidate_json, corrected)
        with np.load(candidate_oof, allow_pickle=False) as archive:
            require(set(archive.files) == set(after) and all(np.array_equal(archive[key], after[key]) for key in after),
                    "corrected OOF failed serialization verification")
        require(sha256(source) == input_sha and sha256(oof_path) == original["oof_predictions"]["sha256"],
                "original source artifacts changed during repair")
        receipt.update(status="verified_candidates", elapsed_seconds=time.monotonic() - started,
            before_primary_warning_count=original["convergence_warning_count"], after_primary_warning_count=0,
            change_checks=change_checks, after_checks={**final_checks, "aggregate_schema_validated": True},
            archive={"detection_json": str((destination / "original" / source.name).relative_to(source.parent)),
                     "oof": str((destination / "original" / oof_path.name).relative_to(source.parent))},
            corrected_detection_sha256=sha256(candidate_json), corrected_oof_sha256=sha256(candidate_oof),
            corrected_detection_path=str(candidate_json), corrected_oof_path=str(candidate_oof))
        atomic_json(receipt_path, receipt)
        if args.apply:
            new_oof = source.with_name(new_oof_name)
            require(not new_oof.exists(), "corrected sibling OOF already exists")
            temporary = new_oof.with_suffix(new_oof.suffix + ".tmp")
            shutil.copyfile(candidate_oof, temporary)
            require(sha256(temporary) == corrected["oof_predictions"]["sha256"], "OOF installation copy failed")
            temporary.replace(new_oof)
            atomic_json(source, corrected)
            require(sha256(source) == receipt["corrected_detection_sha256"], "installed detection JSON hash mismatch")
            receipt.update(status="installed", installed_detection=str(source), installed_oof=str(new_oof),
                           installed_unix=time.time())
            atomic_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path),
            "repaired_fits": len(slots), "remaining_primary_warnings": 0,
            "next_step": "Regenerate aggregate figures and paired analysis from the installed detection.json/OOF pair."}), flush=True)
        return 0
    except Exception as exc:
        receipt.update(status="failed_no_install", error=f"{type(exc).__name__}: {exc}",
                       elapsed_seconds=time.monotonic() - started)
        atomic_json(receipt_path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--plan-only", action="store_true", help="inspect warned fits without reading codes/OOF or writing files")
    parser.add_argument("--apply", action="store_true", help="install only after every fit and consistency gate passes")
    return repair(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
