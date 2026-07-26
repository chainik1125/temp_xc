"""Isolated CPU audit of the backtracking positional-SAE probe.

This diagnostic reuses the frozen cell's ordered positional-SAE codes and
outer-fold contract, but writes to a separate output directory.  It does not
modify ``result.json``, primary predictions, code caches, or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


SENSITIVITY_VERSION = "2026-07-24.positional-sae.1"
DEFAULT_S_GRID = (32, 64, 128, 192, 256)
DEFAULT_C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


@dataclass(frozen=True)
class OuterFold:
    fold: int
    test_indices: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    txc_probability: np.ndarray


def _csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("feature budgets must be positive")
    return parsed


def _csv_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("regularization values must be positive")
    return parsed


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_predictions(payload: dict[str, np.ndarray], path: Path) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_isolated_output(cell_dir: Path, output_dir: Path) -> None:
    """Refuse any output location inside the frozen primary cell."""

    source = cell_dir.resolve()
    target = output_dir.resolve()
    if target == source or source in target.parents:
        raise ValueError(
            "sensitivity output must be outside the primary cell directory"
        )


def scale_stable_effect(
    matrix: sparse.csr_matrix, y: np.ndarray
) -> np.ndarray:
    """Return absolute class-mean differences measured in train-fold SD units."""

    labels = np.asarray(y)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("scale-stable ranking requires both binary classes")
    matrix = matrix.tocsr()
    positive = matrix[labels == 1]
    negative = matrix[labels == 0]
    positive_mean = np.asarray(positive.mean(axis=0)).ravel()
    negative_mean = np.asarray(negative.mean(axis=0)).ravel()
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    standard_deviation = np.sqrt(np.maximum(second - mean * mean, 0.0))
    effect = np.divide(
        np.abs(positive_mean - negative_mean),
        standard_deviation,
        out=np.zeros_like(standard_deviation, dtype=np.float64),
        where=standard_deviation > 0,
    )
    effect[~np.isfinite(effect)] = 0.0
    return effect


def ranked_features(
    matrix: sparse.csr_matrix, y: np.ndarray
) -> np.ndarray:
    """Rank by scale-stable effect, breaking exact ties by feature index."""

    effect = scale_stable_effect(matrix, y)
    return np.lexsort((np.arange(len(effect), dtype=np.int64), -effect))


def grouped_inner_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    placeholder = np.zeros((len(y), 1), dtype=np.int8)
    return list(splitter.split(placeholder, y, groups))


def _scaled_matrices(
    matrix: sparse.csr_matrix,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    selected: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    scaler = StandardScaler(with_mean=False)
    train = scaler.fit_transform(matrix[train_indices][:, selected]).tocsr()
    test = scaler.transform(matrix[test_indices][:, selected]).tocsr()
    return train, test


def _fit_probability(
    train: sparse.csr_matrix,
    y_train: np.ndarray,
    test: sparse.csr_matrix,
    *,
    c_value: float,
    seed: int,
) -> np.ndarray:
    classifier = LogisticRegression(
        penalty="l1",
        C=float(c_value),
        solver="liblinear",
        max_iter=2_000,
        random_state=seed,
    ).fit(train, y_train)
    return classifier.predict_proba(test)[:, 1]


def tune_regularization(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    *,
    s_grid: tuple[int, ...],
    c_grid: tuple[float, ...],
    inner_folds: int,
    seed: int,
) -> dict[int, dict]:
    """Tune C in grouped inner folds, recomputing ranking and scaling each fold."""

    train_y = y[train_indices]
    train_groups = groups[train_indices]
    splits = grouped_inner_splits(
        train_y, train_groups, folds=inner_folds, seed=seed
    )
    scores = {
        int(budget): {float(c_value): [] for c_value in c_grid}
        for budget in s_grid
    }
    for inner_fold, (inner_train_local, inner_valid_local) in enumerate(splits):
        inner_train = train_indices[inner_train_local]
        inner_valid = train_indices[inner_valid_local]
        ranking = ranked_features(matrix[inner_train], y[inner_train])
        for budget in s_grid:
            selected = ranking[: min(int(budget), matrix.shape[1])]
            fitted_train, fitted_valid = _scaled_matrices(
                matrix, inner_train, inner_valid, selected
            )
            for c_value in c_grid:
                probability = _fit_probability(
                    fitted_train,
                    y[inner_train],
                    fitted_valid,
                    c_value=c_value,
                    seed=seed + inner_fold,
                )
                scores[int(budget)][float(c_value)].append(
                    float(
                        average_precision_score(
                            y[inner_valid], probability
                        )
                    )
                )

    result = {}
    for budget in s_grid:
        by_c = {
            float(c_value): {
                "fold_values": values,
                "mean_pr_auc": float(np.mean(values)),
            }
            for c_value, values in scores[int(budget)].items()
        }
        ordered_c = sorted(by_c)
        best_c = max(
            ordered_c,
            key=lambda c_value: (
                by_c[c_value]["mean_pr_auc"],
                -c_value,
            ),
        )
        result[int(budget)] = {
            "selected_c": float(best_c),
            "by_c": {str(c_value): by_c[c_value] for c_value in ordered_c},
        }
    return result


def _load_outer_contract(
    cell_dir: Path,
) -> tuple[dict, np.ndarray, np.ndarray, list[OuterFold]]:
    result_path = cell_dir / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete":
        raise ValueError(f"source cell is not complete: {result_path}")
    txc_summary = max(
        result["probes"]["txc"], key=lambda row: int(row["n_features"])
    )
    primary_budget = int(txc_summary["n_features"])
    n_rows = int(result["n_rows"])
    y = np.empty(n_rows, dtype=np.int8)
    groups = np.empty(n_rows, dtype=object)
    seen = np.zeros(n_rows, dtype=np.int8)
    outer_folds = []
    for fold in range(int(result["folds"])):
        path = (
            cell_dir
            / "predictions"
            / "txc"
            / f"S{primary_budget}_fold{fold}.npz"
        )
        with np.load(path) as payload:
            indices = payload["test_indices"].astype(np.int64, copy=True)
            fold_y = payload["y"].astype(np.int8, copy=True)
            fold_groups = payload["groups"].astype(str, copy=True)
            probability = payload["ordered"].astype(np.float64, copy=True)
        if np.any(indices < 0) or np.any(indices >= n_rows):
            raise ValueError(f"invalid test indices in {path}")
        if np.any(seen[indices]):
            raise ValueError(f"outer folds overlap in {path}")
        seen[indices] = 1
        y[indices] = fold_y
        groups[indices] = fold_groups
        outer_folds.append(
            OuterFold(
                fold=fold,
                test_indices=indices,
                y=fold_y,
                groups=fold_groups,
                txc_probability=probability,
            )
        )
    if not np.all(seen == 1):
        raise ValueError("persisted outer folds do not cover every code row")
    result["primary_txc_budget"] = primary_budget
    return result, y, groups.astype(str), outer_folds


def _fold_payload(
    outer: OuterFold, probability: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "test_indices": outer.test_indices,
        "y": outer.y,
        "groups": outer.groups,
        "ordered": np.asarray(probability, dtype=np.float64),
    }


def paired_question_bootstrap_many(
    outer_folds: list[OuterFold],
    candidates: dict[str, list[dict[str, np.ndarray]]],
    *,
    repeats: int,
    seed: int,
) -> dict[str, dict]:
    """Compare fixed OOF TXC and SAE probabilities on paired question draws."""

    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    original = {name: [] for name in candidates}
    group_rows = []
    for fold, reference in enumerate(outer_folds):
        lookup = {
            group: np.flatnonzero(reference.groups == group)
            for group in np.unique(reference.groups)
        }
        group_rows.append(lookup)
        for name, payloads in candidates.items():
            candidate = payloads[fold]
            for key in ("test_indices", "y", "groups"):
                expected = getattr(reference, key)
                if not np.array_equal(expected, candidate[key]):
                    raise ValueError(
                        f"candidate alignment mismatch: {name}/fold={fold}/{key}"
                    )
            reference_ap = average_precision_score(
                reference.y, reference.txc_probability
            )
            candidate_ap = average_precision_score(
                candidate["y"], candidate["ordered"]
            )
            original[name].append(float(reference_ap - candidate_ap))

    rng = np.random.default_rng(seed)
    draws = {name: [] for name in candidates}
    for _ in range(repeats):
        replicate = {name: [] for name in candidates}
        for fold, reference in enumerate(outer_folds):
            lookup = group_rows[fold]
            unique_groups = np.asarray(sorted(lookup))
            sampled = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            indices = np.concatenate([lookup[group] for group in sampled])
            sampled_y = reference.y[indices]
            if len(np.unique(sampled_y)) < 2:
                continue
            reference_ap = average_precision_score(
                sampled_y, reference.txc_probability[indices]
            )
            for name, payloads in candidates.items():
                candidate_ap = average_precision_score(
                    sampled_y, payloads[fold]["ordered"][indices]
                )
                replicate[name].append(float(reference_ap - candidate_ap))
        for name, values in replicate.items():
            if values:
                draws[name].append(float(np.mean(values)))

    return {
        name: {
            "point_estimate": float(np.mean(original[name])),
            "fold_values": original[name],
            "repeats": len(draws[name]),
            "lower_95": float(np.quantile(draws[name], 0.025)),
            "median": float(np.median(draws[name])),
            "upper_95": float(np.quantile(draws[name], 0.975)),
        }
        for name in candidates
    }


def run_sensitivity(
    *,
    cell_dir: Path,
    output_dir: Path,
    s_grid: tuple[int, ...],
    c_grid: tuple[float, ...],
    inner_folds: int,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict:
    validate_isolated_output(cell_dir, output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"partial sensitivity output exists; choose a fresh path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()

    source, y, groups, outer_folds = _load_outer_contract(cell_dir)
    code_path = cell_dir / "codes" / "sae_positional_ordered.npz"
    matrix = sparse.load_npz(code_path).tocsr()
    if matrix.shape[0] != len(y):
        raise ValueError(
            f"code/label row mismatch: {matrix.shape[0]} != {len(y)}"
        )
    window = int(source["window"])
    if matrix.shape[1] % window:
        raise ValueError("positional code width is not divisible by window")
    per_position_width = matrix.shape[1] // window

    all_indices = np.arange(len(y), dtype=np.int64)
    fold_rows = []
    candidate_folds = {f"S{budget}": [] for budget in s_grid}
    joint_folds = []
    for outer in outer_folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[outer.test_indices] = False
        train_indices = all_indices[train_mask]
        tuning = tune_regularization(
            matrix,
            y,
            groups,
            train_indices,
            s_grid=s_grid,
            c_grid=c_grid,
            inner_folds=inner_folds,
            seed=int(source["seed"]) + 10_000 + outer.fold,
        )
        ranking = ranked_features(matrix[train_indices], y[train_indices])
        joint_budget = max(
            s_grid,
            key=lambda budget: (
                tuning[int(budget)]["by_c"][
                    str(tuning[int(budget)]["selected_c"])
                ]["mean_pr_auc"],
                -int(budget),
            ),
        )
        outer_rows = []
        fold_probabilities = {}
        for budget in s_grid:
            selected = ranking[: min(int(budget), matrix.shape[1])]
            fitted_train, fitted_test = _scaled_matrices(
                matrix, train_indices, outer.test_indices, selected
            )
            selected_c = float(tuning[int(budget)]["selected_c"])
            probability = _fit_probability(
                fitted_train,
                y[train_indices],
                fitted_test,
                c_value=selected_c,
                seed=int(source["seed"]) + outer.fold,
            )
            fold_probabilities[int(budget)] = probability
            payload = _fold_payload(outer, probability)
            candidate_folds[f"S{budget}"].append(payload)
            _atomic_predictions(
                {
                    **payload,
                    "selected_c": np.asarray(selected_c),
                    "selected_features": selected.astype(np.int64),
                },
                prediction_dir / f"S{budget}_fold{outer.fold}.npz",
            )
            position_counts = np.bincount(
                selected // per_position_width, minlength=window
            )
            candidate_ap = float(
                average_precision_score(outer.y, probability)
            )
            txc_ap = float(
                average_precision_score(
                    outer.y, outer.txc_probability
                )
            )
            outer_rows.append(
                {
                    "budget": int(budget),
                    "selected_c": selected_c,
                    "inner_cv": tuning[int(budget)],
                    "position_counts": position_counts.tolist(),
                    "positional_sae_pr_auc": candidate_ap,
                    "txc_pr_auc": txc_ap,
                    "txc_minus_positional_sae": txc_ap - candidate_ap,
                }
            )
        joint_payload = _fold_payload(
            outer, fold_probabilities[int(joint_budget)]
        )
        joint_folds.append(joint_payload)
        fold_rows.append(
            {
                "fold": outer.fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(outer.test_indices)),
                "joint_selected_budget": int(joint_budget),
                "joint_selected_c": float(
                    tuning[int(joint_budget)]["selected_c"]
                ),
                "budgets": outer_rows,
            }
        )

    candidate_folds["joint_inner_selected"] = joint_folds
    bootstrap = paired_question_bootstrap_many(
        outer_folds,
        candidate_folds,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    summaries = []
    for budget in s_grid:
        rows = [
            next(
                row
                for row in fold["budgets"]
                if int(row["budget"]) == int(budget)
            )
            for fold in fold_rows
        ]
        summaries.append(
            {
                "budget": int(budget),
                "selected_c_by_fold": [
                    float(row["selected_c"]) for row in rows
                ],
                "positional_sae_pr_auc": {
                    "fold_values": [
                        float(row["positional_sae_pr_auc"]) for row in rows
                    ],
                    "mean": float(
                        np.mean(
                            [
                                row["positional_sae_pr_auc"]
                                for row in rows
                            ]
                        )
                    ),
                },
                "txc_pr_auc": {
                    "fold_values": [
                        float(row["txc_pr_auc"]) for row in rows
                    ],
                    "mean": float(
                        np.mean([row["txc_pr_auc"] for row in rows])
                    ),
                },
                "txc_minus_positional_sae": bootstrap[f"S{budget}"],
            }
        )

    joint_rows = []
    for fold, payload in zip(fold_rows, joint_folds, strict=True):
        outer = outer_folds[int(fold["fold"])]
        joint_rows.append(
            {
                "fold": int(fold["fold"]),
                "budget": int(fold["joint_selected_budget"]),
                "c": float(fold["joint_selected_c"]),
                "pr_auc": float(
                    average_precision_score(payload["y"], payload["ordered"])
                ),
                "txc_pr_auc": float(
                    average_precision_score(
                        outer.y, outer.txc_probability
                    )
                ),
            }
        )
    output = {
        "status": "complete",
        "sensitivity_version": SENSITIVITY_VERSION,
        "source_cell": str(cell_dir.resolve()),
        "source_result_sha256": _sha256(cell_dir / "result.json"),
        "source_code": str(code_path.resolve()),
        "window": window,
        "seed": int(source["seed"]),
        "primary_txc_budget": int(source["primary_txc_budget"]),
        "n_rows": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "s_grid": list(s_grid),
        "c_grid": list(c_grid),
        "inner_folds": int(inner_folds),
        "bootstrap": {
            "unit": (
                "question group, resampled within each fixed outer test fold"
            ),
            "repeats_requested": int(bootstrap_repeats),
            "seed": int(bootstrap_seed),
        },
        "method": {
            "ranking": (
                "absolute class-mean difference divided by train-fold "
                "feature standard deviation"
            ),
            "scaling": (
                "StandardScaler(with_mean=False), fit inside each train fold"
            ),
            "classifier": "L1 logistic regression with grouped inner-CV C",
            "feature_budget": (
                "each S is audited; a joint S,C choice is also selected "
                "inside each outer training fold"
            ),
            "interpretation": (
                "post-hoc diagnostic of positional-SAE probe sensitivity; "
                "the frozen primary result is unchanged"
            ),
        },
        "summaries": summaries,
        "joint_inner_selected": {
            "folds": joint_rows,
            "positional_sae_pr_auc_mean": float(
                np.mean([row["pr_auc"] for row in joint_rows])
            ),
            "txc_pr_auc_mean": float(
                np.mean([row["txc_pr_auc"] for row in joint_rows])
            ),
            "txc_minus_positional_sae": bootstrap[
                "joint_inner_selected"
            ],
        },
        "folds": fold_rows,
    }
    _atomic_json(output, result_path)
    _atomic_text(_markdown(output), output_dir / "summary.md")
    return output


def _markdown(result: dict) -> str:
    lines = [
        "# Positional-SAE sensitivity audit",
        "",
        (
            "This is an isolated post-hoc diagnostic. It leaves the frozen "
            "backtracking cell and its primary metrics unchanged."
        ),
        "",
        (
            "| S | Positional SAE PR-AUC | TXC PR-AUC "
            "| TXC − SAE [95% question CI] | C by outer fold |"
        ),
        "|---:|---:|---:|---:|---|",
    ]
    for row in result["summaries"]:
        gap = row["txc_minus_positional_sae"]
        c_values = ", ".join(f"{value:g}" for value in row["selected_c_by_fold"])
        lines.append(
            f"| {row['budget']} | {row['positional_sae_pr_auc']['mean']:.4f} "
            f"| {row['txc_pr_auc']['mean']:.4f} "
            f"| {gap['point_estimate']:+.4f} "
            f"[{gap['lower_95']:+.4f}, {gap['upper_95']:+.4f}] "
            f"| {c_values} |"
        )
    joint = result["joint_inner_selected"]
    gap = joint["txc_minus_positional_sae"]
    choices = ", ".join(
        f"S={row['budget']}, C={row['c']:g}" for row in joint["folds"]
    )
    lines.extend(
        [
            "",
            "## Joint inner-selected sensitivity",
            "",
            (
                f"Positional SAE PR-AUC `{joint['positional_sae_pr_auc_mean']:.4f}`; "
                f"TXC PR-AUC `{joint['txc_pr_auc_mean']:.4f}`; paired gap "
                f"`{gap['point_estimate']:+.4f}` with 95% question-bootstrap "
                f"CI `[{gap['lower_95']:+.4f}, {gap['upper_95']:+.4f}]`."
            ),
            "",
            f"Outer-fold choices: {choices}.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--s-grid",
        type=_csv_ints,
        default=DEFAULT_S_GRID,
        help="comma-separated positional-SAE feature budgets",
    )
    parser.add_argument(
        "--c-grid",
        type=_csv_floats,
        default=DEFAULT_C_GRID,
        help="comma-separated L1 logistic-regression C values",
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_sensitivity(
        cell_dir=args.cell_dir,
        output_dir=args.output_dir,
        s_grid=tuple(args.s_grid),
        c_grid=tuple(args.c_grid),
        inner_folds=args.inner_folds,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str((args.output_dir / "result.json").resolve()),
                "summary": str((args.output_dir / "summary.md").resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
