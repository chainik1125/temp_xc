#!/usr/bin/env python3
"""Paired, within-fold prompt-cluster bootstrap of fixed C7 OOF predictions.

Requires all 15 completed v2 detection cells and their hash-verified OOF NPZs.
Outputs core 32K and T-SAE 16K sensitivity CSVs separately. Intervals condition
on the three fitted dictionaries/probes: neither models nor probes are refit,
and training-seed variability is reported separately as paired-delta sample SD.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import summarize_detection as summary

BUDGETS = (8, 32)
BASE = ("txc_base", 32768, "native")
COMPARATORS = tuple((arch, width, view) for arch, width, views in (
    ("topk_sae", 32768, ("last", "mean", "max")),
    ("tsae_paper", 32768, ("last", "mean", "max")),
    ("stacked_sae", 32768, ("position_identity",)),
    ("tsae_paper", 16384, ("last", "mean", "max")),
) for view in views)
FIELDS = ("comparison_role", "comparator_arch", "comparator_width", "comparator_view",
          "probe_mode", "S", "txc_mean_fold_ap", "comparator_mean_fold_ap", "paired_delta_mean",
          "delta_seed1", "delta_seed2", "delta_seed42", "paired_delta_seed_sample_sd",
          "conditional_ci_low", "conditional_ci_high", "bootstrap_mean", "bootstrap_sd",
          "n_bootstrap", "bootstrap_seed", "n_prompt_groups", "n_folds", "n_dictionary_seeds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_ap(labels: np.ndarray, scores: np.ndarray, group_index: np.ndarray) -> dict[str, np.ndarray]:
    """Sort once; combine each score tie before integrating precision/recall."""
    labels, scores, group_index = map(np.asarray, (labels, scores, group_index))
    summary.require(labels.ndim == scores.ndim == group_index.ndim == 1
                    and len(labels) == len(scores) == len(group_index) and len(labels) > 0,
                    "AP inputs must be nonempty aligned vectors")
    summary.require(np.isin(labels, (0, 1)).all() and np.isfinite(scores).all(), "invalid AP labels/scores")
    summary.require(np.issubdtype(group_index.dtype, np.integer) and group_index.min() >= 0,
                    "invalid AP group indices")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    return {"groups": group_index[order], "positive": labels[order].astype(np.float64), "tie_ends": ends}


def weighted_ap(prepared: dict[str, np.ndarray], counts: np.ndarray) -> np.ndarray:
    """Exact average precision for integer cluster multiplicities, including ties.

    Each row of counts is one resample; all sentences in a group share its
    multiplicity. A sample with no positive weight gets AP=0, matching sklearn.
    This equals physically repeating rows but avoids constructing that dataset.
    """
    counts = np.asarray(counts)
    summary.require(counts.ndim == 2 and counts.shape[1] > prepared["groups"].max()
                    and np.isfinite(counts).all() and (counts >= 0).all(), "invalid cluster multiplicities")
    weights = counts[:, prepared["groups"]]
    ends = prepared["tie_ends"]
    total = np.cumsum(weights, axis=1, dtype=np.float64)[:, ends]
    positive = np.cumsum(weights * prepared["positive"], axis=1, dtype=np.float64)[:, ends]
    precision = np.divide(positive, total, out=np.zeros_like(positive), where=total != 0)
    increments = np.diff(positive, axis=1, prepend=0)
    integral = np.sum(increments * precision, axis=1)
    return np.divide(integral, positive[:, -1], out=np.zeros_like(integral), where=positive[:, -1] != 0)


def load_inputs(root: Path) -> tuple[dict, dict, dict, list[dict], int]:
    cells, _ = summary.collect(root)
    expected = {(a, w, s) for a, w in summary.SERIES for s in summary.SEEDS}
    actual = {(c["result"]["arch"], c["result"]["d_sae"], c["result"]["seed"]) for c in cells}
    summary.require(actual == expected, "paired analysis requires all 15 completed v2 cells; missing " + str(sorted(expected - actual)))
    common, predictions, recorded, receipts = None, {}, {}, []
    array_fields = ("labels", "question_ids", "canonical_group_ids", "sentence_keys", "fold_id", "S_grid")
    for cell in cells:
        result = cell["result"]
        source = root / cell["source_path"]
        sidecar = result["oof_predictions"]
        summary.require(Path(sidecar["path"]).name == sidecar["path"], "OOF path must be a sibling filename")
        path = source.with_name(sidecar["path"])
        summary.require(path.is_file() and sha256(path) == sidecar["sha256"], f"missing or modified OOF: {path}")
        with np.load(path, allow_pickle=False) as archive:
            current = {name: archive[name] for name in array_fields}
            if common is None:
                common = current
            else:
                for name in array_fields:
                    summary.require(np.array_equal(common[name], current[name]), f"OOF {name} differs in {source}")
            key_hash = hashlib.sha256(json.dumps(current["sentence_keys"].tolist()).encode()).hexdigest()
            summary.require(key_hash == result["provenance"]["sentence_keys_sha256"], f"OOF sentence-key hash differs in {source}")
            summary.require(np.array_equal(current["S_grid"], summary.S_GRID), "OOF S grid differs")
            for profile in (BASE, *COMPARATORS):
                arch, width, view = profile
                if (arch, width) != (result["arch"], result["d_sae"]):
                    continue
                for mode in summary.MODES:
                    values = archive[f"{view}__{mode}"]
                    summary.require(values.shape == (25204, 6) and np.isfinite(values).all()
                                    and ((0 <= values) & (values <= 1)).all(), "invalid primary OOF probabilities")
                    for budget in BUDGETS:
                        key = (profile, mode, budget, result["seed"])
                        summary.require(key not in predictions, f"duplicated prediction profile: {key}")
                        predictions[key] = values[:, list(summary.S_GRID).index(budget)].copy()
                        recorded[key] = result["views"][view]["probes"][mode]["metrics"][str(budget)]
        receipts.append({"cell": summary.cell_id(result["arch"], result["d_sae"], result["seed"]),
                         "detection_json": cell["source_path"], "detection_sha256": cell["source_sha256"],
                         "oof_path": str(path.relative_to(root)), "oof_sha256": sidecar["sha256"],
                         "evaluator_sha256": result["provenance"]["evaluator_sha256"]})
    labels = common["labels"]
    summary.require(labels.shape == (25204,) and np.isin(labels, (0, 1)).all() and labels.sum() == 3169,
                    "OOF labels differ from the frozen cohort")
    summary.require(all(common[name].shape == labels.shape for name in array_fields if name != "S_grid"),
                    "OOF row metadata shape differs")
    summary.require(len(np.unique(common["question_ids"])) == 300
                    and len(np.unique(common["canonical_group_ids"])) == 213, "OOF group counts differ")
    summary.require(set(np.unique(common["fold_id"])) == set(range(5)), "OOF must contain exactly five folds")
    _, groups = np.unique(common["canonical_group_ids"], return_inverse=True)
    group_fold = np.empty(213, dtype=np.int16)
    for group in range(213):
        folds = np.unique(common["fold_id"][groups == group])
        summary.require(len(folds) == 1, "a canonical problem group crosses OOF folds")
        group_fold[group] = folds[0]
    common.update(group_index=groups, group_fold=group_fold)
    warnings = sum(c["result"]["convergence_warning_count"] for c in cells)
    return common, predictions, recorded, receipts, warnings


def bootstrap(common: dict, predictions: dict, recorded: dict, *, n_bootstrap: int,
              seed: int, chunk_size: int) -> tuple[dict, dict, dict]:
    """Reuse one cluster-resampling matrix for every method and dictionary seed."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_bootstrap, 213), dtype=np.int32)
    fold_groups = []
    for fold in range(5):
        groups = np.flatnonzero(common["group_fold"] == fold)
        summary.require(len(groups) > 0, "empty fold")
        counts[:, groups] = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=n_bootstrap)
        fold_groups.append(len(groups))
    points, resampled = {}, {}
    zero_positive = []
    unit_counts = np.ones((1, 213), dtype=np.int32)
    for fold in range(5):
        rows = np.flatnonzero(common["fold_id"] == fold)
        positives_per_group = np.bincount(common["group_index"][rows], weights=common["labels"][rows], minlength=213)
        zero_positive.append(int(np.count_nonzero(counts @ positives_per_group == 0)))
        for key, scores in predictions.items():
            prepared = prepare_ap(common["labels"][rows], scores[rows], common["group_index"][rows])
            point = float(weighted_ap(prepared, unit_counts)[0])
            summary.require(np.isclose(point, recorded[key]["fold_average_precision"][fold], rtol=0, atol=1e-10),
                            f"OOF AP does not reproduce the saved fold metric: {key}, fold{fold}")
            points[key] = points.get(key, 0.0) + point / 5
            accumulator = resampled.setdefault(key, np.zeros(n_bootstrap, dtype=np.float64))
            for start in range(0, n_bootstrap, chunk_size):
                stop = min(start + chunk_size, n_bootstrap)
                accumulator[start:stop] += weighted_ap(prepared, counts[start:stop]) / 5
        print(json.dumps({"phase": "paired_bootstrap", "fold_complete": fold + 1, "folds": 5}), flush=True)
    for key, point in points.items():
        summary.require(np.isclose(point, recorded[key][summary.METRIC], rtol=0, atol=1e-10),
                        f"replayed mean-fold AP differs: {key}")
    return points, resampled, {"fold_prompt_counts": fold_groups,
        "zero_positive_bootstrap_samples_per_fold": zero_positive,
        "resampling_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest()}


def contrasts(points: dict, resampled: dict, *, n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for comparator in COMPARATORS:
        arch, width, view = comparator
        for mode in summary.MODES:
            for budget in BUDGETS:
                base = np.array([points[(BASE, mode, budget, s)] for s in summary.SEEDS])
                other = np.array([points[(comparator, mode, budget, s)] for s in summary.SEEDS])
                deltas = base - other
                bootstrap_deltas = np.mean([resampled[(BASE, mode, budget, s)]
                                             - resampled[(comparator, mode, budget, s)] for s in summary.SEEDS], axis=0)
                low, high = np.quantile(bootstrap_deltas, (.025, .975), method="linear")
                rows.append({"comparison_role": "T-SAE 16K sensitivity" if width == 16384 else "32K core",
                    "comparator_arch": arch, "comparator_width": width, "comparator_view": view,
                    "probe_mode": mode, "S": budget, "txc_mean_fold_ap": float(base.mean()),
                    "comparator_mean_fold_ap": float(other.mean()), "paired_delta_mean": float(deltas.mean()),
                    **{f"delta_seed{s}": float(value) for s, value in zip(summary.SEEDS, deltas)},
                    "paired_delta_seed_sample_sd": float(deltas.std(ddof=1)),
                    "conditional_ci_low": float(low), "conditional_ci_high": float(high),
                    "bootstrap_mean": float(bootstrap_deltas.mean()), "bootstrap_sd": float(bootstrap_deltas.std(ddof=1)),
                    "n_bootstrap": n_bootstrap, "bootstrap_seed": seed, "n_prompt_groups": 213,
                    "n_folds": 5, "n_dictionary_seeds": 3})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/backtracking/results"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=32, help="Bound temporary weighted-AP matrices.")
    args = parser.parse_args(argv)
    summary.require(args.n_bootstrap >= 100 and 1 <= args.chunk_size <= 256, "require at least 100 replicates and chunk size 1–256")
    root = args.root.resolve()
    common, predictions, recorded, receipts, warnings = load_inputs(root)
    points, resampled, audit = bootstrap(common, predictions, recorded, n_bootstrap=args.n_bootstrap,
                                        seed=args.bootstrap_seed, chunk_size=args.chunk_size)
    rows = contrasts(points, resampled, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)
    output = args.output_dir.resolve() if args.output_dir else root / "publication" / "paired_detection"
    output.mkdir(parents=True, exist_ok=True)
    partitions = {"core32k": [r for r in rows if r["comparator_width"] == 32768],
                  "tsae16k_sensitivity": [r for r in rows if r["comparator_width"] == 16384]}
    for name, partition in partitions.items():
        with (output / f"paired_detection_{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(partition)
    payload = {"schema": "c7-paired-detection-v1", "detection_protocol": summary.PROTOCOL,
        "status": "complete_with_warnings" if warnings else "complete", "metric": summary.METRIC,
        "difference": "TXC-base 32K minus comparator; equal mean of five fold APs and three fixed dictionary seeds",
        "grouping": summary.GROUPING, "n_prompt_groups": 213, "n_original_question_ids": 300,
        "n_sentences": 25204, "S_grid": list(BUDGETS), "probe_modes": list(summary.MODES),
        "dictionary_seeds": list(summary.SEEDS), "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.bootstrap_seed, "interval": "95% percentile, NumPy linear quantiles",
        "conditioning": "Fixed trained dictionaries, fixed fitted probes, fixed OOF folds and the observed seed trio; no refitting or seed resampling.",
        "resampling": "Canonical prompt clusters sampled with replacement independently within each original fold; shared multiplicities across every method, seed, mode and S.",
        "seed_uncertainty": "paired_delta_seed_sample_sd is a separate sample SD across seeds 1/2/42, not the bootstrap CI",
        "limitations": ["Conditional evaluation-sample uncertainty; not total retraining/probe-selection uncertainty.",
                        "Prompt normalization groups exact normalized text, not semantic paraphrases.",
                        "Pointwise intervals across the prespecified comparisons; no multiple-comparison correction.",
                        "No-positive resampled folds have AP=0, matching sklearn average precision."],
        "convergence_warning_count": warnings, "audit": audit, "sources": receipts,
        "paired_script_sha256": sha256(Path(__file__)), "numpy_version": np.__version__, **partitions}
    (output / "paired_detection.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text("""# Paired Backtracking detection contrasts

The two CSVs separate the 32K core from the 16K T-SAE width sensitivity. Positive differences favor TXC-base. Raw-C1 and scaled-C1 remain distinct rows; S=8 and S=32 are prespecified. All 15 completed v2 cells and their hash-verified OOF files are required. Original question-ID diagnostics are excluded.

For each bootstrap draw, 213 canonical prompt groups are resampled with replacement **within their original five folds**. Every sentence in a selected group shares its multiplicity, and the same draws are used for all methods and all three dictionary seeds. Weighted average precision handles tied scores exactly. The statistic averages five fold AP differences and then the three fixed seed differences, preserving the declared metric rather than substituting pooled OOF AP.

The 95% percentile interval measures **conditional evaluation-sample uncertainty** for these already trained dictionaries and fitted probes. There is no refitting, feature reselection, fold reassignment or training-seed resampling. It is not a confidence interval over training seeds or the full learning procedure. `paired_delta_seed_sample_sd` separately reports variation in the paired differences across seeds 1/2/42. Intervals are pointwise and are not corrected for multiple comparisons.

Every saved fold AP is replayed from its OOF predictions before bootstrapping. Labels, sentence order, canonical group identities and fold assignments must agree across all cells; no canonical group may cross folds. The JSON records hashes, resampling seed/count, zero-positive resamples and convergence warnings. Review any convergence warnings before publication.

Regenerate with `python paired_detection.py --root /workspace/backtracking/results`. Default: 1000 deterministic replicates, seed 42. Use `--n-bootstrap 2000` for a larger final run. No GPU, checkpoint download, inference, judging or plot generation occurs here. The Nord seed/SD plots are generated separately by `summarize_detection.py`; their bands must not be relabeled as these bootstrap intervals.
""")
    print(json.dumps({"status": payload["status"], "comparisons": len(rows),
                      "n_bootstrap": args.n_bootstrap, "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
