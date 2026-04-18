"""Shared probe-fit utility — one-vs-rest binary logistic regression
with per-example prediction persistence + item-8 sanity check.

Used by both `mlc_probing.py` (our custom MLC path) and
`probing_runner.py` (the SAE/TempXC path, as a second probe fit to
emit predictions alongside SAEBench's stock aggregate reporting).

Sklearn config mirrors what SAEBench's stock probing uses:
  - LogisticRegression with default solver (lbfgs), max_iter=1000
  - Deterministic at fixed random_state

See:
  - docs/aniket/bench_harness/bug_ledger_check.md § Item 8
  - eval_infra_lessons.md L6 (encode-once, mask by class)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


PREDICTIONS_DIR = "results/saebench/predictions"


def fit_one_vs_rest_probes(
    train_feats: np.ndarray,          # (N_train, d_sae_eff)
    train_cls: np.ndarray,            # (N_train,) class indices
    test_feats: np.ndarray,           # (N_test, d_sae_eff)
    test_cls: np.ndarray,             # (N_test,) class indices
    class_names: list[str],
    k_values: tuple[int, ...],
    dataset_name: str,
    random_seed: int = 42,
) -> tuple[dict[str, float], list[dict]]:
    """One-vs-rest logistic regression per class per k.

    Returns:
        aggregate_accs: {sae_top_K_test_accuracy: float}
        per_example_preds: list of {example_id, class, k, pred, prob, label}
            suitable for persistence.
    """
    per_class_test_accs: dict[int, list[float]] = {kv: [] for kv in k_values}
    per_example_preds: list[dict] = []

    for class_idx, class_name in enumerate(class_names):
        train_labels = (train_cls == class_idx).astype(np.int64)
        test_labels = (test_cls == class_idx).astype(np.int64)

        for kv in k_values:
            # Class-separation score: |mean(pos) - mean(neg)| per feature.
            pos_mean = train_feats[train_labels == 1].mean(axis=0)
            neg_mean = train_feats[train_labels == 0].mean(axis=0)
            score = np.abs(pos_mean - neg_mean)
            top_k_feat_idx = np.argsort(-score)[:kv]

            X_train = train_feats[:, top_k_feat_idx]
            X_test = test_feats[:, top_k_feat_idx]

            probe = LogisticRegression(max_iter=1000, random_state=random_seed)
            probe.fit(X_train, train_labels)
            acc = probe.score(X_test, test_labels)
            per_class_test_accs[kv].append(float(acc))

            # Per-example predictions for the TEST split.
            test_preds = probe.predict(X_test)
            test_probs = probe.predict_proba(X_test)[:, 1]
            for eid_idx, (p, prob, y) in enumerate(
                zip(test_preds, test_probs, test_labels)
            ):
                per_example_preds.append({
                    "example_id": f"{dataset_name}__test_{eid_idx}",
                    "class": class_name,
                    "k": int(kv),
                    "pred": int(p),
                    "prob": float(prob),
                    "label": int(y),
                })

    aggregate_accs = {}
    for kv in k_values:
        if per_class_test_accs[kv]:
            aggregate_accs[f"sae_top_{kv}_test_accuracy"] = float(
                np.mean(per_class_test_accs[kv])
            )
    return aggregate_accs, per_example_preds


def write_predictions(
    run_id: str,
    dataset_name: str,
    rows: list[dict],
    example_id_to_text: dict[str, str],
) -> str:
    """Write preds + sibling texts.jsonl (text deduped across class/k).

    Storage: ~70 MB per (task, arch, aggregation). Total for a full
    10-arch × 4-aggregation × 2-shuffle × 8-task sweep: ~45 GB. See
    L6 in eval_infra_lessons.md for why this isn't duplicated text.
    """
    safe_task = dataset_name.replace("/", "_")
    out_dir = Path(PREDICTIONS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / f"{safe_task}.jsonl"
    with open(preds_path, "w") as fout:
        for r in rows:
            fout.write(json.dumps(r) + "\n")

    texts_path = out_dir / f"{safe_task}__texts.jsonl"
    if not texts_path.exists():
        with open(texts_path, "w") as fout:
            for eid, text in example_id_to_text.items():
                fout.write(json.dumps({"example_id": eid, "text": text}) + "\n")
    return str(preds_path)


def sanity_check_persistence(
    run_id: str,
    dataset_name: str,
    expected: dict,
    k_values: tuple[int, ...],
    rtol: float = 1e-10,
) -> None:
    """Recompute aggregate accuracy from persisted predictions, assert
    machine-precision match against the probing loop's reported values.

    Catches silent persistence-layer drift — predictions misaligned
    with labels, ordering drift between fit-time and save-time, etc.
    Raises AssertionError on mismatch; fail-fast.
    """
    safe_task = dataset_name.replace("/", "_")
    path = Path(PREDICTIONS_DIR) / run_id / f"{safe_task}.jsonl"
    if not path.exists():
        raise AssertionError(
            f"item 8 sanity check: predictions JSONL missing for "
            f"{run_id}/{safe_task}. Expected at {path}."
        )
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

    per_class: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        per_class[r["k"]][r["class"]].append(int(r["pred"] == r["label"]))

    for kv in k_values:
        key = f"sae_top_{kv}_test_accuracy"
        if key not in expected:
            continue
        per_class_accs = [np.mean(v) for v in per_class[kv].values() if v]
        if not per_class_accs:
            raise AssertionError(
                f"item 8: no persisted predictions for k={kv} on {dataset_name}."
            )
        recomputed = float(np.mean(per_class_accs))
        reported = float(expected[key])
        if abs(recomputed - reported) > rtol * max(1.0, abs(reported)):
            raise AssertionError(
                f"item 8 FAILED: persistence drift on {dataset_name} at k={kv}. "
                f"reported={reported:.12f}, recomputed={recomputed:.12f}, "
                f"|Δ|={abs(recomputed - reported):.3e}. "
                f"See bug_ledger_check.md § Item 8."
            )


def cross_check_probe_aggregates(
    ours: dict,
    saebench: dict,
    dataset_name: str,
    k_values: tuple[int, ...],
    rtol: float = 1e-2,
) -> None:
    """Compare our sklearn fit's aggregate to SAEBench's stock fit.

    Used on the hybrid SAE/TempXC path where we run both fits on the
    same cached activations. If aggregates disagree beyond `rtol`,
    something is structurally different (sklearn version, solver,
    feature selection order, etc) — fail fast with a clear message
    rather than persist predictions that don't match the reported
    accuracy. `rtol=1e-2` allows for minor numerical differences
    between our fit and SAEBench's.
    """
    for kv in k_values:
        key = f"sae_top_{kv}_test_accuracy"
        if key not in ours or key not in saebench:
            continue
        if saebench[key] is None or ours[key] is None:
            continue
        diff = abs(ours[key] - saebench[key])
        if diff > rtol * max(1.0, abs(saebench[key])):
            raise AssertionError(
                f"probe fit divergence on {dataset_name} at k={kv}: "
                f"SAEBench={saebench[key]:.6f}, ours={ours[key]:.6f}, "
                f"|Δ|={diff:.3e} (rtol={rtol}). Our per-example "
                f"predictions would not match the reported aggregate. "
                f"Align sklearn config in probe_fit.py with SAEBench's, "
                f"or loosen rtol if the spread is acceptable."
            )
