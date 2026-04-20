"""Probe fit+eval on cached features — decoupled from SAE/arch training.

Reads per-(run, task, aggregation) feature caches produced by
`extract_features.py` (scipy CSR format), fits top-k class-separation
+ L1 logistic regression probes, and writes `probing_results.jsonl`
with both test AUC and test accuracy. Also emits the two required
baselines (raw-last-token L2-LR, attention-pooled Eq. 2) per task —
these run directly from the probe_cache.

Re-running this script does NOT re-encode SAE features. That makes
iterating on the probe fit / metric / k_feat cheap (~minutes) instead
of tens of minutes per arch.

Output JSONL schema:
    {run_id, arch, task_name, dataset_key, aggregation, k_feat,
     test_auc, test_acc, n_train, n_test, elapsed_s}
    baselines: run_id=BASELINE_*, arch=baseline_*, k_feat=null.

Usage:
    PHASE5_REPO=/home/elysium/temp_xc TQDM_DISABLE=1 \
      .venv/bin/python experiments/phase5_downstream_utility/probing/fit_probes.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase5_downstream_utility.probing.run_probing import (
    _load_task_cache,
    last_token_lr_metrics,
    attn_pool_metrics,
    _last_token,
    K_VALUES,
)


REPO = Path(os.environ.get("PHASE5_REPO", Path(__file__).resolve().parents[3]))
PROBE_CACHE = REPO / "experiments/phase5_downstream_utility/results/probe_cache"
FEATURE_CACHE = REPO / "experiments/phase5_downstream_utility/results/feature_cache"
OUT_JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"


def _load_feat(path: Path) -> dict:
    z = np.load(path)
    Ztr = scipy.sparse.csr_matrix(
        (z["Ztr_data"].astype(np.float32), z["Ztr_idx"], z["Ztr_ptr"]),
        shape=tuple(z["Ztr_shape"]),
    )
    Zte = scipy.sparse.csr_matrix(
        (z["Zte_data"].astype(np.float32), z["Zte_idx"], z["Zte_ptr"]),
        shape=tuple(z["Zte_shape"]),
    )
    return {
        "Z_train": Ztr,
        "Z_test":  Zte,
        "y_train": z["y_train"],
        "y_test":  z["y_test"],
    }


def top_k_by_class_sep_sparse(Z_train: scipy.sparse.csr_matrix,
                              y_train: np.ndarray, k: int) -> np.ndarray:
    """Top-k features by |mean_pos - mean_neg| on sparse Z_train."""
    pos_mask = (y_train == 1)
    neg_mask = (y_train == 0)
    n_pos = max(1, int(pos_mask.sum()))
    n_neg = max(1, int(neg_mask.sum()))
    pos_mean = np.asarray(Z_train[pos_mask].sum(axis=0)).ravel() / n_pos
    neg_mean = np.asarray(Z_train[neg_mask].sum(axis=0)).ravel() / n_neg
    diff = np.abs(pos_mean - neg_mean)
    k = min(k, Z_train.shape[1])
    top = np.argpartition(-diff, k - 1)[:k]
    return np.sort(top)


def sae_probe_metrics_sparse(
    Z_train: scipy.sparse.csr_matrix, y_train: np.ndarray,
    Z_test: scipy.sparse.csr_matrix, y_test: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Sparse-aware top-k class-sep + L1 LR. Returns (auc, acc)."""
    if Z_train.shape[0] < 5 or len(np.unique(y_train)) < 2:
        return 0.5, 0.5
    idx = top_k_by_class_sep_sparse(Z_train, y_train, k)
    Xtr = Z_train[:, idx].toarray()
    Xte = Z_test[:, idx].toarray()
    scaler = StandardScaler(with_mean=False)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    try:
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000, C=1.0,
        )
        clf.fit(Xtr_s, y_train)
    except Exception:
        return 0.5, 0.5
    if len(np.unique(y_test)) < 2:
        return 0.5, 0.5
    preds_score = clf.decision_function(Xte_s)
    preds_cls = clf.predict(Xte_s)
    return (
        float(roc_auc_score(y_test, preds_score)),
        float(accuracy_score(y_test, preds_cls)),
    )


def _arch_from_run(run_id: str) -> str:
    return run_id.split("__", 1)[0]


def _read_existing_keys(jsonl_path: Path) -> set[tuple]:
    if not jsonl_path.exists():
        return set()
    keys = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            keys.add((
                r.get("run_id"), r.get("task_name"),
                r.get("aggregation"), r.get("k_feat"),
            ))
    return keys


def run_baselines(
    task_names: list[str] | None,
    aggregations: tuple[str, ...],
    out_f,
    existing_keys: set[tuple],
    dataset_keys: dict[str, str],
    device: torch.device,
) -> None:
    task_dirs = [
        d for d in sorted(PROBE_CACHE.iterdir())
        if d.is_dir()
        and (d / "acts_anchor.npz").exists()
        and (d / "acts_mlc.npz").exists()
    ]
    if task_names:
        task_dirs = [d for d in task_dirs if d.name in task_names]
    for d in task_dirs:
        task_name = d.name
        tc = _load_task_cache(d)
        dataset_keys[task_name] = tc["meta"]["dataset_key"]
        for aggregation in aggregations:
            key = ("BASELINE_last_token_lr", task_name, aggregation, None)
            if key not in existing_keys:
                X_last_tr = _last_token(tc["anchor_train"], tc["train_last_idx"])
                X_last_te = _last_token(tc["anchor_test"], tc["test_last_idx"])
                t0 = time.time()
                auc, acc = last_token_lr_metrics(
                    X_last_tr, tc["train_labels"],
                    X_last_te, tc["test_labels"],
                )
                out_f.write(json.dumps({
                    "run_id": "BASELINE_last_token_lr",
                    "arch": "baseline_last_token_lr",
                    "task_name": task_name,
                    "dataset_key": tc["meta"]["dataset_key"],
                    "aggregation": aggregation,
                    "k_feat": None,
                    "test_auc": auc, "test_acc": acc,
                    "n_train": int(tc["train_labels"].size),
                    "n_test": int(tc["test_labels"].size),
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                out_f.flush()
                print(f"  {task_name} [{aggregation}] last-LR  auc={auc:.4f} acc={acc:.4f}")

            key = ("BASELINE_attn_pool", task_name, aggregation, None)
            if key not in existing_keys:
                t0 = time.time()
                auc, acc = attn_pool_metrics(
                    tc["anchor_train"], tc["train_last_idx"], tc["train_labels"],
                    tc["anchor_test"], tc["test_last_idx"], tc["test_labels"],
                    device,
                )
                out_f.write(json.dumps({
                    "run_id": "BASELINE_attn_pool",
                    "arch": "baseline_attn_pool",
                    "task_name": task_name,
                    "dataset_key": tc["meta"]["dataset_key"],
                    "aggregation": aggregation,
                    "k_feat": None,
                    "test_auc": auc, "test_acc": acc,
                    "n_train": int(tc["train_labels"].size),
                    "n_test": int(tc["test_labels"].size),
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                out_f.flush()
                print(f"  {task_name} [{aggregation}] attn-pool auc={auc:.4f} acc={acc:.4f}")


def fit_probes(
    run_ids: list[str] | None = None,
    task_names: list[str] | None = None,
    aggregations: tuple[str, ...] = ("last_position", "full_window"),
    k_values: tuple[int, ...] = tuple(K_VALUES),
    include_baselines: bool = True,
    reset: bool = False,
) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if reset and OUT_JSONL.exists():
        OUT_JSONL.unlink()
    existing_keys = _read_existing_keys(OUT_JSONL)
    dataset_keys: dict[str, str] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with OUT_JSONL.open("a") as out_f:
        if include_baselines:
            run_baselines(
                task_names, aggregations, out_f, existing_keys,
                dataset_keys, device,
            )

        run_dirs = sorted(FEATURE_CACHE.iterdir()) if FEATURE_CACHE.exists() else []
        if run_ids:
            run_dirs = [d for d in run_dirs if d.name in run_ids]

        for run_dir in run_dirs:
            run_id = run_dir.name
            arch = _arch_from_run(run_id)
            files = sorted(run_dir.glob("*.npz"))
            print(f"=== {run_id} ({arch})  {len(files)} feature files ===")
            for fp in files:
                stem = fp.stem
                if "__" not in stem:
                    continue
                task_name, aggregation = stem.rsplit("__", 1)
                if aggregation not in aggregations:
                    continue
                if task_names and task_name not in task_names:
                    continue

                all_in = all(
                    (run_id, task_name, aggregation, k) in existing_keys
                    for k in k_values
                )
                if all_in:
                    continue

                try:
                    feat = _load_feat(fp)
                except Exception as e:
                    print(f"  {task_name} [{aggregation}]: load FAIL: {e}")
                    continue
                if task_name not in dataset_keys:
                    meta_path = PROBE_CACHE / task_name / "meta.json"
                    if meta_path.exists():
                        dataset_keys[task_name] = json.loads(
                            meta_path.read_text()
                        )["dataset_key"]
                    else:
                        dataset_keys[task_name] = task_name

                Ztr, Zte = feat["Z_train"], feat["Z_test"]
                ytr, yte = feat["y_train"], feat["y_test"]
                for k in k_values:
                    if (run_id, task_name, aggregation, k) in existing_keys:
                        continue
                    t0 = time.time()
                    auc, acc = sae_probe_metrics_sparse(Ztr, ytr, Zte, yte, k)
                    out_f.write(json.dumps({
                        "run_id": run_id, "arch": arch,
                        "task_name": task_name,
                        "dataset_key": dataset_keys.get(task_name, task_name),
                        "aggregation": aggregation,
                        "k_feat": k,
                        "test_auc": auc, "test_acc": acc,
                        "n_train": int(ytr.size), "n_test": int(yte.size),
                        "elapsed_s": time.time() - t0,
                    }) + "\n")
                    out_f.flush()
                print(
                    f"  {task_name} [{aggregation}]: "
                    f"d_feat={Ztr.shape[1]}  nnz={Ztr.nnz}"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument(
        "--aggregations", nargs="+",
        default=["last_position", "full_window"],
    )
    ap.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument(
        "--reset", action="store_true",
        help="delete the existing probing_results.jsonl before writing",
    )
    args = ap.parse_args()
    fit_probes(
        run_ids=args.run_ids,
        task_names=args.tasks,
        aggregations=tuple(args.aggregations),
        k_values=tuple(args.k_values),
        include_baselines=not args.skip_baselines,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
