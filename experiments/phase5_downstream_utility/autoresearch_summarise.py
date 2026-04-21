"""Autoresearch val-delta reporter.

Reads probing_results.jsonl and reports, for each `aggregation=*_val`
arch, mean AUC across the 36 probing tasks at each k, and the delta
vs its baseline arch.

Phase 5.7 plan §"Candidate scoring":
    Flag finalist if Δ_val > +0.010 (1 pp).
    Discard if Δ_val < -0.020.
    Retry at 50k steps if -0.020 < Δ_val < +0.010 (ambiguous zone).

Baseline mapping:
    txcdr_contrastive_t5 -> txcdr_t5
    txcdr_rotational_t5 -> txcdr_t5
    matryoshka_txcdr_contrastive_t5 -> matryoshka_t5
    mlc_temporal_t3 -> mlc
    txcdr_basis_expansion_t5 -> txcdr_t5
    time_layer_contrastive_t5 -> time_layer_crosscoder_t5
    (others -> txcdr_t5 by default)

Also emits an autoresearch_index.jsonl row summarising the candidate.

Usage:
    .venv/bin/python -m experiments.phase5_downstream_utility.autoresearch_summarise \\
        --candidate txcdr_contrastive_t5 [--seed 42] [--k 5] [--write-index]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path("/workspace/temp_xc")
RESULTS = REPO / "experiments/phase5_downstream_utility/results"
JSONL = RESULTS / "probing_results.jsonl"
INDEX = RESULTS / "autoresearch_index.jsonl"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}

DEFAULT_BASE = {
    "txcdr_contrastive_t5": "txcdr_t5",
    "txcdr_rotational_t5": "txcdr_t5",
    "txcdr_basis_expansion_t5": "txcdr_t5",
    "txcdr_film_t5": "txcdr_t5",
    "txcdr_smoothness_t5": "txcdr_t5",
    "txcdr_dynamics_t5": "txcdr_t5",
    "matryoshka_txcdr_contrastive_t5": "matryoshka_t5",
    "matryoshka_feature_idx_t5": "matryoshka_t5",
    "mlc_temporal_t3": "mlc",
    "time_layer_contrastive_t5": "time_layer_crosscoder_t5",
}


def _records():
    rows = []
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _val_per_task(rows, arch: str, seed: int, k: int):
    """Return {task: AUC} for last_position_val rows of (arch, seed, k)."""
    out: dict[str, float] = {}
    rid_target = f"{arch}__seed{seed}"
    for r in rows:
        if r.get("error") or r.get("test_auc") is None:
            continue
        if r.get("aggregation") != "last_position_val":
            continue
        if r.get("k_feat") != k:
            continue
        if r.get("run_id") != rid_target:
            continue
        task = r.get("task_name")
        v = float(r["test_auc"])
        if task in FLIP_TASKS:
            v = max(v, 1.0 - v)
        out[task] = v   # last-write-wins on duplicates
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="arch name e.g. txcdr_contrastive_t5")
    ap.add_argument("--base", default=None,
                    help="baseline arch (defaults from DEFAULT_BASE map)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5,
                    help="k_feat to summarise (default 5)")
    ap.add_argument("--write-index", action="store_true",
                    help="append a summary row to autoresearch_index.jsonl")
    ap.add_argument("--notes", default="",
                    help="free-text notes for the index row")
    args = ap.parse_args()

    base = args.base or DEFAULT_BASE.get(args.candidate, "txcdr_t5")
    rows = _records()
    cand_per_task = _val_per_task(rows, args.candidate, args.seed, args.k)
    base_per_task = _val_per_task(rows, base, args.seed, args.k)

    if not cand_per_task:
        print(f"[FATAL] no last_position_val rows for "
              f"{args.candidate}__seed{args.seed} k={args.k}")
        return
    if not base_per_task:
        print(f"[WARN] no last_position_val rows for "
              f"{base}__seed{args.seed} k={args.k}; "
              "report cand-only; train baseline at last_position_val first.")

    common = sorted(set(cand_per_task) & set(base_per_task))
    cand_mean = float(np.mean(list(cand_per_task.values())))
    cand_std = float(np.std(list(cand_per_task.values())))
    if base_per_task:
        base_vals = np.array([base_per_task[t] for t in common])
        cand_vals_common = np.array([cand_per_task[t] for t in common])
        delta_per_task = cand_vals_common - base_vals
        delta_mean = float(delta_per_task.mean())
        delta_std = float(delta_per_task.std() / np.sqrt(len(delta_per_task)))
        wins = int((delta_per_task > 0.005).sum())
        losses = int((delta_per_task < -0.005).sum())
        ties = len(delta_per_task) - wins - losses
        # paired t-test (manual; no scipy dep)
        t_stat = float(delta_mean / (delta_per_task.std() /
                                     np.sqrt(len(delta_per_task))) if delta_per_task.std() > 0 else 0.0)
    else:
        delta_mean = float("nan")
        delta_std = float("nan")
        wins = losses = ties = 0
        t_stat = float("nan")

    # Verdict per plan thresholds
    if delta_mean > 0.010:
        verdict = "FINALIST"
    elif delta_mean < -0.020:
        verdict = "DISCARD"
    else:
        verdict = "AMBIGUOUS (retry at 50k steps recommended)"

    print()
    print(f"=== {args.candidate} (seed={args.seed}, k={args.k}, agg=last_position_val) ===")
    print(f"  candidate mean_val_AUC = {cand_mean:.4f}  ±{cand_std:.4f}  "
          f"(n={len(cand_per_task)} tasks)")
    if base_per_task:
        print(f"  baseline ({base})   = "
              f"{float(np.mean(list(base_per_task.values()))):.4f}  "
              f"(n={len(base_per_task)} tasks)")
        print(f"  paired Δ_val          = {delta_mean:+.4f}  "
              f"±{delta_std:.4f}  (n_paired={len(delta_per_task)})  "
              f"t={t_stat:.2f}")
        print(f"  wins/ties/losses (|Δ|>0.005) = {wins} / {ties} / {losses}")
    print(f"  VERDICT: {verdict}")

    if args.write_index:
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "candidate": args.candidate,
            "base": base,
            "seed": args.seed,
            "k": args.k,
            "n_tasks_cand": len(cand_per_task),
            "n_tasks_base": len(base_per_task),
            "n_paired": len(common),
            "mean_val_auc_cand": cand_mean,
            "mean_val_auc_base": (
                float(np.mean(list(base_per_task.values()))) if base_per_task else None
            ),
            "delta_val_mean": delta_mean,
            "delta_val_stderr": delta_std,
            "delta_val_t": t_stat,
            "wins": wins, "ties": ties, "losses": losses,
            "verdict": verdict,
            "notes": args.notes,
        }
        with INDEX.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  -> wrote row to {INDEX}")


if __name__ == "__main__":
    main()
