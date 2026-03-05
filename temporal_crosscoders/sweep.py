#!/usr/bin/env python3
"""
sweep.py — Run the full (dataset × k × T) sweep.

Usage:
    # Full sweep (all combos):
    python sweep.py

    # Single run (for tmux parallelism):
    python sweep.py --dataset iid --k 4 --T 8 --model sae
    python sweep.py --dataset markov --k 4 --T 8 --model txcdr

    # Only SAE runs (useful since SAE is T-independent):
    python sweep.py --model sae

    # Only TXCDR runs:
    python sweep.py --model txcdr

    # Quick test (1000 steps):
    python sweep.py --steps 1000
"""

import argparse
import itertools
import json
import os
import sys
import time

import torch

from config import (
    DATASETS, SWEEP_K, SWEEP_T, TRAIN_STEPS, LOG_DIR,
    DEVICE, SEED, should_skip, NUM_FEATS,
)
from data import build_toy_model, get_true_features, CachedDataSource
from train import train_sae, train_txcdr


def parse_args():
    p = argparse.ArgumentParser(description="Temporal Crosscoder sweep")
    p.add_argument("--dataset", type=str, default=None,
                   choices=DATASETS, help="Single dataset (default: all)")
    p.add_argument("--k", type=int, default=None,
                   help="Single k value (default: sweep all)")
    p.add_argument("--T", type=int, default=None,
                   help="Single T value (default: sweep all)")
    p.add_argument("--model", type=str, default=None,
                   choices=["sae", "txcdr", "both"],
                   help="Which model to run (default: both)")
    p.add_argument("--steps", type=int, default=None,
                   help=f"Override TRAIN_STEPS (default: {TRAIN_STEPS})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sweep plan without training")
    return p.parse_args()


def build_sweep_plan(args) -> list[dict]:
    """
    Build the list of (model_type, dataset, k, T) jobs.

    Key optimization: SAE is T-independent, so we only run it once per
    (dataset, k) pair with the smallest T in the grid.  All TXCDR runs
    reference that SAE baseline.
    """
    datasets = [args.dataset] if args.dataset else DATASETS
    ks = [args.k] if args.k else SWEEP_K
    Ts = [args.T] if args.T else SWEEP_T
    model_filter = args.model or "both"

    jobs = []
    sae_done = set()  # (dataset, k) already scheduled

    for k, T, dataset in itertools.product(ks, Ts, datasets):
        if should_skip(k, T):
            print(f"  SKIP  k={k} × T={T} = {k*T} >= {NUM_FEATS}")
            continue

        # SAE: one per (dataset, k) — use T for data gen but result is T-agnostic
        if model_filter in ("sae", "both"):
            sae_key = (dataset, k)
            if sae_key not in sae_done:
                jobs.append(dict(model="sae", dataset=dataset, k=k, T=T))
                sae_done.add(sae_key)

        # TXCDR: one per (dataset, k, T)
        if model_filter in ("txcdr", "both"):
            jobs.append(dict(model="txcdr", dataset=dataset, k=k, T=T))

    return jobs


def main():
    args = parse_args()
    n_steps = args.steps or TRAIN_STEPS

    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(SEED)

    jobs = build_sweep_plan(args)
    total = len(jobs)

    print("=" * 72)
    print(f"  TEMPORAL CROSSCODER SWEEP")
    print(f"  Device: {DEVICE}")
    print(f"  Steps per run: {n_steps:,}")
    print(f"  Total jobs: {total}")
    print("=" * 72)

    if args.dry_run:
        for i, j in enumerate(jobs, 1):
            print(f"  [{i:3d}/{total}]  {j['model']:5s}  dataset={j['dataset']:8s}  k={j['k']}  T={j['T']}")
        print("\n  (dry run — no training)")
        return

    # Build the shared toy model ONCE
    print("\nBuilding toy model...")
    toy_model = build_toy_model(seed=SEED)
    true_features = get_true_features(toy_model)
    print(f"  True features shape: {true_features.shape}")

    # Build one CachedDataSource per dataset (shared across all k, T runs)
    datasets_in_sweep = sorted(set(j["dataset"] for j in jobs))
    caches: dict[str, CachedDataSource] = {}
    for ds in datasets_in_sweep:
        print(f"  Generating cached long chains for '{ds}'...")
        caches[ds] = CachedDataSource(ds, toy_model)
        print(f"    {caches[ds].act_chains.shape} activations cached on {DEVICE}")

    os.makedirs(LOG_DIR, exist_ok=True)

    # Summary collector
    summary_rows = []
    t_sweep_start = time.time()

    for i, job in enumerate(jobs, 1):
        model_type = job["model"]
        dataset = job["dataset"]
        k = job["k"]
        T = job["T"]

        print(f"\n{'─' * 72}")
        print(f"  [{i}/{total}]  {model_type.upper()}  dataset={dataset}  k={k}  T={T}")
        print(f"{'─' * 72}")

        t0 = time.time()

        if model_type == "sae":
            model, history = train_sae(
                dataset, k, T, toy_model, true_features,
                cache=caches[dataset], n_steps=n_steps,
            )
        else:
            model, history = train_txcdr(
                dataset, k, T, toy_model, true_features,
                cache=caches[dataset], n_steps=n_steps,
            )

        elapsed = time.time() - t0
        final = history[-1] if history else {}

        row = {
            "model": model_type,
            "dataset": dataset,
            "k": k,
            "T": T,
            "final_auc": final.get("auc", 0),
            "final_loss": final.get("loss", 0),
            "final_r90": final.get("recovery_90", 0),
            "elapsed_sec": round(elapsed, 1),
        }
        summary_rows.append(row)

        print(f"  Done in {elapsed:.0f}s | AUC={row['final_auc']:.4f} "
              f"| loss={row['final_loss']:.4f} | R@0.9={row['final_r90']:.2f}")

    # Save sweep summary
    summary_path = os.path.join(LOG_DIR, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    total_time = time.time() - t_sweep_start
    print(f"\n{'=' * 72}")
    print(f"  SWEEP COMPLETE  |  {total} jobs  |  {total_time/3600:.1f}h total")
    print(f"  Summary: {summary_path}")
    print(f"  Run `python viz.py` to generate visualizations")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
