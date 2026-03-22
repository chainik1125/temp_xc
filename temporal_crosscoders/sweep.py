#!/usr/bin/env python3
"""
sweep.py — Run the full (rho × k × T × model) sweep.

Usage:
    # Full sweep (all combos):
    python sweep.py

    # Single run:
    python sweep.py --rho 0.9 --k 5 --T 2 --model stacked_sae

    # Only one model type:
    python sweep.py --model txcdr

    # Quick test (1000 steps):
    python sweep.py --steps 1000

    # Dry run:
    python sweep.py --dry-run
"""

import argparse
import itertools
import json
import os
import time

import torch

from config import (
    SWEEP_RHO, SWEEP_K, SWEEP_T, TRAIN_STEPS, LOG_DIR,
    DEVICE, SEED, should_skip, NUM_FEATS,
)
from data import build_toy_model, get_true_features, CachedDataSource
from train import train_stacked_sae, train_txcdr


def parse_args():
    p = argparse.ArgumentParser(description="Temporal Crosscoder sweep (v8: stacked SAE baseline)")
    p.add_argument("--rho", type=float, default=None,
                   help="Single rho value (default: sweep all)")
    p.add_argument("--k", type=int, default=None,
                   help="Single k value (default: sweep all)")
    p.add_argument("--T", type=int, default=None,
                   help="Single T value (default: sweep all)")
    p.add_argument("--model", type=str, default=None,
                   choices=["stacked_sae", "txcdr", "both"],
                   help="Which model to run (default: both)")
    p.add_argument("--steps", type=int, default=None,
                   help=f"Override TRAIN_STEPS (default: {TRAIN_STEPS})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sweep plan without training")
    return p.parse_args()


def build_sweep_plan(args) -> list[dict]:
    """Build the list of (model_type, rho, k, T) jobs."""
    rhos = [args.rho] if args.rho is not None else SWEEP_RHO
    ks = [args.k] if args.k else SWEEP_K
    Ts = [args.T] if args.T else SWEEP_T
    model_filter = args.model or "both"

    jobs = []

    for rho, k, T in itertools.product(rhos, ks, Ts):
        if should_skip(k, T):
            print(f"  SKIP  k={k} × T={T} = {k*T} > {NUM_FEATS}")
            continue

        if model_filter in ("stacked_sae", "both"):
            jobs.append(dict(model="stacked_sae", rho=rho, k=k, T=T))
        if model_filter in ("txcdr", "both"):
            jobs.append(dict(model="txcdr", rho=rho, k=k, T=T))

    return jobs


def main():
    args = parse_args()
    n_steps = args.steps or TRAIN_STEPS

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(SEED)

    jobs = build_sweep_plan(args)
    total = len(jobs)

    print("=" * 72)
    print(f"  TEMPORAL CROSSCODER SWEEP (v8: stacked SAE baseline)")
    print(f"  Device: {DEVICE}")
    print(f"  Steps per run: {n_steps:,}")
    print(f"  Total jobs: {total}")
    print(f"  Rho values: {sorted(set(j['rho'] for j in jobs))}")
    print("=" * 72)

    if args.dry_run:
        for i, j in enumerate(jobs, 1):
            print(f"  [{i:3d}/{total}]  {j['model']:12s}  rho={j['rho']:.1f}  k={j['k']}  T={j['T']}")
        print("\n  (dry run — no training)")
        return

    # Build the shared toy model ONCE
    print("\nBuilding toy model...")
    toy_model = build_toy_model(seed=SEED)
    true_features = get_true_features(toy_model)
    print(f"  True features shape: {true_features.shape}")

    # Build one CachedDataSource per rho (shared across all k, T runs)
    rhos_in_sweep = sorted(set(j["rho"] for j in jobs))
    caches: dict[float, CachedDataSource] = {}
    for rho in rhos_in_sweep:
        print(f"  Generating cached long chains for rho={rho:.1f}...")
        caches[rho] = CachedDataSource(rho, toy_model)
        print(f"    {caches[rho].act_chains.shape} activations cached on {DEVICE}")

    os.makedirs(LOG_DIR, exist_ok=True)

    summary_rows = []
    t_sweep_start = time.time()

    for i, job in enumerate(jobs, 1):
        model_type = job["model"]
        rho = job["rho"]
        k = job["k"]
        T = job["T"]

        print(f"\n{'─' * 72}")
        print(f"  [{i}/{total}]  {model_type.upper()}  rho={rho:.1f}  k={k}  T={T}")
        print(f"{'─' * 72}")

        t0 = time.time()

        if model_type == "stacked_sae":
            model, history = train_stacked_sae(
                rho, k, T, toy_model, true_features,
                cache=caches[rho], n_steps=n_steps,
            )
        else:
            model, history = train_txcdr(
                rho, k, T, toy_model, true_features,
                cache=caches[rho], n_steps=n_steps,
            )

        elapsed = time.time() - t0
        final = history[-1] if history else {}

        row = {
            "model": model_type,
            "rho": rho,
            "k": k,
            "T": T,
            "final_auc": final.get("auc", 0),
            "final_loss": final.get("loss", 0),
            "final_l0": final.get("window_l0", 0),
            "final_r90": final.get("recovery_90", 0),
            "elapsed_sec": round(elapsed, 1),
        }
        summary_rows.append(row)

        print(f"  Done in {elapsed:.0f}s | AUC={row['final_auc']:.4f} "
              f"| loss={row['final_loss']:.4f} | L0={row['final_l0']:.0f} "
              f"| R@0.9={row['final_r90']:.2f}")

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
