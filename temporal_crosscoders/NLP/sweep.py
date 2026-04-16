#!/usr/bin/env python3
"""
sweep.py — Run the full (layer x k x T x architecture) sweep over cached
Gemma 2 2B activations.

tmux-friendly: supports --tmux to generate a launch script, and --job-index
for individual job execution from tmux panes.

Usage:
    # Full sweep (all combos):
    python sweep.py

    # Single run:
    python sweep.py --layer mid_res --k 50 --T 10 --model stacked_sae

    # Subset:
    python sweep.py --layer mid_res final_res --k 50 100

    # Quick test (1000 steps):
    python sweep.py --steps 1000

    # Dry run:
    python sweep.py --dry-run

    # Generate tmux launch script:
    python sweep.py --tmux

    # Run a specific job index (for tmux parallel execution):
    python sweep.py --job-index 3
"""

import argparse
import itertools
import json
import os
import sys
import time

import torch

from config import (
    SWEEP_LAYERS, SWEEP_K, SWEEP_T, SWEEP_ARCHITECTURES,
    TRAIN_STEPS, LOG_DIR, CHECKPOINT_DIR,
    DEVICE, SEED, LAYER_SPECS, run_name,
)
from data import CachedActivationSource
from train import train_stacked_sae, train_txcdr


def parse_args():
    p = argparse.ArgumentParser(
        description="NLP Temporal Crosscoder sweep over Gemma 2 2B activations",
    )
    p.add_argument(
        "--layer", nargs="+", default=None,
        choices=list(LAYER_SPECS.keys()),
        help="Which layers to sweep (default: all)",
    )
    p.add_argument(
        "--k", nargs="+", type=int, default=None,
        help="Which k values to sweep (default: all)",
    )
    p.add_argument(
        "--T", nargs="+", type=int, default=None,
        help="Which T values to sweep (default: all)",
    )
    p.add_argument(
        "--model", nargs="+", default=None,
        choices=SWEEP_ARCHITECTURES,
        help="Which architectures to sweep (default: all)",
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help=f"Override TRAIN_STEPS (default: {TRAIN_STEPS})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print sweep plan without training",
    )
    p.add_argument(
        "--tmux", action="store_true",
        help="Print tmux launch script and exit",
    )
    p.add_argument(
        "--job-index", type=int, default=None,
        help="Run only the job at this index (0-based, for tmux parallel)",
    )
    p.add_argument(
        "--no-checkpoint", action="store_true",
        help="Skip saving model checkpoints",
    )
    return p.parse_args()


def build_sweep_plan(args) -> list[dict]:
    """Build the list of (model_type, layer, k, T) jobs."""
    layers = args.layer or SWEEP_LAYERS
    ks = args.k or SWEEP_K
    Ts = args.T or SWEEP_T
    models = args.model or SWEEP_ARCHITECTURES

    jobs = []
    for layer, k, T, model_type in itertools.product(layers, ks, Ts, models):
        jobs.append(dict(model=model_type, layer=layer, k=k, T=T))

    return jobs


def print_tmux_script(jobs: list[dict], n_steps: int) -> None:
    """Print a bash script that launches each job in a tmux pane.

    For large sweeps, creates one tmux window per layer with jobs tiled inside.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    total = len(jobs)
    print("#!/bin/bash")
    print("# Auto-generated tmux launch script for NLP sweep")
    print(f"# Total jobs: {total}")
    print("set -euo pipefail")
    print()
    print("SESSION=nlp_txcdr_sweep")
    print('tmux kill-session -t "$SESSION" 2>/dev/null || true')
    print()

    # Group jobs by layer for manageable windows
    from collections import defaultdict
    by_layer: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for i, job in enumerate(jobs):
        by_layer[job["layer"]].append((i, job))

    first_window = True
    for layer, layer_jobs in by_layer.items():
        if first_window:
            print(f'tmux new-session -d -s "$SESSION" -n "{layer}"')
            first_window = False
        else:
            print(f'tmux new-window -t "$SESSION" -n "{layer}"')

        for pane_idx, (job_i, job) in enumerate(layer_jobs):
            cmd = (
                f"cd {script_dir} && python sweep.py "
                f"--job-index {job_i} --steps {n_steps}"
            )
            if pane_idx == 0:
                print(f'tmux send-keys -t "$SESSION:{layer}" "{cmd}" Enter')
            else:
                print(f'tmux split-window -t "$SESSION:{layer}"')
                print(f'tmux send-keys -t "$SESSION:{layer}" "{cmd}" Enter')
                if (pane_idx + 1) % 4 == 0:
                    print(f'tmux select-layout -t "$SESSION:{layer}" tiled')

        print(f'tmux select-layout -t "$SESSION:{layer}" tiled')
        print()

    print(f'echo "Launched $SESSION with {total} jobs across {len(by_layer)} windows"')
    print('echo "Attach with: tmux attach -t $SESSION"')


def main():
    args = parse_args()
    n_steps = args.steps or TRAIN_STEPS
    save_ckpt = not args.no_checkpoint

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(SEED)

    jobs = build_sweep_plan(args)
    total = len(jobs)

    # tmux mode: print launch script and exit
    if args.tmux:
        print_tmux_script(jobs, n_steps)
        return

    # Dry run: print plan and exit
    if args.dry_run:
        print("=" * 80)
        print(f"  NLP TEMPORAL CROSSCODER SWEEP (Gemma 2 2B)")
        print(f"  Device: {DEVICE}")
        print(f"  Steps per run: {n_steps:,}")
        print(f"  Total jobs: {total}")
        print("=" * 80)
        for i, j in enumerate(jobs):
            print(
                f"  [{i:3d}/{total}]  {j['model']:12s}  "
                f"layer={j['layer']:12s}  k={j['k']:>3d}  T={j['T']:>2d}"
            )
        print(f"\n  (dry run — no training)")
        return

    # If --job-index, run only that job
    if args.job_index is not None:
        if args.job_index >= total:
            print(f"ERROR: --job-index {args.job_index} >= total jobs {total}")
            sys.exit(1)
        jobs = [jobs[args.job_index]]
        total = 1

    print("=" * 80)
    print(f"  NLP TEMPORAL CROSSCODER SWEEP (Gemma 2 2B)")
    print(f"  Device: {DEVICE}")
    print(f"  Steps per run: {n_steps:,}")
    print(f"  Total jobs: {total}")
    print("=" * 80)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Group jobs by layer so we load one dataset at a time, train all
    # (k, T, arch) combos, then free GPU memory before the next layer.
    import gc
    from collections import OrderedDict

    layers_ordered = list(OrderedDict.fromkeys(j["layer"] for j in jobs))
    jobs_by_layer: dict[str, list[dict]] = {l: [] for l in layers_ordered}
    for job in jobs:
        jobs_by_layer[job["layer"]].append(job)

    summary_rows: list[dict] = []
    t_sweep_start = time.time()
    job_counter = 0

    for layer_key in layers_ordered:
        layer_jobs = jobs_by_layer[layer_key]

        print(f"\n{'=' * 80}")
        print(f"  Loading activations for layer: {layer_key} "
              f"({len(layer_jobs)} jobs)")
        print(f"{'=' * 80}")

        source = CachedActivationSource(layer_key)

        for job in layer_jobs:
            job_counter += 1
            model_type = job["model"]
            k = job["k"]
            T = job["T"]
            if k*T > 1024:
                print(f"  Skipping k={k}, T={T} (k*T={k*T} > 512)")
                continue

            print(f"\n{'─' * 80}")
            print(f"  [{job_counter}/{total}]  {model_type.upper()}  "
                  f"layer={layer_key}  k={k}  T={T}")
            print(f"{'─' * 80}")

            t0 = time.time()

            if model_type == "stacked_sae":
                model, history = train_stacked_sae(
                    layer_key, k, T, source,
                    n_steps=n_steps, save_checkpoint=save_ckpt,
                )
            else:
                model, history = train_txcdr(
                    layer_key, k, T, source,
                    n_steps=n_steps, save_checkpoint=save_ckpt,
                )

            elapsed = time.time() - t0
            final = history[-1] if history else {}

            row = {
                "model": model_type,
                "layer": layer_key,
                "k": k,
                "T": T,
                "final_loss": final.get("loss", 0),
                "final_l0": final.get("window_l0", 0),
                "final_fvu": final.get("fvu", 0),
                "final_entropy": final.get("entropy", 0),
                "elapsed_sec": round(elapsed, 1),
            }
            summary_rows.append(row)

            print(
                f"  Done in {elapsed:.0f}s | loss={row['final_loss']:.4f} "
                f"| L0={row['final_l0']:.0f} | FVU={row['final_fvu']:.4f} "
                f"| H={row['final_entropy']:.3f}"
            )

            # Free trained model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Free this layer's activations before loading the next ──
        print(f"\n  Releasing {layer_key} activations from GPU...")
        source.data = None
        del source
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Save sweep summary (append to existing if present)
    summary_path = os.path.join(LOG_DIR, "sweep_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        summary_rows = existing + summary_rows

    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    total_time = time.time() - t_sweep_start
    print(f"\n{'=' * 80}")
    print(f"  SWEEP COMPLETE  |  {total} jobs  |  {total_time / 3600:.1f}h total")
    print(f"  Summary: {summary_path}")
    print(f"  Run `python viz.py` to generate visualizations")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
