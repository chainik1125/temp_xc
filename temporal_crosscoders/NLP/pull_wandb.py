#!/usr/bin/env python3
"""
pull_wandb.py — Backfill local convergence logs from wandb.

The local sweep logs only have final metrics (we restored them from
summary_table.txt). The full step-by-step convergence histories live on wandb.
This script queries the wandb API and writes one JSON per run to logs/.

Usage:
    python pull_wandb.py
    python pull_wandb.py --project temporal-crosscoders-nlp-gemma2-it
    python pull_wandb.py --entity <user> --project <proj>
    python pull_wandb.py --dry-run
"""

import argparse
import json
import os
import re
import sys

from tqdm.auto import tqdm

from config import (
    WANDB_PROJECT, WANDB_ENTITY, LOG_DIR, run_name,
)


METRIC_KEYS = ["loss", "fvu", "l0", "window_l0", "entropy"]

# run name format: {model}__{layer}__k{k}__T{T}
RUN_NAME_RE = re.compile(r"^(?P<model>[a-z_]+)__(?P<layer>[a-z_]+)__k(?P<k>\d+)__T(?P<T>\d+)$")


def parse_run_name(name: str) -> dict | None:
    """Parse run name like 'stacked_sae__mid_res__k100__T5' into components."""
    m = RUN_NAME_RE.match(name)
    if not m:
        return None
    return {
        "model_type": m.group("model"),
        "layer": m.group("layer"),
        "k": int(m.group("k")),
        "T": int(m.group("T")),
    }


def main():
    parser = argparse.ArgumentParser(description="Pull convergence histories from wandb")
    parser.add_argument("--entity", type=str, default=WANDB_ENTITY,
                        help=f"wandb entity (default: {WANDB_ENTITY or 'default user'})")
    parser.add_argument("--project", type=str, default=WANDB_PROJECT,
                        help=f"wandb project (default: {WANDB_PROJECT})")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help=f"output dir (default: {LOG_DIR})")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing logs (default: skip if exists)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list runs without writing files")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. uv add wandb")
        sys.exit(1)

    api = wandb.Api()

    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    print(f"Querying wandb project: {project_path}")

    try:
        runs = list(api.runs(project_path))
    except Exception as e:
        print(f"ERROR: failed to query runs: {e}")
        print("Hint: set WANDB_ENTITY env var or pass --entity <user>")
        sys.exit(1)

    print(f"Found {len(runs)} runs")
    if not runs:
        sys.exit(0)

    os.makedirs(args.log_dir, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_failed = 0

    for run in tqdm(runs, desc="Pulling histories"):
        cfg = run.config
        model_type = cfg.get("model_type")
        layer = cfg.get("layer")
        k = cfg.get("k")
        T = cfg.get("T")

        # Fall back to parsing the run name if config is missing
        if not all([model_type, layer, k is not None, T is not None]):
            parsed = parse_run_name(run.name)
            if parsed is None:
                tqdm.write(f"  skip {run.name}: can't parse name and config is empty")
                n_failed += 1
                continue
            model_type = parsed["model_type"]
            layer = parsed["layer"]
            k = parsed["k"]
            T = parsed["T"]

        rn = run_name(model_type, layer, int(k), int(T))
        out_path = os.path.join(args.log_dir, f"{rn}.json")

        if os.path.exists(out_path) and not args.overwrite:
            # Check if existing log has more than 1 entry (i.e. real history, not stub)
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                if len(existing) > 1:
                    n_skipped += 1
                    continue
            except Exception:
                pass

        if args.dry_run:
            tqdm.write(f"  would write {rn} (state={run.state})")
            n_written += 1
            continue

        try:
            history = run.history(
                keys=METRIC_KEYS,
                pandas=False,
                samples=10_000,
            )
        except Exception as e:
            tqdm.write(f"  fail {rn}: {e}")
            n_failed += 1
            continue

        rows = []
        for row in history:
            entry = {"step": row.get("_step", 0)}
            for key in METRIC_KEYS:
                if key in row and row[key] is not None:
                    entry[key] = row[key]
            if "loss" in entry:
                rows.append(entry)

        if not rows:
            tqdm.write(f"  empty {rn}: no metric rows")
            n_failed += 1
            continue

        with open(out_path, "w") as f:
            json.dump(rows, f, indent=1)
        n_written += 1
        tqdm.write(f"  ✓ {rn}: {len(rows)} steps → {out_path}")

    print(f"\nDone: wrote {n_written}, skipped {n_skipped}, failed {n_failed}")


if __name__ == "__main__":
    main()
