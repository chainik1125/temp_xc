"""Multi-seed verification of the post-hill-climb winner cell.

Han's W brief requires the headline cell to be stable across seeds within
±0.27 (his observed σ for the Gemma-2-2b steering case). For backtracking
we don't have a pre-measured σ, so we report mean ± std across seeds and
compute a relative dispersion (σ/μ) as the stability headline.

Algorithm:
    winner = state.current_best.cell_id   (from hillclimb_state.json)
    seeds_to_check = [1, 2]               (seed=42 already has metric)
    for seed in seeds_to_check (parallel across N GPUs):
        cell = winner.with_(seed=seed)
        evaluate_cell(cell)               (train + mine + B1 + metric)
    write verify_seeds.json with:
        winner_cell_id, seeds_checked,
        primary_kw_at_coh_per_seed, mean, std, rel_dispersion,
        verdict: "stable" if rel_dispersion < 0.20 else "unstable"

Usage:
    python -m experiments.ward_backtracking_txc.verify_seeds
        [--seeds 1 2]
        [--num-gpus 2]
        [--cell <override>]      # use this cell instead of hillclimb winner
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.ward_backtracking_txc.cell_id import Cell, cell_metric_path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.verify_seeds")


def _dispatch_seeds(cells: list[Cell], num_gpus: int) -> dict[str, dict | None]:
    """Spawn evaluate_cell × len(cells), round-robin across GPUs."""
    in_flight: list[tuple[subprocess.Popen, Cell, int]] = []
    queued = list(cells)
    results: dict[str, dict | None] = {}

    metrics_dir = Path("results/ward_backtracking_txc/cell_metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    while queued or in_flight:
        while queued and len(in_flight) < num_gpus:
            cell = queued.pop(0)
            mp = cell_metric_path(cell, metrics_dir)
            if mp.exists():
                log.info("[reuse] %s (existing metric)", cell.id)
                results[cell.id] = json.loads(mp.read_text()); continue
            in_use = {g for _, _, g in in_flight}
            gpu = next(g for g in range(num_gpus) if g not in in_use)
            log_path = Path("/tmp") / f"verify_{cell.id}_gpu{gpu}.log"
            log.info("[launch] %s on cuda:%d → %s", cell.id, gpu, log_path)
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    [sys.executable, "-m",
                     "experiments.ward_backtracking_txc.evaluate_cell",
                     "--cell", cell.id],
                    env=env, stdout=lf, stderr=subprocess.STDOUT,
                )
            in_flight.append((proc, cell, gpu))

        time.sleep(30)
        still = []
        for proc, cell, gpu in in_flight:
            if proc.poll() is None:
                still.append((proc, cell, gpu))
            else:
                if proc.returncode != 0:
                    log.error("[FAIL] %s rc=%d", cell.id, proc.returncode)
                    results[cell.id] = None
                else:
                    mp = cell_metric_path(cell, metrics_dir)
                    results[cell.id] = json.loads(mp.read_text()) if mp.exists() else None
                    if results[cell.id]:
                        log.info("[done] %s primary=%.4f", cell.id,
                                 results[cell.id]["primary_kw_at_coh"])
        in_flight = still
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--num-gpus", type=int, default=None)
    p.add_argument("--cell", type=str, default=None,
                   help="explicit cell id to verify; default = hill-climb winner")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    num_gpus = args.num_gpus
    if num_gpus is None:
        try:
            res = subprocess.check_output(["nvidia-smi", "-L"]).decode()
            num_gpus = len([l for l in res.splitlines() if l.strip()])
        except Exception:
            num_gpus = 1

    # Resolve the cell to verify.
    if args.cell:
        winner_id = args.cell
    else:
        state_path = Path(cfg["paths"]["root"]) / "hillclimb_state.json"
        if not state_path.exists():
            log.error("[fatal] no --cell given and %s missing", state_path)
            return 1
        winner_id = json.loads(state_path.read_text())["current_best"]["cell_id"]
    base_cell = Cell.from_id(winner_id)
    log.info("[verify] base cell = %s", winner_id)

    # Build the seed variants. Always include the base seed for dispersion.
    base_metric_path = cell_metric_path(
        base_cell, Path(cfg["paths"]["root"]) / "cell_metrics"
    )
    if not base_metric_path.exists():
        log.error("[fatal] base cell metric %s missing — re-run hill-climb",
                  base_metric_path); return 1
    base_metric = json.loads(base_metric_path.read_text())

    new_cells = [base_cell.with_(seed=s) for s in args.seeds if s != base_cell.seed]
    log.info("[verify] dispatching %d new seeds: %s", len(new_cells),
             [c.id for c in new_cells])
    results = _dispatch_seeds(new_cells, num_gpus)

    all_metrics = [(base_cell.seed, base_metric)] + \
                  [(c.seed, results[c.id]) for c in new_cells if results[c.id]]
    if len(all_metrics) < 2:
        log.error("[fatal] only %d seed(s) succeeded; cannot compute σ", len(all_metrics))
        return 1

    primary_vals = [m["primary_kw_at_coh"] for _, m in all_metrics]
    mu = sum(primary_vals) / len(primary_vals)
    var = sum((v - mu) ** 2 for v in primary_vals) / max(1, len(primary_vals) - 1)
    sigma = math.sqrt(var)
    rel_disp = sigma / max(abs(mu), 1e-6)

    verdict = "stable" if rel_disp < 0.20 else "unstable"
    out = {
        "base_cell_id": winner_id,
        "seeds_checked": [s for s, _ in all_metrics],
        "primary_kw_at_coh_per_seed": {str(s): m["primary_kw_at_coh"]
                                        for s, m in all_metrics},
        "best_magnitude_per_seed": {str(s): m["best_magnitude"]
                                     for s, m in all_metrics},
        "best_source_per_seed": {str(s): m["best_source"] for s, m in all_metrics},
        "frac_coherent_per_seed": {str(s): m["frac_coherent"]
                                    for s, m in all_metrics},
        "mean": mu,
        "std": sigma,
        "rel_dispersion": rel_disp,
        "verdict": verdict,
    }
    out_path = Path(cfg["paths"]["root"]) / "verify_seeds.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("=" * 70)
    log.info("[verify] cell %s across seeds %s:", winner_id, out["seeds_checked"])
    log.info("  primary_kw_at_coh: %s", out["primary_kw_at_coh_per_seed"])
    log.info("  μ = %.4f  σ = %.4f  rel_disp = %.2f%%  → %s",
             mu, sigma, rel_disp * 100, verdict)
    log.info("[saved] %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
