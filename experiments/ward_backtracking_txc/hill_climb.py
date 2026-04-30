"""Phase 2 — greedy local hill-climb on Stage B's grid.

Adapted from Han's `agent_w/brief.md` pattern (sweep → identify winner →
greedy local search). For backtracking, the perturbation axes are:

    - arch:           swap to one of the other 3 archs in arch_list
    - hookpoint:      swap to an adjacent hookpoint within enabled set
    - k_per_position: scale by ×0.5 or ×2, clamped to [k_min, k_max]

(seed perturbation is treated as a separate "verify" pass — not part of
the hill-climb axis since reseeding doesn't search a new architecture.)

Algorithm:
    state = {current_best: <Cell>, history: [], iteration: 0,
             evaluated: {cell_id: metric_dict}}

    repeat up to MAX_ITER times:
        neighbors = enumerate_perturbations(state.current_best, state.evaluated)
        if not neighbors: stop("local maximum: all neighbors evaluated, none beats current")
        for batch_of_N_GPUs in neighbors:
            launch evaluate_cell parallel × N
            wait
        best_neighbor = max(neighbors, key=primary_kw_at_coh)
        if best_neighbor.metric > state.current_best.metric * (1 + IMPROVEMENT_THRESHOLD):
            state.current_best = best_neighbor
            state.iteration += 1
            state.history.append({step: i, accepted: best_neighbor.cell_id, ...})
        else:
            stop("no neighbor improves current best by > IMPROVEMENT_THRESHOLD")

State JSON: <root>/hillclimb_state.json (resumable: re-run continues from
where it left off).

Usage:
    python -m experiments.ward_backtracking_txc.hill_climb --start-from <cell_id>
        [--max-iter 4]
        [--num-gpus 2]
        [--improvement-threshold 0.05]   # 5% relative improvement required
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from experiments.ward_backtracking_txc.cell_id import (
    Cell, cell_metric_path,
)
from experiments.ward_backtracking_txc import metrics as metrics_mod

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.hill_climb")


# ─── Perturbation enumeration ───────────────────────────────────────────────────

def enumerate_neighbors(cell: Cell, cfg: dict, seen: set[str]) -> list[Cell]:
    """All single-axis perturbations of `cell` not yet evaluated.

    Axes:
      - arch:           any other arch in cfg.txc.arch_list
      - hookpoint:      any other enabled hookpoint
      - k_per_position: cell.k * 0.5 (clamped ≥ 4) and cell.k * 2 (clamped ≤ 256)
    """
    out = []

    # arch axis
    for arch in cfg["txc"].get("arch_list", []):
        if arch == cell.arch: continue
        n = cell.with_(arch=arch)
        if n.id not in seen: out.append(n)

    # hookpoint axis
    for hp in cfg["hookpoints"]:
        if not hp.get("enabled", True): continue
        if hp["key"] == cell.hookpoint_key: continue
        n = cell.with_(hookpoint_key=hp["key"])
        if n.id not in seen: out.append(n)

    # k_per_position axis
    for factor, op in [(0.5, max), (2.0, min)]:
        new_k = int(round(cell.k_per_position * factor))
        new_k = max(4, min(256, new_k))                  # global clamp
        if new_k == cell.k_per_position: continue
        n = cell.with_(k_per_position=new_k)
        if n.id not in seen: out.append(n)

    return out


# ─── State persistence ─────────────────────────────────────────────────────────

def state_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["root"]) / "hillclimb_state.json"


def load_state(cfg: dict) -> dict:
    p = state_path(cfg)
    if not p.exists():
        return {"current_best": None, "iteration": 0, "history": [], "evaluated": {}}
    return json.loads(p.read_text())


def save_state(state: dict, cfg: dict) -> None:
    p = state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ─── Cell evaluation dispatch (parallel across GPUs) ───────────────────────────

def dispatch_cells(cells: list[Cell], num_gpus: int, cfg: dict) -> dict[str, dict | None]:
    """Spawn evaluate_cell subprocesses, round-robin across GPUs, pool-size=num_gpus.
    Returns {cell_id: metric_dict_or_None}."""
    results: dict[str, dict | None] = {}
    pids: list[tuple[int, Cell]] = []           # (subprocess_pid, cell)
    procs: dict[int, subprocess.Popen] = {}     # pid -> Popen
    cell_for_pid: dict[int, Cell] = {}

    metrics_dir = Path(cfg["paths"]["root"]) / "cell_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    def metric_for(cell: Cell) -> dict | None:
        mp = cell_metric_path(cell, metrics_dir)
        if not mp.exists():
            return None
        try:
            return json.loads(mp.read_text())
        except Exception:
            return None

    log.info("[dispatch] %d cells across %d GPUs", len(cells), num_gpus)
    queued = list(cells)
    in_flight: list[tuple[subprocess.Popen, Cell, int]] = []

    while queued or in_flight:
        # Launch up to num_gpus jobs.
        while queued and len(in_flight) < num_gpus:
            cell = queued.pop(0)
            # If the cell metric already exists, short-circuit.
            existing = metric_for(cell)
            if existing is not None:
                log.info("[reuse] %s (existing metric)", cell.id)
                results[cell.id] = existing
                continue
            gpu = len(in_flight) % num_gpus      # rough — may collide if a slot is busy
            # Pick the lowest-numbered GPU not currently in_flight
            in_use = {g for _, _, g in in_flight}
            gpu = next(g for g in range(num_gpus) if g not in in_use)
            log_path = Path("/tmp") / f"hillclimb_{cell.id}_gpu{gpu}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log.info("[launch] %s on cuda:%d → %s", cell.id, gpu, log_path)
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "experiments.ward_backtracking_txc.evaluate_cell",
                     "--cell", cell.id],
                    env=env, stdout=lf, stderr=subprocess.STDOUT,
                )
            in_flight.append((proc, cell, gpu))

        # Poll every 30s; reap finished.
        time.sleep(30)
        still: list[tuple[subprocess.Popen, Cell, int]] = []
        for proc, cell, gpu in in_flight:
            if proc.poll() is None:
                still.append((proc, cell, gpu))
            else:
                rc = proc.returncode
                if rc != 0:
                    log.error("[FAIL] %s rc=%d (see log /tmp/hillclimb_%s_gpu%d.log)",
                              cell.id, rc, cell.id, gpu)
                    results[cell.id] = None
                else:
                    results[cell.id] = metric_for(cell)
                    log.info("[done] %s primary=%.4f", cell.id,
                             results[cell.id]["primary_kw_at_coh"] if results[cell.id] else float("nan"))
        in_flight = still

    return results


# ─── Main hill-climb loop ───────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--start-from", type=str, default=None,
                   help="cell id of the Phase 1 winner (read from rank_phase1.json if omitted)")
    p.add_argument("--max-iter", type=int, default=4,
                   help="cap on hill-climb iterations (Han's W brief uses 4)")
    p.add_argument("--num-gpus", type=int, default=None,
                   help="default: detect via nvidia-smi -L")
    p.add_argument("--improvement-threshold", type=float, default=0.05,
                   help="relative improvement required to accept a neighbor (default 5%)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())

    # Number of GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        try:
            res = subprocess.check_output(["nvidia-smi", "-L"]).decode()
            num_gpus = len([l for l in res.splitlines() if l.strip()])
        except Exception:
            num_gpus = 1
    log.info("[hill_climb] num_gpus=%d max_iter=%d threshold=%.2f%%",
             num_gpus, args.max_iter, args.improvement_threshold * 100)

    state = load_state(cfg)

    # Resolve starting cell
    if state["current_best"] is None:
        if args.start_from:
            start_id = args.start_from
        else:
            rank_path = Path(cfg["paths"]["root"]) / "rank_phase1.json"
            if not rank_path.exists():
                log.error("[fatal] no --start-from and rank_phase1.json missing. "
                          "Run rank_phase1.py first or pass --start-from <cell_id>.")
                return 1
            ranks = json.loads(rank_path.read_text())
            start_id = ranks["winner_cell_id"]
            log.info("[start] using Phase 1 winner from %s: %s", rank_path, start_id)
        # Bootstrap state with the winner cell's existing metric.
        winner_cell = Cell.from_id(start_id)
        winner_metric = json.loads(cell_metric_path(
            winner_cell, Path(cfg["paths"]["root"]) / "cell_metrics"
        ).read_text()) if cell_metric_path(
            winner_cell, Path(cfg["paths"]["root"]) / "cell_metrics"
        ).exists() else None
        if winner_metric is None:
            log.info("[bootstrap] evaluating winner cell %s to seed hill-climb", start_id)
            res = dispatch_cells([winner_cell], num_gpus, cfg)
            winner_metric = res.get(start_id)
            if winner_metric is None:
                log.error("[fatal] could not evaluate winner cell"); return 1
        state["current_best"] = {"cell_id": start_id, "metric": winner_metric}
        state["evaluated"][start_id] = winner_metric
        save_state(state, cfg)

    # Hill-climb loop
    while state["iteration"] < args.max_iter:
        current_cell = Cell.from_id(state["current_best"]["cell_id"])
        current_score = state["current_best"]["metric"]["primary_kw_at_coh"]
        log.info("[iter %d/%d] current=%s primary=%.4f",
                 state["iteration"] + 1, args.max_iter, current_cell.id, current_score)

        seen = set(state["evaluated"].keys())
        neighbors = enumerate_neighbors(current_cell, cfg, seen)
        if not neighbors:
            log.info("[stop] all neighbors of %s already evaluated; local maximum",
                     current_cell.id)
            break

        log.info("[neighbors] %d to evaluate: %s", len(neighbors),
                 [c.id for c in neighbors])
        results = dispatch_cells(neighbors, num_gpus, cfg)

        # Update evaluated set + pick best successful neighbor
        best_n_cell = None; best_n_score = -float("inf")
        for c in neighbors:
            m = results.get(c.id)
            if m is None: continue
            state["evaluated"][c.id] = m
            if m["primary_kw_at_coh"] > best_n_score:
                best_n_score = m["primary_kw_at_coh"]; best_n_cell = c
        save_state(state, cfg)

        if best_n_cell is None:
            log.info("[stop] no neighbor evaluated successfully")
            break

        rel_improvement = (best_n_score - current_score) / max(abs(current_score), 1e-6)
        log.info("[best-neighbor] %s primary=%.4f (Δ=%+.2f%% vs current)",
                 best_n_cell.id, best_n_score, rel_improvement * 100)

        if rel_improvement > args.improvement_threshold:
            state["history"].append({
                "iteration": state["iteration"] + 1,
                "from": current_cell.id, "to": best_n_cell.id,
                "from_score": current_score, "to_score": best_n_score,
                "rel_improvement": rel_improvement,
            })
            state["current_best"] = {"cell_id": best_n_cell.id, "metric": results[best_n_cell.id]}
            state["iteration"] += 1
            save_state(state, cfg)
        else:
            log.info("[stop] best neighbor improvement (%.2f%%) below threshold (%.2f%%)",
                     rel_improvement * 100, args.improvement_threshold * 100)
            break

    log.info("[done] hill-climb final winner: %s (primary=%.4f) over %d iterations",
             state["current_best"]["cell_id"],
             state["current_best"]["metric"]["primary_kw_at_coh"],
             state["iteration"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
