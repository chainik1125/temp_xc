"""Single-cell pipeline: train → mine → B1 → metric.

Used by the hill-climb orchestrator; can also be invoked manually:

    CUDA_VISIBLE_DEVICES=0 python -m experiments.ward_backtracking_txc.evaluate_cell \
        --cell tsae__ln1_L10__k16__s42

Each stage is idempotent — re-running on a partially-completed cell skips
already-computed artifacts. Outputs:

    <ckpt_dir>/<cell_id>.pt
    <features_dir>/<cell_id>.npz
    <root>/steering_per_cell/b1__<cell_id>.json
    <root>/cell_metrics/<cell_id>.json   ← what hill_climb.py reads

The metric file has the schema described in metrics.py (primary_kw_at_coh,
peak_kw_no_coh, best_magnitude, best_source, frac_coherent).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from experiments.ward_backtracking_txc.cell_id import (
    Cell, ckpt_path, features_path, b1_per_cell_path, cell_metric_path,
)
from experiments.ward_backtracking_txc import metrics

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.eval_cell")


def _run(cmd: list[str], desc: str) -> int:
    log.info("[exec %s] %s", desc, " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        log.error("[exec %s] FAILED rc=%d", desc, rc)
    return rc


def evaluate_cell(cell: Cell, cfg: dict, no_dom: bool = True) -> dict | None:
    """Train + mine + B1 + metric for one cell. Returns the metric dict or None on failure.

    `no_dom=True` (default) skips DoM in the per-cell B1 because it's already
    in the canonical Phase 1 sweep results — saves ~2 sources × 9 mags × 20
    prompts × max_new_tokens of generation per cell.
    """
    paths = cfg["paths"]
    ckpt_dir = Path(paths["ckpt_dir"])
    features_dir = Path(paths["features_dir"])
    metrics_dir = Path(paths["root"]) / "cell_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cp = ckpt_path(cell, ckpt_dir)
    fp = features_path(cell, features_dir)
    b1p = b1_per_cell_path(cell, Path(paths["root"]) / "steering_per_cell")
    mp = cell_metric_path(cell, metrics_dir)

    # 1. Train (skip if ckpt exists)
    if not cp.exists():
        if _run([sys.executable, "-m", "experiments.ward_backtracking_txc.train_txc",
                 "--cell", cell.id], "train") != 0:
            return None
    else:
        log.info("[skip-train] %s exists", cp)

    # 2. Mine (skip if features file exists)
    if not fp.exists():
        if _run([sys.executable, "-m", "experiments.ward_backtracking_txc.mine_features",
                 "--cell", cell.id], "mine") != 0:
            return None
    else:
        log.info("[skip-mine] %s exists", fp)

    # 3. B1 (skip if per-cell JSON exists)
    if not b1p.exists():
        b1_cmd = [sys.executable, "-m", "experiments.ward_backtracking_txc.b1_steer_eval",
                  "--cell", cell.id]
        if no_dom:
            b1_cmd.append("--no-dom")
        if _run(b1_cmd, "b1") != 0:
            return None
    else:
        log.info("[skip-b1] %s exists", b1p)

    # 4. Compute metric
    rows = metrics.load_b1_rows(b1p)
    source_tags = sorted({r["source"] for r in rows if r["source"].startswith(cell.arch + "_")})
    metric = metrics.cell_metric(rows, source_tags)
    metric["cell_id"] = cell.id
    metric["arch"] = cell.arch
    metric["hookpoint"] = cell.hookpoint_key
    metric["k_per_position"] = cell.k_per_position
    metric["seed"] = cell.seed
    metric["n_sources"] = len(source_tags)
    mp.write_text(json.dumps(metric, indent=2))
    log.info("[metric] %s primary=%.4f at mag=%+.0f (frac_coherent=%.2f)",
             cell.id, metric["primary_kw_at_coh"], metric["best_magnitude"],
             metric["frac_coherent"])
    return metric


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--cell", type=str, required=True,
                   help="cell id <arch>__<hp>__k<k>__s<seed>")
    p.add_argument("--with-dom", action="store_true",
                   help="include DoM baselines in the per-cell B1 (default: skip; "
                        "DoM is in the Phase 1 canonical results already).")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    cell = Cell.from_id(args.cell)
    metric = evaluate_cell(cell, cfg, no_dom=(not args.with_dom))
    return 0 if metric is not None else 1


if __name__ == "__main__":
    sys.exit(main())
