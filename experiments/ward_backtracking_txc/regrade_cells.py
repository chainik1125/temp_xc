"""Re-grade existing cell metrics under Sonnet 4.6 coherence floor.

For each cell that already has a per-cell B1 JSON, fetches/loads its
Sonnet grades and rewrites cell_metrics/<cell_id>.json with both the
legacy `primary_kw_at_coh` (max-run heuristic) and the new
`primary_kw_at_sonnet_coh` (Sonnet 0-3 grade ≥ 2 floor).

Two modes:
  --grade        run grade_sonnet first to populate grades JSONs (default off)
  --no-grade     just re-compute metrics from existing grades

Cells without a grades file get only the legacy metric.

Usage:
    python -m experiments.ward_backtracking_txc.regrade_cells --grade --concurrency 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

from experiments.ward_backtracking_txc.cell_id import (
    Cell, b1_per_cell_path, cell_metric_path, sonnet_grades_path,
)
from experiments.ward_backtracking_txc import metrics as metrics_mod
from experiments.ward_backtracking_txc.grade_sonnet import grade_rows, load_prompt_lookup

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.regrade")


def discover_cells_with_b1(b1_dir: Path) -> list[Cell]:
    """Find all per-cell B1 JSONs and return their Cell objects."""
    cells: list[Cell] = []
    for p in sorted(b1_dir.glob("b1__*.json")):
        cell_id = p.name[len("b1__"):-len(".json")]
        try:
            cells.append(Cell.from_id(cell_id))
        except Exception as e:
            log.warning("[skip] %s — %s", p.name, e)
    return cells


async def grade_all(cells: list[Cell], cfg: dict, concurrency: int):
    paths = cfg["paths"]
    b1_dir = Path(paths["root"]) / "steering_per_cell"
    grades_dir = Path(paths["root"]) / "coherence_grades"
    grades_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompt_lookup(Path(paths["stageA_prompts"]))

    for cell in cells:
        b1p = b1_per_cell_path(cell, b1_dir)
        if not b1p.exists():
            log.warning("[skip] %s — no B1 file", cell.id)
            continue
        gp = sonnet_grades_path(cell, grades_dir)
        rows = json.loads(b1p.read_text())["rows"]
        log.info("[grade] %s (%d rows) → %s", cell.id, len(rows), gp)
        await grade_rows(rows, prompts, gp, concurrency=concurrency)


def regrade_metrics(cells: list[Cell], cfg: dict):
    """Recompute cell_metrics/<cell>.json for every cell that has B1 + grades."""
    paths = cfg["paths"]
    b1_dir = Path(paths["root"]) / "steering_per_cell"
    grades_dir = Path(paths["root"]) / "coherence_grades"
    metrics_dir = Path(paths["root"]) / "cell_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    n_with_sonnet = 0; n_legacy_only = 0; n_skip = 0
    for cell in cells:
        b1p = b1_per_cell_path(cell, b1_dir)
        if not b1p.exists():
            n_skip += 1; continue
        rows = json.loads(b1p.read_text())["rows"]
        source_tags = sorted({r["source"] for r in rows
                              if r["source"].startswith(cell.arch + "_")})
        gp = sonnet_grades_path(cell, grades_dir)
        sonnet = metrics_mod.load_sonnet_grades(gp)
        metric = metrics_mod.cell_metric(rows, source_tags, sonnet_grades=sonnet)
        metric["cell_id"] = cell.id
        metric["arch"] = cell.arch
        metric["hookpoint"] = cell.hookpoint_key
        metric["k_per_position"] = cell.k_per_position
        metric["seed"] = cell.seed
        metric["rank"] = cell.rank
        metric["n_sources"] = len(source_tags)
        cell_metric_path(cell, metrics_dir).write_text(json.dumps(metric, indent=2))
        if sonnet:
            n_with_sonnet += 1
            log.info(
                "[metric] %s legacy=%.4f sonnet=%.4f frac_coh_sonnet=%.2f best_src=%s mag=%+.0f",
                cell.id,
                metric["primary_kw_at_coh"], metric["primary_kw_at_sonnet_coh"],
                metric.get("frac_coh_sonnet", 0.0),
                metric.get("best_source_sonnet"),
                metric.get("best_magnitude_sonnet", 0.0),
            )
        else:
            n_legacy_only += 1
            log.info("[metric] %s legacy=%.4f (no sonnet)",
                     cell.id, metric["primary_kw_at_coh"])
    log.info("[summary] sonnet=%d legacy_only=%d skip=%d total=%d",
             n_with_sonnet, n_legacy_only, n_skip, len(cells))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--grade", action="store_true",
                   help="run Sonnet grading on all per-cell B1 files first")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--cell", type=str, default=None,
                   help="restrict to a single cell id (default: all cells with B1)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["paths"]
    b1_dir = Path(paths["root"]) / "steering_per_cell"
    cells = ([Cell.from_id(args.cell)] if args.cell
             else discover_cells_with_b1(b1_dir))
    log.info("[discover] %d cells with B1 files", len(cells))

    if args.grade:
        asyncio.run(grade_all(cells, cfg, concurrency=args.concurrency))

    regrade_metrics(cells, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
