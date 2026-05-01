"""Run the behavioral backtracking judge across ALL B1 files (canonical
+ per-cell), recompute per-cell `genuine_backtracking_rate` and write
back to cell_metrics/<cell>.json.

Per-cell metric:
  genuine_backtracking_rate = mean(genuine_count >= 1) over rows that pass
                              (kw_rate - baseline_kw > kw_floor) AND
                              (sonnet_grade >= 2)
                              and that belong to this cell's source set.

This is the "share of sentences both >coherent threshold and have genuine
backtracking" metric Dmitry asked for.

Usage:
    python -m experiments.ward_backtracking_txc.regrade_backtracking --judge --concurrency 12

Run with --judge to call the API (resumable; ~$16 for the full sweep).
Run without --judge to just recompute metrics from existing judgements.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.regrade_bt")


def _judgements_path(b1_name: str, root: Path) -> Path:
    return root / "backtracking_judgements" / b1_name


async def judge_all(b1_files: list[Path], cfg: dict, concurrency: int):
    """Run the behavioral judge on every (b1, coherence_grades) pair."""
    from experiments.ward_backtracking_txc.grade_backtracking import grade_rows
    paths = cfg["paths"]
    grades_dir = Path(paths["root"]) / "coherence_grades"
    out_dir = Path(paths["root"]) / "backtracking_judgements"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_a = json.loads(Path(paths["stageA_prompts"]).read_text())
    prompts = {p["id"]: p.get("question") or p.get("prompt") or p.get("text", "")
               for p in stage_a}

    for b1f in b1_files:
        gp = grades_dir / b1f.name
        if not gp.exists():
            log.warning("[skip] %s — no coherence grades", b1f.name)
            continue
        coherence_grades = json.loads(gp.read_text())
        out_path = out_dir / b1f.name
        rows = json.loads(b1f.read_text())["rows"]
        log.info("[judge] %s (%d rows)", b1f.name, len(rows))
        await grade_rows(
            rows, prompts, out_path,
            concurrency=concurrency,
            kw_floor=0.005,
            coherence_grades=coherence_grades,
            coherence_floor=2,
        )


def recompute_metrics(b1_files: list[Path], cfg: dict):
    """For each cell, add `genuine_backtracking_rate` to cell_metrics JSON."""
    from experiments.ward_backtracking_txc.cell_id import (
        Cell, cell_metric_path, sonnet_grades_path,
    )
    paths = cfg["paths"]
    metrics_dir = Path(paths["root"]) / "cell_metrics"
    judgements_dir = Path(paths["root"]) / "backtracking_judgements"
    grades_dir = Path(paths["root"]) / "coherence_grades"

    for b1f in b1_files:
        name = b1f.name
        if not name.startswith("b1__") or not name.endswith(".json"):
            continue
        cell_id = name[len("b1__"):-len(".json")]
        try:
            cell = Cell.from_id(cell_id)
        except Exception:
            continue
        mp = cell_metric_path(cell, metrics_dir)
        if not mp.exists():
            log.warning("[skip] no cell_metrics for %s", cell_id)
            continue
        jp = judgements_dir / name
        gp = grades_dir / name
        if not jp.exists() or not gp.exists():
            continue
        rows = json.loads(b1f.read_text())["rows"]
        judgements = json.loads(jp.read_text())
        grades = json.loads(gp.read_text())

        # Restrict to the cell's own sources.
        src_pred = lambda s: s.startswith(cell.arch + "_")

        n_eligible = 0
        n_genuine = 0
        for idx, r in enumerate(rows):
            if not src_pred(r["source"]): continue
            kw = float(r.get("keyword_rate", 0.0))
            if kw - 0.007 < 0.005: continue
            g = grades.get(str(idx))
            if g is None or g.get("grade", -1) < 2: continue
            n_eligible += 1
            j = judgements.get(str(idx))
            if j is not None and j.get("genuine_count", -1) >= 1:
                n_genuine += 1

        m = json.loads(mp.read_text())
        m["genuine_backtracking_rate"] = (n_genuine / n_eligible) if n_eligible else 0.0
        m["n_eligible_for_bt_judge"] = n_eligible
        m["n_genuine_backtracking"] = n_genuine
        mp.write_text(json.dumps(m, indent=2))
        log.info("[metric] %-55s eligible=%d genuine=%d rate=%.3f",
                 cell_id, n_eligible, n_genuine,
                 m["genuine_backtracking_rate"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--judge", action="store_true",
                   help="run the Sonnet behavioral judge first (calls API)")
    p.add_argument("--concurrency", type=int, default=12)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["paths"]

    # Discover B1 files: canonical + per-cell.
    canonical = Path(paths["steering"])
    per_cell_dir = Path(paths["root"]) / "steering_per_cell"
    b1_files = []
    if canonical.exists():
        b1_files.append(canonical)
    if per_cell_dir.exists():
        b1_files.extend(sorted(per_cell_dir.glob("b1__*.json")))
    log.info("[discover] %d B1 files", len(b1_files))

    if args.judge:
        asyncio.run(judge_all(b1_files, cfg, args.concurrency))

    recompute_metrics(b1_files, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
