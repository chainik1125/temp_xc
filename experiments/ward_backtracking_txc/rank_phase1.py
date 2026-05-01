"""Rank Phase 1 sweep cells by the hill-climb metric and pick the winner.

Phase 1's B1 results live in `<paths.steering>` (the canonical merged
JSON across all archs/hookpoints). We re-aggregate per-cell from that
JSON without re-running B1, write each cell's metric to
`<root>/cell_metrics/<cell_id>.json`, and emit the leaderboard at
`<root>/rank_phase1.json`.

The Phase 1 sweep uses sprint-style filenames (no k_per_position or seed
in the source tag — they're implicit from cfg). For consistency with
Phase 2's cell-id scheme we synthesize cell ids using cfg's defaults
(k = txc.k_per_position, seed = txc.seed).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import yaml

from experiments.ward_backtracking_txc.cell_id import (
    Cell, cell_metric_path, sonnet_grades_path,
)
from experiments.ward_backtracking_txc import metrics as metrics_mod

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.rank_phase1")


# Sprint-era source tag pattern: <arch>_<hp_component>_L<layer>_f<id>_<mode>
SPRINT_TAG = re.compile(r"^(?P<arch>(?:topk_sae|stacked_sae|tsae|txc))_"
                        r"(?P<hp>[a-z]+_L\d+)_f(?P<fid>\d+)_(?P<mode>pos0|union)$")


def _group_sources_by_cell(rows: list[dict], default_k: int, default_seed: int):
    """Group source tags by (arch, hookpoint) cell. Returns
    {cell_key (str): set of source tags} where cell_key uses the paper-budget
    cell-id format with default k and seed substituted in.
    """
    cells = defaultdict(set)
    for r in rows:
        m = SPRINT_TAG.match(r["source"])
        if not m: continue
        arch = m.group("arch"); hp = m.group("hp")
        c = Cell(arch=arch, hookpoint_key=hp,
                 k_per_position=default_k, seed=default_seed)
        cells[c.id].add(r["source"])
    return cells


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--metric-key", type=str, default="primary_kw_at_coh",
                   choices=("primary_kw_at_coh", "primary_kw_at_sonnet_coh"),
                   help="rank cells by this metric (Sonnet variant requires "
                        "coherence_grades populated for canonical B1 JSON)")
    p.add_argument("--out-name", type=str, default="rank_phase1.json",
                   help="leaderboard output filename under <root>/")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    paths = cfg["paths"]
    txc = cfg["txc"]

    b1_path = Path(paths["steering"])
    if not b1_path.exists():
        log.error("[fatal] %s missing — run Phase 1 first", b1_path); return 1

    rows = metrics_mod.load_b1_rows(b1_path)
    cells = _group_sources_by_cell(rows, default_k=int(txc["k_per_position"]),
                                   default_seed=int(txc["seed"]))
    if not cells:
        log.error("[fatal] no Phase-1 cells parsed from %s", b1_path); return 1

    metrics_dir = Path(paths["root"]) / "cell_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    grades_dir = Path(paths["root"]) / "coherence_grades"
    canonical_grades = grades_dir / Path(b1_path).name
    sonnet_grades = metrics_mod.load_sonnet_grades(canonical_grades)
    if args.metric_key == "primary_kw_at_sonnet_coh" and sonnet_grades is None:
        log.warning("[warn] --metric-key sonnet but %s missing; "
                    "rank will fall back to legacy floor.", canonical_grades)

    leaderboard = []
    for cell_id, source_tags in cells.items():
        cell = Cell.from_id(cell_id)
        m = metrics_mod.cell_metric(rows, list(source_tags),
                                    sonnet_grades=sonnet_grades)
        m["cell_id"] = cell_id
        m["arch"] = cell.arch
        m["hookpoint"] = cell.hookpoint_key
        m["k_per_position"] = cell.k_per_position
        m["seed"] = cell.seed
        m["n_sources"] = len(source_tags)
        cell_metric_path(cell, metrics_dir).write_text(json.dumps(m, indent=2))
        leaderboard.append(m)

    leaderboard.sort(key=lambda m: -m.get(args.metric_key, 0.0))
    log.info("[leaderboard] (top 5 by %s)", args.metric_key)
    for m in leaderboard[:5]:
        sonnet_str = (
            f" sonnet={m.get('primary_kw_at_sonnet_coh', 0.0):.4f} "
            f"frac_coh_s={m.get('frac_coh_sonnet', 0.0):.2f}"
            if "primary_kw_at_sonnet_coh" in m else ""
        )
        log.info("  %-50s primary=%.4f mag=%+.0f frac_coh=%.2f n_src=%d%s",
                 m["cell_id"], m["primary_kw_at_coh"],
                 m["best_magnitude"], m["frac_coherent"], m["n_sources"],
                 sonnet_str)

    out = {"winner_cell_id": leaderboard[0]["cell_id"],
           "winner_metric": leaderboard[0],
           "metric_key": args.metric_key,
           "leaderboard": leaderboard,
           "n_cells": len(leaderboard)}
    out_path = Path(paths["root"]) / args.out_name
    out_path.write_text(json.dumps(out, indent=2))
    log.info("[saved] %s — winner: %s", out_path, leaderboard[0]["cell_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
