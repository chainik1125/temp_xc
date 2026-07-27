"""em-redo Phase A driver — trains + evals every frozen cell via the
canonical runner (hard rule 1: everything through
``temp_bench.core.runner.run_experiment``; leaderboard rows are the
deliverable, eval = the detection-3.0.0 port in ``evals/em.py``).

Cell table: ``em_redo_cells.py`` (frozen). Idempotent — the runner
cache-hits on eval_key (leaderboard) and train_key (checkpoints), so
re-running after an interruption resumes where it left off.

Dirty-tree stance: the freeze commit pins the code; the sweep itself
runs with TEMP_BENCH_ALLOW_DIRTY=1 because the leaderboard/manifest
appends of cell N dirty the tree for cell N+1 (the established practice
— 7031/7116 existing rows carry dirty=true for this reason). The wall
log lands OUTSIDE the repo so results/ stays clean of untracked files.

Run:  .venv/bin/python -m experiments.explorations.conversion_depth.run_em_panel
      (optionally: ... run_em_panel panel | anchors | layer13 | layer15 | layer9)

``layerN`` selects that layer's panel + anchor cells — used to PARTITION
the 51 cells across three concurrent driver processes (disjoint
train/eval keys, so no duplicate-row races; the leaderboard append is
flock-protected). Added after cell 1 timed at ~31 min serial (26 h for
the full panel); the frozen cell table is untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TEMP_BENCH_ALLOW_DIRTY", "1")
os.environ.setdefault("AGENT_NAME", "runpod-c")

from temp_bench.core.runner import run_experiment

from experiments.explorations.conversion_depth.em_redo_cells import all_cells

WALL_LOG = Path("/workspace/conv_depth_caches/em_redo_results/runs_log.jsonl")


def main(which: str = "all"):
    WALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    cells = list(all_cells(include_anchors=True))
    if which == "anchors":
        cells = [c for c in cells if c["cell_id"].endswith("_anchor")]
    elif which == "panel":
        cells = [c for c in cells if not c["cell_id"].endswith("_anchor")]
    elif which.startswith("layer"):
        layer = int(which.removeprefix("layer"))
        cells = [c for c in cells if c["layer"] == layer]
    print(f"[panel] {len(cells)} cells ({which})", flush=True)
    n_ok = n_fail = 0
    for i, c in enumerate(cells):
        tag = f"{c['cell_id']}/L{c['layer']}/s{c['seed']}"
        t0 = time.time()
        rec = {"cell": tag, "arch": c["arch"], "layer": c["layer"],
               "seed": c["seed"], "datasource": c["datasource"],
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            res = run_experiment(
                experiment="em",
                arch_name=c["arch"], seed=c["seed"],
                datasource_name=c["datasource"],
                training_cfg=c["training_cfg"],
                eval_cfg={},
                agent=os.environ.get("AGENT_NAME", "runpod-c"),
                allow_dirty=True,
            )
            wall = time.time() - t0
            row = res.row
            if hasattr(row, "model_dump"):
                row = row.model_dump()
            m = (row or {}).get("metrics", {})
            rec.update(ok=True, wall_s=round(wall, 1),
                       train_key=res.train_key, eval_key=res.eval_key,
                       train_cached=res.train_cached,
                       eval_cached=res.eval_cached,
                       pr_auc_S16=m.get("pr_auc_S16"),
                       l0_per_token=m.get("l0_per_token"))
            n_ok += 1
            print(f"[{i + 1}/{len(cells)}] {tag} OK {wall:.0f}s "
                  f"pr_auc_S16={m.get('pr_auc_S16')} "
                  f"l0/tok={m.get('l0_per_token')}", flush=True)
        except Exception as e:
            wall = time.time() - t0
            rec.update(ok=False, wall_s=round(wall, 1), error=repr(e))
            n_fail += 1
            print(f"[{i + 1}/{len(cells)}] {tag} FAILED after {wall:.0f}s: "
                  f"{e}", flush=True)
            traceback.print_exc()
        with WALL_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"[panel] DONE ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
