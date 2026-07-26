"""ACTMIX P2 driver — trains + evals the frozen EM btk-only cells via
the canonical runner (hard rule 1); one process per lane on GPU 2.

Usage:
  .venv/bin/python -m experiments.explorations.actmix_em.run_cells \
      --lane a --pin <freeze_sha>

Pin discipline (actmix-shared): --pin is the freeze commit sha taken
from ORIGIN's history at launch (rev-parse origin/arxiv — never
hand-typed). The driver refuses to start unless HEAD == pin AND the
pin is an ancestor of origin/arxiv. Idempotent: the runner cache-hits
on eval_key/train_key, so relaunching resumes where it left off.

Dirty-tree stance (em-redo precedent): the freeze commit pins the
code; cells run TEMP_BENCH_ALLOW_DIRTY=1 because leaderboard/manifest
appends of cell N dirty the tree for cell N+1 (7031/7116 existing
rows carry dirty=true). The wall log lands under /workspace/logs/
(outside the repo).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

os.environ.setdefault("TEMP_BENCH_ALLOW_DIRTY", "1")
os.environ.setdefault("AGENT_NAME", "runpod-2")

from temp_bench.core.runner import run_experiment

from experiments.explorations.actmix_em.cells import LANES

LOG_DIR = Path("/workspace/logs")


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=True).stdout.strip()


def assert_pinned(pin: str) -> None:
    head = _sh(["git", "rev-parse", "HEAD"])
    if head != pin:
        raise SystemExit(f"[pin] HEAD {head} != pin {pin} — refusing to run")
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", pin, "origin/arxiv"]).returncode
    if rc != 0:
        raise SystemExit(f"[pin] {pin} is not in origin/arxiv history — "
                         "refusing to run")
    dirty = _sh(["git", "status", "--porcelain", "--untracked-files=no"])
    if dirty:
        print(f"[pin] tracked modifications present at launch:\n{dirty}\n"
              "[pin] (allowed only for leaderboard/manifest appends)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=sorted(LANES))
    ap.add_argument("--pin", required=True)
    args = ap.parse_args()
    assert_pinned(args.pin)

    cells = LANES[args.lane]()
    wall_log = LOG_DIR / f"actmix_em_runs_{args.lane}.jsonl"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[lane {args.lane}] {len(cells)} cells; pin {args.pin[:12]}",
          flush=True)

    n_ok = n_fail = 0
    for i, c in enumerate(cells):
        tag = f"{c['cell_id']}/s{c['seed']}"
        t0 = time.time()
        rec = {"cell": tag, "arch": c["arch"], "seed": c["seed"],
               "datasource": c["datasource"], "lane": args.lane,
               "pin": args.pin,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            res = run_experiment(
                experiment="em",
                arch_name=c["arch"], seed=c["seed"],
                datasource_name=c["datasource"],
                training_cfg=c["training_cfg"],
                eval_cfg={},
                agent="runpod-2",
                allow_dirty=True,
            )
            wall = time.time() - t0
            row = res.row
            if hasattr(row, "model_dump"):
                row = row.model_dump()
            m = (row or {}).get("metrics", {})
            rec.update(
                ok=True, wall_s=round(wall, 1),
                train_key=res.train_key, eval_key=res.eval_key,
                train_cached=res.train_cached,
                pr_auc_S16=m.get("pr_auc_S16"),
                pr_auc_shuffled_S16=m.get("pr_auc_shuffled_S16"),
                shuffle_gap_S16=m.get("shuffle_gap_S16"),
                l0_per_window=m.get("l0_per_window"),
                l0_per_token=m.get("l0_per_token"),
            )
            n_ok += 1
            print(f"[{i + 1}/{len(cells)}] OK  {tag}  "
                  f"pr_auc_S16={m.get('pr_auc_S16')}  "
                  f"l0/tok={m.get('l0_per_token')}  "
                  f"({wall / 60:.1f} min)", flush=True)
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}",
                       tb=traceback.format_exc()[-2000:],
                       wall_s=round(time.time() - t0, 1))
            n_fail += 1
            print(f"[{i + 1}/{len(cells)}] FAIL {tag}: {e}", flush=True)
        with wall_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"[lane {args.lane}] DONE ok={n_ok} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
