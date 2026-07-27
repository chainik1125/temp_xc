"""ACTMIX RLHF btk-only driver — canonical runner, one lane process.

Usage:
  .venv/bin/python -m experiments.explorations.actmix_rlhf.run_cells \
      --lane r --pin <sha>

Same pin/dirty discipline as actmix_em/run_cells.py. Wall log:
/workspace/logs/actmix_rlhf_runs_<lane>.jsonl. The first cell
(sae_k500) is the smoke/neg_frac gate — inspect its train log +
realized l0 before trusting the rest (CARD § 2).
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

_frac = os.environ.get("TEMP_BENCH_GPU_FRACTION")
if _frac:
    import torch
    torch.cuda.set_per_process_memory_fraction(float(_frac))

from temp_bench.core.runner import run_experiment

from experiments.explorations.actmix_rlhf.cells import LANES

LOG_DIR = Path("/workspace/logs")


def _sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          check=True).stdout.strip()


def assert_pinned(pin: str) -> None:
    head = _sh(["git", "rev-parse", "HEAD"])
    if head != pin:
        raise SystemExit(f"[pin] HEAD {head} != pin {pin}")
    rc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", pin, "origin/arxiv"]).returncode
    if rc != 0:
        raise SystemExit(f"[pin] {pin} not in origin/arxiv history")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=sorted(LANES))
    ap.add_argument("--pin", required=True)
    args = ap.parse_args()
    assert_pinned(args.pin)

    cells = LANES[args.lane]()
    wall_log = LOG_DIR / f"actmix_rlhf_runs_{args.lane}.jsonl"
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
                experiment="rlhf",
                arch_name=c["arch"], seed=c["seed"],
                datasource_name=c["datasource"],
                training_cfg=c["training_cfg"],
                eval_cfg={},
                agent=os.environ.get("AGENT_NAME", "runpod-2"),
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
                       preference_auc_k20=m.get("preference_auc_k20"),
                       shuffle_gap_auc_k20=m.get("shuffle_gap_auc_k20"),
                       mass_at_20=m.get("mass_at_20"),
                       l0_per_unit=m.get("l0_per_unit"))
            n_ok += 1
            print(f"[{i + 1}/{len(cells)}] OK  {tag}  "
                  f"auc={m.get('preference_auc_k20')}  "
                  f"l0={m.get('l0_per_unit')}  ({wall / 60:.1f} min)",
                  flush=True)
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
