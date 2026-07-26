"""GAP-B of crossratify/MINI_CARD.md — raw gate at the claims' T and on
the 8B replication model (mac-b, salvage W2).

Everything measured is Andrii's code verbatim (`raw_arms`, `task_rows`,
`score_task`, `build_cache`); this file only restricts the task list to
the novelty pair and writes under `crossratify/results/` instead of
their `results/`. Gate criterion verbatim theirs:
CANDIDATE iff max(gap_window, gap_mean) > 0.03.

Run (in the pinned container; needs GPU for the cache build):
  .venv/bin/python -m experiments.explorations.txcwin.crossratify.rawgate_fill \
      --model gpt2 --layer 6 --t-ladder 8 --tag gpt2_L6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.explorations.txcwin.rawgate import raw_arms
from experiments.explorations.txcwin.sweep import (
    TASKS, build_cache, load_cache, task_rows,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
NOVELTY_TASKS = [t for t in TASKS if t[0] in ("novelty_resid",
                                              "novelty_rate")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--t-ladder", required=True)
    ap.add_argument("--max-rows", type=int, default=6000)  # = their gate
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    Ts = [int(x) for x in a.t_ladder.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    mm, meta = load_cache(build_cache(a.model, "novelty_fineweb", a.layer))
    out = {"meta": {"card": "crossratify/MINI_CARD.md GAP-B",
                    "model": a.model, "layer": a.layer, "t_ladder": Ts,
                    "max_rows": a.max_rows, "cache_tag": meta["tag"],
                    "n_tokens": meta["n_tokens"], "d": meta["d"]},
           "cells": []}
    for name, stem, field, kind, desc in NOVELTY_TASKS:
        for T in Ts:
            t0 = time.time()
            starts, y, docs = task_rows(stem, field, a.model, T, a.max_rows)
            arms = raw_arms(mm, meta, starts, y, docs, T, kind)
            gap = arms["raw_window"]["skill"] - arms["raw_last"]["skill"]
            gap_mean = arms["raw_mean"]["skill"] - arms["raw_last"]["skill"]
            out["cells"].append({
                "task": name, "desc": desc, "T": T, "kind": kind,
                "gap_window_minus_last": round(gap, 4),
                "gap_mean_minus_last": round(gap_mean, 4),
                "candidate": bool(max(gap, gap_mean) > 0.03),
                "rows": int(len(starts)),
                "seconds": round(time.time() - t0, 1), **arms})
            print(f"[{a.tag}] {name:14s} T={T:<3} "
                  f"last={arms['raw_last']['skill']:+.3f} "
                  f"mean={arms['raw_mean']['skill']:+.3f} "
                  f"win={arms['raw_window']['skill']:+.3f} "
                  f"gap={gap:+.3f} gap_mean={gap_mean:+.3f} "
                  f"{'CANDIDATE' if out['cells'][-1]['candidate'] else 'per-token-available'}",
                  flush=True)
            p = OUT / f"rawgate_fill_{a.tag}.json"   # incremental, resumable
            p.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / f'rawgate_fill_{a.tag}.json'}")


if __name__ == "__main__":
    main()
