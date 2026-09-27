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

import numpy as np

from experiments.explorations.txcwin.rawgate import raw_arms
from experiments.explorations.txcwin.sweep import (
    TASKS, build_cache, load_cache, score_task, task_rows,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
NOVELTY_TASKS = [t for t in TASKS if t[0] in ("novelty_resid",
                                              "novelty_rate")]
# The flattened-window ridge builds a (T*d)^2 Gram; above ~46341 the
# element count exceeds int32 and numpy's bundled BLAS segfaults
# (observed 3x as exit 139 on the 8B at T=16, flatten dim 65536, at
# both 48G and 128G — not a memory limit). For such cells compute only
# the two d-dim arms (raw_last / raw_mean — the gate doc's
# "dimension-matched, directly comparable" arm) and disclose the
# omission in the cell. Post-freeze amendment, disclosed in
# CROSSRATIFY.md; claims-T cells (T=8) are unaffected.
FLATTEN_DIM_MAX = 40000


def lean_arms(mm, meta, starts, y, docs, T, kind):
    d = meta["d"]
    last = np.ascontiguousarray(mm[starts + T - 1]).astype(np.float32)
    idx = (starts[:, None] + np.arange(T)[None, :]).reshape(-1)
    mean = np.ascontiguousarray(mm[idx]).astype(np.float32) \
        .reshape(len(starts), T, d).mean(1)
    return {
        "raw_last": score_task(last, y, docs, kind),
        "raw_window": None,
        "raw_mean": score_task(mean, y, docs, kind),
        "raw_window_omitted": (
            f"flatten dim {T * d} > {FLATTEN_DIM_MAX}: BLAS int32 "
            "overflow segfault; gate evaluated on gap_mean only"),
    }


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
            lean = T * meta["d"] > FLATTEN_DIM_MAX
            arms = (lean_arms if lean else raw_arms)(
                mm, meta, starts, y, docs, T, kind)
            gap = (arms["raw_window"]["skill"] - arms["raw_last"]["skill"]
                   if arms["raw_window"] is not None else None)
            gap_mean = arms["raw_mean"]["skill"] - arms["raw_last"]["skill"]
            out["cells"].append({
                "task": name, "desc": desc, "T": T, "kind": kind,
                "gap_window_minus_last": (round(gap, 4)
                                          if gap is not None else None),
                "gap_mean_minus_last": round(gap_mean, 4),
                "candidate": bool(max(gap if gap is not None else -9,
                                      gap_mean) > 0.03),
                "rows": int(len(starts)),
                "seconds": round(time.time() - t0, 1), **arms})
            win_s = (f"{arms['raw_window']['skill']:+.3f}"
                     if arms["raw_window"] is not None else "  n/a ")
            gap_s = f"{gap:+.3f}" if gap is not None else "  n/a "
            print(f"[{a.tag}] {name:14s} T={T:<3} "
                  f"last={arms['raw_last']['skill']:+.3f} "
                  f"mean={arms['raw_mean']['skill']:+.3f} "
                  f"win={win_s} gap={gap_s} gap_mean={gap_mean:+.3f} "
                  f"{'CANDIDATE' if out['cells'][-1]['candidate'] else 'per-token-available'}",
                  flush=True)
            p = OUT / f"rawgate_fill_{a.tag}.json"   # incremental, resumable
            p.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / f'rawgate_fill_{a.tag}.json'}")


if __name__ == "__main__":
    main()
