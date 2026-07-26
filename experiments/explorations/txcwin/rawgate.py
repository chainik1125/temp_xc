"""RAW gate — the precondition every candidate task must pass before dictionaries.

Why this file exists. On the switch-clock task, TXC-post's trained code beat the
per-token SAE's code by +0.15 (6 sigma, audited). It looked like the result the
paper needs. Then this check showed that a linear probe on the RAW activation at
the single last position already reaches r = 0.33 — higher than any dictionary.
So the signal was never non-local: the per-token *dictionary* simply threw more of
it away. A window-versus-per-token comparison between dictionaries says nothing
about temporal structure unless the raw activations show the same asymmetry.

The gate, per task and window size T, all on identical rows:
    raw_last     linear probe on the activation at the label position only
    raw_window   linear probe on the flattened T-position window
    raw_mean     linear probe on the window average (order-free)
    gap          raw_window - raw_last

A task can support the paper's claim only if `gap > 0` by more than the noise
floor: the window must contain something one position does not. If `raw_last`
already matches `raw_window`, the task is per-token-available and any dictionary
difference is about code efficiency, not temporal structure.

Note the dimensionality trap, visible in the numbers: a fixed-budget ridge on
T*d features degrades as T grows, so `raw_window` can fall with T for purely
statistical reasons. The gate therefore also reports `raw_mean`, which has the
same dimension as `raw_last` and so is directly comparable to it.

Run:  .venv/bin/python -m experiments.explorations.txcwin.rawgate --model gpt2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.explorations.txcwin.sweep import (
    LABELS, RESULTS, TASKS, _npz_key, build_cache, load_cache, score_task,
    task_rows,
)


def raw_arms(mm, meta, starts, y, docs, T, kind):
    d = meta["d"]
    last = np.ascontiguousarray(mm[starts + T - 1]).astype(np.float32)
    idx = (starts[:, None] + np.arange(T)[None, :]).reshape(-1)
    win = np.ascontiguousarray(mm[idx]).astype(np.float32).reshape(len(starts), T * d)
    mean = win.reshape(len(starts), T, d).mean(1)
    return {
        "raw_last": score_task(last, y, docs, kind),
        "raw_window": score_task(win, y, docs, kind),
        "raw_mean": score_task(mean, y, docs, kind),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--t-ladder", default="4,16")
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    Ts = [int(x) for x in a.t_ladder.split(",")]
    tag = a.tag or f"rawgate_{a.model.split('/')[-1]}_L{a.layer}"
    RESULTS.mkdir(parents=True, exist_ok=True)

    caches: dict[str, tuple] = {}
    out = {"meta": {"model": a.model, "layer": a.layer, "t_ladder": Ts,
                    "max_rows": a.max_rows}, "cells": []}
    print(f"{'task':16s} {'T':>3}  {'raw_last':>9} {'raw_mean':>9} "
          f"{'raw_window':>11}  {'gap(win-last)':>13}  verdict")
    print("-" * 84)
    for name, stem, field, kind, desc in TASKS:
        p = LABELS / f"{stem}_{_npz_key(stem, a.model)}.npz"
        if not p.exists():
            continue
        if stem not in caches:
            caches[stem] = load_cache(build_cache(a.model, stem, a.layer))
        mm, meta = caches[stem]
        for T in Ts:
            t0 = time.time()
            starts, y, docs = task_rows(stem, field, a.model, T, a.max_rows)
            arms = raw_arms(mm, meta, starts, y, docs, T, kind)
            gap = arms["raw_window"]["skill"] - arms["raw_last"]["skill"]
            gap_mean = arms["raw_mean"]["skill"] - arms["raw_last"]["skill"]
            # a task is a candidate only if the window carries MORE than one
            # position, by either the flatten or the dimension-matched mean
            live = max(gap, gap_mean) > 0.03
            out["cells"].append({
                "task": name, "desc": desc, "T": T, "kind": kind,
                "gap_window_minus_last": round(gap, 4),
                "gap_mean_minus_last": round(gap_mean, 4),
                "candidate": bool(live), "rows": int(len(starts)),
                "seconds": round(time.time() - t0, 1),
                **{k: v for k, v in arms.items()}})
            print(f"{name:16s} {T:>3}  {arms['raw_last']['skill']:>+9.3f} "
                  f"{arms['raw_mean']['skill']:>+9.3f} "
                  f"{arms['raw_window']['skill']:>+11.3f}  {gap:>+13.3f}  "
                  f"{'CANDIDATE' if live else 'per-token-available'}", flush=True)
    p = RESULTS / f"{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    cands = sorted({c["task"] for c in out["cells"] if c["candidate"]})
    print(f"tasks where a window carries more than one position: {cands or 'NONE'}")


if __name__ == "__main__":
    main()
