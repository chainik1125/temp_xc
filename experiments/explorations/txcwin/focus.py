"""Focused head-to-head on ONE task, trained properly, with seeds.

The sweep is a triage at 400 steps, where trained dictionaries score about the
same as randomly initialised ones — so its ranking is suggestive at best. This
script takes one task and runs the panel the way it has to be run for a claim:
more steps, several seeds, an untrained control per cell, and matched realized
code rate.

Reported per (architecture, T):
    skill_trained  mean +/- spread over seeds, with a bootstrap CI per seed
    skill_init     the same architecture at random initialisation
    learned        skill_trained - skill_init   <- the quantity that matters
    l0             realized active latents per read (the fairness check)

Run:
  .venv/bin/python -m experiments.explorations.txcwin.focus \
      --task switch_clock --model gpt2 --steps 4000 --seeds 1,2,42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.txcwin.sweep import (
    PANEL, TASKS, RESULTS, build_cache, calibrate_k, encode_rows, load_cache,
    score_task, task_rows, train_one, _npz_key,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--t-ladder", default="2,4,8")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k-pos", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", default="1,2,42")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--match", default="nnz", choices=["nnz", "pertoken"],
                    help="nnz = equal features for the probe; pertoken = equal "
                         "atoms per token of the stream")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    spec = next(t for t in TASKS if t[0] == a.task)
    _, stem, field, kind, desc = spec
    seeds = [int(s) for s in a.seeds.split(",")]
    tag = a.tag or f"focus_{a.task}_{a.model.split('/')[-1]}_s{a.steps}"
    RESULTS.mkdir(parents=True, exist_ok=True)

    mm, meta = load_cache(build_cache(a.model, stem, a.layer))
    print(f"[focus] task={a.task} ({desc})")
    print(f"[focus] {meta['model']} L{a.layer} d={meta['d']} "
          f"tokens={meta['n_tokens']:,} steps={a.steps} seeds={seeds}")

    out = {"meta": {"task": a.task, "desc": desc, "kind": kind,
                    "model": a.model, "layer": a.layer, "steps": a.steps,
                    "d_sae": a.d_sae, "k_pos": a.k_pos, "batch": a.batch,
                    "seeds": seeds, "max_rows": a.max_rows,
                    "n_tokens": meta["n_tokens"], "d": meta["d"],
                    "triage_only": False}, "cells": []}

    Ts = [int(x) for x in a.t_ladder.split(",")]
    for arch_name, path, fixedT, blurb in PANEL:
        for T in ([1] if fixedT == 1 else Ts):
            k, got_nnz = calibrate_k(mm, meta, arch_name, path, T, a.d_sae,
                                     a.k_pos, a.match)
            print(f"  [calib] {arch_name} T={T}: nominal k={k} -> "
                  f"{a.match} budget {got_nnz:.1f} (target {a.k_pos})", flush=True)
            starts, y, docs = task_rows(stem, field, a.model, T, a.max_rows)
            for trained in (True, False):
                for seed in (seeds if trained else seeds[:1]):
                    t0 = time.time()
                    arch = train_one(mm, meta, arch_name, path, T, a.d_sae, k,
                                     a.steps if trained else 0, a.batch, seed)
                    codes, l0 = encode_rows(arch, mm, meta, starts, T)
                    r = score_task(codes, y, docs, kind)
                    out["cells"].append({
                        "arch": arch_name, "family": blurb, "T": T,
                        "task": a.task, "kind": kind, "trained": trained,
                        "seed": seed, "l0": round(l0, 2),
                        "rows": int(len(starts)),
                        "seconds": round(time.time() - t0, 1),
                        "nominal_k": k, "calibrated_nnz": round(got_nnz, 2),
                        "match": a.match, **r})
                    print(f"  {arch_name:20s} T={T:<2} "
                          f"{'trained' if trained else 'init   '} s{seed:<3} "
                          f"skill={r['skill']:+.4f} "
                          f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                          f"l0={l0:.1f} {round(time.time()-t0,1)}s", flush=True)
                    del arch
                    torch.cuda.empty_cache()

    p = RESULTS / f"{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n[focus] wrote {p}")

    # ── summary: what actually beats what ────────────────────────────
    print("\n=== per architecture (mean over seeds) ===")
    rows = []
    for arch_name, _, fixedT, blurb in PANEL:
        for T in ([1] if fixedT == 1 else Ts):
            tr = [c["skill"] for c in out["cells"]
                  if c["arch"] == arch_name and c["T"] == T and c["trained"]]
            un = [c["skill"] for c in out["cells"]
                  if c["arch"] == arch_name and c["T"] == T and not c["trained"]]
            l0 = [c["l0"] for c in out["cells"]
                  if c["arch"] == arch_name and c["T"] == T and c["trained"]]
            if not tr:
                continue
            rows.append((arch_name, T, float(np.mean(tr)), float(np.std(tr)),
                         float(np.mean(un)) if un else float("nan"),
                         float(np.mean(l0))))
    for arch_name, T, m, s, u, l0 in rows:
        print(f"  {arch_name:20s} T={T:<2} trained {m:+.4f} ± {s:.4f}  "
              f"init {u:+.4f}  learned {m-u:+.4f}  l0={l0:.1f}")
    base = max((m for n, T, m, s, u, l0 in rows if n in
                ("batchtopk_sae", "tsae")), default=float("nan"))
    win = max(((m, n, T, u) for n, T, m, s, u, l0 in rows
               if n not in ("batchtopk_sae", "tsae") and T >= 2),
              default=(float("nan"),) * 4)
    print(f"\n  best per-token baseline : {base:+.4f}")
    print(f"  best window arch        : {win[0]:+.4f}  ({win[1]} @T{win[2]})")
    print(f"  window advantage        : {win[0]-base:+.4f}")
    print(f"  of which learned        : {win[0]-win[3]:+.4f}")


if __name__ == "__main__":
    main()
