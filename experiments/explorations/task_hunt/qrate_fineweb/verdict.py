"""Score the frozen CARD.md KEEP/KILL clauses for both punctint faces.

Reads `results/screen_<model>.json` and emits, per face and model, every
quantity the card's § 7 rules read — so the verdict is computed from the
artifacts rather than eyeballed:

  gap(T)            = window-MEAN − per-token             (the ladder)
  conversion_frac   = (tok − floor) / (best_window − floor)
                      the floor-relative diagnostic recommended in the
                      novelty LOG entry (an absolute gap cannot separate
                      "converted with residue" from "window-only")
  g_order(T)        = flatten − mean          (≤ 0 ⇒ order-free)
  shuffle_drop(T)   = flatten − context-shuffled  (≈ 0 ⇒ immune)
  anchor_gap        = ambient anchor's window-MEAN − per-token at T=16
  anchor_diff       = face gap(T=16) − anchor_gap
                      NOTE the metric mismatch, reported not hidden: the
                      face is 3-class (chance 1/3) and the anchor binary
                      (chance 1/2), so this difference is directional
                      evidence, not a calibrated quantity.
  3σ_null           = pooled over every permutation-null cell

Run: .venv/bin/python -m experiments.explorations.task_hunt.qrate_fineweb.verdict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")
FACES = ("q", "list")
TS = (4, 8, 16, 32, 64)
FLAT_TS = (4, 8, 16, 32)


def main() -> None:
    out: dict = {"models": {}, "pooled": {}}
    nulls = []
    for m in MODELS:
        p = RES / f"screen_{m}.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text())["cells"]

        def g(k):
            return c[k]["acc_test"] if k in c else float("nan")

        rec: dict = {}
        for face in FACES:
            tok = g(f"{face}/tok_linear")
            floor = g(f"{face}/position_floor")
            means = {T: g(f"{face}/T{T}/win_mean_linear") for T in TS}
            gaps = {T: means[T] - tok for T in TS}
            finite = [v for v in means.values() if np.isfinite(v)]
            best = max(finite) if finite else float("nan")
            a_tok = g(f"{face}_anchor/tok_linear")
            a_win = g(f"{face}_anchor/T16/win_mean_linear")
            rec[face] = {
                "tok": tok, "position_floor": floor,
                "means": means, "gaps": gaps, "best_window": best,
                "conversion_frac": (tok - floor) / (best - floor)
                if np.isfinite(best) and best > floor else float("nan"),
                "g_order": {T: g(f"{face}/T{T}/win_linear") - means[T]
                            for T in FLAT_TS},
                "shuffle_drop": {T: g(f"{face}/T{T}/win_linear")
                                 - g(f"{face}/T{T}/win_shuf_linear")
                                 for T in FLAT_TS},
                "anchor_tok": a_tok, "anchor_win_T16": a_win,
                "anchor_gap": a_win - a_tok,
                "anchor_diff_T16": gaps[16] - (a_win - a_tok),
                "gap_grows": bool(np.isfinite(gaps[64]) and
                                  gaps[64] >= gaps[4] + 0.02),
                "clears_floor": bool(best - floor >= 0.05),
            }
            for k in (f"{face}/null_tok_linear",
                      f"{face}/T16/null_win_linear"):
                if np.isfinite(g(k)):
                    nulls.append(g(k))
        out["models"][m] = rec

    if nulls:
        sd = float(np.std(nulls, ddof=1))
        out["pooled"] = {"null_cells": len(nulls), "null_mean":
                         float(np.mean(nulls)), "null_sd": sd,
                         "three_sigma": 3 * sd}

    (RES / "verdict.json").write_text(json.dumps(out, indent=2))

    ts = out["pooled"].get("three_sigma", float("nan"))
    print(f"pooled 3sigma_null = {ts:.4f}  (n={out['pooled'].get('null_cells')})\n")
    for face in FACES:
        print(f"===== face: {face} =====")
        hdr = f"{'model':12s} {'tok':>6} {'floor':>6} " + \
              " ".join(f"g{T:<4d}" for T in TS) + \
              f" {'conv%':>6} {'anchgap':>8} {'anchdiff':>9} {'grows':>6}"
        print(hdr)
        for m, rec in out["models"].items():
            r = rec[face]
            print(f"{m:12s} {r['tok']:6.3f} {r['position_floor']:6.3f} "
                  + " ".join(f"{r['gaps'][T]:+.3f}" for T in TS)
                  + f" {r['conversion_frac']*100:5.0f}% {r['anchor_gap']:+8.3f}"
                  f" {r['anchor_diff_T16']:+9.3f} {str(r['gap_grows']):>6}")
        n_keep = sum(1 for rec in out["models"].values()
                     if max(rec[face]["gaps"].values()) >= 0.05
                     and rec[face]["gap_grows"]
                     and rec[face]["clears_floor"]
                     and rec[face]["anchor_diff_T16"] >= 0.02)
        print(f"  models satisfying every KEEP clause: {n_keep}/"
              f"{len(out['models'])}  (KEEP needs >= 2)\n")
    print(f"wrote {RES / 'verdict.json'}")


if __name__ == "__main__":
    main()
