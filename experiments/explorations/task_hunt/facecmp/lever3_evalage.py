"""LEVER 3 on the corpus that already has the narrow `w` — `evalage`.

## Why this is the missing cell, and not a repeat

The hub's un-parking (00:20) names one priority: *"re-label an existing
corpus at T=16 and measure arm−floor BEFORE generating anything."*
**I ran that sweep already** — `lever3_test.py`, frozen `bf7aa3b5f`,
T ∈ {4,8,16,32,64} on `gemma2_2b`@512, **0/5 KEEP-shaped**:

    T     arm-floor    gain-0.05     min(both)
    4      +0.0360      -0.0400       -0.0400
    8      +0.0267      -0.0291       -0.0291
    16     -0.0107      -0.0249       -0.0249   <- closest
    32     -0.0599      +0.0010       -0.0599
    64     -0.1175      +0.0588       -0.1175

**But that was `retryesc_gen`, where `w = 25`.** The floor's horizon is
`T + w`, so even at T=16 it still sees **41** tokens. I then measured the
`w` half by counterfactual (`d8dd927f4`) and reported **"do not fund a
narrow-event corpus"** — 0.08 sigma, optimistically biased.

**That verdict was about GENERATING a corpus. We already own one.**
`evalage` has **w = 13** (measured on 3 legs, `floor_predictor_test`), so
its floor horizon at T=16 is **29, not 41** — the `w` lever applied for
free to a corpus that exists. **Neither of my lever-3 scripts tested
this**: both held the corpus fixed and moved a knob. This moves the
corpus and reuses the frozen instrument unchanged.

It is also the right corpus on the merits: `evalage` is the **rescue
candidate** — it cleared the width null, its visible floor, and the
within-conversation control on every leg, and failed only the hardcoded
`gain >= +0.05`, by ~0.5 sigma on its best leg. `retryesc_gen` was
dropped as *"wrong on all three levers"*.

## Pre-registered, written and committed BEFORE the numbers

  P1  **evalage's floor is much lower than retryesc's at every T.**
      Measured floor_excess at the screened T: **0.064 (evalage gemma2)
      vs 0.275 (retryesc gemma2)** — w=13 and lower event density.
      **Falsifier: if evalage's floor is NOT materially lower, my
      corrected floor law (`fe ~ P(masked turn-token in T+w)`) is wrong
      and I report that against myself.**
  P2  therefore `arm - floor` should be positive at MORE T values than
      retryesc's 2/5 (which were T4 and T8 only).
  P3  **the binding constraint FLIPS.** retryesc was FLOOR-bound (it had
      gain at large T but lost to the floor). evalage should be
      GAIN-bound: its screened gain was only +0.0460 at SEQ_LEN 128, so
      clearing +0.05 is the hard half here, not beating the floor.
  P4  deliverable is the T maximising `min(arm - floor, gain - 0.05)` —
      **not** the T that maximises either. If no T satisfies both, that
      is a real negative about this corpus and it is reported as one.
  P5  **if a cell DOES clear both, it is a RESCUED candidate at
      corrected geometry and must be disclosed as rescued** — screen
      verdict WEAK, re-measured because the miss was inside the noise
      and the geometry was wrong. Never quoted as though it had passed
      the original screen.

**P3 and P5 are the ones I most want on record.** P3 because if the
constraint does not flip, my whole account of the two horizons is
suspect. P5 because this is the first corpus in the program with a real
chance of clearing both bars, and that is exactly when the temptation to
launder a rescue into a pass appears.

Instrument frozen: `arm_test.build_rows` / `screen` unchanged except that
the grid filename is now a parameter (`GRID_PAT`) instead of a hardcode;
the default is byte-identical, verified, so the frozen retryesc results
stand.

Run: PYTHONPATH=. python -m experiments.explorations.task_hunt.facecmp.lever3_evalage
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TS = [4, 8, 16, 32, 64]
CHANCE = 1.0 / 3.0


def main():
    import experiments.explorations.task_hunt.facecmp.arm_test as at
    import experiments.explorations.task_hunt.facecmp.face_battery as fb

    root = os.environ.get("FACECMP_CACHE_ROOT")
    if root:
        at.CACHE_ROOT = Path(root)
    at.GRIDS = (Path(__file__).resolve().parents[1] / "evalage" / "grids")
    at.GRID_PAT = "elicit_evalage_screen_{tag}.npz"

    key = os.environ.get("FACECMP_MODEL", "gemma2_2b")
    tag = os.environ.get("FACECMP_TAG", "evalage_gemma2_512")

    out_dir = Path(__file__).resolve().parent / "results" / "lever3"
    out_dir.mkdir(parents=True, exist_ok=True)

    at.FACE, at.H = f"L3EV_{tag}", 64
    at.AX_TS, at.FOREIGN_TS = TS, [16, 64]
    at.RES = out_dir
    at.rate_face = fb.f_age
    at.screen(key)

    p = out_dir / f"arm_test_{key}.json"
    d = json.loads(p.read_text())
    c, F = d["cells"], at.FACE
    tok = max(c[f"{F}/tok_linear"]["acc_test"], c[f"{F}/tok_mlp"]["acc_test"])

    rows = []
    print(f"\n{'T':>4}{'arm':>9}{'floor':>9}{'arm-floor':>11}"
          f"{'gain':>9}{'gain-0.05':>11}{'min(both)':>11}")
    print("-" * 64)
    for T in TS:
        arm = max(c[f"{F}/T{T}/actxmean_linear"]["acc_test"],
                  c[f"{F}/T{T}/actxmean_mlp"]["acc_test"])
        fl = c[f"{F}/T{T}/visible_evidence_floor"]["acc_test"]
        gain = arm - tok
        both = min(arm - fl, gain - 0.05)
        rows.append({"T": T, "arm": arm, "floor": fl,
                     "arm_minus_floor": arm - fl, "gain": gain,
                     "gain_margin": gain - 0.05, "min_both": both,
                     "floor_excess": fl - CHANCE})
        print(f"{T:>4}{arm:>9.4f}{fl:>9.4f}{arm - fl:>+11.4f}"
              f"{gain:>+9.4f}{gain - 0.05:>+11.4f}{both:>+11.4f}")

    best = max(rows, key=lambda r: r["min_both"])
    keep = [r for r in rows if r["arm_minus_floor"] > 0 and r["gain"] >= 0.05]
    print(f"\n  tok baseline: {tok:.4f}")
    print(f"  best T by min(arm-floor, gain-0.05): T{best['T']} "
          f"({best['min_both']:+.4f})")
    print(f"  cells satisfying BOTH (KEEP-shaped): {len(keep)}/{len(rows)}"
          + (f" -> T{[r['T'] for r in keep]}" if keep else "  <- none"))

    # P1/P3 scored explicitly against the frozen retryesc run
    ref = out_dir / "lever3_gemma2_512.json"
    if ref.exists():
        rr = {r["T"]: r for r in json.loads(ref.read_text())["rows"]}
        print(f"\n  {'T':>4}{'floor evalage':>15}{'floor retryesc':>16}"
              f"{'P1 lower?':>11}")
        for r in rows:
            o = rr.get(r["T"])
            if o:
                print(f"  {r['T']:>4}{r['floor']:>15.4f}{o['floor']:>16.4f}"
                      f"{('YES' if r['floor'] < o['floor'] else 'NO'):>11}")
        pos = sum(1 for r in rows if r["arm_minus_floor"] > 0)
        opos = sum(1 for t, o in rr.items() if o["arm_minus_floor"] > 0)
        print(f"  P2: arm>floor at {pos}/5 T (retryesc {opos}/5) -> "
              f"{'HELD' if pos > opos else 'NOT HELD'}")
        nf = sum(1 for r in rows if r["arm_minus_floor"] <= 0)
        ng = sum(1 for r in rows if r["gain"] < 0.05)
        print(f"  P3: floor-bound cells {nf}, gain-bound cells {ng} -> "
              f"{'FLIPPED (gain-bound)' if ng > nf else 'still floor-bound'}")

    (out_dir / f"lever3_{tag}.json").write_text(json.dumps(
        {"model": key, "tag": tag, "corpus": "evalage", "w": 13, "tok": tok,
         "rows": rows, "best_T_by_min_both": best, "keep_shaped_cells": keep,
         "note": "lever 3 with the w half applied for FREE — evalage already "
                 "has w=13 vs retryesc's 25. Instrument frozen; only the "
                 "corpus moved. A KEEP here is a RESCUE and is disclosed as "
                 "one (P5)."}, indent=2))
    print(f"\nwrote {out_dir / f'lever3_{tag}.json'}")


if __name__ == "__main__":
    main()
