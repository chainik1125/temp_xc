"""LEVER 3 — shrink the FLOOR's horizon. The cheap step the hub asked for
before any pod: re-label an existing corpus across T and measure arm−floor.

## Why this is the binding lever

Three levers came out of tonight's matrix, and two are already measured:

  1 CONTEXT  SEQ_LEN 128 -> 512      gpt2 +0.0596 -> +0.0928
  2 MODEL    124M -> 2.6B            alone: nothing; with (1): -> +0.1088
  3 FLOOR    shrink T + w            UNTESTED, and now binding

The floor is a probe on `(sage_floor(T), dose_window_count(event_mask, T))`,
so **its horizon is `T + w`** and depends on NOTHING else — not the model,
not the context length. Measured proof: it did not move across four
model x context combinations (gpt2 0.5859/0.5932, gemma2 0.6048/0.6081).
**That makes it the one term in this benchmark that is purely ours to set.**

At the screened setting T=64, w=25 -> floor horizon 89, against a readable
horizon of ~100 (gpt2) — a band of factor 1.1, which is why nothing ever
cleared. **At T=16 the floor horizon drops to 41**, while gemma2_2b at 512
context reads past 262. Band becomes ~(41, 262): **factor 6.4.**

## What this measures, and what it cannot

Sweeps T over an EXISTING corpus and reports `arm − floor` per T. `w` is
fixed at 25 by that corpus, so this tests the T half of lever 3 only; the
`w` half needs a generated corpus with narrow event turns and is NOT
tested here.

**Pre-registered before running:**

  P1  floor falls steeply with T -- it is ~P(masked token within T+w), and
      at T=16 that horizon is 41 vs 89. Expect floor_excess roughly halved.
  P2  arm falls too, but LESS steeply, because each activation in the
      window still carries up to 512 tokens of context regardless of T.
  P3  therefore `arm - floor` should be LEAST NEGATIVE at small T, and may
      turn positive. On gpt2@128 the trend was already visible and
      monotone: T16 -0.017, T32 -0.077, T64 -0.148.
  P4  the gain-vs-tok bar moves the OPPOSITE way -- small T means less
      window to aggregate. **The KEEP needs both at once**, so the
      deliverable of this script is the T that maximises
      `min(arm - floor, gain - 0.05)`, not the T that maximises either.

**P4 is the whole difficulty and I want it on record before the numbers:**
if no T satisfies both, that is a real negative about the corpus geometry
and it should be reported as one, not narrated around by picking whichever
T looks best on one axis.

Run: PYTHONPATH=. python -m experiments.explorations.task_hunt.facecmp.lever3_test
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CACHE_ENV = "FACECMP_CACHE_ROOT"
TS = [4, 8, 16, 32, 64]
CHANCE = 1.0 / 3.0


def main():
    import experiments.explorations.task_hunt.facecmp.arm_test as at
    import experiments.explorations.task_hunt.facecmp.face_battery as fb

    root = os.environ.get(CACHE_ENV)
    if root:
        at.CACHE_ROOT = Path(root)
    key = os.environ.get("FACECMP_MODEL", "gemma2_2b")
    tag = os.environ.get("FACECMP_TAG", "gemma2_512")

    out_dir = Path(__file__).resolve().parent / "results" / "lever3"
    out_dir.mkdir(parents=True, exist_ok=True)

    at.FACE, at.H = f"L3_{tag}", 64
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
        rows.append({"T": T, "arm": arm, "floor": fl, "arm_minus_floor": arm - fl,
                     "gain": gain, "gain_margin": gain - 0.05, "min_both": both})
        print(f"{T:>4}{arm:>9.4f}{fl:>9.4f}{arm - fl:>+11.4f}"
              f"{gain:>+9.4f}{gain - 0.05:>+11.4f}{both:>+11.4f}")

    best = max(rows, key=lambda r: r["min_both"])
    keep = [r for r in rows if r["arm_minus_floor"] > 0 and r["gain"] >= 0.05]
    print(f"\n  tok baseline: {tok:.4f}")
    print(f"  best T by min(arm-floor, gain-0.05): T{best['T']} "
          f"({best['min_both']:+.4f})")
    print(f"  cells satisfying BOTH (KEEP-shaped): "
          f"{len(keep)}/{len(rows)}"
          + (f" -> T{[r['T'] for r in keep]}" if keep else "  <- none"))

    (out_dir / f"lever3_{tag}.json").write_text(json.dumps(
        {"model": key, "tag": tag, "tok": tok, "rows": rows,
         "best_T_by_min_both": best, "keep_shaped_cells": keep,
         "note": "feasibility probe on a BORROWED corpus; w fixed at 25, so "
                 "only the T half of lever 3 is tested"}, indent=2))
    print(f"\nwrote {out_dir / f'lever3_{tag}.json'}")


if __name__ == "__main__":
    main()
