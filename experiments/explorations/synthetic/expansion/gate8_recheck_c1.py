"""Cycle-2 rider — gate-8 recheck of the two Cycle-1 PROCEED mirrors.

The Cycle-1 review added guardrail 8 (non-fitted-moment mirror gate) after
those mirrors were fit; this recheck applies it retroactively from the STORED
held-out validations (`records/<name>/calibration_stats.json`) — no new
labels, no refit, pure analysis. Record pass/fail per mirror.

**Preregistered here, before computing** (the moments and tolerances are
fixed by this script's constants, chosen from what each fit does NOT target):

- `assumption-then-consequence` (markov): the MLE transition fit targets
  next-symbol counts, not the self-match autocorrelation → moment =
  **self-match ACF(1)** held-out real vs synthetic, tolerance **±0.05 abs**.
- `uncertainty-hedging-drift` (ar1+trend): the fit targets the lag-1 residual
  persistence (ρ) + trend, so lag-1 of the raw ACF is fit-adjacent; the first
  genuinely non-fitted shape moment is **ACF(2)**, tolerance **±0.05 abs**.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.gate8_recheck_c1
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHECKS = {
    "assumption-then-consequence": {
        "process": "markov", "moment": "selfmatch_acf[lag1]", "curve": "acf",
        "idx": 0, "tol_abs": 0.05,
        "rationale": "MLE transition fit targets next-symbol counts, not the "
                     "self-match autocorrelation curve"},
    "uncertainty-hedging-drift": {
        "process": "ar1+trend", "moment": "acf[lag2]", "curve": "acf",
        "idx": 1, "tol_abs": 0.05,
        "rationale": "fit targets lag-1 residual persistence (rho) + trend; "
                     "ACF(2) is the first non-fitted shape moment"},
}


def main():
    out = {"date": "2026-07-14", "note": "retroactive gate-8 check from stored "
           "held-out validations; moments+tolerances preregistered in "
           "gate8_recheck_c1.py before computation", "checks": {}}
    for name, spec in CHECKS.items():
        blob = json.loads((HERE / "records" / name / "calibration_stats.json").read_text())
        mv = blob["mirror"]["validation"]
        rv = float(mv["real"][spec["curve"]][spec["idx"]])
        sv = float(mv["synthetic"][spec["curve"]][spec["idx"]])
        err = abs(rv - sv)
        res = dict(spec, real_heldout=rv, synthetic=sv, abs_err=err,
                   passes=bool(err <= spec["tol_abs"]))
        out["checks"][name] = res
        print(f"{name:32} {spec['moment']:20} real={rv:+.4f} syn={sv:+.4f} "
              f"|err|={err:.4f} tol={spec['tol_abs']} -> "
              f"{'PASS' if res['passes'] else 'FAIL'}")
    (HERE / "results" / "gate8_recheck_cycle1.json").write_text(json.dumps(out, indent=2))
    print("-> results/gate8_recheck_cycle1.json")


if __name__ == "__main__":
    main()
