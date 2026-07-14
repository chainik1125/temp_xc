"""Cycle-2 rider (follow-on) — menu-upgrade attempt for the hedging mirror.

The retroactive gate-8 check FAILED the `hedging_drift` ar1+trend mirror on
ACF(2) (real 0.155 vs syn 0.085, |err| 0.071 > 0.05): the AR(1) collapses the
real stream's slow-decay plateau. The measurement verdict (TEMPORAL,
DC-slow-drift) is untouched — this is a mirror-fidelity repair, menu-first
per the README guardrails ("generators from the menu; bespoke requires
written justification").

**Preregistered here, before fitting:** try `semi_markov` (Appendix B row
"heavy-tailed dwell") on the 3-symbol ordinal stream. Acceptance = BOTH:
- gate-8 moment: **ACF(2)** held-out real vs syn within **±0.05 abs** (the
  same moment+tolerance the ar1 failed, for comparability);
- matched-persistence check: **ACF(1)** within **±0.05 abs**.
Also reported (diagnostics, not gates): MI(1), dwell mean/CV, and the
position-profile correlation — semi-Markov has NO drift term, so the trend
(0.68→0.97), which ar1+trend matched, becomes deliberately NOT matched. If
semi_markov passes, the spec gets a dated amendment swapping the mirror and
logging exactly that trade; if it fails too, the spec's mirror is recorded
INVALID pending a better process (a Cycle-3 item).

Pure analysis of cached labels — no API calls, no new data.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.mirror_upgrade_hedging
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explorations.synthetic.expansion import mirrors
from explorations.synthetic.expansion import signature as sig

HERE = Path(__file__).resolve().parent
SPLIT_SEED = 1000  # fresh 70/30 doc split (C1's internal rng state not replayed)
TOL = 0.05


def main():
    blob = json.loads((HERE / "records/uncertainty-hedging-drift/labels.json").read_text())
    seqs = [np.array(x, dtype=np.int8) for x in blob["labels"] if x is not None]
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(seqs))
    cut = int(0.7 * len(seqs))
    train = [seqs[i] for i in idx[:cut]]
    ev = [seqs[i] for i in idx[cut:]]

    params = mirrors.fit_semi_markov(train, n_symbols=3)
    syn = mirrors.gen_semi_markov(params, [s.size for s in ev], rng)

    evf = [s.astype(float) for s in ev]
    synf = [np.asarray(s, dtype=float) for s in syn]
    acf_r, acf_s = sig.acf(evf), sig.acf(synf)
    mi_r = sig.mi_vs_lag(ev, 12, 3)
    mi_s = sig.mi_vs_lag([np.asarray(s, dtype=np.int8) for s in syn], 12, 3)
    dw_r, dw_s = sig.dwell_stats(ev), sig.dwell_stats(syn)
    prof_r = sig.position_profile(evf)
    prof_s = sig.position_profile(synf)

    checks = {
        "gate8_acf2": {"real": float(acf_r[1]), "syn": float(acf_s[1]),
                       "abs_err": float(abs(acf_r[1] - acf_s[1])), "tol": TOL,
                       "passes": bool(abs(acf_r[1] - acf_s[1]) <= TOL)},
        "matched_acf1": {"real": float(acf_r[0]), "syn": float(acf_s[0]),
                         "abs_err": float(abs(acf_r[0] - acf_s[0])), "tol": TOL,
                         "passes": bool(abs(acf_r[0] - acf_s[0]) <= TOL)},
    }
    diagnostics = {
        "acf_lag1_8_real": acf_r[:8].tolist(), "acf_lag1_8_syn": acf_s[:8].tolist(),
        "mi1": {"real": float(mi_r[0]), "syn": float(mi_s[0])},
        "dwell": {"real": dw_r, "syn": dw_s},
        "position_profile_corr": float(np.corrcoef(prof_r, prof_s)[0, 1]),
        "trend_not_matched_note": "semi_markov has no drift term; the position "
                                  "trend ar1+trend matched is NOT matched here",
    }
    ok = all(c["passes"] for c in checks.values())
    out = {"date": "2026-07-14", "process": "semi_markov", "split_seed": SPLIT_SEED,
           "n_train": len(train), "n_eval": len(ev), "params": params,
           "checks": checks, "diagnostics": diagnostics,
           "verdict": "PASS — swap mirror via dated spec amendment" if ok
                      else "FAIL — spec mirror INVALID pending better process"}
    (HERE / "results/mirror_upgrade_hedging.json").write_text(
        json.dumps(out, indent=2, default=float))
    for k, c in checks.items():
        print(f"{k:14} real={c['real']:+.4f} syn={c['syn']:+.4f} "
              f"|err|={c['abs_err']:.4f} -> {'PASS' if c['passes'] else 'FAIL'}")
    print(f"ACF lags1-8 real: {[round(x, 3) for x in acf_r[:8]]}")
    print(f"ACF lags1-8 syn : {[round(x, 3) for x in acf_s[:8]]}")
    print(f"pos-profile corr: {diagnostics['position_profile_corr']:.3f}  "
          f"MI1 {mi_r[0]:.3f}/{mi_s[0]:.3f}")
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
