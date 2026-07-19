"""Cycle-3 rider — second mirror re-fit for `uncertainty-hedging-drift`.

Executes the prereg frozen in the card's 2026-07-19 amendment
(`amend_cards_c3.py`, machine side `results/amendments_cycle3.json`):
fit the **`hier_ar1`** Appendix-B extension (pooled position trend +
per-document latent level + within-document AR(1)) on the cached C1 labels,
and gate it on the two NON-fitted moments the short-memory mirrors failed —

- gate-8: held-out ACF(2) AND ACF(4), each within the uniform C3 tolerance
  (±20% of the held-out real magnitude, floor 0.01);
- matched-moment sanity check: ACF(1) within ±0.05 abs.

PASS ⇒ `SPEC*`→`SPEC` via a dated amendment to
`synthetic/hedging_drift/bench_spec.md`; FAIL ⇒ stays `SPEC*`, mirror
INVALID. Pure analysis of cached labels — no API calls.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.mirror_upgrade_hedging_c3
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explorations.synthetic.expansion import mirrors
from explorations.synthetic.expansion import signature as sig

HERE = Path(__file__).resolve().parent


def main():
    amend = json.loads((HERE / "results/amendments_cycle3.json").read_text())
    spec = amend["hedging_refit"]
    tol_rel = amend["tol_rule"]["rel"]
    floor = amend["tol_rule"]["floors"]["acf"]

    blob = json.loads((HERE / "records/uncertainty-hedging-drift/labels.json").read_text())
    seqs = [np.array(x, dtype=float) for x in blob["labels"] if x is not None]
    rng = np.random.default_rng(spec["split_seed"])
    idx = rng.permutation(len(seqs))
    cut = int(0.7 * len(seqs))
    train = [seqs[i] for i in idx[:cut]]
    ev = [seqs[i] for i in idx[cut:]]

    params = mirrors.fit_hier_ar1(train, position=True)
    syn = mirrors.gen_hier_ar1(params, [s.size for s in ev], rng)

    acf_r = sig.acf(ev)
    acf_s = sig.acf([np.asarray(s, dtype=float) for s in syn])

    def check(lag_idx, tol_abs=None):
        r, s = float(acf_r[lag_idx]), float(acf_s[lag_idx])
        tol = tol_abs if tol_abs is not None else max(tol_rel * abs(r), floor)
        return {"lag": lag_idx + 1, "real": r, "syn": s,
                "abs_err": abs(r - s), "tol_eff": tol,
                "rule": ("abs" if tol_abs is not None
                         else f"max({tol_rel}*|real|, {floor})"),
                "passes": bool(abs(r - s) <= tol)}

    g8 = {f"gate8_acf{c['idx'] + 1}": check(c["idx"]) for c in spec["gate8"]}
    matched = {"matched_acf1": check(spec["matched_check"]["idx"],
                                     tol_abs=spec["matched_check"]["tol_abs"])}
    ok = all(c["passes"] for c in g8.values())

    lvl = np.array(params["levels"])
    out = {"date": amend["date"], "process": "hier_ar1",
           "split_seed": spec["split_seed"], "n_train": len(train), "n_eval": len(ev),
           "params_summary": {"mu": params["mu"], "beta_position": params["beta_position"],
                              "rho": params["rho"], "sigma": params["sigma"],
                              "level_sd": float(lvl.std()), "n_levels": int(lvl.size)},
           "params": params, "gate8": g8, "matched_check": matched,
           "diagnostics": {"acf_lag1_8_real": acf_r[:8].tolist(),
                           "acf_lag1_8_syn": acf_s[:8].tolist(),
                           "position_profile_corr": float(np.corrcoef(
                               sig.position_profile(ev),
                               sig.position_profile([np.asarray(s) for s in syn]))[0, 1])},
           "verdict": ("PASS — upgrade SPEC*->SPEC via dated spec amendment" if ok
                       else "FAIL — mirror stays INVALID, spec stays SPEC*")}
    (HERE / "results/mirror_upgrade_hedging_c3.json").write_text(
        json.dumps(out, indent=2, default=float))
    for k, c in {**g8, **matched}.items():
        print(f"{k:14} real={c['real']:+.4f} syn={c['syn']:+.4f} "
              f"|err|={c['abs_err']:.4f} tol={c['tol_eff']:.4f} "
              f"-> {'PASS' if c['passes'] else 'FAIL'}")
    print(f"ACF lags1-8 real: {[round(x, 3) for x in acf_r[:8]]}")
    print(f"ACF lags1-8 syn : {[round(x, 3) for x in acf_s[:8]]}")
    print("VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
