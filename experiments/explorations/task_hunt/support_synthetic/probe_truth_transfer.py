"""Does the mirror's probe-bias map describe the REAL panel? (transfer test)

Reads only committed evidence: the mirror calibration produced by
`probe_truth_calib.py` (exact truth) and runpod-d's committed probe-capacity
diagnostic on the real λ̂ panel
(`lambda_intensity/results/probe_capacity_ward_real_lambda_base_l12.json` —
**reported, under review, not adopted**, per `PROBE_V2_SPEC.md` § 4). Writes
nothing to the leaderboard and re-runs no cell.

**Why this exists.** The mirror shows *how much* each probe under-reports as
a function of (true recovery, p/n, code density). That is a statement about
probes, not about Ward text. Two things make it testable on the real panel
anyway — and both are falsifiable, which is the point:

**Test A — the density prediction.** At FIXED p/n the mirror says the sag
depends strongly on code density: a sparse code leaves little for an
unregularised probe to overfit, a dense one a great deal. So the v2 − v1 gap
should track density at fixed p/n. The real panel contains the matched pair
that tests this directly: at T = 16, p/n = 1.00, `txc_batchtopk_post` runs
BOTH a sparse code (nnz/row ≈ 7.8) and, in the budget-matched re-run, a dense
one (≈ 127.9). If the mirror's mechanism is the operative one, the sparse
cell's gap must be far smaller than the dense cell's.

**Test B — the inversion, and its own consistency check.** Given a real
cell's observed (v1, v2) at known (p/n, density), the mirror's truth → v1 and
truth → v2 curves can each be inverted to ask what true recovery is
consistent with that reading. The two inversions are independent uses of the
same map, so **they must agree** — and if they do not, the mirror's map does
not describe the real code and the inversion is reported as failed rather
than averaged into a number. Agreement is evidence the transfer holds; it is
not proof, and the assumption it rides on is stated in the output:
`signal_dims_caveat`.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.probe_truth_transfer
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
REAL = (HERE.parent / "lambda_intensity" / "results"
        / "probe_capacity_ward_real_lambda_base_l12.json")
T_MAIN = 16
SIGNAL_DIMS_CAVEAT = (
    "The mirror's constructed code carries λ in <= 2 explicit columns among p; "
    "a trained code spreads it over many correlated columns. The inversion "
    "assumes the (p/n, density) pair is what governs the probe's bias, which "
    "is what Test A checks and what the v1/v2 inversion agreement checks "
    "again — but neither rules out a residual dependence on how the signal is "
    "distributed across columns.")


def _mirror_curves(calib) -> dict:
    """(density, p, p/n) -> sorted [(truth, v1, v2)] over the mix/frozen arms."""
    by = defaultdict(lambda: defaultdict(list))
    for c in calib:
        if c["T"] != T_MAIN:
            continue
        g = [q for q in c["grid"] if q["n_windows"] == 1024][0]
        key = (c["density"], c["p"], round(g["p_over_n"], 4))
        by[key][c["arm"]].append((g["truth"], c["v1"], c["v2"]))
    out = {}
    for key, arms in by.items():
        pts = []
        for arm, vals in arms.items():
            if len(vals) < 2:                    # need the seeds to average
                continue
            pts.append((float(np.mean([v[0] for v in vals])),
                        float(np.mean([v[1] for v in vals])),
                        float(np.mean([v[2] for v in vals])),
                        arm, len(vals)))
        if len(pts) >= 2:
            out[key] = sorted(pts)
    return out


def _invert(curve, obs, idx: int):
    """Truth consistent with one observed reading, by linear interpolation.

    ``idx`` selects the probe column (1 = v1, 2 = v2). Returns None when the
    observation falls outside the curve's range — extrapolating a saturating
    map would manufacture precision that is not there.
    """
    xs = [c[idx] for c in curve]
    ts = [c[0] for c in curve]
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ts = [ts[i] for i in order]
    if obs < xs[0] or obs > xs[-1]:
        return None
    return float(np.interp(obs, xs, ts))


def main():
    calib = {}
    for p in sorted(RES.glob("probe_truth_calib*.json")):
        for r in json.loads(p.read_text()):
            calib[(r["arm"], r["T"], r["p_nominal"], r["density"],
                   r["seed"])] = r
    calib = list(calib.values())
    curves = _mirror_curves(calib)

    real = defaultdict(list)
    for r in json.loads(REAL.read_text()):
        real[(r["arch"], r["T"], r["k_pos"], r["n_windows"], r["est"])].append(r)
    cells = {}
    for (arch, T, k, nw, est), rs in real.items():
        cells.setdefault((arch, T, k), {})[(nw, est)] = {
            "r": float(np.mean([q["r"] for q in rs])),
            "p": int(rs[0]["p"]), "n_rows": int(rs[0]["n_rows"]),
            "nnz": float(np.mean([q["nnz_per_row"] for q in rs])),
            # EFFECTIVE feature dimension: columns that ever fire. A column
            # that is identically zero on the train rows contributes no
            # parameter to fit, so the ratio that governs over-fitting is
            # p_eff/n, not d_sae/n. On this panel the two differ by 3-30x
            # (e.g. post/T16/k8: 70 active of 2048), which is why the nominal
            # p/n = 1.00 label is not where those cells actually sit.
            "p_eff": float(np.mean([q["n_active_cols"] for q in rs])),
            "n_seeds": len(rs)}

    # ── Test A — the density prediction at fixed p/n ────────────────────────
    rows = []
    for (arch, T, k), v in sorted(cells.items(), key=str):
        a, b = v.get((1024, "ols")), v.get((8192, "ridge"))
        if not a or not b:
            continue
        rows.append({"arch": arch, "T": T, "k_pos": k, "p": a["p"],
                     "p_over_n": a["p"] / a["n_rows"],
                     "p_eff": a["p_eff"],
                     "p_eff_over_n": a["p_eff"] / a["n_rows"],
                     "density": a["nnz"] / a["p"],
                     "density_eff": a["nnz"] / max(a["p_eff"], 1.0),
                     "nnz_per_row": a["nnz"],
                     "v1": a["r"], "v2": b["r"], "gap": b["r"] - a["r"],
                     "n_seeds": a["n_seeds"]})
    at_pn1 = sorted([r for r in rows if abs(r["p_over_n"] - 1.0) < 1e-6],
                    key=lambda r: r["density"])
    test_a = {"cells_at_p_over_n_1": at_pn1}
    if len(at_pn1) >= 2:
        lo, hi = at_pn1[0], at_pn1[-1]
        test_a.update({
            "sparse_cell": f"{lo['arch']}/T{lo['T']}/k{lo['k_pos']}",
            "sparse_density": lo["density"], "sparse_gap": lo["gap"],
            "dense_cell": f"{hi['arch']}/T{hi['T']}/k{hi['k_pos']}",
            "dense_density": hi["density"], "dense_gap": hi["gap"],
            "prediction": "dense gap > sparse gap at matched p/n",
            "holds": bool(hi["gap"] > lo["gap"])})
    # the same contrast measured on the mirror, for the side-by-side
    mirror_a = {}
    for dens in ("k8", "p6"):
        key = (dens, 2048, 1.0)
        if key in curves:
            mirror_a[dens] = [{"arm": c[3], "truth": c[0], "v1": c[1],
                               "v2": c[2], "gap": c[2] - c[1]}
                              for c in curves[key]]
    test_a["mirror_same_contrast_p2048_pn1"] = mirror_a

    # ── Test B — inversion + its consistency check ──────────────────────────
    inv = []
    for r in rows:
        # Match on the EFFECTIVE ratio p_eff/n (see p_eff above), nearest in
        # log space since the mirror's ladder is geometric; and on density
        # class measured against p_eff for the same reason.
        dens_class = "p6" if r["density_eff"] > 0.02 else "k8"
        cand = [(k, c) for k, c in curves.items() if k[0] == dens_class]
        if not cand:
            cand = list(curves.items())
        if not cand:
            continue
        key, curve = min(cand, key=lambda kc: abs(
            np.log(max(kc[0][2], 1e-6)) - np.log(max(r["p_eff_over_n"], 1e-6))))
        t1, t2 = _invert(curve, r["v1"], 1), _invert(curve, r["v2"], 2)
        rec = {"cell": f"{r['arch']}/T{r['T']}/k{r['k_pos']}",
               "p_over_n": r["p_over_n"], "p_eff": r["p_eff"],
               "p_eff_over_n": r["p_eff_over_n"], "density": r["density"],
               "density_eff": r["density_eff"],
               "mirror_curve": {"density": key[0], "p": key[1],
                                "p_over_n": key[2],
                                "n_points": len(curve),
                                "truths": [round(c[0], 3) for c in curve]},
               "v1": r["v1"], "v2": r["v2"],
               "truth_implied_by_v1": t1, "truth_implied_by_v2": t2}
        if t1 is not None and t2 is not None:
            rec["disagreement"] = abs(t1 - t2)
            rec["consistent"] = bool(abs(t1 - t2) <= 0.10)
            rec["truth_estimate"] = 0.5 * (t1 + t2)
        else:
            rec["consistent"] = None
            rec["note"] = "observation outside the mirror curve's range"
        inv.append(rec)

    out = {"source_real": str(REAL.relative_to(HERE.parents[3]))
           if REAL.is_absolute() else str(REAL),
           "real_status": "runpod-d committed diagnostic — reported, under "
                          "review, NOT adopted (PROBE_V2_SPEC.md § 4)",
           "signal_dims_caveat": SIGNAL_DIMS_CAVEAT,
           "test_A_density_prediction": test_a,
           "test_B_inversion": inv,
           "real_cells": rows}
    (RES / "probe_truth_transfer.json").write_text(json.dumps(out, indent=1))

    print("TEST A — density prediction at matched p/n = 1.00")
    for r in at_pn1:
        print(f"  {r['arch']:<22} T{r['T']:<3} k{r['k_pos']:<4} "
              f"dens={r['density']:.3f} (nnz {r['nnz_per_row']:.1f}) "
              f"p_eff={r['p_eff']:.0f} p_eff/n={r['p_eff_over_n']:.3f} "
              f"v1={r['v1']:+.3f} v2={r['v2']:+.3f} gap={r['gap']:+.3f}")
    if "holds" in test_a:
        print(f"  prediction (dense gap > sparse gap): "
              f"{'HOLDS' if test_a['holds'] else 'FAILS'}")
    for dens, pts in mirror_a.items():
        for q in pts:
            print(f"  [mirror {dens}] {q['arm']:<7} truth={q['truth']:+.3f} "
                  f"v1={q['v1']:+.3f} v2={q['v2']:+.3f} gap={q['gap']:+.3f}")
    print("\nTEST B — inversion (truth implied by each probe; they must agree)")
    for r in inv:
        t1, t2 = r["truth_implied_by_v1"], r["truth_implied_by_v2"]
        f = (lambda v: "  n/a" if v is None else f"{v:+.3f}")
        print(f"  {r['cell']:<28} p_eff/n={r['p_eff_over_n']:.3f} "
              f"(nominal {r['p_over_n']:.2f}) dens={r['density']:.3f} "
              f"→curve p/n={r['mirror_curve']['p_over_n']:.3f} "
              f"{r['mirror_curve']['density']}  v1→truth {f(t1)}  "
              f"v2→truth {f(t2)}  "
              f"{'consistent' if r.get('consistent') else r.get('note', 'INCONSISTENT')}")
    print(f"-> {RES/'probe_truth_transfer.json'}")


if __name__ == "__main__":
    main()
