"""C6 verification battery — calibrated segment-composition extraction.

Runs the frozen estimator card's batteries 1–5 (see
``prereg/estimator-card-c6-segment-extraction.md``) for the two candidates
(`seg_hier_categorical_cal`, `seg_hier_categorical_deflate`) with the r3
raw estimator reported alongside, applies the frozen selection rule, and
writes ``results/estimator_battery_c6.json``. Zero API spend: committed r3
labels + synthetic toys only. Run BEFORE the r4 calibration:

    .venv/bin/python -m experiments.explorations.synthetic.expansion.estimator_battery_c6
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explorations.synthetic.expansion import mirrors
from explorations.synthetic.expansion import signature as sig

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "estimator_battery_c6.json"
GEN_SEEDS = (9100, 9101, 9102)      # measurement reps (distinct from the
                                    # cal fit's internal 9000+rep seeds)
CANDS = ("seg_hier_categorical_cal", "seg_hier_categorical_deflate")
ALL_EST = ("seg_hier_categorical",) + CANDS      # r3 raw reported alongside


def _moments(seqs):
    return mirrors._seg_moments(seqs)


def _syn_moments(params, lengths):
    """Mean moments over the fixed measurement replicates."""
    acc = {"mi2": [], "acf4": []}
    for seed in GEN_SEEDS:
        syn = mirrors.gen_seg_hier_categorical(
            params, lengths, np.random.default_rng(seed))
        m = _moments(syn)
        for k in acc:
            acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _fit(est, seqs, k):
    fit_fn, _ = mirrors.MENU[est]
    return fit_fn(seqs, n_symbols=k)


def _seg_hier_truth(conc_val=0.9, tilt_seg=0.85):
    """The committed harness three-timescale toy (tests/test_expansion_harness
    ._seg_hier_truth), parameterized so the card's WEAK variant (conc 0.60,
    tilt_seg 0.70) is the same object with two knobs turned."""
    def conc(d):
        v = np.full(4, (1.0 - conc_val) / 3)
        v[d] = conc_val
        return v.tolist()
    seg_tables, doc_props = [], []
    for t in range(4):
        table = [[12, conc(t)], [16, conc(t)], [20, conc(t)],
                 [12, conc((t + 1) % 4)], [14, conc((t + 2) % 4)],
                 [10, conc((t + 3) % 4)]]
        seg_tables += [table] * 50
        doc_props += [list(np.roll([0.55, 0.15, 0.15, 0.15], t))] * 50
    return {"process": "seg_hier_categorical", "n_symbols": 4,
            "jump_P": ((np.ones((4, 4)) - np.eye(4)) / 3).tolist(),
            "tilt_seg": tilt_seg, "tilt_doc": 0.05,
            "doc_props": doc_props, "seg_tables": seg_tables,
            "dwell": [[1, 1, 2, 2, 3]] * 4}


def _heavy_dwell_null():
    """The harness doc-homogeneous heavy-dwell null the r3 estimator fails."""
    return {"process": "hier_categorical", "n_symbols": 4, "alpha": 0.85,
            "jump_P": ((np.ones((4, 4)) - np.eye(4)) / 3).tolist(),
            "doc_props": sum(([list(np.roll([0.70, 0.10, 0.10, 0.10], t))] * 50
                              for t in range(4)), []),
            "dwell": [[1, 1, 2, 2, 3, 4, 6, 9]] * 4}


def _contrast(params):
    """Length-weighted total segment concentration Σ len·KL(table ‖ pi_doc)
    of a fit's (deconvolved) segment tables — the generation-side contrast."""
    tot = 0.0
    for j, table in enumerate(params["seg_tables"]):
        pi_doc = np.asarray(params["doc_props"][j], dtype=float)
        for length, pi in table:
            tot += length * mirrors._kl(np.asarray(pi, dtype=float), pi_doc)
    return tot


def battery1_real_null(ok, k):
    """Null-safety on real material: fit on run-permuted committed streams,
    insertion vs the permuted streams' own moments, null-referenced tol."""
    perm = mirrors.run_permuted_streams(ok)
    perm_m = _moments(perm)
    tol = mirrors._null_tol(perm_m)
    lengths = [s.size for s in ok]
    out = {"perm_moments": perm_m, "tol": tol, "estimators": {}}
    for est in ALL_EST:
        params = _fit(est, perm, k)
        syn_m = _syn_moments(params, lengths)
        ins = {m: abs(syn_m[m] - perm_m[m]) for m in perm_m}
        out["estimators"][est] = {
            "syn_nullfit": syn_m, "insertion": ins,
            "pass": bool(all(ins[m] <= tol[m] for m in ins)),
            **({"lam": params.get("lam"),
                "uncalibratable": params.get("uncalibratable")}
               if est.endswith("_cal") else {}),
            **({"mean_deflation": params.get("mean_deflation"),
                "frac_collapsed": params.get("frac_collapsed")}
               if est.endswith("_deflate") else {})}
    return out


def battery2_heavy_dwell(rng):
    """Null-safety on the harness heavy-dwell null (r3 provably fails):
    insertion on acf4 ≤ max(0.2·acf_null4, 0.01)."""
    seqs_n = mirrors.gen_hier_categorical(_heavy_dwell_null(), [120] * 200, rng)
    acf_n4 = float(sig.selfmatch_acf(seqs_n)[3])
    perm_n = mirrors.run_permuted_streams(seqs_n)
    perm_m = _moments(perm_n)
    bound = max(0.2 * acf_n4, 0.01)
    out = {"acf_null4": acf_n4, "perm_acf4": perm_m["acf4"], "bound": bound,
           "estimators": {}}
    for est in ALL_EST:
        params = _fit(est, perm_n, 4)
        syn_m = _syn_moments(params, [120] * 200)
        ins = abs(syn_m["acf4"] - perm_m["acf4"])
        out["estimators"][est] = {"syn_acf4": syn_m["acf4"], "insertion": ins,
                                  "pass": bool(ins <= bound)}
    return out


def battery34_truth(conc_val, tilt_seg, data_seed):
    """Planted-truth round-trip (strong = gate 3; weak = report-only)."""
    rng = np.random.default_rng(data_seed)
    truth = _seg_hier_truth(conc_val, tilt_seg)
    seqs = mirrors.gen_seg_hier_categorical(truth, [120] * 200, rng)
    real_m = _moments(seqs)
    out = {"real_moments": real_m, "estimators": {}}
    for est in ALL_EST:
        params = _fit(est, seqs, 4)
        syn_m = _syn_moments(params, [120] * 200)
        err = abs(syn_m["acf4"] - real_m["acf4"])
        out["estimators"][est] = {
            "syn": syn_m, "acf4_abs_err": err,
            "acf4_rel_err": err / max(real_m["acf4"], 1e-9),
            "pass_20pct": bool(err <= 0.20 * real_m["acf4"]),
            "tilt_seg": params["tilt_seg"],
            **({"lam": params.get("lam")} if est.endswith("_cal") else {})}
    return out


def battery5_variance(n_rep=5):
    """Variance penalty: mean |acf4 round-trip err| + retained contrast
    across replicate draws of the strong toy (report-only)."""
    errs = {est: [] for est in ALL_EST}
    contrast = {est: [] for est in ALL_EST}
    for r in range(n_rep):
        rng = np.random.default_rng(100 + r)
        seqs = mirrors.gen_seg_hier_categorical(
            _seg_hier_truth(), [120] * 200, rng)
        real4 = _moments(seqs)["acf4"]
        raw_c = None
        for est in ALL_EST:
            params = _fit(est, seqs, 4)
            syn_m = _syn_moments(params, [120] * 200)
            errs[est].append(abs(syn_m["acf4"] - real4))
            c = _contrast(params)
            if est == "seg_hier_categorical":
                raw_c = c
            contrast[est].append(c / max(raw_c, 1e-9))
    return {est: {"mean_acf4_abs_err": float(np.mean(errs[est])),
                  "retained_contrast_vs_r3": float(np.mean(contrast[est]))}
            for est in ALL_EST}


def main():
    # ── real material (committed r3 labels; zero API) ──
    labels = json.loads(
        (HERE / "records" / "proof-operation-phase-runs-r3" / "labels.json")
        .read_text())
    ok = [np.array(x, dtype=np.int8) for x in labels["labels"] if x is not None]
    k = int(max(int(s.max()) for s in ok)) + 1
    print(f"[battery] real streams: {len(ok)} docs, "
          f"{sum(s.size for s in ok)} sentences, k={k}")

    print("[battery] 1/5 null-safety on real material …")
    b1 = battery1_real_null(ok, k)
    for est, r in b1["estimators"].items():
        print(f"  {est}: ins mi2={r['insertion']['mi2']:.4f} "
              f"acf4={r['insertion']['acf4']:.4f} "
              f"(tol {b1['tol']['mi2']:.4f}/{b1['tol']['acf4']:.4f}) "
              f"-> {'PASS' if r['pass'] else 'FAIL'}"
              + (f"  lam={r.get('lam')}" if "lam" in r else ""))

    print("[battery] 2/5 heavy-dwell null …")
    b2 = battery2_heavy_dwell(np.random.default_rng(31))
    for est, r in b2["estimators"].items():
        print(f"  {est}: ins acf4={r['insertion']:.4f} "
              f"(bound {b2['bound']:.4f}) -> {'PASS' if r['pass'] else 'FAIL'}")

    print("[battery] 3/5 strong planted truth …")
    b3 = battery34_truth(0.9, 0.85, data_seed=21)
    for est, r in b3["estimators"].items():
        print(f"  {est}: acf4 err={r['acf4_abs_err']:.4f} "
              f"({r['acf4_rel_err']:.0%}) -> "
              f"{'PASS' if r['pass_20pct'] else 'FAIL'}")

    print("[battery] 4/5 weak planted truth (report-only) …")
    b4 = battery34_truth(0.60, 0.70, data_seed=22)
    for est, r in b4["estimators"].items():
        print(f"  {est}: acf4 err={r['acf4_abs_err']:.4f} "
              f"({r['acf4_rel_err']:.0%})")

    print("[battery] 5/5 variance penalty (report-only) …")
    b5 = battery5_variance()
    for est, r in b5.items():
        print(f"  {est}: mean acf4 err={r['mean_acf4_abs_err']:.4f}  "
              f"retained contrast={r['retained_contrast_vs_r3']:.2f}")

    # ── frozen selection rule (card): gates 1–3; lower mean |acf4 err|
    # across batteries 3+4; tie (<10% rel) → cal ──
    passing = [est for est in CANDS
               if b1["estimators"][est]["pass"]
               and b2["estimators"][est]["pass"]
               and b3["estimators"][est]["pass_20pct"]]
    sel, reason = None, ""
    if len(passing) == 1:
        sel = passing[0]
        reason = "only candidate passing gates 1-3"
    elif len(passing) == 2:
        score = {est: 0.5 * (b3["estimators"][est]["acf4_abs_err"]
                             + b4["estimators"][est]["acf4_abs_err"])
                 for est in passing}
        lo = min(score.values())
        close = [e for e in passing if score[e] <= lo * 1.10]
        sel = ("seg_hier_categorical_cal"
               if "seg_hier_categorical_cal" in close else
               min(score, key=score.get))
        reason = (f"both pass; sensitivity score (mean acf4 err, b3+b4) "
                  f"{ {e: round(v, 4) for e, v in score.items()} }; "
                  + ("tie<10% -> cal" if len(close) > 1 else "lower error"))
    else:
        reason = ("NEITHER candidate passes gates 1-3: estimator family "
                  "uncalibratable under the frozen card; NO r4 run "
                  "(card's third branch)")
    print(f"[battery] SELECTION: {sel or 'NONE'} — {reason}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(
        {"battery1_real_null": b1, "battery2_heavy_dwell": b2,
         "battery3_strong_truth": b3, "battery4_weak_truth": b4,
         "battery5_variance": b5,
         "gen_seeds": GEN_SEEDS,
         "selection": {"selected": sel, "passing": passing,
                       "reason": reason}},
        indent=1, default=float))
    print(f"[battery] written -> {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
