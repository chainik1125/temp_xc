"""C7 acceptance battery — monotone segment-composition extraction.

Runs the frozen C7 card exactly (`prereg/estimator-card-c7-monotone-
extraction.md`): the monotonicity pre-check on real material BEFORE any
gate, then C6 gates 1–3 verbatim under the variance-aware margin rule
(2·SE decision band, replicate escalation R = 6 → 12 → 24, conservative
FAIL at the boundary), batteries 4–5 report-only, and the pre-specified
fork resolution (r4 / close). Zero API spend — committed r3 labels +
synthetic toys only. Committed BEFORE first execution (strict
commit-then-run). Writes ``results/estimator_battery_c7.json``.

    .venv/bin/python -m experiments.explorations.synthetic.expansion.estimator_battery_c7
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explorations.synthetic.expansion import mirrors
from explorations.synthetic.expansion import signature as sig
from experiments.explorations.synthetic.expansion import (
    estimator_battery_c6 as c6)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "estimator_battery_c7.json"
SEED_FAMILY = list(range(9100, 9124))       # 24 preregistered measurement
R_SCHEDULE = (6, 12, 24)                    # seeds; first 3 are C6's
PRECHECK_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
INERT_LAM = 0.85
EST = "seg_hier_categorical_mono"


def _replicate_moments(params, lengths, keys, R):
    vals = {m: [] for m in keys}
    for seed in SEED_FAMILY[:R]:
        syn = mirrors.gen_seg_hier_categorical(
            params, lengths, np.random.default_rng(seed))
        mm = mirrors._seg_moments(syn)
        for m in keys:
            vals[m].append(mm[m])
    return vals


def decide_gate(params, lengths, refs: dict, tols: dict) -> dict:
    """The card's variance-aware decision: statistic s = |mean − ref|,
    SE from the replicates; PASS needs s ≤ tol − 2·SE, FAIL s > tol + 2·SE,
    else escalate R; boundary at R_max ⇒ conservative FAIL."""
    rows = {}
    for R in R_SCHEDULE:
        vals = _replicate_moments(params, lengths, list(refs), R)
        states = []
        for m in refs:
            v = np.asarray(vals[m], dtype=float)
            s = abs(float(v.mean()) - refs[m])
            se = float(v.std(ddof=1)) / np.sqrt(len(v))
            rows[m] = {"stat": s, "se": se, "tol": tols[m], "R": R,
                       "ref": refs[m], "mean_syn": float(v.mean())}
            states.append("pass" if s <= tols[m] - 2 * se else
                          "fail" if s > tols[m] + 2 * se else "zone")
        if "fail" in states:
            return {"pass": False, "decided_at_R": R, "boundary_fail": False,
                    "rows": rows}
        if "zone" not in states:
            return {"pass": True, "decided_at_R": R, "boundary_fail": False,
                    "rows": rows}
    return {"pass": False, "decided_at_R": R_SCHEDULE[-1],
            "boundary_fail": True, "rows": rows}


def precheck_monotone(ok, k):
    """Card § pre-check: generated ACF(4) vs λ on the real streams —
    (i) no adjacent increase beyond max(0.5·SE_pool, 0.002);
    (ii) span m(0) − m(1) ≥ 5·SE_pool. Scored before any gate."""
    from scipy.stats import spearmanr
    base, u_docs = mirrors._mono_stages(
        [np.asarray(s).astype(int) for s in ok], k, 400, 4, 13)
    lengths = [s.size for s in ok]
    means, ses = [], []
    for lam in PRECHECK_GRID:
        params = mirrors._mono_tables(base, u_docs, lam)
        v = np.asarray(_replicate_moments(params, lengths, ["acf4"], 6)["acf4"])
        means.append(float(v.mean()))
        ses.append(float(v.std(ddof=1)) / np.sqrt(len(v)))
    se_pool = float(np.mean(ses))
    diffs = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    inc_bound = max(0.5 * se_pool, 0.002)
    no_increase = all(d <= inc_bound for d in diffs)
    span = means[0] - means[-1]
    moves = span >= 5.0 * se_pool
    rho = float(spearmanr(PRECHECK_GRID, means).statistic)
    return {"grid": PRECHECK_GRID, "mean_acf4": means, "se": ses,
            "se_pool": se_pool, "adjacent_diffs": diffs,
            "increase_bound": inc_bound, "no_material_increase": bool(no_increase),
            "span": span, "span_bound": 5.0 * se_pool, "knob_moves": bool(moves),
            "spearman_rho": rho, "pass": bool(no_increase and moves)}


def main():
    labels = json.loads(
        (HERE / "records" / "proof-operation-phase-runs-r3" / "labels.json")
        .read_text())
    ok = [np.array(x, dtype=np.int8) for x in labels["labels"] if x is not None]
    k = int(max(int(s.max()) for s in ok)) + 1
    lengths = [s.size for s in ok]
    print(f"[c7] real streams: {len(ok)} docs, {sum(lengths)} sentences, k={k}")
    out = {}

    # ── 0: monotonicity pre-check (before ANY gate; fail ⇒ close) ──
    print("[c7] 0/5 monotonicity pre-check on real material …")
    pre = precheck_monotone(ok, k)
    out["precheck"] = pre
    print(f"  m(λ): {[round(m, 4) for m in pre['mean_acf4']]}  "
          f"(SE_pool {pre['se_pool']:.4f})")
    print(f"  no-increase: {pre['no_material_increase']} "
          f"(bound {pre['increase_bound']:.4f}; diffs "
          f"{[round(d, 4) for d in pre['adjacent_diffs']]})  "
          f"knob-moves: {pre['knob_moves']} (span {pre['span']:.4f} ≥ "
          f"{pre['span_bound']:.4f})  ρ={pre['spearman_rho']:.2f} "
          f"-> {'PASS' if pre['pass'] else 'FAIL — concept dead, go to close'}")

    # ── 0b: the real-material calibrated fit (λ*_real + inert check) ──
    print("[c7] 0b real-material calibrated fit (λ*_real) …")
    fit_real = mirrors.fit_seg_hier_categorical_mono(ok, n_symbols=k)
    out["real_fit"] = {"lam_star": fit_real["lam"],
                       "uncalibratable": fit_real["uncalibratable"],
                       "null_moments": fit_real["null_moments"],
                       "null_tol": fit_real["null_tol"],
                       "lam_scan": fit_real["lam_scan"]}
    print(f"  λ*_real = {fit_real['lam']:.3f}  "
          f"uncalibratable={fit_real['uncalibratable']}  "
          f"(inert threshold {INERT_LAM})")

    gates_pass = None
    if pre["pass"]:
        # ── gate 1: null-safety on run-permuted real streams ──
        print("[c7] 1/5 gate 1 — null-safety on real material …")
        perm = mirrors.run_permuted_streams(
            [np.asarray(s).astype(int) for s in ok])
        perm_m = mirrors._seg_moments(perm)
        tol = mirrors._null_tol(perm_m)
        fit1 = mirrors.fit_seg_hier_categorical_mono(perm, n_symbols=k)
        g1 = decide_gate(fit1, lengths, perm_m, tol)
        g1["lam"] = fit1["lam"]
        out["gate1"] = g1
        for m, r in g1["rows"].items():
            print(f"  {m}: ins={r['stat']:.4f} ±2SE {2 * r['se']:.4f} "
                  f"tol={r['tol']:.4f} R={r['R']}")
        print(f"  gate 1 -> {'PASS' if g1['pass'] else 'FAIL'}"
              f"{' (boundary ⇒ conservative FAIL)' if g1['boundary_fail'] else ''}"
              f"  [fit λ={fit1['lam']:.3f}]")

        # ── gate 2: heavy-dwell null ──
        print("[c7] 2/5 gate 2 — heavy-dwell null …")
        rng = np.random.default_rng(31)
        seqs_n = mirrors.gen_hier_categorical(
            c6._heavy_dwell_null(), [120] * 200, rng)
        acf_n4 = float(sig.selfmatch_acf(seqs_n)[3])
        perm_n = mirrors.run_permuted_streams(seqs_n)
        perm_nm = mirrors._seg_moments(perm_n)
        fit2 = mirrors.fit_seg_hier_categorical_mono(perm_n, n_symbols=4)
        g2 = decide_gate(fit2, [120] * 200, {"acf4": perm_nm["acf4"]},
                         {"acf4": max(0.2 * acf_n4, 0.01)})
        g2["lam"] = fit2["lam"]
        out["gate2"] = g2
        r = g2["rows"]["acf4"]
        print(f"  acf4: ins={r['stat']:.4f} ±2SE {2 * r['se']:.4f} "
              f"bound={r['tol']:.4f} R={r['R']} -> "
              f"{'PASS' if g2['pass'] else 'FAIL'}  [fit λ={fit2['lam']:.3f}]")

        # ── gate 3: strong planted truth sensitivity ──
        print("[c7] 3/5 gate 3 — strong planted truth …")
        seqs_s = mirrors.gen_seg_hier_categorical(
            c6._seg_hier_truth(), [120] * 200, np.random.default_rng(21))
        real4 = float(sig.selfmatch_acf(seqs_s)[3])
        fit3 = mirrors.fit_seg_hier_categorical_mono(seqs_s, n_symbols=4)
        g3 = decide_gate(fit3, [120] * 200, {"acf4": real4},
                         {"acf4": 0.20 * real4})
        g3["lam"] = fit3["lam"]
        out["gate3"] = g3
        r = g3["rows"]["acf4"]
        print(f"  acf4: err={r['stat']:.4f} ±2SE {2 * r['se']:.4f} "
              f"bound={r['tol']:.4f} R={r['R']} -> "
              f"{'PASS' if g3['pass'] else 'FAIL'}  [fit λ={fit3['lam']:.3f}]")

        gates_pass = g1["pass"] and g2["pass"] and g3["pass"]

        # ── 4: weak truth (report-only) ──
        print("[c7] 4/5 weak planted truth (report-only) …")
        seqs_w = mirrors.gen_seg_hier_categorical(
            c6._seg_hier_truth(0.60, 0.70), [120] * 200,
            np.random.default_rng(22))
        real_w4 = float(sig.selfmatch_acf(seqs_w)[3])
        b4 = {}
        for est in ("seg_hier_categorical", EST):
            fit_fn, _ = mirrors.MENU[est]
            p = fit_fn(seqs_w, n_symbols=4)
            v = np.asarray(_replicate_moments(
                p, [120] * 200, ["acf4"], 6)["acf4"])
            b4[est] = {"acf4_abs_err": abs(float(v.mean()) - real_w4),
                       "rel_err": abs(float(v.mean()) - real_w4) / real_w4,
                       **({"lam": p["lam"]} if est == EST else {})}
            print(f"  {est}: err={b4[est]['acf4_abs_err']:.4f} "
                  f"({b4[est]['rel_err']:.0%})")
        out["battery4_weak"] = b4

        # ── 5: variance panel (report-only) ──
        print("[c7] 5/5 variance panel (report-only) …")
        errs, contr = {e: [] for e in ("seg_hier_categorical", EST)}, \
                      {e: [] for e in ("seg_hier_categorical", EST)}
        for rr in range(5):
            seqs5 = mirrors.gen_seg_hier_categorical(
                c6._seg_hier_truth(), [120] * 200,
                np.random.default_rng(100 + rr))
            r4v = float(sig.selfmatch_acf(seqs5)[3])
            raw_c = None
            for est in ("seg_hier_categorical", EST):
                fit_fn, _ = mirrors.MENU[est]
                p = fit_fn(seqs5, n_symbols=4)
                v = np.asarray(_replicate_moments(
                    p, [120] * 200, ["acf4"], 6)["acf4"])
                errs[est].append(abs(float(v.mean()) - r4v))
                cv = c6._contrast(p)
                if est == "seg_hier_categorical":
                    raw_c = cv
                contr[est].append(cv / max(raw_c, 1e-9))
        out["battery5_variance"] = {
            e: {"mean_acf4_abs_err": float(np.mean(errs[e])),
                "retained_contrast_vs_r3": float(np.mean(contr[e]))}
            for e in errs}
        for e, r in out["battery5_variance"].items():
            print(f"  {e}: mean err={r['mean_acf4_abs_err']:.4f}  "
                  f"retained contrast={r['retained_contrast_vs_r3']:.2f}")

    # ── the pre-specified fork (card, verbatim) ──
    inert = fit_real["lam"] > INERT_LAM or fit_real["uncalibratable"]
    if not pre["pass"]:
        branch, why = "close", "monotonicity pre-check FAILED — concept dead"
    elif not gates_pass:
        branch, why = "close", "battery gates 1-3 not all passed"
    elif inert:
        branch, why = "close", (f"λ*_real={fit_real['lam']:.3f} > {INERT_LAM} "
                                "— calibrated to ≈ inert on real material")
    else:
        branch, why = "r4", (f"pre-check + gates 1-3 pass, "
                             f"λ*_real={fit_real['lam']:.3f} ≤ {INERT_LAM}")
    out["fork"] = {"branch": branch, "reason": why,
                   "gates_pass": gates_pass, "precheck_pass": pre["pass"],
                   "lam_star_real": fit_real["lam"], "inert": bool(inert)}
    print(f"[c7] FORK -> {branch.upper()}: {why}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"[c7] written -> {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
