"""Loss-dissection analysis — the mechanical application of CARD § 5 (frozen).

Reads the per-bench dissection grid dumps + the canonical leaderboard's
txc_batchtopk_post anchor rows, and emits results/dissection_table.{json,md}.
Verdicts are READ OFF the frozen rules, never chosen:

- paired per-seed differences vs txc_post_plain at each of the 9 (T,k_pos)
  slice cells; D=mean, SE=std(ddof=1)/sqrt(3);
- a cell passes the bar iff |D| > max(2*SE, delta_floor)
  (delta_floor: 0.05 recovery/eauc, 0.02 nmse; nmse sign flipped so
  positive = improvement);
- bench verdict: HELPS iff >=2/9 cells pass positively AND 0 negatively;
  HURTS mirror; MIXED if both directions; NEUTRAL otherwise;
- Gate B (graft validity): |mean plain - mean anchor| <= max(2*SD_pool, 0.10)
  on >=7/9 cells (primary metric) — FAIL blocks all component claims for
  that bench;
- untrained guard: rows identical across variants per (T, seed);
- interaction I = D_both - D_mat - D_ctr (descriptive only).

Key-mapping note (mechanical, recorded here and in RECORD.md): the card's
"gauc" capability metric is the leaderboard's ``eauc`` key (the
feature-direction recovery AUC; the row-level ``primary_metric: "gauc"``
field is the same quantity's legacy label).

    .venv/bin/python -m experiments.explorations.synthetic.loss_dissection.analyze
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEADERBOARD = HERE.parents[3] / "results" / "leaderboard.jsonl"

DELTA_FLOOR = {"default": 0.05, "nmse": 0.02}
LOWER_BETTER = {"nmse"}
COMPONENTS = ("mat", "ctr", "both")
TS, KS, SEEDS = (2, 4, 8), (1, 2, 4), (1, 2, 42)

# family switch (CARD § 9 amendment): argv[1] in {post, pre}, default post.
FAMILY = "post"
VARIANT = {c: f"txc_{FAMILY}_{c}" for c in COMPONENTS}
PLAIN = f"txc_{FAMILY}_plain"
ANCHOR = f"txc_batchtopk_{FAMILY}"
SUFFIX = ""


def set_family(family: str) -> None:
    global FAMILY, VARIANT, PLAIN, ANCHOR, SUFFIX
    assert family in ("post", "pre")
    FAMILY = family
    VARIANT = {c: f"txc_{family}_{c}" for c in COMPONENTS}
    PLAIN = f"txc_{family}_plain"
    ANCHOR = f"txc_batchtopk_{family}"
    SUFFIX = "" if family == "post" else "_pre"

# bench -> (datasource, primary, [reported metrics])
BENCHES = {
    "backtracking": ("toy_backtracking_selfexcite_d64", "lambda_recovery",
                     ["lambda_recovery", "nmse", "eauc"]),
    "frequency": ("toy_cyclic_circle_M101_d128", "velocity_recovery",
                  ["velocity_recovery", "nmse", "eauc"]),
    "phasepair": ("toy_phasepair_M101_d24", "sign_recovery",
                  ["sign_recovery", "pair_recovery", "nmse", "eauc"]),
    "recipe_instruction_phase_runs": ("toy_recipe_instruction_d64",
                                      "equality_residual_recovery",
                                      ["equality_residual_recovery",
                                       "phase_recovery", "nmse", "eauc"]),
    "multilane": ("toy_multilane_circle_M101_d24", "multilane_recovery",
                  ["multilane_recovery", "nmse", "eauc"]),
}
RECOVERY_METRICS = {m for _, _, ms in BENCHES.values() for m in ms} - {"nmse", "eauc"}


def _load_dissect(bench):
    p = HERE / "results" / f"{bench}_dissect{SUFFIX}_grid_results.json"
    rows = json.loads(p.read_text())
    bad = [r for r in rows if not r.get("ok")]
    return rows, bad


def _index(rows, kind):
    """(arch, T, k_pos, seed) -> metrics, for trained; (arch, T, seed) for untrained."""
    out = {}
    for r in rows:
        if not r.get("ok") or r.get("kind") != kind:
            continue
        key = ((r["arch"], r["T"], r["k_pos"], r["seed"]) if kind == "trained"
               else (r["arch"], r["T"], r["seed"]))
        out[key] = r["metrics"]
    return out


def _anchor_rows(ds, F):
    out = {}
    for line in open(LEADERBOARD):
        r = json.loads(line)
        if r.get("arch") != ANCHOR or r.get("datasource") != ds:
            continue
        if r["training_cfg"].get("n_steps", 0) == 0:
            continue
        o = r["training_cfg"].get("arch_hparams_override", {})
        if o.get("d_sae") != F:
            continue
        key = (o.get("T"), o.get("k_pos"), r.get("seed"))
        if key[0] in TS and key[1] in KS and key[2] in SEEDS:
            out[key] = r["metrics"]  # later rows overwrite earlier (same cell re-runs)
    return out


def _std(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _delta(metric, variant_val, plain_val):
    if metric in LOWER_BETTER:
        return plain_val - variant_val   # positive = improvement
    return variant_val - plain_val


def _floor(metric):
    return DELTA_FLOOR.get(metric, DELTA_FLOOR["default"])


def analyze():
    report = {"benches": {}, "untrained_guard": {}, "gate_b": {},
              "verdicts": {}, "notes": []}
    for bench, (ds, primary, metrics) in BENCHES.items():
        rows, bad = _load_dissect(bench)
        if bad:
            report["notes"].append(f"{bench}: {len(bad)} FAILED cells (excluded, listed)")
        trained = _index(rows, "trained")
        untrained = _index(rows, "untrained")

        # ── untrained guard: identical across variants per (T, seed) ──
        guard_ok, guard_max = True, 0.0
        for T in TS:
            for s in SEEDS:
                base = untrained.get((PLAIN, T, s))
                if base is None:
                    guard_ok = False
                    continue
                for c in COMPONENTS:
                    other = untrained.get((VARIANT[c], T, s))
                    if other is None:
                        guard_ok = False
                        continue
                    for m in metrics:
                        if m in base and m in other:
                            d = abs(base[m] - other[m])
                            guard_max = max(guard_max, d)
                            if d > 1e-6:
                                guard_ok = False
        report["untrained_guard"][bench] = {"ok": guard_ok, "max_abs_diff": guard_max}

        # ── Gate B: plain vs anchor on the primary metric ──
        anchor = _anchor_rows(ds, {"backtracking": 20, "frequency": 101,
                                   "phasepair": 101,
                                   "recipe_instruction_phase_runs": 20,
                                   "multilane": 101}[bench])
        gb_cells, gb_pass = [], 0
        for T in TS:
            for k in KS:
                pv = [trained.get((PLAIN, T, k, s), {}).get(primary) for s in SEEDS]
                av = [anchor.get((T, k, s), {}).get(primary) for s in SEEDS]
                if any(v is None for v in pv + av):
                    gb_cells.append({"T": T, "k_pos": k, "status": "missing"})
                    continue
                diff = abs(sum(pv) / 3 - sum(av) / 3)
                sd_pool = math.sqrt((_std(pv) ** 2 + _std(av) ** 2) / 2)
                tol = max(2 * sd_pool, 0.10)
                ok = diff <= tol
                gb_pass += ok
                gb_cells.append({"T": T, "k_pos": k, "plain_mean": sum(pv) / 3,
                                 "anchor_mean": sum(av) / 3, "abs_diff": diff,
                                 "tol": tol, "ok": ok})
        # ok iff <=2 evaluable cells exceed tolerance (== >=7/9 when all 9
        # slice cells are dict-feasible; the pre family loses (T=8,k=4) at
        # F=20 — CARD § 9: thresholds counted over PRESENT cells).
        n_eval = sum(1 for c in gb_cells if c.get("status") != "missing")
        gate_b_ok = (n_eval - gb_pass) <= 2 and n_eval > 0
        report["gate_b"][bench] = {"ok": gate_b_ok, "pass_cells": gb_pass,
                                   "n_evaluable": n_eval, "cells": gb_cells}

        # ── component effects ──
        bench_out = {}
        for c in COMPONENTS:
            for m in metrics:
                cells = []
                for T in TS:
                    for k in KS:
                        ds_ = [
                            (trained.get((VARIANT[c], T, k, s), {}).get(m),
                             trained.get((PLAIN, T, k, s), {}).get(m))
                            for s in SEEDS
                        ]
                        if any(a is None or b is None for a, b in ds_):
                            cells.append({"T": T, "k_pos": k, "status": "missing"})
                            continue
                        deltas = [_delta(m, a, b) for a, b in ds_]
                        D = sum(deltas) / 3
                        SE = _std(deltas) / math.sqrt(3)
                        bar = max(2 * SE, _floor(m))
                        cells.append({"T": T, "k_pos": k, "deltas": deltas,
                                      "D": D, "SE": SE, "bar": bar,
                                      "pass_pos": D > bar, "pass_neg": -D > bar})
                n_pos = sum(1 for x in cells if x.get("pass_pos"))
                n_neg = sum(1 for x in cells if x.get("pass_neg"))
                if n_pos >= 2 and n_neg == 0:
                    verdict = "HELPS"
                elif n_neg >= 2 and n_pos == 0:
                    verdict = "HURTS"
                elif n_pos >= 1 and n_neg >= 1:
                    verdict = "MIXED"
                else:
                    verdict = "NEUTRAL"
                if not gate_b_ok:
                    verdict = f"BLOCKED-GATE-B ({verdict})"
                bench_out[f"{c}:{m}"] = {"verdict": verdict, "n_pos": n_pos,
                                         "n_neg": n_neg, "cells": cells}
        # interaction (descriptive)
        inter = {}
        for m in metrics:
            vals = []
            for T in TS:
                for k in KS:
                    Ds = {}
                    for c in COMPONENTS:
                        cell = next((x for x in bench_out[f"{c}:{m}"]["cells"]
                                     if x.get("T") == T and x.get("k_pos") == k
                                     and "D" in x), None)
                        Ds[c] = cell["D"] if cell else None
                    if all(v is not None for v in Ds.values()):
                        vals.append({"T": T, "k_pos": k,
                                     "I": Ds["both"] - Ds["mat"] - Ds["ctr"]})
            inter[m] = vals
        report["benches"][bench] = {"effects": bench_out, "interaction": inter,
                                    "failed_cells": [
                                        {k: r.get(k) for k in
                                         ("arch", "T", "k_pos", "seed", "error")}
                                        for r in bad]}
        report["verdicts"][bench] = {
            key: v["verdict"] for key, v in bench_out.items()}

    out_json = HERE / "results" / f"dissection_table{SUFFIX}.json"
    out_json.write_text(json.dumps(report, indent=1, default=float))

    # ── markdown component table ──
    lines = [f"# Loss-dissection component table — {FAMILY} family "
             "(mechanical, CARD § 5" + (" + § 9)" if SUFFIX else ")"), ""]
    lines.append("| bench | metric | +matryoshka | +contrastive | +both |")
    lines.append("|---|---|---|---|---|")
    for bench, (_, primary, metrics) in BENCHES.items():
        for m in metrics:
            row = [bench, m + (" (primary)" if m == primary else "")]
            for c in COMPONENTS:
                e = report["benches"][bench]["effects"][f"{c}:{m}"]
                best = max((x for x in e["cells"] if "D" in x),
                           key=lambda x: abs(x["D"]), default=None)
                d_str = (f"{e['verdict']} (max|D| {best['D']:+.3f}"
                         f"±{2 * best['SE']:.3f})" if best else e["verdict"])
                row.append(d_str)
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Gates")
    for bench in BENCHES:
        gb = report["gate_b"][bench]
        ug = report["untrained_guard"][bench]
        lines.append(f"- {bench}: Gate B {'PASS' if gb['ok'] else 'FAIL'} "
                     f"({gb['pass_cells']}/{gb.get('n_evaluable', 9)} cells); "
                     f"untrained guard {'PASS' if ug['ok'] else 'FAIL'} "
                     f"(max |diff| {ug['max_abs_diff']:.2e})")
    (HERE / "results" / f"dissection_table{SUFFIX}.md").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        set_family(sys.argv[1])
    analyze()
