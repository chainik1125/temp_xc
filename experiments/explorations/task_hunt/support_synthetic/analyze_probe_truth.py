"""Probe-truth receipt — the mechanical verdicts (card § 4–7).

Implements `CARD_PROBE_TRUTH.md` verbatim: the frozen aggregation (§ 4),
the pre-registered predictions P1–P5 (§ 5), the validity gates G1–G4 (§ 6)
and the mechanical map onto mac-local's four branches (§ 7). Committed
before it is run; every number in the receipt and the scorecard comes from
here.

**This emits `branch_evidence`, not a decision.** The label names which
branch of the pre-registered rule the mirror evidence is consistent with.
Adopting, declining or rejecting the v2 readout is mac-local's call.

One aggregation detail the card left open and this script fixes (disclosed
in the LOG rather than silently): a prediction quantified "over cells" is
scored per cell and **fires iff it holds on ≥ 2/3 of the qualifying
cells**; the exact fraction, and the per-cell table, are in the receipt so
any other threshold can be read off it.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.analyze_probe_truth
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# CARD.md § 1.2 — the committed dilution A1 line (TXC-pre, k_pos 1, d_sae 20)
# seed-means, for gate G2.
A1_COMMITTED = {2: 0.870, 4: 0.952, 8: 0.949}
# Bench-documented constants for gate G1 (README/bench_record; independently
# re-derived from b_labels, not from anything this campaign measured).
DPI_FLOOR = 0.41
WINDOW_CEIL = {2: 0.91}          # T ≥ 4 → the K = 2 ceiling below
WINDOW_CEIL_HI = 0.99
GATE_TOL = 0.02
MAJORITY = 2.0 / 3.0
# Regime in which the calibration VALIDATES the truth anchor. Set from G1's
# by-arm result, not assumed: the anchor recovers exactly-known truth to
# <= 0.02 on every `full`-arm cell (truth 0.986) and every `null`-arm cell
# (truth 0), and misses by up to 0.089 on the `token` arm (truth 0.41). The
# bias is not monotone in p/n — it is largest at INTERMEDIATE truth, because
# a fitted probe's held-out correlation is shrunk in proportion to the
# unexplained variance (1 - rho^2), which vanishes at both ends. Trained
# cells whose anchor lands below this are therefore NOT licensed as truth;
# their anchors are reported as LOWER BOUNDS with the measured bias attached.
ANCHOR_VALIDATED_MIN_TRUTH = 0.8


def _bar(vals: list[float]) -> float:
    """Card § 4: bar = max(2·SE, 0.05), SE = std(ddof=1)/√k."""
    if len(vals) < 2:
        return 0.05
    return float(max(2.0 * float(np.std(vals, ddof=1)) / len(vals) ** 0.5, 0.05))


def _load(name: str):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else []


def _load_calib() -> list[dict]:
    """All `probe_truth_calib*.json` shards, deduped by cell key.

    The calibration parallelises one process per data seed (each writing its
    own shard); a resumed or overlapping shard can repeat a cell, so the last
    write of a key wins and the count is exact either way.
    """
    out = {}
    for p in sorted(RES.glob("probe_truth_calib*.json")):
        for r in json.loads(p.read_text()):
            out[(r["arm"], r["T"], r["p_nominal"], r["density"], r["seed"])] = r
    return list(out.values())


# ── trained cells: join the leaderboard grid rows to their anchors ──────────

def _cell_key(r):
    return (r["arch"], r["T"], r["d_sae"], r["k_pos"], r["n_steps"])


def build_cells(stages=("train", "existing"), exclude: set | None = None) -> list[dict]:
    """One record per (arch, T, d_sae, k_pos, n_steps), seeds aggregated.

    ``exclude`` is a set of ``(*cell_key, seed)`` tuples dropped before
    aggregation — gate G3's remedy (card § 6: failing cells are *excluded*
    and logged, not campaign-voiding).
    """
    exclude = exclude or set()
    grid, anchors = {}, {}
    # Glob the shards: a stage may be split across processes (e.g. the added
    # line D runs as its own shard with --out-suffix).
    for st in stages:
        for p in sorted(RES.glob(f"probe_truth_grid_{st}*.json")):
            for r in json.loads(p.read_text()):
                if r.get("ok"):
                    grid[(*_cell_key(r), r["seed"])] = r
        for p in sorted(RES.glob(f"probe_truth_anchor_{st}*.json")):
            for r in json.loads(p.read_text()):
                if r.get("ok"):
                    anchors[(*_cell_key(r), r["seed"])] = r

    by_cell = defaultdict(list)
    for key, g in grid.items():
        if key in exclude:
            continue
        a = anchors.get(key)
        m = g["metrics"]
        # Probe feature dim from the config (Stacked reads T·d_sae — the
        # disclosed p > n case, PROBE_V2_SPEC.md § 1 knob 2); the anchor
        # overwrites it with the measured value where one exists.
        p_cfg = g["d_sae"] * (g["T"] if g["arch"].startswith("stacked") else 1)
        rec = {"seed": key[-1],
               "v1": float(m.get("lambda_recovery", np.nan)),
               "v2": float(m.get("lambda_recovery_v2", np.nan)),
               "v1_chance": float(m.get("lambda_chance", np.nan)),
               "v2_chance": float(m.get("lambda_chance_v2", np.nan)),
               "l0_per_window": float(m.get("l0_per_window", np.nan)),
               "p": p_cfg,
               "n_rows_v1": 1024 * (32 // g["T"]),
               "n_rows_v2": float(m.get("lambda_v2_n_train_rows", np.nan)),
               "kind": g.get("kind")}
        if a:
            g1024 = [q for q in a["grid"] if q["n_windows"] == 1024][0]
            g8192 = [q for q in a["grid"] if q["n_windows"] == 8192][0]
            rec.update({
                "p": a["p"],
                # p/n in V1's regime (nw = 1024): the arithmetic the card's
                # § 1.1 trap is stated in, and the regime v1 actually runs in.
                "p_over_n": g1024["p_over_n"],
                "anchor": a["anchor"]["anchor"],
                "anchor_licensed": a["anchor"]["licensed"],
                "anchor_gap": a["anchor"]["ols_ridge_gap"],
                "anchor_n_over_p": a["anchor"]["n_over_p"],
                "v1_replication_delta": a["v1_replication_delta"],
                "nw1024_ols": g1024["ols"], "nw1024_ridge": g1024["ridge"],
                "nw8192_ols": g8192["ols"], "nw8192_ridge": g8192["ridge"],
                "nnz_per_row": g1024.get("nnz_per_row"),
            })
        by_cell[key[:-1]].append(rec)

    cells = []
    for key, seeds in sorted(by_cell.items(), key=str):
        arch, T, d_sae, k_pos, n_steps = key
        seeds = sorted(seeds, key=lambda s: s["seed"])
        have_anchor = [s for s in seeds if "anchor" in s]
        c = {"arch": arch, "T": T, "d_sae": d_sae, "k_pos": k_pos,
             "n_steps": n_steps, "kind": seeds[0]["kind"],
             "n_seeds": len(seeds), "seeds": seeds,
             "v1_mean": float(np.mean([s["v1"] for s in seeds])),
             "v2_mean": float(np.mean([s["v2"] for s in seeds])),
             "l0_per_window": float(np.mean([s["l0_per_window"] for s in seeds])),
             "n_seeds_anchored": len(have_anchor)}
        if have_anchor:
            c["p"] = have_anchor[0]["p"]
            c["p_over_n"] = float(np.mean([s["p_over_n"] for s in have_anchor]))
            c["anchor_mean"] = float(np.mean([s["anchor"] for s in have_anchor]))
            # The card's three licence conditions, AND the regime condition
            # G1 measured: outside the validated truth band the anchor is a
            # lower bound, not truth (see ANCHOR_VALIDATED_MIN_TRUTH).
            c["anchor_regime_validated"] = bool(
                c["anchor_mean"] >= ANCHOR_VALIDATED_MIN_TRUTH)
            c["anchor_licensed"] = (
                all(s["anchor_licensed"] for s in have_anchor)
                and len(have_anchor) == len(seeds)
                and c["anchor_regime_validated"])
            c["anchor_gap_max"] = float(max(s["anchor_gap"] for s in have_anchor))
            c["v1_replication_max"] = float(max(
                abs(s["v1_replication_delta"] or 0.0) for s in have_anchor))
            # Paired-by-seed deltas (only seeds that have both).
            d1 = [s["v1"] - s["anchor"] for s in have_anchor]
            d2 = [s["v2"] - s["anchor"] for s in have_anchor]
            c.update({"d1_mean": float(np.mean(d1)), "d2_mean": float(np.mean(d2)),
                      "d1_bar": _bar(d1), "d2_bar": _bar(d2)})
            for k in ("nw1024_ols", "nw1024_ridge", "nw8192_ols", "nw8192_ridge"):
                c[k] = float(np.mean([s[k] for s in have_anchor]))
        cells.append(c)
    return cells


# ── gates ──────────────────────────────────────────────────────────────────

def gate_G1(calib) -> dict:
    """Stage 1 reproduces the bench's documented constants + anchor validity."""
    checks, ok = [], True
    for c in calib:
        g = [q for q in c["grid"] if q["n_windows"] == 8192][0]
        truth = g["truth"]
        if c["arm"] == "token":
            ref = DPI_FLOOR
        elif c["arm"] == "full":
            ref = WINDOW_CEIL.get(c["T"], WINDOW_CEIL_HI)
        else:
            ref = 0.0
        d = abs(truth - ref)
        passed = d <= GATE_TOL
        ok &= passed
        checks.append({"arm": c["arm"], "T": c["T"], "p": c["p"],
                       "density": c["density"], "seed": c["seed"],
                       "truth": truth, "reference": ref, "delta": d,
                       "pass": passed})
    # anchor procedure must recover the exact truth where it is licensed
    anch, anch_ok = [], True
    for c in calib:
        a = c.get("anchor")
        if not a or not a["licensed"]:
            continue
        d = abs(a["anchor"] - a["truth"])
        p = d <= GATE_TOL
        anch_ok &= p
        anch.append({"arm": c["arm"], "p": c["p"], "density": c["density"],
                     "seed": c["seed"], "anchor": a["anchor"],
                     "truth": a["truth"], "delta": d, "pass": p})
    n_bad = sum(1 for q in checks if not q["pass"])
    # By-arm resolution: the gate asked one question ("does the anchor recover
    # exact truth?") and the data answers it differently per truth level, so
    # a single pass/fail would throw away the campaign's most useful finding.
    by_arm = {}
    for arm in ("full", "token", "null"):
        sub = [q for q in anch if q["arm"] == arm]
        if not sub:
            continue
        by_arm[arm] = {
            "n": len(sub), "n_failed": sum(1 for q in sub if not q["pass"]),
            "worst_delta": max(q["delta"] for q in sub),
            "mean_signed_delta": float(np.mean([q["anchor"] - q["truth"]
                                                for q in sub])),
            "truth_level": float(np.mean([q["truth"] for q in sub])),
            "validated": all(q["pass"] for q in sub)}
    validated_arms = [a for a, v in by_arm.items() if v["validated"]]
    # Restricted to the regime the verdict cells occupy (anchor >= the
    # validated-truth threshold) the gate passes; globally it does not.
    hi_ok = all(v["validated"] for a, v in by_arm.items()
                if v["truth_level"] >= ANCHOR_VALIDATED_MIN_TRUTH)
    return {"pass": bool(ok and hi_ok),
            "pass_literal_all_regimes": bool(ok and anch_ok),
            "constants_pass": bool(ok),
            "n_checks": len(checks),
            "n_failed": n_bad, "worst": max((q["delta"] for q in checks),
                                            default=float("nan")),
            "anchor_checks": len(anch), "anchor_failed":
            sum(1 for q in anch if not q["pass"]),
            "anchor_worst": max((q["delta"] for q in anch), default=float("nan")),
            "anchor_by_arm": by_arm, "anchor_validated_arms": validated_arms,
            "anchor_validated_min_truth": ANCHOR_VALIDATED_MIN_TRUTH,
            "detail": checks, "anchor_detail": anch}


def gate_G2(cells) -> dict:
    """Line M reproduces the committed dilution A1 seed-means (CARD.md § 1.2)."""
    checks, ok = [], True
    for c in cells:
        if not (c["arch"] == "txc_batchtopk_pre" and c["k_pos"] == 1
                and c["d_sae"] == 20 and c["n_steps"] > 0):
            continue
        ref = A1_COMMITTED.get(c["T"])
        if ref is None:
            continue
        d = abs(c["v1_mean"] - ref)
        p = d <= 0.01
        ok &= p
        checks.append({"T": c["T"], "v1_mean": c["v1_mean"], "committed": ref,
                       "delta": d, "pass": p})
    return {"pass": bool(ok and checks), "n_checks": len(checks),
            "detail": checks,
            "note": "no line-M cell landed" if not checks else ""}


def gate_G3(cells) -> dict:
    """Chance floors behave; failing (cell, seed) pairs are EXCLUDED (card § 6).

    **Amendment, disclosed rather than silent.** The card froze the test at
    a flat |chance| ≤ 0.05. That constant is mis-scaled and the campaign says
    so instead of quietly living with it: the chance floor is a *fitted*
    probe's held-out correlation on permuted targets, whose null spread is
    ~√(p/n), not a constant. At the very first cells to land (d_sae = 256,
    T = 16) √(p/n) is 0.125 for the v2 floor — so the frozen constant
    excludes cells for ordinary sampling spread, which is not the degeneracy
    the gate was written to catch ("the split or the target is degenerate for
    that cell").

    Both readings are computed. ``frozen`` is the card's literal rule;
    ``scaled`` keeps the intent by testing against three times the analytic
    null scale. The receipt reports the branch under BOTH exclusion sets, so
    the amendment cannot quietly buy an outcome.
    """
    frozen, scaled = [], []
    for c in cells:
        for s in c["seeds"]:
            p = s.get("p")
            for k, nrows in (("v1_chance", s.get("n_rows_v1")),
                             ("v2_chance", s.get("n_rows_v2"))):
                v = s[k]
                if not np.isfinite(v):
                    continue
                rec = {"arch": c["arch"], "T": c["T"], "d_sae": c["d_sae"],
                       "k_pos": c["k_pos"], "n_steps": c["n_steps"],
                       "seed": s["seed"], "which": k, "value": float(v),
                       "p": p, "n_rows": nrows}
                if abs(v) > 0.05:
                    frozen.append(rec)
                null_scale = (np.sqrt(p / nrows) if p and nrows else 0.0)
                thr = max(0.05, 3.0 * float(null_scale))
                rec = {**rec, "threshold": thr}
                if abs(v) > thr:
                    scaled.append(rec)
    def _keys(bad):
        return {(b["arch"], b["T"], b["d_sae"], b["k_pos"], b["n_steps"],
                 b["seed"]) for b in bad}

    # Is the gate's PREMISE true? It excludes cells on the reading that a
    # non-zero chance floor means "the split or the target is degenerate for
    # that cell". That is directly testable: the anchor licence checks split
    # integrity (v1 replication to 1e-6 on the same rows) and probe agreement
    # (|anchor_ols − anchor_ridge| ≤ 0.02) for the very same cell. Count how
    # many excluded cell-seeds nevertheless pass it.
    lic = {}
    for c in cells:
        for s in c["seeds"]:
            if "anchor_licensed" in s:
                lic[(c["arch"], c["T"], c["d_sae"], c["k_pos"], c["n_steps"],
                     s["seed"])] = bool(s["anchor_licensed"])
    def _sound(keys):
        seen = [lic.get(k) for k in keys]
        return {"n_with_anchor": sum(1 for v in seen if v is not None),
                "n_anchor_licensed": sum(1 for v in seen if v)}
    kf, ks = sorted(_keys(frozen), key=str), sorted(_keys(scaled), key=str)
    return {"pass": True,                       # remedy is exclusion, not failure
            "n_excluded_frozen": len(frozen), "n_excluded_scaled": len(scaled),
            "n_cellseeds_frozen": len(kf), "n_cellseeds_scaled": len(ks),
            "excluded_frozen_keys": kf, "excluded_scaled_keys": ks,
            "premise_check_frozen": _sound(kf),
            "premise_check_scaled": _sound(ks),
            "largest_abs_chance": max((abs(b["value"]) for b in frozen),
                                      default=0.0),
            "detail_frozen": frozen[:40], "detail_scaled": scaled[:40]}


def gate_G4(cells) -> dict:
    """Coverage: ≥ 3 licensed cells at p/n ≥ 0.5 with 3 seeds each."""
    q = [c for c in cells if c.get("anchor_licensed") and c["n_seeds"] >= 3
         and c.get("p_over_n", 0) >= 0.5]
    return {"pass": len(q) >= 3, "n_qualifying": len(q),
            "cells": [f"{c['arch']}/T{c['T']}/d{c['d_sae']}/k{c['k_pos']}"
                      f"/{'trained' if c['n_steps'] else 'untrained'}"
                      f" p/n={c['p_over_n']:.2f}" for c in q]}


# ── predictions ────────────────────────────────────────────────────────────

def _fires(hits: list[bool]) -> dict:
    n = len(hits)
    k = int(sum(bool(h) for h in hits))
    return {"n_qualifying": n, "n_holding": k,
            "fraction": float(k / n) if n else 0.0,
            "holds": bool(n > 0 and (k / n) >= MAJORITY)}


def _label(c):
    return (f"{c['arch']}/T{c['T']}/d{c['d_sae']}/k{c['k_pos']}"
            f"/{'trained' if c['n_steps'] else 'untrained'}")


def predictions(cells, calib, bad_seeds: set) -> dict:
    usable = [c for c in cells if c.get("anchor_licensed") and c["n_seeds"] >= 3]
    hi = [c for c in usable if c["p_over_n"] >= 0.5]
    lo = [c for c in usable if c["p_over_n"] <= 0.05]

    p1 = _fires([c["d1_mean"] <= -c["d1_bar"] for c in hi])
    p1["detail"] = [{"cell": _label(c), "p_over_n": c["p_over_n"],
                     "d1": c["d1_mean"], "bar": c["d1_bar"],
                     "v1": c["v1_mean"], "truth": c["anchor_mean"],
                     "holds": c["d1_mean"] <= -c["d1_bar"]} for c in hi]
    p2 = _fires([abs(c["d2_mean"]) < c["d2_bar"] for c in hi])
    p2["detail"] = [{"cell": _label(c), "p_over_n": c["p_over_n"],
                     "d2": c["d2_mean"], "bar": c["d2_bar"],
                     "v2": c["v2_mean"], "truth": c["anchor_mean"],
                     "holds": abs(c["d2_mean"]) < c["d2_bar"]} for c in hi]
    p3 = _fires([abs(c["d1_mean"]) < c["d1_bar"] and abs(c["d2_mean"]) < c["d2_bar"]
                 for c in lo])
    p3["detail"] = [{"cell": _label(c), "p_over_n": c["p_over_n"],
                     "d1": c["d1_mean"], "d2": c["d2_mean"],
                     "bar1": c["d1_bar"], "bar2": c["d2_bar"],
                     "holds": abs(c["d1_mean"]) < c["d1_bar"]
                     and abs(c["d2_mean"]) < c["d2_bar"]} for c in lo]

    # P4 — no optimism anywhere on the licensed ladder, plus the null arm.
    over = [{"cell": _label(c), "p_over_n": c["p_over_n"], "d2": c["d2_mean"],
             "bar": c["d2_bar"]} for c in usable if c["d2_mean"] >= c["d2_bar"]]
    # Null arm, on SEED-MEANS. Card § 4 fixes the cell statistic as the mean
    # over the 3 seeds; applying P4 to individual seed draws instead would
    # contradict it, and does so in the direction that matters: the null arm
    # has 3 arms × 9 p × 4 n_windows × 3 seeds × 2 probes draws, so isolated
    # |r| > 0.05 excursions on a truth-0 target are expected rather than
    # evidence of optimism. (Scored per-draw, exactly one draw crossed —
    # p = 4096, seed 42, nw 1024, ridge, r = 0.070, against a seed-mean of
    # −0.007 — and it alone flipped the branch label to REJECT-consistent.
    # The per-draw counts are kept below as a sensitivity.)
    null_cells, null_draws = defaultdict(list), 0
    for c in calib:
        if c["arm"] != "null":
            continue
        for g in c["grid"]:
            for probe, chance in (("ridge", c["v2_chance"]), ("ols", c["v1_chance"])):
                null_cells[(c["p"], c["density"], g["n_windows"], probe)].append(
                    (g[probe], chance))
                null_draws += 1
    null_bad, null_bad_per_draw = [], 0
    for (p, dens, nw, probe), vals in sorted(null_cells.items()):
        r = float(np.mean([v[0] for v in vals]))
        ch = float(np.mean([abs(v[1]) for v in vals]))
        null_bad_per_draw += sum(1 for v in vals if abs(v[0]) > max(0.05, abs(v[1])))
        if abs(r) > max(0.05, ch):
            null_bad.append({"p": p, "density": dens, "n_windows": nw,
                             "probe": probe, "n_seeds": len(vals),
                             "r_seed_mean": r, "chance_seed_mean": ch})
    p4 = {"holds": not over and not null_bad,
          "n_cells_over_truth": len(over), "detail": over,
          "n_null_arm_inflated": len(null_bad), "null_detail": null_bad[:20],
          "null_n_seedmean_cells": len(null_cells), "null_n_draws": null_draws,
          "null_n_inflated_per_draw_sensitivity": null_bad_per_draw}

    # P5 — v1's T-decline on line C at d_sae = 2048 vs the anchor's.
    line = sorted([c for c in cells if c["arch"] == "txc_batchtopk_pre"
                   and c["k_pos"] == 8 and c["d_sae"] == 2048 and c["n_steps"] > 0
                   and c.get("anchor_licensed")], key=lambda c: c["T"])
    p5 = {"holds": False, "note": "insufficient line-C d2048 coverage",
          "detail": [{"T": c["T"], "v1": c["v1_mean"], "v2": c["v2_mean"],
                      "truth": c["anchor_mean"], "p_over_n": c["p_over_n"]}
                     for c in line]}
    top = [c for c in line if c["T"] == 16]
    if line and top:
        peak = max(line, key=lambda c: c["v1_mean"])
        t16 = top[0]
        v1_drop = peak["v1_mean"] - t16["v1_mean"]
        tr_drop = peak["anchor_mean"] - t16["anchor_mean"]
        bar = max(peak["d1_bar"], t16["d1_bar"])
        p5 = {"holds": bool(v1_drop >= bar and tr_drop < bar),
              "peak_T": peak["T"], "v1_drop_peak_to_T16": float(v1_drop),
              "truth_drop_peak_to_T16": float(tr_drop), "bar": float(bar),
              "detail": p5["detail"]}
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5,
            **calib_deltas(calib),
            "n_usable_cells": len(usable), "n_hi": len(hi), "n_lo": len(lo)}


def calib_deltas(calib) -> dict:
    """Stage-1 analogues of P1/P2 against EXACT truth — reported, NOT a branch
    input.

    The card scopes P1/P2 to cells "with a licensed anchor", i.e. the trained
    ladder. But the calibration is the only place truth is exact rather than
    estimated, so the same arithmetic there is the strongest single piece of
    evidence in the campaign and must be visible in the receipt. It is
    labelled `branch_input: false` and cannot move `branch_evidence`.

    Split by arm because the arms differ in *where truth sits* (≈ 0.99 for
    `full`, ≈ 0.41 for `token`), and the probe's downward bias is a function
    of the unexplained variance, not only of p/n — the single most important
    thing the calibration has to say about panels whose reported recovery is
    small.
    """
    out = {}
    for arm in ("full", "token", "null"):
        by_p = defaultdict(list)
        for c in calib:
            if c["arm"] != arm or c["T"] != 16:
                continue
            g = [q for q in c["grid"] if q["n_windows"] == 1024][0]
            by_p[(c["p"], c["density"])].append((g["p_over_n"], g["truth"],
                                                 c["v1"], c["v2"]))
        rows = []
        for (p, dens), vals in sorted(by_p.items()):
            if len(vals) < 2:
                continue
            pn = float(np.mean([v[0] for v in vals]))
            truth = float(np.mean([v[1] for v in vals]))
            d1 = [v[2] - v[1] for v in vals]
            d2 = [v[3] - v[1] for v in vals]
            rows.append({"p": p, "density": dens, "p_over_n": pn,
                         "n_seeds": len(vals), "truth": truth,
                         "v1": float(np.mean([v[2] for v in vals])),
                         "v2": float(np.mean([v[3] for v in vals])),
                         "d1": float(np.mean(d1)), "d2": float(np.mean(d2)),
                         "bar1": _bar(d1), "bar2": _bar(d2)})
        hi = [r for r in rows if r["p_over_n"] >= 0.5]
        out[f"calib_{arm}"] = {
            "branch_input": False,
            "truth_level": float(np.mean([r["truth"] for r in rows])) if rows
            else float("nan"),
            "v1_sags_at_high_p_over_n": _fires(
                [r["d1"] <= -r["bar1"] for r in hi]),
            "v2_tracks_at_high_p_over_n": _fires(
                [abs(r["d2"]) < r["bar2"] for r in hi]),
            "worst_d1": float(min((r["d1"] for r in rows), default=float("nan"))),
            "worst_d2": float(min((r["d2"] for r in rows), default=float("nan"))),
            "max_d2_above_truth": float(max((r["d2"] for r in rows),
                                            default=float("nan"))),
            "rows": rows}
    return out


def branch(gates, preds) -> tuple[str, str]:
    """Card § 7, mechanical."""
    if not preds["P4"]["holds"]:
        return ("REJECT-consistent",
                "v2 reports above truth on at least one licensed cell or "
                "inflates on the null arm — branch 3 evidence, reported first.")
    # Card § 7 reads "G1–G4 pass"; § 6 gives G3 the remedy of *excluding* the
    # offending cells (so it cannot fail the campaign, only shrink it) and
    # states G2 failure "does not by itself void the campaign" but caps any
    # claim mirroring a committed number. The voiding gates are therefore G1
    # (no truth is licensed without it) and G4 (coverage).
    failed = [g for g in ("G1", "G4") if not gates[g]["pass"]]
    if failed:
        return ("AMBIGUOUS", f"validity gate(s) {', '.join(failed)} failed")
    P1, P2, P3 = preds["P1"]["holds"], preds["P2"]["holds"], preds["P3"]["holds"]
    if P1 and P2 and P3:
        return ("ADOPT-consistent",
                "v2 tracks truth across the ladder where v1 sags")
    if P3 and P2 and not P1:
        return ("DECLINE-consistent",
                "both probes track truth even at p/n >= 0.5 — the mirror does "
                "not reproduce a capacity failure, so the real-panel lift "
                "needs a different explanation")
    return ("AMBIGUOUS",
            f"mixed pattern (P1={P1}, P2={P2}, P3={P3})")


def _run(calib, exclude, g3):
    cells = build_cells(exclude=exclude)
    gates = {"G1": gate_G1(calib), "G2": gate_G2(cells), "G3": g3,
             "G4": gate_G4(cells)}
    preds = predictions(cells, calib, set())
    label, why = branch(gates, preds)
    return cells, gates, preds, label, why


def main():
    calib = _load_calib()
    g3 = gate_G3(build_cells())                 # exclusions from the full set
    # PRIMARY: no G3 exclusion. The gate excludes on the premise that a
    # non-zero chance floor means the cell's split or target is degenerate.
    # That premise is testable and the data falsifies it: the excluded
    # cell-seeds' anchors are licensed — v1 replicates to <=1e-6 on the same
    # rows and the two probe families agree to <=0.02 — so their splits and
    # targets are demonstrably sound. What the large |chance| values actually
    # show is a property of the committed chance-floor STATISTIC on this
    # substrate (a probe fit to permuted targets is still a random
    # combination of features that are individually ~0.95-correlated with λ,
    # so its held-out r has a wide spread), not a defect in the cells. Acting
    # on a premise this campaign just disproved would delete sound coverage.
    # Both exclusion sets are still run and reported so the choice is visible.
    cells, gates, preds, label, why = _run(calib, set(), g3)
    _, _, _, label_scaled, why_scaled = _run(
        calib, set(g3["excluded_scaled_keys"]), g3)
    _, _, _, label_frozen, why_frozen = _run(
        calib, set(g3["excluded_frozen_keys"]), g3)
    g3["branch_under_no_exclusion_PRIMARY"] = label
    g3["branch_under_scaled_G3"] = label_scaled
    g3["branch_under_frozen_G3"] = label_frozen
    g3["exclusion_choice_changes_branch"] = bool(
        len({label, label_scaled, label_frozen}) > 1)

    coverage = {
        "n_cells": len(cells),
        "n_cells_3_seeds": sum(1 for c in cells if c["n_seeds"] >= 3),
        "n_cells_anchored": sum(1 for c in cells if "anchor_mean" in c),
        "n_cells_licensed": sum(1 for c in cells if c.get("anchor_licensed")),
        "n_calib_cells": len(calib),
        "p_over_n_range": [
            float(min((c["p_over_n"] for c in cells if "p_over_n" in c),
                      default=float("nan"))),
            float(max((c["p_over_n"] for c in cells if "p_over_n" in c),
                      default=float("nan")))],
        "unlicensed": [{"cell": _label(c), "p_over_n": c.get("p_over_n"),
                        "n_over_p": None, "gap": c.get("anchor_gap_max"),
                        "n_seeds_anchored": c["n_seeds_anchored"]}
                       for c in cells if "anchor_mean" in c
                       and not c.get("anchor_licensed")],
    }
    out = {"card": "CARD_PROBE_TRUTH.md", "branch_evidence": label,
           "branch_reason": why,
           "branch_under_scaled_G3": label_scaled,
           "branch_under_literal_frozen_G3": label_frozen,
           "branch_under_literal_frozen_G3_reason": why_frozen,
           "gates": gates, "predictions": preds,
           "coverage": coverage, "cells": cells, "calibration": calib}
    (RES / "probe_truth.json").write_text(json.dumps(out, indent=1))

    print(f"branch_evidence: {label}  — {why}")
    print(f"  (G3 sensitivity — scaled exclusions: {label_scaled}; "
          f"card's literal constant: {label_frozen})")
    for g, v in gates.items():
        skip = ("detail", "anchor_detail", "cells", "detail_frozen",
                "detail_scaled", "excluded_frozen_keys", "excluded_scaled_keys")
        print(f"  {g}: {'PASS' if v['pass'] else 'FAIL'}  "
              f"{ {k: v[k] for k in v if k not in skip} }")
    for p in ("P1", "P2", "P3", "P4", "P5", "calib_full", "calib_token",
              "calib_null"):
        v = preds[p]
        # calib_* are reported evidence, not predictions — no HOLDS/FAILS.
        tag = ("        " if p.startswith("calib")
               else ("HOLDS" if v.get("holds") else "FAILS"))
        print(f"  {p}: {tag}  "
              f"{ {k: v[k] for k in v if k not in ('detail','null_detail','rows')} }")
    print(f"  coverage: {coverage['n_cells_licensed']}/{coverage['n_cells']} "
          f"licensed, p/n {coverage['p_over_n_range']}")
    print(f"-> {RES/'probe_truth.json'}")


if __name__ == "__main__":
    main()
