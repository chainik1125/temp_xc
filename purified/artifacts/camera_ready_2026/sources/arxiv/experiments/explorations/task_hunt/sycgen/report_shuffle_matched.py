"""SHUFFLE ABLATION — merge shards, apply the frozen rules, print the verdict.

Mechanises `SHUFFLE_MATCHED_CARD.md` so the verdict cannot drift after
the numbers exist. Every threshold here was written before any cell ran
(git history is the receipt); this file only *applies* them.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.report_shuffle_matched

THE CLAIM: **TXC gap > STACKED gap at matched measured budget.** Never
against pooled — mean-pooling is permutation-invariant, so pooled's gap
is exactly 0 and "beating" it is a mathematical identity (card §1).

THE COMPARATOR RULE IS A BRACKET, NOT A SINGLE SIDE (card §2b). Item 6
took "the best point with l0 <= TXC's l0", which is defensible in words
and biased in arithmetic: on a grid whose steps are 40-75% it silently
compares against a 38%-cheaper baseline. That moved a shipped verdict
from "above 3/4" to "above 2/4". Here: report the point below AND the
cheapest point above, interpolate to TXC's exact l0, and print the
budget ratio — **if the ratio is not ~1.0 the word "matched" is not
earned.** If the two ends disagree, that IS the finding; it is printed,
not resolved by picking.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

GAP = "gap_fixedprobe"            # PRIMARY instrument (card §4)
GAP2 = "gap_refitprobe"           # SECONDARY, declared secondary pre-data
L0 = "realized_l0_per_window_ordered"


def load() -> dict:
    """Merge shards. A missing shard is named, never silently averaged over."""
    single = RES / "shuffle_matched.json"
    shards = sorted(RES.glob("shuffle_matched.shard*.json"))
    rows, gates, timings, seen = [], [], [], set()
    if shards:
        for p in shards:
            d = json.loads(p.read_text())
            i, n = d.get("shard", [None, None])
            seen.add(i)
            rows += d["rows"]; gates += d["gates"]; timings += d.get("timings", [])
        missing = [i for i in range(n) if i not in seen] if shards else []
        if missing:
            raise SystemExit(
                f"REFUSING TO REPORT: shards {missing} of {n} are absent. "
                "A verdict computed on a partial grid is not the verdict "
                "the card pre-registered.")
    elif single.exists():
        d = json.loads(single.read_text())
        rows, gates, timings = d["rows"], d["gates"], d.get("timings", [])
        if d.get("smoke"):
            raise SystemExit("REFUSING: this is SMOKE output (random "
                             "activations). It is not a result.")
    else:
        raise SystemExit(f"no results in {RES}")
    return {"rows": rows, "gates": gates, "timings": timings}


def gate_report(gates: list) -> bool:
    """Card §4b. The INPUT-side gate; pooled's zero cannot do this job."""
    print("\n=== INSTRUMENT GATE (card §4b) — identity-row count vs band ===")
    ok = True
    for g in sorted(gates, key=lambda g: (g["T"], g["seed"], g["draw"])):
        lo, hi = g["band"]
        good = lo <= g["identity_rows"] <= hi
        ok &= good
        bc = " [by construction]" if g.get("by_construction") else ""
        print(f"  T{g['T']:<3} s{g['seed']:<3} {g['draw']:7s} "
              f"identity {g['identity_rows']:>5}/{g['n_rows']:<6} "
              f"band {lo}..{hi}  {'PASS' if good else 'FAIL'}{bc}")
    print(f"  -> gate {'PASS' if ok else 'FAIL'}")
    return ok


def pooled_identity(rows: list) -> bool:
    """Card §1. Pooled's gap must be 0. NOT a shuffle check — an arm check.

    A non-zero value here means the pooled arm has become
    position-sensitive and the framing of the whole lane is wrong.
    """
    p = [r for r in rows if r["arm"] == "pooled"]
    worst = max((abs(r[GAP]) for r in p), default=0.0)
    ok = worst < 1e-8
    print(f"\n=== POOLED IDENTITY (card §1) === max |gap| {worst:.3e} "
          f"over {len(p)} rows -> {'PASS' if ok else 'VOID'}")
    if not ok:
        print("  ⚑ pooled is NOT permutation-invariant. The run is VOID.")
    return ok


def l0_invariance(rows: list) -> bool:
    """Card §6 — a PREDICTION, therefore measured rather than asserted."""
    bad = [r for r in rows if r["arm"] in ("pooled", "stacked")
           and abs(r[L0] - r["realized_l0_per_window_shuffled"]) > 1e-6]
    print(f"=== SAE l0 PERMUTATION-INVARIANCE (card §6) === "
          f"{len(bad)} violations -> {'PASS' if not bad else 'FAIL'}")
    return not bad


def _bracket(cands: list, target: float) -> dict:
    """Card §2b: below + cheapest above + interpolation to TXC's exact l0."""
    below = [c for c in cands if c["l0"] <= target]
    above = [c for c in cands if c["l0"] > target]
    lo = max(below, key=lambda c: c["l0"]) if below else None
    hi = min(above, key=lambda c: c["l0"]) if above else None
    out = {"below": lo, "above": hi, "target_l0": target}
    if lo and hi and hi["l0"] > lo["l0"]:
        w = (target - lo["l0"]) / (hi["l0"] - lo["l0"])
        out["interp_gap"] = lo["gap"] + w * (hi["gap"] - lo["gap"])
        out["bracket_width_frac"] = (hi["l0"] - lo["l0"]) / target
    elif lo:
        out["interp_gap"] = lo["gap"]
        out["bracket_width_frac"] = None
    elif hi:
        out["interp_gap"] = hi["gap"]
        out["bracket_width_frac"] = None
    return out


def _monotone(cands: list) -> tuple[bool, float]:
    """Interpolation is only valid on a monotone arm — printed, not assumed."""
    s = sorted(cands, key=lambda c: c["l0"])
    worst = 0.0
    for a, b in zip(s, s[1:]):
        worst = min(worst, b["gap"] - a["gap"])
    return worst >= -2e-3, worst


def verdict(rows: list, draw: str, gapkey: str) -> list:
    """Apply the frozen (a)-(d) rules. No threshold is chosen here."""
    out = []
    Ts = sorted({r["T"] for r in rows})
    for T in Ts:
        per_seed, ratios, widths, monos = [], [], [], []
        for seed in sorted({r["seed"] for r in rows}):
            def sel(arm, weights="trained"):
                return [r for r in rows if r["arm"] == arm and r["T"] == T
                        and r["seed"] == seed and r["draw"] == draw
                        and r["weights"] == weights]
            txc = sel("txc")
            if not txc:
                continue
            tgt = txc[0][L0]
            cands = [{"l0": r[L0], "gap": r[gapkey], "k": r["k_tok"]}
                     for r in sel("stacked")]
            if not cands:
                continue
            mono, worst = _monotone(cands)
            monos.append((mono, worst))
            br = _bracket(cands, tgt)
            if "interp_gap" not in br:
                continue
            per_seed.append(txc[0][gapkey] - br["interp_gap"])
            widths.append(br.get("bracket_width_frac"))
            used = br["below"] or br["above"]
            ratios.append(used["l0"] / tgt if tgt else float("nan"))
            # A2 twin gate: trained gap must exceed the untrained twin's.
            tw = sel("txc", "untrained")
            per_seed[-1] = (per_seed[-1],
                            txc[0][gapkey] - (tw[0][gapkey] if tw else None)
                            if tw else None)
        if not per_seed:
            continue
        deltas = [d for d, _ in per_seed]
        twins = [t for _, t in per_seed if t is not None]
        sd = st.pstdev(deltas) if len(deltas) > 1 else 0.0
        mean = sum(deltas) / len(deltas)
        signs_ok = all(d > 0 for d in deltas)
        margin_ok = abs(mean) > sd
        twin_ok = bool(twins) and all(t > 0 for t in twins)
        if signs_ok and margin_ok and twin_ok:
            state = "(a) TXC ABOVE"
        elif mean < -sd and all(d < 0 for d in deltas):
            state = "(c) TXC BELOW"
        elif twins and not twin_ok:
            state = "(b) ARCHITECTURAL (twin gate failed)"
        else:
            state = "(d) INDISTINGUISHABLE"
        out.append({"T": T, "mean_delta": mean, "sd": sd, "deltas": deltas,
                    "signs_ok": signs_ok, "margin_ok": margin_ok,
                    "twin_ok": twin_ok, "state": state,
                    "budget_ratios": ratios, "bracket_widths": widths,
                    "monotone": all(m for m, _ in monos),
                    "worst_mono": min((w for _, w in monos), default=0.0)})
    return out


def main() -> int:
    d = load()
    rows, gates = d["rows"], d["gates"]
    print(f"[report] {len(rows)} rows, {len(gates)} gate receipts")

    g_ok = gate_report(gates)
    p_ok = pooled_identity(rows)
    l_ok = l0_invariance(rows)
    if not (g_ok and p_ok):
        raise SystemExit("\n⛔ INSTRUMENT CHECKS FAILED — no verdict is "
                         "computed. A fast wrong answer is the worst "
                         "outcome available here.")
    if not l_ok:
        print("  ⚑ l0 invariance violated — disclosed, verdict continues "
              "under protest (it bears on the budget axis, not the gate).")

    for draw in ("plain", "redraw"):
        for label, gk in (("PRIMARY fixed-probe", GAP),
                          ("SECONDARY refit-probe", GAP2)):
            v = verdict(rows, draw, gk)
            if not v:
                continue
            print(f"\n=== {label} | draw={draw} "
                  f"{'(cross-T statements allowed)' if draw == 'redraw' else '(plain: carries 1-1/T!, fixed-T only)'} ===")
            print(f"  {'T':>3} {'delta':>9} {'sd':>8} {'ratio':>7} "
                  f"{'width':>7} {'mono':>5}  verdict")
            for r in v:
                rr = [x for x in r["budget_ratios"] if x == x]
                ratio = sum(rr) / len(rr) if rr else float("nan")
                w = [x for x in r["bracket_widths"] if x is not None]
                width = sum(w) / len(w) if w else float("nan")
                flag = "" if 0.9 <= ratio <= 1.1 else "  ⚑ NOT MATCHED"
                print(f"  {r['T']:>3} {r['mean_delta']:>+9.4f} {r['sd']:>8.4f} "
                      f"{ratio:>7.3f} {width:>7.3f} "
                      f"{'yes' if r['monotone'] else 'NO':>5}  "
                      f"{r['state']}{flag}")
    print("\nPTR — pending team review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
